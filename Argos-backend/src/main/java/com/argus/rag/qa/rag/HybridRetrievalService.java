package com.argus.rag.qa.rag;

import com.argus.rag.common.exception.BusinessException;
import com.argus.rag.engine.pgvector.PgVectorRetriever;
import com.argus.rag.ingestion.mapper.DocumentChunkMapper;
import com.argus.rag.ingestion.model.entity.DocumentChunkEntity;
import com.argus.rag.qa.model.EvidenceLevel;
import com.argus.rag.qa.model.QueryPlanResult;
import com.argus.rag.qa.service.QueryPlanningService;
import com.argus.rag.engine.elasticsearch.ElasticsearchService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.*;

/**
 * 混合文档切片检索服务。
 * <p>
 * 核心检索引擎，融合向量语义检索和关键词检索两个通道，
 * 通过 RRF（Reciprocal Rank Fusion）算法融合排序，
 * 并支持邻居窗口扩展和证据充分度评估。
 * </p>
 * <h3>检索流程</h3>
 * <ol>
 * <li>查询规划：由 {@link QueryPlanningService} 分析问题并生成检索语句</li>
 * <li>双通道检索：向量检索 + 关键词检索</li>
 * <li>RRF 融合排序：合并两通道结果并按 RRF 评分排序</li>
 * <li>聚类分组：将连续的切片聚合为类簇</li>
 * <li>邻居窗口扩展：扩展上下文窗口以提供更完整的证据</li>
 * <li>证据充分度评估：根据检索结果评估证据质量</li>
 * </ol>
 */
@Service
@Slf4j
public class HybridRetrievalService {

    /** 默认邻居窗口大小（向前向后各扩展的切片数） */
    private static final int DEFAULT_NEIGHBOR_WINDOW = 1;
    /** 每个检索通道返回的最大候选数 */
    private static final int CHANNEL_TOP_K = 50;
    /**
     * RRF 融合算法的 k 参数。
     */
    private static final int RRF_K = 60;
    /** 向量检索通道权重 */
    private static final double VECTOR_WEIGHT = 0.3;
    /** 关键词检索通道权重 */
    private static final double KEYWORD_WEIGHT = 0.7;

    /** 向量检索适配器 */
    private final PgVectorRetriever vectorRetriever;
    /** Elasticsearch 关键词检索服务 */
    private final ElasticsearchService elasticsearchService;
    /** 文档切片数据访问层 */
    private final DocumentChunkMapper documentChunkMapper;
    /** 查询规划服务 */
    private final QueryPlanningService queryPlanningService;
    /** 邻居窗口大小 */
    private final int neighborWindow;

    /**
     * 主构造函数（Spring 注入），使用默认邻居窗口。
     */
    @Autowired
    public HybridRetrievalService(
            PgVectorRetriever vectorRetriever,
            ElasticsearchService elasticsearchService,
            DocumentChunkMapper documentChunkMapper,
            QueryPlanningService queryPlanningService) {
        this(
                vectorRetriever,
                elasticsearchService,
                documentChunkMapper,
                queryPlanningService,
                DEFAULT_NEIGHBOR_WINDOW);
    }

    /**
     * 构造函数，支持自定义邻居窗口大小。
     */
    public HybridRetrievalService(
            PgVectorRetriever vectorRetriever,
            ElasticsearchService elasticsearchService,
            DocumentChunkMapper documentChunkMapper,
            QueryPlanningService queryPlanningService,
            int neighborWindow) {
        this.vectorRetriever = vectorRetriever;
        this.elasticsearchService = elasticsearchService;
        this.documentChunkMapper = documentChunkMapper;
        this.queryPlanningService = queryPlanningService;
        this.neighborWindow = Math.max(0, neighborWindow);
    }

    /**
     * 执行混合检索，返回包含证据文档和证据等级的完整检索结果。
     * <p>
     * 流程：查询规划 → 双通道检索 → RRF 融合排序 → 聚类分组 → 窗口扩展 → 证据评估。
     * </p>
     *
     * @param groupId  群组 ID，限定检索范围
     * @param question 用户问题
     * @param topK     返回的最大文档数
     * @return 检索证据束
     */
    public RetrievedEvidenceBundle retrieve(Long groupId, String question, int topK) {
        long startNano = System.nanoTime();
        Long validGroupId = requirePositiveGroupId(groupId);
        String normalizedQuestion = requireQuestion(question);
        int validTopK = topK > 0 ? topK : 5;

        // 1. 查询规划：LLM 决定直接/改写/拆解
        log.info("混合检索开始: groupId={}, topK={}, questionLength={}", validGroupId, validTopK,
                normalizedQuestion.length());
        QueryPlanResult queryPlan = queryPlanningService.plan(normalizedQuestion);
        log.info("查询规划完成: groupId={}, strategy={}, queries={}", validGroupId, queryPlan.strategy(),
                queryPlan.queries());

        // 2. 双路检索：对每个规划查询分别走向量和关键词通道，合并候选
        Map<Long, RetrievalCandidate> candidates = new LinkedHashMap<>();
        for (String plannedQuery : queryPlan.queries()) {
            mergeVectorHits(candidates, validGroupId, plannedQuery);
            mergeKeywordHits(candidates, validGroupId, plannedQuery);
        }

        long vectorHitCount = candidates.values().stream().filter(c -> c.vectorMatched).count();
        long keywordHitCount = candidates.values().stream().filter(c -> c.keywordMatched).count();
        log.info("双路检索完成: groupId={}, candidateCount={}, vectorHits={}, keywordHits={}",
                validGroupId, candidates.size(), vectorHitCount, keywordHitCount);

        if (candidates.isEmpty()) {
            long elapsedMs = (System.nanoTime() - startNano) / 1_000_000;
            log.info("混合检索结果为空: groupId={}, elapsedMs={}", validGroupId, elapsedMs);
            return RetrievedEvidenceBundle.empty();
        }

        // 3. RRF 融合排序 + 聚类（合并同文档连续 chunk）
        List<RetrievalCandidate> rankedCandidates = candidates.values().stream()
                .sorted(Comparator
                        .comparingDouble(RetrievalCandidate::rankingScore).reversed()
                        .thenComparing(RetrievalCandidate::chunkId))
                .limit(validTopK)
                .toList();
        List<RetrievalCluster> rankedClusters = buildClusters(rankedCandidates);
        log.info("RRF融合排序完成: groupId={}, rankedCandidates={}, clusters={}",
                validGroupId, rankedCandidates.size(), rankedClusters.size());

        // 4. 证据组装：查询 chunk 元数据，按聚类构建 Document（含邻居窗口扩展）
        List<Long> chunkIds = rankedCandidates.stream().map(RetrievalCandidate::chunkId).toList();
        Map<Long, Map<String, Object>> rowByChunkId = indexRows(
                documentChunkMapper.selectQaReadyChunksByIds(validGroupId, chunkIds));
        Map<Long, List<DocumentChunkEntity>> chunkWindowCache = new LinkedHashMap<>();
        List<Document> documents = new ArrayList<>();
        int evidenceIndex = 1;
        for (RetrievalCluster cluster : rankedClusters) {
            Map<String, Object> row = rowByChunkId.get(cluster.primaryChunkId());
            if (row == null) {
                continue;
            }
            Document document = toDocument("E" + evidenceIndex, row, cluster, chunkWindowCache);
            if (document == null) {
                continue;
            }
            documents.add(document);
            evidenceIndex++;
        }
        if (documents.isEmpty()) {
            long elapsedMs = (System.nanoTime() - startNano) / 1_000_000;
            log.info("混合检索证据组装为空: groupId={}, elapsedMs={}", validGroupId, elapsedMs);
            return RetrievedEvidenceBundle.empty();
        }
        // 5. 证据等级评估：NONE/WEAK/PARTIAL/SUFFICIENT
        EvidenceLevel evidenceLevel = evaluateEvidenceLevel(documents);
        long elapsedMs = (System.nanoTime() - startNano) / 1_000_000;
        log.info("混合检索完成: groupId={}, evidenceCount={}, evidenceLevel={}, elapsedMs={}",
                validGroupId, documents.size(), evidenceLevel, elapsedMs);
        return new RetrievedEvidenceBundle(documents, evidenceLevel, buildEvidenceGuidance(evidenceLevel));
    }

    /** 合并向量检索结果到候选集合，累加 RRF 评分 */
    private void mergeVectorHits(
            Map<Long, RetrievalCandidate> candidates,
            Long groupId,
            String query) {
        // 1. 向量通道检索 top-K
        List<PgVectorRetriever.VectorHit> vectorHits = vectorRetriever.search(groupId, query, CHANNEL_TOP_K);
        for (int index = 0; index < vectorHits.size(); index++) {
            PgVectorRetriever.VectorHit hit = vectorHits.get(index);
            // 2 遍历向量检索结果，每个chunkId包装为一个检索候选项
            RetrievalCandidate candidate = candidates.computeIfAbsent(
                    hit.chunkId(),
                    chunkId -> RetrievalCandidate.fromVectorHit(hit));
            // 3. 给候选项计算RRF分数：1/(rank+60)
            candidate.mergeVectorHit(hit, index + 1);
        }
    }

    /** 合并关键词检索结果到候选集合，累加 RRF 评分 */
    private void mergeKeywordHits(
            Map<Long, RetrievalCandidate> candidates,
            Long groupId,
            String query) {
        // 1. 关键词通道检索 top-K
        List<ElasticsearchService.KeywordHit> keywordHits = elasticsearchService.search(groupId,
                query, CHANNEL_TOP_K);
        for (int index = 0; index < keywordHits.size(); index++) {
            ElasticsearchService.KeywordHit hit = keywordHits.get(index);
            // 2. 按 chunkId 去重，已有则复用（与向量通道共享同一 candidates Map）
            RetrievalCandidate candidate = candidates.computeIfAbsent(
                    hit.chunkId(),
                    chunkId -> RetrievalCandidate.fromKeywordHit(hit));
            // 3. 累加 RRF 分数，同时被两个通道命中的 chunk 分数叠加，排名更靠前
            candidate.mergeKeywordHit(hit, index + 1);
        }
    }

    /** 将数据库查询结果按 chunkId 索引，便于快速查找 */
    private Map<Long, Map<String, Object>> indexRows(List<Map<String, Object>> rows) {
        Map<Long, Map<String, Object>> rowByChunkId = new LinkedHashMap<>();
        for (Map<String, Object> row : rows) {
            rowByChunkId.put(requireLong(getValue(row, "chunkId"), "chunkId"), row);
        }
        return rowByChunkId;
    }

    /** 将检索候选和数据库数据组装为 Spring AI 的 Document 对象 */
    private Document toDocument(
            String evidenceId,
            Map<String, Object> row,
            RetrievalCluster cluster,
            Map<Long, List<DocumentChunkEntity>> chunkWindowCache) {
        Long documentId = requireLong(getValue(row, "documentId"), "documentId");
        Integer chunkIndex = requireInteger(getValue(row, "chunkIndex"), "chunkIndex");
        if (!documentId.equals(cluster.documentId()) || !chunkIndex.equals(cluster.primaryChunkIndex())) {
            throw new BusinessException("检索结果与文档切片不一致");
        }
        Long chunkId = requireLong(getValue(row, "chunkId"), "chunkId");
        String fileName = requireText(getValue(row, "fileName"), "fileName");
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("evidenceId", evidenceId);
        metadata.put("groupId", requireLong(getValue(row, "groupId"), "groupId"));
        metadata.put("documentId", documentId);
        metadata.put("chunkId", chunkId);
        metadata.put("chunkIndex", chunkIndex);
        metadata.put("primaryChunkId", cluster.primaryChunkId());
        metadata.put("primaryChunkIndex", cluster.primaryChunkIndex());
        metadata.put("startChunkIndex", cluster.expandedStartChunkIndex(neighborWindow));
        metadata.put("endChunkIndex", cluster.expandedEndChunkIndex(neighborWindow));
        metadata.put("fileName", fileName);
        double normalizedScore = normalizeScore(cluster.rankingScore());
        metadata.put("score", normalizedScore);
        metadata.put("retrievalSource", cluster.source());
        metadata.put("vectorScore", cluster.vectorScore());
        metadata.put("keywordScore", cluster.keywordScore());
        metadata.put("hybridScore", cluster.rankingScore());
        String evidenceText = buildEvidenceWindow(row, cluster, chunkWindowCache);
        if (!StringUtils.hasText(evidenceText)) {
            return null;
        }
        return Document.builder()
                .id(evidenceId)
                .text(evidenceText)
                .metadata(metadata)
                .build();
    }

    /** 构建证据窗口文本：根据聚类的切片范围和邻居窗口拼接切片文本 */
    private String buildEvidenceWindow(
            Map<String, Object> row,
            RetrievalCluster cluster,
            Map<Long, List<DocumentChunkEntity>> chunkWindowCache) {
        Long groupId = requireLong(getValue(row, "groupId"), "groupId");
        Long documentId = requireLong(getValue(row, "documentId"), "documentId");
        String fileName = requireText(getValue(row, "fileName"), "fileName");
        List<DocumentChunkEntity> chunks = chunkWindowCache.computeIfAbsent(
                documentId,
                ignored -> documentChunkMapper.selectReadyActiveChunksByDocumentId(groupId, documentId));
        if (chunks.isEmpty()) {
            return null;
        }
        int startIndex = cluster.expandedStartChunkIndex(neighborWindow);
        int endIndex = cluster.expandedEndChunkIndex(neighborWindow);
        StringBuilder builder = new StringBuilder();
        for (DocumentChunkEntity chunk : chunks) {
            if (chunk.getChunkIndex() != null
                    && chunk.getChunkIndex() >= startIndex
                    && chunk.getChunkIndex() <= endIndex
                    && StringUtils.hasText(chunk.getChunkText())) {
                if (!builder.isEmpty()) {
                    builder.append("\n");
                }
                builder.append(chunk.getChunkText().trim());
            }
        }
        if (builder.isEmpty()) {
            return null;
        }
        return "文件名：" + fileName + "\n" + builder;
    }

    /** 将候选切片聚合为类簇：同一文档中连续的切片合并为一个类簇 */
    private List<RetrievalCluster> buildClusters(List<RetrievalCandidate> rankedCandidates) {
        Map<Long, List<RetrievalCandidate>> candidatesByDocumentId = new LinkedHashMap<>();
        for (RetrievalCandidate candidate : rankedCandidates) {
            candidatesByDocumentId.computeIfAbsent(candidate.documentId(), ignored -> new ArrayList<>()).add(candidate);
        }
        List<RetrievalCluster> clusters = new ArrayList<>();
        for (List<RetrievalCandidate> sameDocumentCandidates : candidatesByDocumentId.values()) {
            List<RetrievalCandidate> sortedByChunkIndex = sameDocumentCandidates.stream()
                    .sorted(Comparator.comparing(RetrievalCandidate::chunkIndex))
                    .toList();
            RetrievalCluster currentCluster = null;
            for (RetrievalCandidate candidate : sortedByChunkIndex) {
                if (currentCluster == null || !currentCluster.isContinuousWith(candidate)) {
                    currentCluster = new RetrievalCluster(candidate);
                    clusters.add(currentCluster);
                    continue;
                }
                currentCluster.add(candidate);
            }
        }
        return clusters.stream()
                .sorted(Comparator
                        .comparingDouble(RetrievalCluster::rankingScore).reversed()
                        .thenComparing(RetrievalCluster::primaryChunkId))
                .toList();
    }

    /** 评估检索结果的证据充分度等级 */
    private EvidenceLevel evaluateEvidenceLevel(List<Document> documents) {
        if (documents.isEmpty()) {
            return EvidenceLevel.NONE;
        }
        boolean hasBothSource = documents.stream()
                .map(document -> document.getMetadata().get("retrievalSource"))
                .anyMatch("BOTH"::equals);
        boolean hasVectorEvidence = documents.stream()
                .map(document -> document.getMetadata().get("retrievalSource"))
                .anyMatch(source -> "VECTOR".equals(source) || "BOTH".equals(source));
        double topScore = documents.stream()
                .map(document -> document.getMetadata().get("score"))
                .filter(Double.class::isInstance)
                .map(Double.class::cast)
                .max(Double::compareTo)
                .orElse(0D);
        // 归一化后 score ≥ 0.85 对应双通道 rank-1 或多次高排名命中
        if (documents.size() >= 2 && (hasBothSource || (hasVectorEvidence && topScore >= 0.85D))) {
            return EvidenceLevel.SUFFICIENT;
        }
        if (hasBothSource || documents.size() >= 2) {
            return EvidenceLevel.PARTIAL;
        }
        return EvidenceLevel.WEAK;
    }

    /** 根据证据等级生成对应的回答指导语 */
    private String buildEvidenceGuidance(EvidenceLevel evidenceLevel) {
        return switch (evidenceLevel) {
            case NONE -> "当前没有可用证据，必须直接拒答。";
            case WEAK -> "当前证据相关性有限，只能谨慎回答，必须明确说明依据有限，不能给出确定性结论。";
            case PARTIAL -> "当前证据只覆盖部分问题，只能回答证据明确支持的部分，未覆盖部分必须明确说明不足。";
            case SUFFICIENT -> "当前证据较充分，可以正常回答，但仍然不得超出证据进行臆测。";
        };
    }

    /**
     * 将原始 RRF 融合评分归一化到 [0, 1] 区间。
     * <p>
     * 使用指数饱和函数 {@code 1 - e^(-x)}，具有以下特性：
     * <ul>
     * <li>边际递减：每增加一份独立证据，分数增幅逐渐变小，符合证据积累直觉</li>
     * <li>自动适配：无需硬编码除数，不受查询条数和 topK 变化影响</li>
     * <li>渐进逼近 1：体现检索系统固有的不确定性</li>
     * </ul>
     * <p>
     * 典型映射（k=0 时）：
     * <ul>
     * <li>单通道 rank-1 命中（原始分 1.0） → 0.63</li>
     * <li>双通道 rank-1 命中（原始分 2.0） → 0.86</li>
     * <li>多查询多次命中（原始分 ≥ 3.0） → ≥ 0.95</li>
     * </ul>
     *
     * @param rawRrfScore 原始 RRF 累加评分（k=0 时值域 [0, +∞)）
     * @return 归一化到 [0, 1) 的评分
     */
    private double normalizeScore(double rawRrfScore) {
        return 1.0 - Math.exp(-rawRrfScore);
    }

    /** 从 Row Map 中获取值，支持驼峰和小写两种键名 */
    private Object getValue(Map<String, Object> row, String field) {
        Object value = row.get(field);
        if (value != null) {
            return value;
        }
        return row.get(field.toLowerCase());
    }

    /** 校验 groupId 为正数 */
    private Long requirePositiveGroupId(Long groupId) {
        if (groupId == null || groupId <= 0) {
            throw new BusinessException("groupId 非法");
        }
        return groupId;
    }

    /** 校验问题非空并 trim */
    private String requireQuestion(String question) {
        if (!StringUtils.hasText(question)) {
            throw new BusinessException("问题不能为空");
        }
        return question.trim();
    }

    /** 安全转换为 Long，失败时抛出业务异常 */
    private Long requireLong(Object value, String field) {
        if (value instanceof Number number) {
            return number.longValue();
        }
        throw new BusinessException("检索结果缺少字段: " + field);
    }

    /** 安全转换为 Integer，失败时抛出业务异常 */
    private Integer requireInteger(Object value, String field) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        throw new BusinessException("检索结果缺少字段: " + field);
    }

    /** 安全转换为非空字符串，失败时抛出业务异常 */
    private String requireText(Object value, String field) {
        if (value instanceof String text && StringUtils.hasText(text)) {
            return text.trim();
        }
        throw new BusinessException("检索结果缺少字段: " + field);
    }

    /**
     * 检索候选项，代表一个切片在检索结果中的命中信息。
     * <p>
     * 记录该切片在向量检索和关键词检索中的命中情况、各通道评分和 RRF 融合评分。
     * </p>
     */
    static final class RetrievalCandidate {

        /** 所属文档 ID */
        private final Long documentId;
        /** 切片 ID */
        private final Long chunkId;
        /** 切片在文档中的序号 */
        private final Integer chunkIndex;
        /** 向量检索评分（取多次命中的最大值） */
        private double vectorScore;
        /** 关键词检索评分（取多次命中的最大值） */
        private double keywordScore;
        /** RRF 融合评分（累加值） */
        private double rankingScore;
        /** 是否在向量检索中命中 */
        private boolean vectorMatched;
        /** 是否在关键词检索中命中 */
        private boolean keywordMatched;

        private RetrievalCandidate(Long documentId, Long chunkId, Integer chunkIndex) {
            this.documentId = documentId;
            this.chunkId = chunkId;
            this.chunkIndex = chunkIndex;
        }

        /** 从向量检索命中结果创建候选项 */
        static RetrievalCandidate fromVectorHit(PgVectorRetriever.VectorHit hit) {
            return new RetrievalCandidate(hit.documentId(), hit.chunkId(), hit.chunkIndex());
        }

        /** 从关键词检索命中结果创建候选项 */
        static RetrievalCandidate fromKeywordHit(ElasticsearchService.KeywordHit hit) {
            return new RetrievalCandidate(hit.documentId(), hit.chunkId(), hit.chunkIndex());
        }

        /** 合并向量检索命中：更新评分并累加 RRF 分数 */
        void mergeVectorHit(PgVectorRetriever.VectorHit hit, int rank) {
            this.vectorMatched = true;
            this.vectorScore = Math.max(this.vectorScore, hit.score());
            this.rankingScore += reciprocalRank(rank) * VECTOR_WEIGHT;
        }

        /** 合并关键词检索命中：更新评分并累加 RRF 分数 */
        void mergeKeywordHit(ElasticsearchService.KeywordHit hit, int rank) {
            this.keywordMatched = true;
            this.keywordScore = Math.max(this.keywordScore, hit.normalizedScore());
            this.rankingScore += reciprocalRank(rank) * KEYWORD_WEIGHT;
        }

        Long documentId() {
            return documentId;
        }

        Long chunkId() {
            return chunkId;
        }

        Integer chunkIndex() {
            return chunkIndex;
        }

        double vectorScore() {
            return vectorScore;
        }

        double keywordScore() {
            return keywordScore;
        }

        double rankingScore() {
            return rankingScore;
        }

        /** 获取检索来源：BOTH（双通道）、VECTOR 或 KEYWORD */
        String source() {
            if (vectorMatched && keywordMatched) {
                return "BOTH";
            }
            return vectorMatched ? "VECTOR" : "KEYWORD";
        }

        /** 计算 RRF 倒数排名分数：1 / (k + rank) */
        private double reciprocalRank(int rank) {
            return 1D / (RRF_K + Math.max(rank, 1));
        }
    }

    /**
     * 检索类簇，代表同一文档中连续切片的聚合组。
     * <p>
     * 将连续的切片合并为一个证据单元，跟踪主要候选项（评分最高者）和切片范围。
     * </p>
     */
    static final class RetrievalCluster {

        /** 所属文档 ID */
        private final Long documentId;
        /** 类簇内的所有候选切片 */
        private final List<RetrievalCandidate> members = new ArrayList<>();
        /** 主要候选项（评分最高的切片） */
        private RetrievalCandidate primaryCandidate;
        /** 类簇的起始切片序号 */
        private int startChunkIndex;
        /** 类簇的结束切片序号 */
        private int endChunkIndex;
        /** 类簇的 RRF 融合评分（取成员最大值） */
        private double rankingScore;
        /** 类簇的向量检索评分（取成员最大值） */
        private double vectorScore;
        /** 类簇的关键词检索评分（取成员最大值） */
        private double keywordScore;
        /** 类簇是否包含向量检索命中的成员 */
        private boolean hasVectorSource;
        /** 类簇是否包含关键词检索命中的成员 */
        private boolean hasKeywordSource;

        private RetrievalCluster(RetrievalCandidate seed) {
            this.documentId = seed.documentId();
            this.startChunkIndex = seed.chunkIndex();
            this.endChunkIndex = seed.chunkIndex();
            add(seed);
        }

        /** 判断候选切片是否与当前类簇连续（同文档且序号相邻） */
        boolean isContinuousWith(RetrievalCandidate candidate) {
            return documentId.equals(candidate.documentId()) && candidate.chunkIndex() == endChunkIndex + 1;
        }

        /** 添加候选切片到类簇，更新范围、评分和主要候选项 */
        void add(RetrievalCandidate candidate) {
            members.add(candidate);
            endChunkIndex = Math.max(endChunkIndex, candidate.chunkIndex());
            startChunkIndex = Math.min(startChunkIndex, candidate.chunkIndex());
            rankingScore = Math.max(rankingScore, candidate.rankingScore());
            vectorScore = Math.max(vectorScore, candidate.vectorScore());
            keywordScore = Math.max(keywordScore, candidate.keywordScore());
            hasVectorSource = hasVectorSource || "VECTOR".equals(candidate.source())
                    || "BOTH".equals(candidate.source());
            hasKeywordSource = hasKeywordSource || "KEYWORD".equals(candidate.source())
                    || "BOTH".equals(candidate.source());
            if (primaryCandidate == null
                    || candidate.rankingScore() > primaryCandidate.rankingScore()
                    || (candidate.rankingScore() == primaryCandidate.rankingScore()
                            && candidate.chunkIndex() < primaryCandidate.chunkIndex())) {
                primaryCandidate = candidate;
            }
        }

        Long documentId() {
            return documentId;
        }

        Long primaryChunkId() {
            return primaryCandidate.chunkId();
        }

        Integer primaryChunkIndex() {
            return primaryCandidate.chunkIndex();
        }

        double rankingScore() {
            return rankingScore;
        }

        double vectorScore() {
            return vectorScore;
        }

        double keywordScore() {
            return keywordScore;
        }

        /** 获取扩展后的起始切片序号（向前扩展 neighborWindow 个切片，最小为 0） */
        int expandedStartChunkIndex(int neighborWindow) {
            return Math.max(0, startChunkIndex - Math.max(0, neighborWindow));
        }

        /** 获取扩展后的结束切片序号（向后扩展 neighborWindow 个切片） */
        int expandedEndChunkIndex(int neighborWindow) {
            return endChunkIndex + Math.max(0, neighborWindow);
        }

        /** 获取类簇的检索来源：BOTH（双通道）、VECTOR 或 KEYWORD */
        String source() {
            if (hasVectorSource && hasKeywordSource) {
                return "BOTH";
            }
            return hasVectorSource ? "VECTOR" : "KEYWORD";
        }
    }
}

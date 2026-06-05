package com.argus.rag.engine.elasticsearch;

import com.argus.rag.common.exception.BusinessException;
import com.argus.rag.ingestion.model.entity.DocumentChunkEntity;
import lombok.extern.slf4j.Slf4j;
import org.elasticsearch.action.index.IndexRequest;
import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.RequestOptions;
import org.elasticsearch.client.RestHighLevelClient;
import org.elasticsearch.client.indices.CreateIndexRequest;
import org.elasticsearch.client.indices.GetIndexRequest;
import org.elasticsearch.index.reindex.DeleteByQueryRequest;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.index.query.BoolQueryBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Elasticsearch 文档切片索引服务，兼具 <b>索引管理</b> 和 <b>关键词检索</b> 两大职责。
 * <p>
 * <b>职责一：索引管理（写入）</b>
 * <ul>
 *   <li>将文档切片写入 ES 索引（{@link #indexReadyChunks}）</li>
 *   <li>按 documentId 批量删除过期索引（{@link #deleteDocumentChunks}）</li>
 *   <li>启动时自动检测并创建索引（{@link #ensureIndexInitialized}）</li>
 * </ul>
 * <p>
 * <b>职责二：关键词检索（召回）</b>
 * <ul>
 *   <li>基于 IK 中文分词的全文本检索（{@link #search}）</li>
 *   <li>双层打分：bool 初排 + rescore 二次精排</li>
 *   <li>多字段加权：fileName 和 chunkText 各有 match_phrase / match 两种匹配策略</li>
 *   <li>分数归一化：将对数变换后的原始分数压缩到 [0, 1] 区间</li>
 * </ul>
 * <p>
 * <b>检索安全保障：</b>
 * <ul>
 *   <li>filter 层强制过滤 groupId + status=READY + deleted=false</li>
 *   <li>检索失败不抛异常，降级返回空列表（保证服务不中断）</li>
 * </ul>
 * <p>
 * <b>通信方式：</b>通过 Elasticsearch {@link RestHighLevelClient} 调用 ES REST API。
 *
 * @author Argus-RAG Team
 * @see KeywordHit 关键词检索命中结果记录
 */
@Service
@Slf4j
public class ElasticsearchService {

    /** 关键词分数归一化的参考基准值，用于对数变换 */
    private static final double KEYWORD_SCORE_REFERENCE = 100D;

    /** ES 索引中文档切片的有效状态标识 */
    private static final String READY_STATUS = "READY";

    /** Elasticsearch 高级 REST 客户端 */
    private final RestHighLevelClient client;

    /** ES 索引名称 */
    private final String indexName;

    /** 索引是否已初始化（volatile 保证多线程可见性） */
    private volatile boolean indexInitialized;

    /**
     * Spring 构造注入，通过 {@code @Value} 读取 ES 索引名称配置。
     *
     * @param elasticsearchClient ES 高级 REST 客户端（由 {@link ElasticsearchConfiguration} 提供）
     * @param indexName           ES 索引名称，默认 {@code dd_rag_document_chunks}
     */
    @Autowired
    public ElasticsearchService(
            RestHighLevelClient elasticsearchClient,
            @Value("${elasticsearch.index-name:dd_rag_document_chunks}") String indexName
    ) {
        this.client = elasticsearchClient;
        this.indexName = indexName;
    }

    /**
     * 将一批文档切片写入 ES 索引，用于后续关键词检索。
     * <p>
     * 内部逐条写入，每条切片调用 {@link #indexChunk} 完成序列化和 PUT。
     *
     * @param fileName 原始文件名（写入 ES 供检索时展示）
     * @param chunks   文档切片列表（必须已包含 chunkId / groupId / documentId 等字段）
     */
    public void indexReadyChunks(String fileName, List<DocumentChunkEntity> chunks) {
        if (!StringUtils.hasText(fileName) || chunks == null || chunks.isEmpty()) {
            return;
        }
        ensureIndexInitialized();
        for (DocumentChunkEntity chunk : chunks) {
            indexChunk(fileName, chunk);
        }
    }

    /**
     * 按文档 ID 批量删除该文档在 ES 中的所有切片索引。
     * <p>
     * 使用 ES 的 {@code _delete_by_query} API，条件为 {@code term: {documentId: xxx}}。
     * 删除失败时记录 WARN 日志但不抛异常（索引删除不是关键路径）。
     *
     * @param documentId 要清理的文档 ID
     */
    public void deleteDocumentChunks(Long documentId) {
        if (documentId == null || documentId <= 0) {
            return;
        }
        ensureIndexInitialized();
        try {
            DeleteByQueryRequest request = new DeleteByQueryRequest(indexName);
            request.setQuery(QueryBuilders.termQuery("documentId", documentId));
            client.deleteByQuery(request, RequestOptions.DEFAULT);
            log.info("ES 文档切片索引删除完成: documentId={}", documentId);
        } catch (IOException | RuntimeException exception) {
            log.warn("ES 文档切片索引删除失败: documentId={}, reason={}", documentId, exception.getMessage());
        }
    }

    /**
     * 执行 ES 关键词全文检索，返回与问题最匹配的 topK 个文档切片。
     * <p>
     * <b>检索策略：</b>
     * <ol>
     *   <li><b>Filter 过滤层：</b>限定 groupId + status=READY + deleted=false</li>
     *   <li><b>Should 召回层：</b>同时对 fileName 和 chunkText 做
     *       match_phrase（短语匹配，boost 6~8）和 match（分词匹配，boost 3~4）</li>
     *   <li><b>分数归一化：</b>通过 {@link #normalizeKeywordScore} 将对数变换后的分数
     *       压缩到 [0, 1] 区间</li>
     * </ol>
     * <p>
     * <b>降级策略：</b>检索失败（ES 宕机、网络超时等）不抛异常，
     * 记录 WARN 日志后返回空列表，保证服务整体可用。
     *
     * @param groupId  群组 ID
     * @param question 检索关键词 / 问题文本
     * @param topK     返回的最大结果数
     * @return 关键词匹配的切片列表（含归一化分数），无结果或失败时返回空列表
     */
    public List<KeywordHit> search(Long groupId, String question, int topK) {
        // 1. 参数校验，无效输入直接返回空
        if (groupId == null || groupId <= 0 || !StringUtils.hasText(question) || topK <= 0) {
            return List.of();
        }
        ensureIndexInitialized();
        long startNano = System.nanoTime();
        try {
            // 2. 构建检索请求：bool query
            SearchSourceBuilder sourceBuilder = new SearchSourceBuilder()
                    .size(topK)
                    .fetchSource(
                            new String[]{"groupId", "documentId", "chunkId", "chunkIndex", "fileName", "chunkText"},
                            null
                    )
                    .query(buildBoolQuery(groupId, question));

            SearchRequest request = new SearchRequest(indexName).source(sourceBuilder);
            SearchResponse response = client.search(request, RequestOptions.DEFAULT);

            // 3. 解析 ES 响应，提取文档字段和评分，转为领域对象
            return parseSearchHits(response.getHits().getHits(), groupId, topK, startNano);
        } catch (IOException | RuntimeException exception) {
            // 4. 检索失败时降级返回空结果，不阻断主流程
            long elapsedMs = (System.nanoTime() - startNano) / 1_000_000;
            log.warn(
                    "ES 关键词检索失败，降级为空结果: groupId={}, question='{}', elapsedMs={}, reason={}",
                    groupId,
                    abbreviate(question),
                    elapsedMs,
                    exception.getMessage()
            );
            return List.of();
        }
    }

    /**
     * 解析 ES 搜索响应中的 hits 数组为领域对象列表。
     * <p>
     * 包私有可见性，便于单元测试直接验证解析逻辑。
     *
     * @param hits      ES 返回的 SearchHit 数组
     * @param groupId   群组 ID（用于日志）
     * @param topK      请求的最大结果数（用于日志）
     * @param startNano 检索开始时间戳（用于计算耗时）
     * @return 解析后的 KeywordHit 列表，hits 为空时返回空列表
     */
    List<KeywordHit> parseSearchHits(SearchHit[] hits, Long groupId, int topK, long startNano) {
        if (hits.length == 0) {
            long elapsedMs = (System.nanoTime() - startNano) / 1_000_000;
            log.info("关键词检索完成: groupId={}, topK={}, hitCount=0, elapsedMs={}", groupId, topK, elapsedMs);
            return List.of();
        }
        List<KeywordHit> result = new ArrayList<>();
        for (SearchHit hit : hits) {
            Map<String, Object> source = hit.getSourceAsMap();
            double rawScore = hit.getScore();
            result.add(new KeywordHit(
                    toLong(source.get("documentId")),
                    toLong(source.get("chunkId")),
                    toInt(source.get("chunkIndex")),
                    toStr(source.get("fileName")),
                    toStr(source.get("chunkText")),
                    rawScore,
                    normalizeKeywordScore(rawScore)
            ));
        }
        long elapsedMs = (System.nanoTime() - startNano) / 1_000_000;
        log.info("关键词检索完成: groupId={}, topK={}, hitCount={}, elapsedMs={}",
                groupId, topK, result.size(), elapsedMs);
        return List.copyOf(result);
    }

    /**
     * 将单条文档切片写入 ES。
     * <p>
     * 写入前调用 {@link #validateChunk} 校验必要字段完整性。
     * 使用 {@link IndexRequest} 指定文档 ID 确保相同 chunkId 幂等覆盖。
     *
     * @param fileName 原始文件名
     * @param chunk    文档切片实体
     */
    private void indexChunk(String fileName, DocumentChunkEntity chunk) {
        validateChunk(chunk);
        Map<String, Object> document = new LinkedHashMap<>();
        document.put("chunkId", chunk.getId());
        document.put("groupId", chunk.getGroupId());
        document.put("documentId", chunk.getDocumentId());
        document.put("chunkIndex", chunk.getChunkIndex());
        document.put("fileName", fileName);
        document.put("chunkText", chunk.getChunkText());
        document.put("status", READY_STATUS);
        document.put("deleted", false);
        try {
            IndexRequest request = new IndexRequest(indexName)
                    .id(String.valueOf(chunk.getId()))
                    .source(document);
            client.index(request, RequestOptions.DEFAULT);
        } catch (IOException e) {
            throw new BusinessException("ES 索引写入失败: chunkId=" + chunk.getId(), e);
        }
    }

    /**
     * 校验文档切片的必要字段是否完整。
     * <p>
     * 检查项：chunk 非 null、id/groupId/documentId/chunkIndex 非 null、chunkText 非空。
     *
     * @param chunk 文档切片实体
     * @throws BusinessException 任一必要字段缺失时抛出
     */
    private void validateChunk(DocumentChunkEntity chunk) {
        if (chunk == null
                || chunk.getId() == null
                || chunk.getGroupId() == null
                || chunk.getDocumentId() == null
                || chunk.getChunkIndex() == null
                || !StringUtils.hasText(chunk.getChunkText())) {
            throw new BusinessException("ES 索引写入缺少必要 chunk 字段");
        }
    }

    /**
     * 初始化ES索引（双重检查锁+volatile）
     * 外层判断：已初始化则快速返回，提升并发性能
     * 加锁后二次判断：避免多线程排队重复创建索引
     * 校验并创建索引，最终更新初始化标记
     */
    private void ensureIndexInitialized() {
        if (indexInitialized) {
            return;
        }
        synchronized (this) {
            if (indexInitialized) {
                return;
            }
            if (!indexExists()) {
                createIndex();
            }
            indexInitialized = true;
        }
    }

    /**
     * 检查 ES 索引是否存在。
     * <p>
     * 使用 {@link RestHighLevelClient#indices()} 的 exists API，
     * 返回 true 表示存在，false 表示不存在。
     *
     * @return {@code true} 索引已存在，{@code false} 索引不存在
     * @throws BusinessException ES 通信失败时抛出
     */
    private boolean indexExists() {
        try {
            GetIndexRequest request = new GetIndexRequest(indexName);
            return client.indices().exists(request, RequestOptions.DEFAULT);
        } catch (IOException e) {
            throw new BusinessException("ES 索引检查失败", e);
        }
    }

    /**
     * 创建 ES 索引，使用自定义的 IK 中文分词器配置。
     * <p>
     * 使用 {@link XContentBuilder} 定义 settings（分词器）和 mappings（字段类型）。
     */
    private void createIndex() {
        try (XContentBuilder settings = buildSettingsBuilder();
             XContentBuilder mappings = buildMappingsBuilder()) {
            CreateIndexRequest request = new CreateIndexRequest(indexName);
            request.settings(settings);
            request.mapping(mappings);
            client.indices().create(request, RequestOptions.DEFAULT);
            log.info("ES 索引初始化完成: {}", indexName);
        } catch (IOException e) {
            throw new BusinessException("ES 索引创建失败", e);
        }
    }

    /**
     * 构建 ES 索引 settings（analysis 分词器配置）。
     * <p>
     * <b>Analysis 配置：</b>
     * <ul>
     *   <li>{@code ddrag_ik_index} —— 索引端使用 IK 最大切分（ik_max_word），提高召回覆盖率</li>
     *   <li>{@code ddrag_ik_search} —— 查询端使用 IK 智能切分（ik_smart），提高精确度</li>
     * </ul>
     *
     * @return XContentBuilder 索引 settings 定义
     */
    XContentBuilder buildSettingsBuilder() throws IOException {
        return XContentFactory.jsonBuilder()
                .startObject()
                .startObject("analysis")
                .startObject("analyzer")
                .startObject("ddrag_ik_index")
                .field("type", "custom")
                .field("tokenizer", "ik_max_word")
                .endObject()
                .startObject("ddrag_ik_search")
                .field("type", "custom")
                .field("tokenizer", "ik_smart")
                .endObject()
                .endObject()
                .endObject()
                .endObject();
    }

    /**
     * 构建 ES 索引 mappings（字段类型定义）。
     * <p>
     * <b>Mappings 字段：</b>
     * <ul>
     *   <li>{@code groupId / documentId / chunkId} —— long 类型（精确匹配 + filter）</li>
     *   <li>{@code chunkIndex} —— integer 类型</li>
     *   <li>{@code status} —— keyword 类型（精确 term 过滤）</li>
     *   <li>{@code deleted} —— boolean 类型</li>
     *   <li>{@code fileName} —— text 类型，双字段（原字段 + keyword 子字段）</li>
     *   <li>{@code chunkText} —— text 类型，IK 中文分词</li>
     * </ul>
     *
     * @return XContentBuilder 索引 mappings 定义
     */
    XContentBuilder buildMappingsBuilder() throws IOException {
        return XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject("groupId").field("type", "long").endObject()
                .startObject("documentId").field("type", "long").endObject()
                .startObject("chunkId").field("type", "long").endObject()
                .startObject("chunkIndex").field("type", "integer").endObject()
                .startObject("status").field("type", "keyword").endObject()
                .startObject("deleted").field("type", "boolean").endObject()
                .startObject("fileName")
                .field("type", "text")
                .field("analyzer", "ddrag_ik_index")
                .field("search_analyzer", "ddrag_ik_search")
                .startObject("fields")
                .startObject("keyword")
                .field("type", "keyword")
                .field("ignore_above", 256)
                .endObject()
                .endObject()
                .endObject()
                .startObject("chunkText")
                .field("type", "text")
                .field("analyzer", "ddrag_ik_index")
                .field("search_analyzer", "ddrag_ik_search")
                .endObject()
                .endObject()
                .endObject();
    }

    /**
     * 构建关键词检索的 bool query。
     * <p>
     * 四个打分维度：
     * <ol>
     *   <li>fileName 短语匹配——boost 8（文档名精确命中权重最高）</li>
     *   <li>fileName 分词匹配——boost 4（文档名模糊命中）</li>
     *   <li>chunkText 短语匹配——boost 6（内容精确命中）</li>
     *   <li>chunkText 分词匹配——boost 3（内容模糊命中）</li>
     * </ol>
     *
     * @param groupId 群组 ID
     * @param question 检索关键词
     * @return BoolQueryBuilder
     */
    private BoolQueryBuilder buildBoolQuery(Long groupId, String question) {
        return QueryBuilders.boolQuery()
                .filter(QueryBuilders.termQuery("groupId", groupId))
                .filter(QueryBuilders.termQuery("status", READY_STATUS))
                .filter(QueryBuilders.termQuery("deleted", false))
                .should(QueryBuilders.matchPhraseQuery("fileName", question).boost(8f))
                .should(QueryBuilders.matchQuery("fileName", question).boost(4f))
                .should(QueryBuilders.matchPhraseQuery("chunkText", question).boost(6f))
                .should(QueryBuilders.matchQuery("chunkText", question).boost(3f))
                .minimumShouldMatch(1);
    }

    /**
     * 对日志输出中的问题文本做截断处理，防止超长文本撑爆日志。
     * <p>
     * 规则：合并连续空白 → 超过 120 字符则截断并追加 "..." → 否则原样返回。
     *
     * @param text 原始文本
     * @return 截断后的文本（不超过 123 字符）
     */
    private String abbreviate(String text) {
        if (!StringUtils.hasText(text)) {
            return "";
        }
        String normalized = text.replaceAll("\\s+", " ").trim();
        return normalized.length() <= 120 ? normalized : normalized.substring(0, 120) + "...";
    }

    /**
     * 将 ES 原始BM25分数归一化到 [0, 1] 区间。
     * <p>
     * 使用对数变换 {@code log1p(x)} 压缩分数范围，消除不同查询间分数不可比的问题。
     * 公式：{@code min(1.0, log1p(rawScore) / log1p(KEYWORD_SCORE_REFERENCE))}
     * <p>
     * 如 rawScore=100 时归一化分数约为 1.0，rawScore=10 时约为 0.5。
     *
     * @param rawScore ES 返回的原始 _score（非负）
     * @return 归一化后的分数 [0, 1]
     */
    private double normalizeKeywordScore(double rawScore) {
        if (rawScore <= 0D) {
            return 0D;
        }
        return Math.min(1D, Math.log1p(rawScore) / Math.log1p(KEYWORD_SCORE_REFERENCE));
    }

    /**
     * 将 Object 安全转换为 long，处理 HLRC 反序列化时 Integer/Long 不确定的情况。
     */
    private static long toLong(Object value) {
        if (value instanceof Number n) {
            return n.longValue();
        }
        return 0L;
    }

    /**
     * 将 Object 安全转换为 int，处理 HLRC 反序列化时 Integer/Long 不确定的情况。
     */
    private static int toInt(Object value) {
        if (value instanceof Number n) {
            return n.intValue();
        }
        return 0;
    }

    /**
     * 将 Object 安全转换为 String，null 值返回空字符串。
     */
    private static String toStr(Object value) {
        return value != null ? value.toString() : "";
    }

    /**
     * ES 关键词检索的命中结果记录。
     * <p>
     * 不可变 record，可安全用于下游排序、融合和 LLM 上下文组装。
     * <p>
     * {@code rawScore} 为 ES 返回的原始 BM25 分数（无上界），
     * {@code normalizedScore} 为经 {@link #normalizeKeywordScore} 归一化后的 [0, 1] 分数。
     *
     * @param documentId      所属文档 ID
     * @param chunkId         切片 ID
     * @param chunkIndex      切片序号（从 0 开始）
     * @param fileName        原始文件名
     * @param chunkText       切片文本内容
     * @param rawScore        ES 原始 BM25 分数
     * @param normalizedScore 归一化后的 [0, 1] 分数
     */
    public record KeywordHit(
            Long documentId,
            Long chunkId,
            Integer chunkIndex,
            String fileName,
            String chunkText,
            double rawScore,
            double normalizedScore
    ) {
    }
}
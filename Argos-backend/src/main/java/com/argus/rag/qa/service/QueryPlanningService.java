package com.argus.rag.qa.service;

import com.argus.rag.common.exception.BusinessException;
import com.argus.rag.qa.model.QueryPlanResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 查询规划服务。
 * <p>
 * 调用大模型分析用户问题，决定采用哪种检索策略（直接、重写或分解），
 * 并生成对应的检索语句列表。规划失败时自动回退为 DIRECT 策略。
 * </p>
 */
@Service
public class QueryPlanningService {

    private static final Logger log = LoggerFactory.getLogger(QueryPlanningService.class);

    /** 检索语句最大数量限制 */
    private static final int MAX_QUERY_COUNT = 3;

    private final ChatClient queryPlanningChatClient;
    private final PromptTemplate queryPlanningPromptTemplate;

    /**
     * 构造函数。
     *
     * @param queryPlanningChatClient         查询规划专用的 ChatClient
     * @param queryPlanningPromptTemplate 查询规划用户提示词模板
     */
    public QueryPlanningService(
            @Qualifier("queryPlanningChatClient") ChatClient queryPlanningChatClient,
            @Qualifier("queryPlanningUserPromptTemplate") PromptTemplate queryPlanningPromptTemplate
    ) {
        this.queryPlanningChatClient = queryPlanningChatClient;
        this.queryPlanningPromptTemplate = queryPlanningPromptTemplate;
    }

    /**
     * 对用户问题执行查询规划。
     * <p>
     * 调用大模型分析问题并返回检索策略和检索语句，
     * 失败时回退为 DIRECT 策略。
     * </p>
     *
     * @param question 用户原始问题
     * @return 查询规划结果
     */
    public QueryPlanResult plan(String question) {
        String normalizedQuestion = requireQuestion(question);
        try {
            // 1. 构造提示词，调用大模型进行查询规划，返回策略和检索语句列表
            Prompt planPrompt = queryPlanningPromptTemplate.create(Map.of("question", normalizedQuestion));

            QueryPlanResult rawResult = queryPlanningChatClient.prompt(planPrompt)
                    .call()
                    // 输出结果直接反序列化为 QueryPlanResult 对象
                    .entity(QueryPlanResult.class);
            // 2. 校验 LLM 返回结果，按策略构建最终检索语句
            QueryPlanResult validatedPlan = validatePlan(rawResult, normalizedQuestion);
            log.info("查询规划完成: strategy={}, queries={}", validatedPlan.strategy(), validatedPlan.queries());
            return validatedPlan;
        } catch (RuntimeException exception) {
            log.warn("查询规划失败，回退为直接检索: question={}", normalizedQuestion, exception);
            return QueryPlanResult.fallback(normalizedQuestion);
        }
    }

    /**
     * 校验规划结果，确保策略和检索语句有效。
     * 根据不同策略构建最终的检索语句列表。
     */
    private QueryPlanResult validatePlan(QueryPlanResult rawResult, String originalQuestion) {
        if (rawResult == null || rawResult.strategy() == null) {
            return QueryPlanResult.fallback(originalQuestion);
        }
        Set<String> normalizedQueries = normalizeQueries(rawResult.queries());
        if (normalizedQueries.isEmpty()) {
            return QueryPlanResult.fallback(originalQuestion);
        }
        // 按策略类型构建检索语句：DIRECT 直接用原问题，REWRITE 追加重写语句，DECOMPOSE 用拆解的子问题
        List<String> finalQueries = switch (rawResult.strategy()) {
            case DIRECT -> List.of(originalQuestion);
            case REWRITE -> buildRewriteQueries(originalQuestion, normalizedQueries);
            case DECOMPOSE -> limitQueries(normalizedQueries);
        };
        if (finalQueries.isEmpty()) {
            return QueryPlanResult.fallback(originalQuestion);
        }
        return new QueryPlanResult(rawResult.strategy(), finalQueries);
    }

    /** 构建 REWRITE 策略的检索语句：原始问题 + 第一条改写句 */
    private List<String> buildRewriteQueries(String originalQuestion, Set<String> normalizedQueries) {
        LinkedHashSet<String> rewriteQueries = new LinkedHashSet<>();
        rewriteQueries.add(originalQuestion);
        // 只取第一条改写语句，避免过多冗余
        normalizedQueries.stream().findFirst().ifPresent(rewriteQueries::add);
        return List.copyOf(rewriteQueries);
    }

    /** 规范化检索语句：去除空白、去重 */
    private Set<String> normalizeQueries(List<String> queries) {
        LinkedHashSet<String> normalizedQueries = new LinkedHashSet<>();
        if (queries == null) {
            return normalizedQueries;
        }
        for (String query : queries) {
            if (!StringUtils.hasText(query)) {
                continue;
            }
            String normalized = query.replaceAll("\\s+", " ").trim();
            if (StringUtils.hasText(normalized)) {
                normalizedQueries.add(normalized);
            }
        }
        return normalizedQueries;
    }

    /** 限制检索语句数量不超过 {@link #MAX_QUERY_COUNT} */
    private List<String> limitQueries(Set<String> queries) {
        return queries.stream()
                // 限制取前3条作为最终检索语句
                .limit(MAX_QUERY_COUNT)
                .toList();
    }

    /** 校验问题非空并规范化空白字符 */
    private String requireQuestion(String question) {
        if (!StringUtils.hasText(question)) {
            throw new BusinessException("问题不能为空");
        }
        return question.replaceAll("\\s+", " ").trim();
    }
}

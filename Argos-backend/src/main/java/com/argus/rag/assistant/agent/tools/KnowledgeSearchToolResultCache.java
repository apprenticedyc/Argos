package com.argus.rag.assistant.agent.tools;

import com.argus.rag.qa.model.vo.AskQuestionResponse;

import java.util.List;

/**
 * 暂存知识库检索工具结果
 * <p> 用于在 Agent 执行过程中跨步骤传递知识库检索的结果和引用信息。</p>
 * <p> 在KnowledgeSearchTool工具执行完毕后记录引用列表，
 */
public class KnowledgeSearchToolResultCache {

    private List<AskQuestionResponse.Citation> citations = List.of();
    private KnowledgeSearchState searchState = KnowledgeSearchState.NOT_STARTED;

    /**
     * 记录知识库检索的引用列表。
     * <p>在 {@link KnowledgeSearchTool} 检索完成后调用，
     * 同时将检索状态标记为 COMPLETED</p>
     *
     * @param citations 引用列表，为 {@code null} 时视为空列表
     */
    public void recordCitations(List<AskQuestionResponse.Citation> citations) {
        this.citations = citations == null ? List.of() : List.copyOf(citations);
        this.searchState = KnowledgeSearchState.COMPLETED;
    }

    /**
     * 获取当前的引用列表。
     * <p>如果尚未执行检索，返回空列表。</p>
     *
     */
    public List<AskQuestionResponse.Citation> currentCitations() {
        return citations;
    }

    /**
     * 判断是否已完成知识库检索。
     * <p>用于 {@link KnowledgeSearchTool} 检测重复调用，
     * 防止 Agent 在同一轮对话中多次检索知识库。</p>
     *
     */
    public boolean hasCompletedSearch() {
        return searchState == KnowledgeSearchState.COMPLETED;
    }

    /**
     * 知识库检索状态枚举。
     * <p>用于追踪一次 Agent 调用周期内知识库检索的完成情况。</p>
     */
    enum KnowledgeSearchState {
        /** 尚未开始检索 */
        NOT_STARTED,
        /** 已完成检索 */
        COMPLETED
    }
}

package com.argus.rag.assistant.model.vo.chat;

import com.argus.rag.qa.model.vo.AskQuestionResponse;

import java.util.List;

public record AgentResult(
        /**
         * 助手回复的文本内容
         */
        String reply,
        /**
         * 知识库引用来源列表
         * <p>当工具模式为 KB_SEARCH 时包含检索到的文档引用信息</p>
         */
        List<AskQuestionResponse.Citation> citations
) {

    public AgentResult {
        citations = citations == null ? List.of() : List.copyOf(citations);
    }

    public static AgentResult withoutCitations(String reply) {
        return new AgentResult(reply, List.of());
    }
}

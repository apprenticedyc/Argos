package com.argus.rag.assistant.model.vo.chat;

import com.argus.rag.assistant.model.enums.AssistantChatMode;
import com.argus.rag.qa.model.vo.AskQuestionResponse;

import java.util.List;

/**
 *  SSE 流式聊天的事件载体
 */
public record ChatStreamEvent(
        /**
         * 事件类型名称
         * <p>一共四种事件：start（开始）、chunk（流式增量）、done（完成）、error（错误）</p>
         */
        String event,
        /**
         * 会话ID
         */
        Long sessionId,
        /**
         * 当前对话的工具模式
         */
        AssistantChatMode toolMode,
        /**
         * 知识库组ID（仅 KB_SEARCH 模式有值）
         */
        Long groupId,
        /**
         * 流式增量文本内容，仅在 chunk 事件中有值
         */
        String chunk,
        /**
         * 完成后的消息ID，仅在 done 事件中有值
         */
        Long messageId,
        /**
         * 完成后的完整回复文本，仅在 done 事件中有值
         */
        String fullReply,
        /**
         * 知识库引用来源列表，仅在 done 事件中有值
         */
        List<AskQuestionResponse.Citation> citations,
        /**
         * 错误信息，仅在 error 事件中有值
         */
        String error) {

    public static ChatStreamEvent start(Long sessionId, AssistantChatMode toolMode, Long groupId) {
        return new ChatStreamEvent("start", sessionId, toolMode, groupId, null, null, null, List.of(), null);
    }

    public static ChatStreamEvent chunk(Long sessionId, AssistantChatMode toolMode, Long groupId, String chunk) {
        return new ChatStreamEvent("chunk", sessionId, toolMode, groupId, chunk, null, null, List.of(), null);
    }

    public static ChatStreamEvent done(Long sessionId, AssistantChatMode toolMode, Long groupId, Long messageId,
                                       String reply, List<AskQuestionResponse.Citation> citations) {
        return new ChatStreamEvent("done", sessionId, toolMode, groupId, null, messageId, reply, citations == null ? List.of() : citations, null);
    }

    public static ChatStreamEvent error(Long sessionId, AssistantChatMode toolMode, Long groupId, String error) {
        return new ChatStreamEvent("error", sessionId, toolMode, groupId, null, null, null, List.of(), error);
    }
}

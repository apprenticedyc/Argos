package com.argus.rag.assistant.support.tools;

import com.argus.rag.assistant.model.entity.MessageEntity;
import com.argus.rag.assistant.model.enums.AssistantChatMode;
import com.argus.rag.common.exception.BusinessException;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

/**
 * 助手记忆摘要服务。
 * <p>调用 LLM 生成三种类型的摘要：</p>
 * <ul>
 *   <li>会话记忆（session memory）：基于对话历史生成的长期记忆</li>
 *   <li>压缩摘要（compact summary）：对会话的更精炼压缩</li>
 *   <li>运行时上下文（runtime context）：结合压缩摘要和当前问题的实时压缩</li>
 * </ul>
 */
@Service
public class LLMSummarizer {

    private final ChatClient chatClient;
    private final PromptTemplate assistantSessionMemoryPromptTemplate;
    private final PromptTemplate assistantCompactSummaryPromptTemplate;
    private final PromptTemplate assistantRuntimeCompactPromptTemplate;

    public LLMSummarizer(
            ChatClient.Builder chatClientBuilder,
            @Qualifier("assistantSessionMemoryPromptTemplate") PromptTemplate assistantSessionMemoryPromptTemplate,
            @Qualifier("assistantCompactSummaryPromptTemplate") PromptTemplate assistantCompactSummaryPromptTemplate,
            @Qualifier("assistantRuntimeCompactPromptTemplate") PromptTemplate assistantRuntimeCompactPromptTemplate
    ) {
        this.chatClient = chatClientBuilder.build();
        this.assistantSessionMemoryPromptTemplate = assistantSessionMemoryPromptTemplate;
        this.assistantCompactSummaryPromptTemplate = assistantCompactSummaryPromptTemplate;
        this.assistantRuntimeCompactPromptTemplate = assistantRuntimeCompactPromptTemplate;
    }

    /**
     * 生成会话记忆摘要。
     * <p>调用 LLM，基于现有会话记忆和新增消息列表，结合当前工具模式和知识库组信息，
     * 生成更新后的会话记忆文本。</p>
     *
     * @param existingSessionMemory 现有的会话记忆文本，可能为 {@code null} 或空
     * @param newMessages           新增的消息列表
     * @param toolMode              当前工具模式
     * @param groupId               知识库组 ID，可能为 {@code null}
     * @return 更新后的会话记忆文本
     */
    public String buildSessionMemory(
            String existingSessionMemory,
            List<MessageEntity> newMessages,
            AssistantChatMode toolMode,
            Long groupId
    ) {
        return callLLM(assistantSessionMemoryPromptTemplate.create(Map.of(
                "existingSessionMemory", defaultText(existingSessionMemory),
                "newMessages", formatMessages(newMessages),
                "currentToolMode", toolMode == null ? "UNKNOWN" : toolMode.name(),
                "currentGroupId", groupId == null ? "NONE" : String.valueOf(groupId)
        )), "生成 session memory 失败");
    }

    /**
     * 生成压缩摘要。
     * <p>调用 LLM，基于现有压缩摘要、会话记忆和待压缩消息列表，生成更精炼的压缩摘要。</p>
     *
     * @param existingCompactSummary 现有的压缩摘要文本，可能为 {@code null} 或空
     * @param sessionMemory          当前的会话记忆文本
     * @param messagesToCompact      待压缩的消息列表
     * @return 生成后的压缩摘要文本
     */
    public String buildCompactSummary(
            String existingCompactSummary,
            String sessionMemory,
            List<MessageEntity> messagesToCompact
    ) {
        // 根据已有的压缩摘要、会话记忆和待压缩消息，调用 LLM 生成新的压缩摘要
        return callLLM(assistantCompactSummaryPromptTemplate.create(Map.of(
                "existingCompactSummary", defaultText(existingCompactSummary),
                "sessionMemory", defaultText(sessionMemory),
                "messagesToCompact", formatMessages(messagesToCompact)
        )), "生成 compact summary 失败");
    }

    /**
     * 生成运行时压缩上下文。
     * <p>调用 LLM，结合压缩摘要、会话记忆、最近消息和当前问题，
     * 生成精简的运行时上下文，用于在模型调用前缩减输入 token 量。</p>
     *
     * @param compactSummary 当前的压缩摘要文本
     * @param sessionMemory  当前的会话记忆文本
     * @param recentMessages 最近的消息文本
     * @param currentQuestion 当前用户问题
     * @return 运行时压缩后的上下文文本
     */
    public String summarizeRuntimeContext(
            String compactSummary,
            String sessionMemory,
            String recentMessages,
            String currentQuestion
    ) {
        return callLLM(assistantRuntimeCompactPromptTemplate.create(Map.of(
                "compactSummary", defaultText(compactSummary),
                "sessionMemory", defaultText(sessionMemory),
                "recentMessages", defaultText(recentMessages),
                "currentQuestion", defaultText(currentQuestion)
        )), "生成运行时压缩上下文失败");
    }

    /**
     * 格式化消息列表为纯文本。
     * <p>将消息实体列表转换为 {@code [角色] 内容} 格式的纯文本，用于作为 Prompt 模板的输入变量。</p>
     *
     * @param messages 消息实体列表
     * @return 格式化后的消息文本，若列表为空则返回 {@code "NONE"}
     */
    String formatMessages(List<MessageEntity> messages) {
        if (messages == null || messages.isEmpty()) {
            return "NONE";
        }
        StringBuilder builder = new StringBuilder();
        for (MessageEntity message : messages) {
            if (message == null) {
                continue;
            }
            String content = normalize(message.getContent());
            if (content.isEmpty()) {
                continue;
            }
            if (!builder.isEmpty()) {
                builder.append(System.lineSeparator());
            }
            builder.append('[')
                    .append(defaultText(message.getRole()))
                    .append("] ")
                    .append(content);
        }
        return builder.isEmpty() ? "NONE" : builder.toString();
    }

    /**
     * 调用 ChatClient 获取结果
     * @param prompt
     * @param errorMessage
     * @return
     */
    private String callLLM(Prompt prompt, String errorMessage) {
        try {
            String content = chatClient.prompt(prompt)
                    .call()
                    .content();
            String normalized = normalize(content);
            if (normalized.isEmpty()) {
                throw new BusinessException(errorMessage + "，模型返回为空");
            }
            return normalized;
        } catch (RuntimeException exception) {
            if (exception instanceof BusinessException businessException) {
                throw businessException;
            }
            throw new BusinessException(errorMessage, exception);
        }
    }

    private String normalize(String content) {
        if (content == null) {
            return "";
        }
        return content.replace("\r\n", "\n").trim();
    }

    private String defaultText(String value) {
        return normalize(value).isEmpty() ? "NONE" : normalize(value);
    }
}

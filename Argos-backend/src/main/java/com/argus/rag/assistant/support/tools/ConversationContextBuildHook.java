package com.argus.rag.assistant.support.tools;

import com.alibaba.cloud.ai.graph.RunnableConfig;
import com.alibaba.cloud.ai.graph.agent.hook.HookPosition;
import com.alibaba.cloud.ai.graph.agent.hook.HookPositions;
import com.alibaba.cloud.ai.graph.agent.hook.messages.AgentCommand;
import com.alibaba.cloud.ai.graph.agent.hook.messages.MessagesModelHook;
import com.alibaba.cloud.ai.graph.agent.hook.messages.UpdatePolicy;
import com.argus.rag.assistant.model.enums.MessageRole;
import com.argus.rag.assistant.model.vo.message.MessageVO;
import com.argus.rag.assistant.service.ConversationContextService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.messages.*;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * 上下文管理触发 Hook。
 * <p>作为 ReactAgent 的 BEFORE_MODEL Hook，在每次模型调用前组装上下文消息。</p>
 * <p>通过从数据库中加载会话的压缩摘要、会话记忆和最近消息，构建发送给 LLM 的消息列表，
 * 实现会话上下文的自动注入。</p>
 */
@Component
@HookPositions({HookPosition.BEFORE_MODEL})
public class ConversationContextBuildHook extends MessagesModelHook {

    /** 日志记录器 */
    private static final Logger log = LoggerFactory.getLogger(ConversationContextBuildHook.class);
    /** 加载最近消息的最大条数 */
    private static final int RECENT_MESSAGE_LIMIT = 10;
    /** 触发运行时压缩的 token 阈值 */
    private static final int RUNTIME_TOKEN_THRESHOLD = 50000;

    private final ConversationContextService conversationContextService;

    public ConversationContextBuildHook(ConversationContextService conversationContextService) {
        this.conversationContextService = conversationContextService;
    }

    @Override
    public String getName() {
        return "assistant_short_term_memory_hook";
    }

    @Override
    public AgentCommand beforeModel(List<Message> runtimeMessages, RunnableConfig config) {
        // 前置校验：消息列表或配置为空时直接透传原始消息
        if (runtimeMessages == null || runtimeMessages.isEmpty() || config == null) {
            return new AgentCommand(runtimeMessages, UpdatePolicy.REPLACE);
        }
        // 从运行时配置 config 提取 userId、sessionId
        Long userId = metadataAsLong(config, "userId").orElse(null);
        Long sessionId = metadataAsLong(config, "sessionId").orElse(null);
        // 关键信息缺失时直接透传，不组装上下文
        if (userId == null || sessionId == null) {
            return new AgentCommand(runtimeMessages, UpdatePolicy.REPLACE);
        }
        // 从消息列表末尾提取当前用户问题
        String currentQuestion = extractCurrentQuestion(runtimeMessages);
        if (currentQuestion == null) {
            return new AgentCommand(runtimeMessages, UpdatePolicy.REPLACE);
        }
        // 打印入参日志，便于调试上下文组装过程
        if (log.isDebugEnabled()) {
            log.debug(
                    "ShortTermMemoryHook.beforeModel input. userId={}, sessionId={}, runtimeMessages={}",
                    userId,
                    sessionId,
                    summarizeMessages(runtimeMessages)
            );
        }
        // 组装送进模型的上下文：compact summary -> session memory -> recent messages -> current question
        List<Message> assembledMessages = assembleBeforeModelMessages(
                userId,
                sessionId,
                currentQuestion,
                runtimeMessages
        );
        if (log.isDebugEnabled()) {
            log.debug(
                    "ShortTermMemoryHook.beforeModel output. userId={}, sessionId={}, currentQuestion={}, assembledMessages={}",
                    userId,
                    sessionId,
                    currentQuestion,
                    summarizeMessages(assembledMessages)
            );
        }

        // 用组装后的消息列表整体替换原始消息
        return new AgentCommand(
                assembledMessages,
                UpdatePolicy.REPLACE
        );
    }

    /**
     * 兼容旧版 Agent 框架的 beforeModel 重载。
     * <p>该方法仅透传输入，不做任何处理，确保与旧版 API 兼容。</p>
     *
     * @param ignored 输入对象，直接返回
     * @return 与输入相同的对象
     */
    public Object beforeModel(Object ignored) {
        return ignored;
    }

    /**
     * 组装模型调用前的消息列表（无运行时消息的便捷方法）。
     * <p>等价于 {@code assembleBeforeModelMessages(userId, sessionId, currentQuestion, List.of())}。</p>
     *
     * @param userId          用户 ID
     * @param sessionId       会话 ID
     * @param currentQuestion 当前用户问题
     * @return 组装后的消息列表
     */
    public List<Message> assembleBeforeModelMessages(
            Long userId,
            Long sessionId,
            String currentQuestion
    ) {
        return assembleBeforeModelMessages(
                userId,
                sessionId,
                currentQuestion,
                List.of()
        );
    }

    /**
     * 组装模型调用前的消息列表（完整参数版本）。
     * <p>按以下顺序构建发送给 LLM 的消息列表：</p>
     * <ol>
     *   <li>压缩摘要（短期记忆，作为系统消息注入）</li>
     *   <li>会话记忆（作为系统消息注入）</li>
     *   <li>最近的对话历史</li>
     *   <li>运行时工具消息</li>
     *   <li>当前用户问题</li>
     * </ol>
     *
     * @param userId           用户 ID
     * @param sessionId        会话 ID
     * @param currentQuestion  当前用户问题
     * @param runtimeMessages  运行时工具返回的消息列表
     * @return 组装后的消息列表
     */
    public List<Message> assembleBeforeModelMessages(
            Long userId,
            Long sessionId,
            String currentQuestion,
            List<Message> runtimeMessages
    ) {
        List<Message> messages = new ArrayList<>();
        ConversationContextService.ConversationContext conversationContext =
                conversationContextService.buildSingleTurnContext(userId, sessionId, RECENT_MESSAGE_LIMIT);
        // 记忆层作为系统消息注入，按时间从远到近：compactSummary -> sessionMemory -> recentMessages -> currentQuestion
        // 1. 压缩摘要（短期记忆压缩结果）
        addSystemMemory(messages, "compact summary", conversationContext.compactSummary());
        // 2. 会话记忆（作为系统消息）
        addSystemMemory(messages, "session memory", conversationContext.sessionMemory());
        // 3. 最近的新消息（用户、助手、工具）
        appendRecentMessages(messages, conversationContext.recentMessages(), currentQuestion);
        // 4. 运行时工具消息（Agent 图执行过程中产生的 ToolResponseMessage）
        appendRuntimeToolMessages(messages, runtimeMessages);
        // 5. 当前用户问题(最新消息)
        messages.add(new UserMessage(currentQuestion));
        return messages;
    }

    /**
     * 判断是否需要运行时压缩。
     * <p>当估算 token 数超过阈值 {@value #RUNTIME_TOKEN_THRESHOLD} 时，认为需要触发运行时压缩。</p>
     *
     * @param estimatedTokens 估算的 token 数
     * @return 需要运行时压缩时返回 {@code true}
     */
    public boolean shouldRuntimeCompact(int estimatedTokens) {
        return estimatedTokens > RUNTIME_TOKEN_THRESHOLD;
    }

    /**
     * 执行运行时压缩。
     * <p>保留消息列表末尾最多 3 条消息，丢弃更早的消息以控制 token 数量。</p>
     *
     * @param messages 待压缩的消息列表
     * @return 压缩后的消息列表，若输入为空则返回空列表
     */
    public List<Message> runtimeCompact(List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            return List.of();
        }
        int keepCount = Math.min(3, messages.size());
        return new ArrayList<>(messages.subList(messages.size() - keepCount, messages.size()));
    }

    private void addSystemMemory(List<Message> messages, String label, String content) {
        if (content == null || content.isBlank()) {
            return;
        }
        messages.add(new SystemMessage((label + System.lineSeparator() + content).trim()));
    }

    private void appendRecentMessages(
            List<Message> messages,
            List<MessageVO> recentMessages,
            String currentQuestion
    ) {
        if (recentMessages == null || recentMessages.isEmpty()) {
            return;
        }
        int lastIndex = recentMessages.size() - 1;
        // 遍历最近消息，按角色转换为 Spring AI 的 Message 类型
        for (int index = 0; index < recentMessages.size(); index++) {
            MessageVO recentMessage = recentMessages.get(index);
            // 跳过空消息
            if (recentMessage == null || recentMessage.content() == null || recentMessage.content().isBlank()) {
                continue;
            }
            // 判断是否是当前问题
            boolean isCurrentQuestionEcho = index == lastIndex
                    && recentMessage.role() == MessageRole.USER
                    && currentQuestion.equals(recentMessage.content().trim());
            if (isCurrentQuestionEcho) {
                // 如果是当前问题则跳过，避免重复注入
                continue;
            }
            // 按角色分别转换：USER/UserMessage, ASSISTANT/AssistantMessage, TOOL/AssistantMessage
            switch (recentMessage.role()) {
                case USER -> messages.add(new UserMessage(formatRecentMessage(recentMessage)));
                case ASSISTANT -> messages.add(new AssistantMessage(formatRecentMessage(recentMessage)));
                case TOOL -> messages.add(new AssistantMessage(formatToolMessage(recentMessage.content())));
            }
        }
    }

    private void appendRuntimeToolMessages(List<Message> messages, List<Message> runtimeMessages) {
        if (runtimeMessages == null || runtimeMessages.isEmpty()) {
            return;
        }
        for (Message runtimeMessage : runtimeMessages) {
            // 只把 ToolResponseMessage类型的消息加入上下文，其他类型的运行时消息不处理
            if (!(runtimeMessage instanceof ToolResponseMessage toolResponseMessage)) {
                continue;
            }
            if (toolResponseMessage.getResponses() == null || toolResponseMessage.getResponses().isEmpty()) {
                continue;
            }
            // 将每条工具响应格式化后转为 AssistantMessage 追加到上下文
            for (ToolResponseMessage.ToolResponse response : toolResponseMessage.getResponses()) {
                if (response == null || response.responseData() == null || response.responseData().isBlank()) {
                    continue;
                }
                messages.add(new AssistantMessage(formatToolMessage(response.responseData())));
            }
        }
    }

    private String formatRecentMessage(MessageVO message) {
        String mode = message.toolMode() == null ? "UNKNOWN" : message.toolMode().name();
        return ("[历史消息 | 模式：" + mode + "]" + System.lineSeparator() + message.content()).trim();
    }

    private String formatToolMessage(String content) {
        return ("[工具观察]" + System.lineSeparator() + content).trim();
    }

    private Optional<Long> metadataAsLong(RunnableConfig config, String key) {
        try {
            return config.metadata(key)
                    .map(value -> {
                    if (value instanceof Number number) {
                        return number.longValue();
                    }
                    if (value instanceof String stringValue && !stringValue.isBlank()) {
                        return Long.parseLong(stringValue);
                    }
                    return null;
                });
        } catch (RuntimeException exception) {
            return Optional.empty();
        }
    }

    /**
     * 从消息列表末尾向前扫描，提取最后一个用户消息(最新消息)文本作为当前问题。
     * @param messages
     * @return
     */
    private String extractCurrentQuestion(List<Message> messages) {
        for (int index = messages.size() - 1; index >= 0; index--) {
            Message message = messages.get(index);
            if (message instanceof UserMessage) {
                UserMessage userMessage = (UserMessage) message;
                String text = userMessage.getText();
                if (text != null && !text.isBlank()) {
                    return text.trim();
                }
            }
        }
        return null;
    }

    private String summarizeMessages(List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            return "[]";
        }
        List<String> summaries = new ArrayList<>();
        for (int index = 0; index < messages.size(); index++) {
            Message message = messages.get(index);
            if (message == null) {
                summaries.add(index + ":null");
                continue;
            }
            String type = message.getClass().getSimpleName();
            String text = message.getText();
            String normalized = text == null ? "" : text.replace("\r\n", "\n").replace('\n', ' ').trim();
            if (normalized.length() > 200) {
                normalized = normalized.substring(0, 200) + "...";
            }
            summaries.add(index + ":" + type + "[" + normalized + "]");
        }
        return summaries.toString();
    }
}

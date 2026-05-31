package com.argus.rag.assistant.service;

import com.argus.rag.assistant.constants.SessionContextConstants;
import com.argus.rag.assistant.mapper.MessageMapper;
import com.argus.rag.assistant.mapper.SessionContextMapper;
import com.argus.rag.assistant.support.tools.LLMSummarizer;
import com.argus.rag.assistant.model.entity.MessageEntity;
import com.argus.rag.assistant.model.entity.SessionContextEntity;
import com.argus.rag.assistant.model.enums.AssistantChatMode;
import com.argus.rag.common.exception.BusinessException;
import org.springframework.stereotype.Service;

import java.time.Clock;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * 会话上下文记忆服务。
 * <p>负责在消息持久化后更新会话上下文的各类记忆摘要。</p>
 * <p>核心逻辑：</p>
 * <ul>
 *   <li>会话记忆（sessionMemory）：基于增量消息生成的事实摘要</li>
 *   <li>压缩摘要（compactSummary）：当 token 超阈值时的 LLM 语义压缩</li>
 *   <li>使用乐观锁控制并发更新</li>
 * </ul>
 */
@Service
public class SessionContextService {

    private final MessageMapper messageMapper;
    private final SessionContextMapper sessionContextMapper;
    private final LLMSummarizer LLMSummarizer;
    private final Clock clock;

    /**
     * 构造方法。
     */
    public SessionContextService(
            MessageMapper messageMapper,
            SessionContextMapper sessionContextMapper,
            LLMSummarizer LLMSummarizer
    ) {
        this.messageMapper = messageMapper;
        this.sessionContextMapper = sessionContextMapper;
        this.LLMSummarizer = LLMSummarizer;
        this.clock = Clock.systemDefaultZone();
    }


    /**
     * 在模型响应前执行会话上下文记忆维护。
     */
    public void maintainBeforeResponse(Long sessionId, AssistantChatMode toolMode, Long groupId, Long currentMessageId) {
        maintainSessionContext(sessionId, toolMode, groupId, currentMessageId);
    }

    /**
     * 在模型响应后执行会话上下文记忆维护。
     */
    public void maintainAfterResponse(Long sessionId, AssistantChatMode toolMode, Long groupId, Long currentMessageId) {
        maintainSessionContext(sessionId, toolMode, groupId, currentMessageId);
    }

    /**
     * 维护会话上下文记忆的核心逻辑。
     * <p>三类记忆独立判断、独立更新，各自维护独立的消息区间游标：</p>
     * <ul>
     *   <li>会话记忆（sessionMemory）：基于 sessionMemoryRangeEndMessageId 之后的增量消息</li>
     *   <li>压缩摘要（compactSummary）：基于 compactSummaryRangeEndMessageId 之后的增量消息</li>
     * </ul>
     */
    public void maintainSessionContext(Long sessionId, AssistantChatMode toolMode, Long groupId, Long currentMessageId) {
        List<MessageEntity> allMessages = messageMapper.selectBySessionIdOrderByCreatedAt(sessionId);
        SessionContextEntity existingContext = sessionContextMapper.selectBySessionId(sessionId);

        // 三类记忆各自维护独立的消息区间游标，"new" 的含义因策略而异
        long lastMemoryEndId = existingContext == null || existingContext.getSessionMemoryRangeEndMessageId() == null
                ? 0L
                : existingContext.getSessionMemoryRangeEndMessageId();
        long lastCompactEndId = existingContext == null || existingContext.getCompactSummaryRangeEndMessageId() == null
                ? 0L
                : existingContext.getCompactSummaryRangeEndMessageId();

        // session memory 的增量消息：上次 sessionMemory 游标之后的全部消息
        List<MessageEntity> memoryNewMessages = allMessages.stream()
                .filter(message -> message.getId() != null && message.getId() > lastMemoryEndId)
                .toList();
        if (memoryNewMessages.isEmpty()) {
            return;
        }

        // compact 的增量消息数和 token 数：上次 compactSummary 游标之后的全部消息
        long compactNewCount = allMessages.stream()
                .filter(message -> message.getId() != null && message.getId() > lastCompactEndId)
                .count();
        int compactNewTokens = estimateTokens(allMessages.stream()
                .filter(message -> message.getId() != null && message.getId() > lastCompactEndId)
                .toList());

        SessionContextEntity newSessionContext = existingContext == null
                ? new SessionContextEntity()
                : existingContext;
        newSessionContext.setSessionId(sessionId);
        boolean updated = false;

        // 会话记忆：基于增量消息生成事实摘要
        if (shouldMemory(memoryNewMessages)) {
            updateSessionMemory(newSessionContext, existingContext, memoryNewMessages, toolMode, groupId);
            updated = true;
        }

        // 压缩摘要：当 token 超阈值时的 LLM 语义压缩
        int estimatedTokens = estimateTokens(allMessages);
        if (shouldCompact(estimatedTokens, compactNewCount, compactNewTokens)) {
            updateCompactSummary(newSessionContext, existingContext, allMessages, currentMessageId);
            updated = true;
        }

        // 持久化（仅当至少一类记忆发生更新时）
        if (!updated) {
            return;
        }
        // 乐观锁控制：更新时校验版本号，确保并发安全
        newSessionContext.setUpdatedAt(LocalDateTime.now(clock));
        long expectedVersion = existingContext == null || existingContext.getContextVersion() == null
                ? 0L
                : existingContext.getContextVersion();
        newSessionContext.setContextVersion(expectedVersion + 1);

        int updatedRows;
        if (existingContext == null) {
            updatedRows = sessionContextMapper.upsert(newSessionContext);
        } else {
            updatedRows = sessionContextMapper.updateWithVersion(newSessionContext, expectedVersion);
        }
        if (updatedRows != 1) {
            throw new BusinessException("会话上下文记忆写回失败");
        }
    }

    // ==================== 公共工具方法 ====================

    /**
     * 估算消息列表的 token 数。
     */
    public int estimateTokens(List<MessageEntity> messages) {
        if (messages == null || messages.isEmpty()) {
            return 0;
        }
        int totalChars = messages.stream()
                .map(MessageEntity::getContent)
                .filter(content -> content != null && !content.isBlank())
                .mapToInt(String::length)
                .sum();
        return Math.max(1, totalChars / SessionContextConstants.TOKEN_ESTIMATE_DIVISOR);
    }

    // ==================== 会话记忆（sessionMemory） ====================

    /**
     * 判断是否需要更新会话记忆。
     */
    private boolean shouldMemory(List<MessageEntity> newMessages) {
        if (newMessages == null || newMessages.isEmpty()) {
            return false;
        }
        int estimatedTokens = estimateTokens(newMessages);
        return newMessages.size() >= SessionContextConstants.SESSION_MEMORY_MESSAGE_TRIGGER
                || estimatedTokens >= SessionContextConstants.SESSION_MEMORY_TOKEN_TRIGGER;
    }

    /**
     * 更新会话记忆，基于增量消息生成事实摘要。
     */
    private void updateSessionMemory(
            SessionContextEntity newSessionContext,
            SessionContextEntity existingContext,
            List<MessageEntity> newMessages,
            AssistantChatMode toolMode,
            Long groupId
    ) {
        newSessionContext.setSessionMemory(LLMSummarizer.buildSessionMemory(
                existingContext == null ? null : existingContext.getSessionMemory(),
                newMessages,
                toolMode,
                groupId
        ));
        newSessionContext.setSessionMemoryBaseMessageId(newMessages.getFirst().getId());
        newSessionContext.setSessionMemoryRangeEndMessageId(newMessages.getLast().getId());
    }

    // ==================== 压缩摘要（compactSummary） ====================

    /**
     * 判断是否需要触发压缩摘要。
     */
    private boolean shouldCompact(int estimatedTokens, long newMessageCount, long newTokenCount) {
        return estimatedTokens > SessionContextConstants.COMPACT_TOKEN_THRESHOLD
                && (newMessageCount >= SessionContextConstants.COMPACT_MESSAGE_TRIGGER
                || newTokenCount >= SessionContextConstants.COMPACT_TOKEN_TRIGGER);
    }

    /**
     * 更新压缩摘要，调用 LLM 对历史消息进行语义压缩。
     */
    private void updateCompactSummary(
            SessionContextEntity newSessionContext,
            SessionContextEntity existingContext,
            List<MessageEntity> allMessages,
            Long currentMessageId
    ) {
        newSessionContext.setCompactSummary(LLMSummarizer.buildCompactSummary(
                existingContext == null ? null : existingContext.getCompactSummary(),
                newSessionContext.getSessionMemory(),
                collectMessagesToCompact(allMessages, currentMessageId)
        ));
        newSessionContext.setCompactSummaryBaseMessageId(allMessages.getFirst().getId());
        newSessionContext.setCompactSummaryRangeEndMessageId(allMessages.getLast().getId());
    }

    /**
     * 截取当前消息之前的历史消息用于压缩摘要，最多保留最近 {@value SessionContextConstants#MAX_MESSAGES_TO_COMPACT} 条。
     * <p>allMessages 按时间升序，subList 取尾部保证是最近的消息；更早的消息已由 existingCompactSummary 覆盖，无需重复送给 LLM。</p>
     */
    private List<MessageEntity> collectMessagesToCompact(List<MessageEntity> allMessages, Long currentMessageId) {
        List<MessageEntity> messagesToCompact = new ArrayList<>();
        for (MessageEntity message : allMessages) {
            if (message.getId() != null && currentMessageId != null && message.getId() >= currentMessageId) {
                break;
            }
            messagesToCompact.add(message);
        }
        int size = messagesToCompact.size();
        // 如果待压缩历史消息数未超过20条，直接返回；否则截取最近20条历史消息
        if (size <= SessionContextConstants.MAX_MESSAGES_TO_COMPACT) {
            return messagesToCompact;
        }
        return messagesToCompact.subList(size - SessionContextConstants.MAX_MESSAGES_TO_COMPACT, size);
    }
}
package com.argus.rag.assistant.service;

import com.argus.rag.assistant.mapper.MessageMapper;
import com.argus.rag.assistant.mapper.SessionContextMapper;
import com.argus.rag.assistant.mapper.SessionMapper;
import com.argus.rag.assistant.model.dto.message.MessageCreateDTO;
import com.argus.rag.assistant.model.entity.MessageEntity;
import com.argus.rag.assistant.model.entity.SessionContextEntity;
import com.argus.rag.assistant.model.entity.SessionEntity;
import com.argus.rag.assistant.model.enums.MessageRole;
import com.argus.rag.assistant.model.enums.AssistantChatMode;
import com.argus.rag.assistant.model.vo.conversation.ConversationContextVO;
import com.argus.rag.assistant.model.vo.message.MessageVO;
import com.argus.rag.auth.service.CurrentUserService;
import com.argus.rag.common.exception.BusinessException;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Comparator;
import java.util.List;

/**
 * 助手会话管理服务。
 * <p>管理会话消息的持久化、查询、上下文加载等功能。</p>
 * <p>职责包括：消息保存（用户消息和助手消息）、最近消息加载、会话上下文构建、
 * 短期记忆维护触发等。</p>
 */
@Service
public class ConversationContextService {

    private static final int MAX_RECENT_MESSAGE_LIMIT = 100;

    private final MessageMapper messageMapper;
    private final SessionContextMapper sessionContextMapper;
    private final SessionMapper sessionMapper;
    private final SessionContextService sessionContextService;
    private final SessionService sessionService;
    private final CurrentUserService currentUserService;
    private final ObjectMapper objectMapper;

    public ConversationContextService(
            MessageMapper messageMapper,
            SessionContextMapper sessionContextMapper,
            SessionMapper sessionMapper,
            SessionContextService sessionContextService,
            SessionService sessionService,
            CurrentUserService currentUserService,
            ObjectMapper objectMapper
    ) {
        this.messageMapper = messageMapper;
        this.sessionContextMapper = sessionContextMapper;
        this.sessionMapper = sessionMapper;
        this.sessionContextService = sessionContextService;
        this.sessionService = sessionService;
        this.currentUserService = currentUserService;
        this.objectMapper = objectMapper;
    }

    /**
     * 保存用户消息。
     * <p>将用户发送的消息持久化为 {@code USER} 角色的消息记录，并触发短期记忆维护。</p>
     *
     * @param currentUserId 当前登录用户ID
     * @param dto           消息创建请求体，包含会话ID、工具模式、消息内容等
     * @throws BusinessException 如果会话不存在、参数校验失败或消息保存失败
     */
    @Transactional
    public void saveUserMessage(Long currentUserId, MessageCreateDTO dto) {
        saveMessage(currentUserId, dto, MessageRole.USER);
    }

    /**
     * 保存助手回复消息。
     * <p>将 Agent 生成的回复内容持久化为 {@code ASSISTANT} 角色的消息记录，并触发短期记忆维护。</p>
     *
     * @param currentUserId 当前登录用户ID
     * @param dto           消息创建请求体，包含会话ID、工具模式、回复内容等
     * @return 保存后的消息视图对象
     * @throws BusinessException 如果会话不存在、参数校验失败或消息保存失败
     */
    @Transactional
    public MessageVO saveAssistantMessage(Long currentUserId, MessageCreateDTO dto) {
        return saveMessage(currentUserId, dto, MessageRole.ASSISTANT);
    }

    /**
     * 加载指定会话的最近消息列表。
     * <p>按创建时间升序返回最近的 N 条消息，用于构建对话上下文。
     * {@code limit} 会被规范化到最大值 {@value #MAX_RECENT_MESSAGE_LIMIT} 以内。</p>
     *
     * @param currentUserId 当前登录用户ID，用于校验会话归属权
     * @param sessionId     目标会话ID
     * @param limit         要加载的消息数量，取值范围 [1, {@value #MAX_RECENT_MESSAGE_LIMIT}]
     * @return 按时间升序排列的消息视图对象列表
     * @throws BusinessException 如果会话不存在或参数非法
     */
    public List<MessageVO> loadRecentMessages(Long currentUserId, Long sessionId, int limit) {
        SessionEntity session = requireOwnedSession(requireUserId(currentUserId), requireSessionId(sessionId));
        int safeLimit = normalizeLimit(limit);
        return messageMapper.selectRecentBySessionId(session.getId(), safeLimit).stream()
                .sorted(Comparator.comparing(MessageEntity::getCreatedAt)
                        .thenComparing(MessageEntity::getId))
                .map(this::toMessageVO)
                .toList();
    }

    /**
     * 构建单轮对话上下文信息。
     * 构建包含历史消息摘要、压缩摘要、会话记忆和最近消息的上下文对象。：
     * 优先复用
     * </ol>
     *
     * @param currentUserId 当前登录用户ID，用于校验会话归属权
     * @param sessionId     目标会话ID
     * @param recentLimit   要加载的最近消息数量
     * @return 会话对话上下文对象，包含摘要和最近消息
     * @throws BusinessException 如果会话不存在或参数非法
     */
    public ConversationContext buildSingleTurnContext(Long currentUserId, Long sessionId, int recentLimit) {
        SessionEntity session = requireOwnedSession(requireUserId(currentUserId), requireSessionId(sessionId));
        // 至多加载最近100条
        int recentCount = normalizeLimit(recentLimit);
        SessionContextEntity sessionContext = sessionContextMapper.selectBySessionId(session.getId());
        // 加载最近消息
        List<MessageVO> recentMessages = messageMapper.selectRecentBySessionId(session.getId(), recentCount)
                .stream()
                // 按照消息创建时间升序排序，保证新消息在后面，旧消息在前面
                .sorted(Comparator.comparing(MessageEntity::getCreatedAt)
                        .thenComparing(MessageEntity::getId))
                .map(this::toMessageVO)
                .toList();
        // compactSummary / sessionMemory 统一由 SessionContextService.maintainSessionContext 维护
        String compactSummary = sessionContext == null ? null : normalizeOptionalText(sessionContext.getCompactSummary());
        String sessionMemory = sessionContext == null ? null : normalizeOptionalText(sessionContext.getSessionMemory());
        return new ConversationContext(compactSummary, sessionMemory, recentMessages);
    }

    /**
     * 获取会话对话上下文（对外接口）。
     * <p>自动获取当前登录用户身份，加载指定会话的对话上下文，并转为 VO 返回给前端。</p>
     *
     * @param sessionId   目标会话ID
     * @param recentLimit 要加载的最近消息数量
     * @return 对话上下文视图对象，包含摘要文本和最近消息列表
     * @throws BusinessException 如果会话不存在或参数非法
     */
    public ConversationContextVO getConversationContext(
            Long sessionId,
            int recentLimit
    ) {
        Long currentUserId = currentUserService.requireBusinessUser().userId();
        ConversationContext context = buildSingleTurnContext(currentUserId, sessionId, recentLimit);
        return new ConversationContextVO(context.recentMessages());
    }

    private MessageVO saveMessage(
            Long currentUserId,
            MessageCreateDTO dto,
            MessageRole role
    ) {
        Long userId = requireUserId(currentUserId);
        MessageCreateDTO safeDto = requireCreateDTO(dto);
        SessionEntity session = requireOwnedSession(userId, requireSessionId(safeDto.sessionId()));
        LocalDateTime now = LocalDateTime.now();
        MessageEntity entity = buildMessageEntity(session.getId(), safeDto, role, now);
        // 消息入库
        int affectedRows = messageMapper.insert(entity);
        if (affectedRows != 1 || entity.getId() == null) {
            throw new BusinessException("消息保存失败");
        }
        // 更新会话最后消息时间
        int updatedRows = sessionMapper.updateLastMessageAt(session.getId(), userId, now);
        if (updatedRows != 1) {
            throw new BusinessException("会话更新时间刷新失败");
        }
        // 如果是用户消息且是会话中的第一条消息，则尝试自动为会话命名。
        if (role == MessageRole.USER) {
            Long messageCount = messageMapper.countBySessionId(session.getId());
            if (messageCount != null && messageCount == 1L) {
                sessionService.autoRenameSessionIfNeeded(userId, session.getId(), safeDto.content());
            }
        }
        // 💥💥💥💥💥消息持久化完成后再维护短期记忆，这样短期摘要才包含最新消息
        maintainSessionContext(userId, safeDto, role, entity.getId());
        return toMessageVO(entity);
    }

    private MessageEntity buildMessageEntity(
            Long sessionId,
            MessageCreateDTO dto,
            MessageRole role,
            LocalDateTime createdAt
    ) {
        MessageEntity entity = new MessageEntity();
        entity.setSessionId(sessionId);
        entity.setRole(role.name());
        entity.setToolMode(dto.toolMode().name());
        entity.setGroupId(normalizeGroupId(dto.toolMode(), dto.groupId()));
        entity.setContent(requireContent(dto.content()));
        entity.setStructuredPayload(normalizeStructuredPayload(dto.structuredPayload()));
        entity.setCreatedAt(createdAt);
        return entity;
    }

    private SessionEntity requireOwnedSession(Long currentUserId, Long sessionId) {
        SessionEntity session = sessionMapper.selectByIdAndUserId(sessionId, currentUserId);
        if (session == null) {
            throw new BusinessException("会话不存在");
        }
        return session;
    }

    private MessageCreateDTO requireCreateDTO(MessageCreateDTO dto) {
        if (dto == null) {
            throw new BusinessException("消息请求不能为空");
        }
        if (dto.toolMode() == null) {
            throw new BusinessException("toolMode 不能为空");
        }
        return dto;
    }

    private Long requireUserId(Long currentUserId) {
        if (currentUserId == null || currentUserId <= 0) {
            throw new BusinessException("userId 非法");
        }
        return currentUserId;
    }

    private Long requireSessionId(Long sessionId) {
        if (sessionId == null || sessionId <= 0) {
            throw new BusinessException("sessionId 非法");
        }
        return sessionId;
    }

    private String requireContent(String content) {
        if (content == null || content.isBlank()) {
            throw new BusinessException("content 不能为空");
        }
        return content;
    }

    private String normalizeOptionalText(String text) {
        if (text == null) {
            return null;
        }
        String normalized = text.trim();
        return normalized.isEmpty() ? null : normalized;
    }

    private int normalizeLimit(int limit) {
        if (limit <= 0) {
            throw new BusinessException("limit 非法");
        }
        return Math.min(limit, MAX_RECENT_MESSAGE_LIMIT);
    }

    private Long normalizeGroupId(AssistantChatMode toolMode, Long groupId) {
        if (groupId != null && groupId <= 0) {
            throw new BusinessException("groupId 非法");
        }
        if (toolMode == AssistantChatMode.CHAT) {
            return null;
        }
        if (groupId == null) {
            throw new BusinessException("KB_SEARCH 模式必须提供 groupId");
        }
        return groupId;
    }

    private String normalizeStructuredPayload(String structuredPayload) {
        if (structuredPayload == null || structuredPayload.isBlank()) {
            return null;
        }
        try {
            objectMapper.readTree(structuredPayload);
            return structuredPayload;
        } catch (JsonProcessingException exception) {
            throw new BusinessException("structuredPayload 非法", exception);
        }
    }

    private void maintainSessionContext(
            Long userId,
            MessageCreateDTO dto,
            MessageRole role,
            Long messageId
    ) {
        if (messageId == null) {
            return;
        }
        if (role == MessageRole.USER) {
            sessionContextService.maintainBeforeResponse(
                    dto.sessionId(),
                    dto.toolMode(),
                    dto.groupId(),
                    messageId
            );
            return;
        }
        sessionContextService.maintainAfterResponse(
                dto.sessionId(),
                dto.toolMode(),
                dto.groupId(),
                messageId
        );
    }

    private MessageVO toMessageVO(MessageEntity entity) {
        return new MessageVO(
                entity.getId(),
                entity.getSessionId(),
                MessageRole.valueOf(entity.getRole()),
                entity.getToolMode() == null ? null : AssistantChatMode.valueOf(entity.getToolMode()),
                entity.getGroupId(),
                entity.getContent(),
                entity.getStructuredPayload(),
                entity.getCreatedAt()
        );
    }

    /**
     * 单轮对话上下文信息。
     * <p>封装 Agent 调用所需的完整对话上下文信息，包括摘要层和最近消息层，
     * 避免每次请求都将全量历史消息塞入模型上下文窗口。</p>
     *
     * @param compactSummary 压缩摘要（短期记忆压缩结果），可能为 {@code null}
     * @param sessionMemory  会话记忆文本，可能为 {@code null}
     * @param recentMessages 最近的 N 条消息列表，按时间升序排列
     */
    public record ConversationContext(
            String compactSummary,
            String sessionMemory,
            List<MessageVO> recentMessages
    ) {
    }
}

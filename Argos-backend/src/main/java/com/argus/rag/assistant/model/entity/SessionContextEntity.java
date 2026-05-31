package com.argus.rag.assistant.model.entity;

import com.argus.rag.assistant.service.SessionContextService;
import lombok.Data;

/**
 * 会话完整存档
 * 完整保留所有摘要、记忆，消息区间标记、版本锁、更新时间等管理字段，用于数据持久存储、增量更新、过期校验、并发防护，是长期留存的原始数据载体。
 */
import java.time.LocalDateTime;
@Data
public class SessionContextEntity {

    /**
     * 会话ID，主键
     * <p>必填，关联 {@link SessionEntity} 的主键</p>
     */
    private Long sessionId;
    /**
     * 会话记忆内容，由 LLM 生成的长期记忆摘要
     * <p>选填，为空表示尚未生成会话记忆</p>
     */
    private String sessionMemory;
    /**
     * 压缩摘要，由 LLM 生成的更精炼的会话摘要
     * <p>选填，为空表示尚未生成压缩摘要</p>
     */
    private String compactSummary;
    /**
     * 会话记忆覆盖的起始消息ID（基础消息）
     * <p>选填，用于标记 sessionMemory 覆盖的消息范围起点</p>
     */
    private Long sessionMemoryBaseMessageId;
    /**
     * 会话记忆覆盖的结束消息ID（范围终点）
     * <p>选填，用于标记 sessionMemory 覆盖的消息范围终点</p>
     */
    private Long sessionMemoryRangeEndMessageId;
    /**
     * 压缩摘要覆盖的起始消息ID（基础消息）
     * <p>选填，用于标记 compactSummary 覆盖的消息范围起点</p>
     */
    private Long compactSummaryBaseMessageId;
    /**
     * 压缩摘要覆盖的结束消息ID（范围终点）
     * <p>选填，用于标记 compactSummary 覆盖的消息范围终点</p>
     */
    private Long compactSummaryRangeEndMessageId;
    /**
     * 上下文版本号，用于乐观锁控制并发更新
     * <p>选填，初始为 0，每次更新递增</p>
     */
    private Long contextVersion;
    /**
     * 最后更新时间
     * <p>必填，由系统自动设置</p>
     */
    private LocalDateTime updatedAt;

}

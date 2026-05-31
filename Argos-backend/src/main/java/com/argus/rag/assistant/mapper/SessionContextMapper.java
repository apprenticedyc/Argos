package com.argus.rag.assistant.mapper;

import com.argus.rag.assistant.model.entity.SessionContextEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

/**
 * 助手会话上下文数据访问接口。
 * <p>提供会话上下文（摘要、记忆、版本控制）的持久化操作，支持乐观锁并发更新。</p>
 */
@Mapper
public interface SessionContextMapper {

    /**
     * 根据会话ID查询会话上下文
     *
     * @param sessionId 会话ID，不能为空
     * @return 会话上下文实体，若不存在返回 null
     */
    SessionContextEntity selectBySessionId(@Param("sessionId") Long sessionId);

    /**
     * 插入或更新会话上下文记录
     * <p>依赖于数据库的 INSERT ... ON DUPLICATE KEY UPDATE 语义，以 sessionId 为唯一键</p>
     *
     * @param sessionContextEntity 会话上下文实体，不能为 null
     * @return 受影响的行数
     */
    int upsert(SessionContextEntity sessionContextEntity);

    /**
     * 乐观锁方式更新短期记忆内容
     * <p>仅在数据库中的 contextVersion 与传入的 expectedVersion 一致时才执行更新，用于防止并发覆盖</p>
     *
     * @param sessionContextEntity 会话上下文实体，包含需要更新的字段
     * @param expectedVersion               预期的当前版本号，用于乐观锁判断
     * @return 受影响的行数，若版本不匹配则返回 0
     */
    int updateWithVersion(
            @Param("context") SessionContextEntity sessionContextEntity,
            @Param("expectedVersion") Long expectedVersion
    );

    /**
     * 删除指定会话的上下文记录
     *
     * @param sessionId 会话ID，不能为空
     * @return 被删除的记录数
     */
    int deleteBySessionId(@Param("sessionId") Long sessionId);
}

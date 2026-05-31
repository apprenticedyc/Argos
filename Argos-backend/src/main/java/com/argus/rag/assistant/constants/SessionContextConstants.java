package com.argus.rag.assistant.constants;

public final class SessionContextConstants {

    private SessionContextConstants() {
    }

    /** token 估算时字符数与 token 数的比值 */
    public static final int TOKEN_ESTIMATE_DIVISOR = 2;

    // ---- 会话记忆相关阈值 ----

    /** 触发会话记忆更新的消息数 */
    public static final int SESSION_MEMORY_MESSAGE_TRIGGER = 2;

    /** 触发会话记忆更新的 token 数 */
    public static final int SESSION_MEMORY_TOKEN_TRIGGER = 300;

    // ---- 压缩摘要相关阈值 ----

    /** 触发压缩摘要的总 token 阈值 */
    public static final int COMPACT_TOKEN_THRESHOLD = 2000;

    /** 触发压缩摘要的消息数 */
    public static final int COMPACT_MESSAGE_TRIGGER = 3;

    /** 触发压缩摘要的 token 数 */
    public static final int COMPACT_TOKEN_TRIGGER = 500;

    /** 压缩摘要时最多送给 LLM 的消息条数 */
    public static final int MAX_MESSAGES_TO_COMPACT = 10;
}

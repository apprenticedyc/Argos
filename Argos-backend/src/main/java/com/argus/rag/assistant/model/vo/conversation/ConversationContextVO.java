package com.argus.rag.assistant.model.vo.conversation;

import com.argus.rag.assistant.model.vo.message.MessageVO;

import java.util.List;

public record ConversationContextVO(
        /**
         * 最近的消息列表
         */
        List<MessageVO> recentMessages
) {
}

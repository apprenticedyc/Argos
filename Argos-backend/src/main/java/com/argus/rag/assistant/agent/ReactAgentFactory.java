package com.argus.rag.assistant.agent;

import com.alibaba.cloud.ai.graph.CompileConfig;
import com.alibaba.cloud.ai.graph.agent.ReactAgent;
import com.alibaba.cloud.ai.graph.checkpoint.savers.MemorySaver;
import com.argus.rag.assistant.agent.tools.KnowledgeSearchTool;
import com.argus.rag.assistant.agent.tools.KnowledgeSearchToolResultCache;
import com.argus.rag.assistant.support.tools.ConversationContextBuildHook;
import com.argus.rag.assistant.model.enums.AssistantChatMode;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * 助手 React Agent 工厂。
 * <p>负责创建配置好的 {@link ReactAgent} 实例，根据不同的工具模式（CHAT/KB_SEARCH）
 * 动态添加知识库检索工具和方法工具上下文。</p>
 */
@Component
public class ReactAgentFactory {

    // KB_SEARCH 至少需要经历“模型决定调工具 -> 工具执行 -> 模型基于工具结果生成最终回答”，
    // 递归上限过小会导致图在最终回答前被截断，返回空 AssistantMessage。
    private static final int AGENT_RECURSION_LIMIT = 10;

    private final ChatModel chatModel;
    private final ConversationContextBuildHook conversationContextBuildHook;
    private final KnowledgeSearchTool knowledgeSearchTool;

    public ReactAgentFactory(
            @Qualifier("dashScopeChatModel")
            ChatModel chatModel,
            ConversationContextBuildHook conversationContextBuildHook,
            KnowledgeSearchTool knowledgeSearchTool
    ) {
        this.chatModel = chatModel;
        this.conversationContextBuildHook = conversationContextBuildHook;
        this.knowledgeSearchTool = knowledgeSearchTool;
    }

    /**
     * 创建配置好的 ReactAgent 实例。
     * <p>根据指定的 instruction、工具模式和知识库组 ID 构建 Agent，
     * CHAT 模式下仅配置对话能力，KB_SEARCH 模式下额外注入知识库检索工具及其上下文。</p>
     * <p>Agent 配置了短期记忆 Hook、检查点保存器和递归上限，
     * 确保单次调用能完成"模型推理 → 工具执行 → 最终回复"的完整链路。</p>
     *
     * @param instruction  系统提示词，设定 Agent 的角色和行为规范
     * @param toolMode     工具模式（CHAT 仅对话 / KB_SEARCH 知识库检索）
     * @param groupId      知识库组 ID，仅 KB_SEARCH 模式生效，CHAT 模式可传 {@code null}
     * @param resultHolder 知识库检索结果持有器，用于跨步骤传递检索引用
     * @return 构建完成的 ReactAgent 实例
     */
    public ReactAgent createAgent(
            String instruction,
            AssistantChatMode toolMode,
            Long groupId,
            KnowledgeSearchToolResultCache resultHolder
    ) {
        // 当前“仅对话模式”仍复用 ReactAgent 运行时：
        // system prompt 与短期上下文保留，长期记忆工具已整体移除。
        com.alibaba.cloud.ai.graph.agent.Builder builder = ReactAgent.builder()
                .name("assistant_chat_agent")
                .model(chatModel)
                .instruction(instruction)
                .hooks(conversationContextBuildHook)
                .compileConfig(CompileConfig.builder()
                        .recursionLimit(AGENT_RECURSION_LIMIT)
                        .build())
                // MemorySaver 只负责图运行态的 checkpoint，不是正式业务记忆的事实源。
                .saver(new MemorySaver());
        if (toolMode == AssistantChatMode.KB_SEARCH) {
            builder.methodTools(knowledgeSearchTool)
                    // 传给工具的上下文参数，供工具执行时读取。
                    .toolContext(Map.of(
                            KnowledgeSearchTool.GROUP_ID_CONTEXT_KEY, groupId,
                            KnowledgeSearchTool.RESULT_HOLDER_CONTEXT_KEY, resultHolder
                    ));
        }
        return builder.build();
    }
}

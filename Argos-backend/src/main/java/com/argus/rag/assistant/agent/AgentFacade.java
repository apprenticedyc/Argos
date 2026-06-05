package com.argus.rag.assistant.agent;

import com.alibaba.cloud.ai.graph.NodeOutput;
import com.alibaba.cloud.ai.graph.RunnableConfig;
import com.alibaba.cloud.ai.graph.agent.ReactAgent;
import com.alibaba.cloud.ai.graph.exception.GraphRunnerException;
import com.alibaba.cloud.ai.graph.streaming.OutputType;
import com.alibaba.cloud.ai.graph.streaming.StreamingOutput;
import com.argus.rag.assistant.agent.tools.KnowledgeSearchToolResultCache;
import com.argus.rag.assistant.model.enums.AssistantChatMode;
import com.argus.rag.assistant.model.vo.chat.AgentResult;
import com.argus.rag.assistant.support.config.SystemPromptBuilder;
import com.argus.rag.assistant.support.config.AssistantRunnableConfigFactory;
import com.argus.rag.common.exception.BusinessException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;

import java.util.function.Consumer;

/**
 * 助手 Agent 门面服务。
 * <p>封装 ReactAgent 的调用逻辑，提供同步聊天和流式聊天的统一入口。</p>
 * <p>负责构建运行时 instruction 和 RunnableConfig，处理 Agent 调用结果，
 * 并在调用失败时转换为业务异常抛出。</p>
 */
@Component
public class AgentFacade {

    private static final Logger log = LoggerFactory.getLogger(AgentFacade.class);

    private final ReactAgentFactory reactAgentFactory;
    private final AssistantRunnableConfigFactory assistantRunnableConfigFactory;
    private final SystemPromptBuilder systemPromptBuilder;

    public AgentFacade(ReactAgentFactory reactAgentFactory,
                       AssistantRunnableConfigFactory assistantRunnableConfigFactory,
                       SystemPromptBuilder systemPromptBuilder) {
        this.reactAgentFactory = reactAgentFactory;
        this.assistantRunnableConfigFactory = assistantRunnableConfigFactory;
        this.systemPromptBuilder = systemPromptBuilder;
    }

    /**
     * 同步聊天。
     * <p>基于用户输入、会话上下文和工具模式构建 instruction，通过 ReactAgent
     * 同步执行对话，返回助手的文本回复和引用列表。</p>
     *
     * @param userId      当前用户 ID
     * @param sessionId   当前会话 ID
     * @param toolMode    工具模式（CHAT 仅对话 / KB_SEARCH 知识库检索）
     * @param groupId     知识库组 ID，CHAT 模式下可传 {@code null}
     * @param userMessage 用户输入的文本
     * @return 包含助手回复文本和引用列表的结果对象
     * @throws com.argus.rag.common.exception.BusinessException 当 Agent 调用失败或返回内容为空时
     */
    public AgentResult chat(Long userId, Long sessionId, AssistantChatMode toolMode, Long groupId, String userMessage) {
        // 这里把“仅对话模式”的运行时输入收口成两部分：
        // 1) instruction：系统提示词
        // 2) runnableConfig：session/user/toolMode 等 metadata，供 hooks 在 BEFORE_MODEL 阶段读取
        String instruction = systemPromptBuilder.buildSystemPrompt(userId, sessionId, toolMode, groupId);
        RunnableConfig runnableConfig = assistantRunnableConfigFactory.create(userId, sessionId, toolMode, groupId);
        // 当前虽然是“仅对话模式”，底层仍然使用 ReactAgent，只是 system prompt 已改成纯对话风格。
        KnowledgeSearchToolResultCache toolResult = new KnowledgeSearchToolResultCache();
        ReactAgent agent = reactAgentFactory.createAgent(instruction, toolMode, groupId, toolResult);
        AssistantMessage assistantMessage;
        try {
            assistantMessage = agent.call(userMessage, runnableConfig);
        } catch (GraphRunnerException exception) {
            throw new BusinessException("助手调用失败", exception);
        }
        if (log.isDebugEnabled()) {
            log.debug("AgentFacade.chat result. userId={}, sessionId={}, toolMode={}, groupId={}, text={}, hasToolCalls={}, toolCallCount={}", userId, sessionId, toolMode, groupId, assistantMessage == null ? null : abbreviate(assistantMessage.getText()), assistantMessage != null && assistantMessage.hasToolCalls(), assistantMessage == null || assistantMessage.getToolCalls() == null ? 0 : assistantMessage.getToolCalls()
                    .size());
        }
        if (assistantMessage == null || assistantMessage.getText() == null || assistantMessage.getText().isBlank()) {
            throw new BusinessException("助手返回内容为空");
        }
        // 💥💥💥最后返回结果包含工具结果缓存中的引用列表，供前端展示使用。
        return new AgentResult(assistantMessage.getText(), toolResult.currentCitations());
    }

    /**
     * 流式聊天。
     * <p>与 {@link #chat} 类似，但通过 ReactAgent 的流式接口执行对话，
     * 模型生成的文本增量会通过 {@code sseMessageConsumer} 逐段推送给调用方。</p>
     * <p>内部处理流式节点的输出拼接与去重，确保最终返回完整的助手回复。</p>
     *
     * @param userId        当前用户 ID
     * @param sessionId     当前会话 ID
     * @param toolMode      工具模式（CHAT 仅对话 / KB_SEARCH 知识库检索）
     * @param groupId       知识库组 ID，CHAT 模式下可传 {@code null}
     * @param userMessage   用户输入的文本
     * @param sseMessageConsumer 流式文本增量回调，每个增量片段都会调用此回调
     * @return 包含助手完整回复文本和引用列表的结果对象
     * @throws com.argus.rag.common.exception.BusinessException 当 Agent 调用失败或返回内容为空时
     */
    public AgentResult streamChat(Long userId, Long sessionId, AssistantChatMode toolMode, Long groupId,
                                  String userMessage, Consumer<String> sseMessageConsumer) {
        String instruction = systemPromptBuilder.buildSystemPrompt(userId, sessionId, toolMode, groupId);
        RunnableConfig runnableConfig = assistantRunnableConfigFactory.create(userId, sessionId, toolMode, groupId);
        // 缓存工具调用结果, 给 AgentFacade 最后构建响应时用，把引用传给前端
        KnowledgeSearchToolResultCache toolResult = new KnowledgeSearchToolResultCache();
        ReactAgent agent = reactAgentFactory.createAgent(instruction, toolMode, groupId, toolResult);
        // 将流式增量文本拼接为完整回复
        StringBuilder finalReply = new StringBuilder();
        try {
            // stream() 返回的是图执行过程中的节点输出，这里只抽取模型实际产出的文本 chunk。
            Flux<NodeOutput> stream = agent.stream(userMessage, runnableConfig);
            // blockLast() 会一直等直到流结束，期间每有新的节点输出就调用 handleStreamingOutput() 处理增量文本发给前端。
            stream.doOnNext(output -> handleStreamingOutput(output, sseMessageConsumer, finalReply)).blockLast();
        } catch (GraphRunnerException exception) {
            throw new BusinessException("助手调用失败", exception);
        }
        String reply = finalReply.toString().trim();
        if (reply.isBlank()) {
            throw new BusinessException("助手返回内容为空");
        }
        return new AgentResult(reply, toolResult.currentCitations());
    }

    /**
     * 处理流式输出中的增量文本。
     */
    private void handleStreamingOutput(NodeOutput output, Consumer<String> sseMessageConsumer, StringBuilder finalReply) {
        // 仅处理流式类型输出
        if (!(output instanceof StreamingOutput streamingOutput)) {
            return;
        }
        OutputType type = streamingOutput.getOutputType();
        Message message = streamingOutput.message();
        // 仅处理模型回答消息，工具调用结果等其他类型节点输出不处理。
        if (!(message instanceof AssistantMessage assistantMessage)) {
            return;
        }
        if (type == OutputType.AGENT_MODEL_STREAMING) {
            // 正常流式路径下，模型边生成边向前端透传文本。
            String newChunk = normalizeStreamingChunk(assistantMessage.getText(), finalReply.toString());
            if (newChunk.isBlank()) {
                return;
            }
            // 更新累计回复内容，供后续增量去重使用。
            finalReply.append(newChunk);
            // 将增量文本推送给前端，前端负责拼接展示。
            sseMessageConsumer.accept(newChunk);
            return;
        }
        // 有些模型没有流式输出能力，只有在最终完成时才返回完整文本，在这做兜底处理一次性把文本发给前端。
        if (type == OutputType.AGENT_MODEL_FINISHED && !assistantMessage.hasToolCalls() && finalReply.isEmpty()) {
            String fullText = assistantMessage.getText();
            if (fullText == null || fullText.isBlank()) {
                return;
            }
            finalReply.append(fullText);
            sseMessageConsumer.accept(fullText);
        }
    }


    private String normalizeStreamingChunk(String currentFullContent, String accumulatedContent) {
        // 本轮无内容直接返回空
        if (currentFullContent == null || currentFullContent.isBlank()) {
            return "";
        }
        // 暂无历史内容，直接返回当前文本
        if (accumulatedContent.isEmpty()) {
            return currentFullContent;
        }
        // 截掉已展示的历史前缀，只保留新增部分
        if (currentFullContent.startsWith(accumulatedContent)) {
            return currentFullContent.substring(accumulatedContent.length());
        }
        // 前缀不匹配则原样返回
        return currentFullContent;
    }

    private String abbreviate(String text) {
        if (text == null) {
            return null;
        }
        String normalized = text.replace("\r\n", "\n").replace('\n', ' ').trim();
        if (normalized.length() > 200) {
            return normalized.substring(0, 200) + "...";
        }
        return normalized;
    }
}

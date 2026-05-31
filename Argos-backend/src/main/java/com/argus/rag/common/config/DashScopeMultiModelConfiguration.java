package com.argus.rag.common.config;

import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatOptions;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * DashScope 多模态模型全局配置。
 * <p>
 * 通过自定义 {@link ChatClient.Builder} Bean，将 {@code withMultiModel(true)} 设为默认选项，
 * 使所有通过 Builder 创建的 ChatClient 自动走 DashScope 多模态 API 端点。
 * <p>
 * 这样 gui-plus 等多模态模型可以通过标准 ChatClient 接口调用，
 * 而无需在每个调用点手动设置 DashScopeChatOptions。
 *
 * @author Argus-RAG Team
 */
@Configuration
public class DashScopeMultiModelConfiguration {

    /**
     * 自定义 ChatClient.Builder，全局启用 DashScope 多模态模式。
     * <p>
     * Spring AI 的 {@code ChatClient.Builder} 由自动配置创建，
     * 此处覆盖后注入到所有依赖 {@code ChatClient.Builder} 的地方。
     *
     * @param chatModel Spring AI 自动配置的 ChatModel（DashScopeChatModel）
     * @return 配置了多模态选项的 ChatClient.Builder
     */
    @Bean
    ChatClient.Builder chatClientBuilder(
            @Qualifier("dashScopeChatModel") ChatModel chatModel) {
        DashScopeChatOptions dashScopeOptions = DashScopeChatOptions.builder()
                .withMultiModel(true)
                .build();
        return ChatClient.builder(chatModel)
                .defaultOptions(dashScopeOptions);
    }
}
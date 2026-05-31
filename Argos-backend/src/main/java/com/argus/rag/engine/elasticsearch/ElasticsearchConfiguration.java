package com.argus.rag.engine.elasticsearch;

import org.apache.http.HttpHost;
import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.RestHighLevelClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Elasticsearch 客户端配置，提供 {@link RestHighLevelClient} Bean。
 * <p>
 * 连接参数通过 {@code elasticsearch.*} 配置项注入，与 application.yml 保持一致。
 * 超时设置为 5 秒（connect + socket），匹配原有 HttpClient 行为。
 *
 * @author Argus-RAG Team
 */
@Configuration
public class ElasticsearchConfiguration {

    /**
     * 创建 ES 高级 REST 客户端 Bean。
     * <p>
     * {@code destroyMethod = "close"} 确保应用关闭时释放底层连接池。
     *
     * @param host   ES 主机地址
     * @param port   ES 端口
     * @param scheme ES 协议（http / https）
     * @return RestHighLevelClient 实例
     */
    @Bean(destroyMethod = "close")
    RestHighLevelClient elasticsearchClient(
            @Value("${elasticsearch.host:localhost}") String host,
            @Value("${elasticsearch.port:9200}") int port,
            @Value("${elasticsearch.scheme:http}") String scheme
    ) {
        return new RestHighLevelClient(
                RestClient.builder(new HttpHost(host, port, scheme))
                        .setRequestConfigCallback(configBuilder ->
                                configBuilder
                                        .setConnectTimeout(5000)
                                        .setSocketTimeout(5000)
                        )
        );
    }
}

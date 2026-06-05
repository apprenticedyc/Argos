package com.argus.rag.engine.elasticsearch;

import com.argus.rag.common.exception.BusinessException;
import com.argus.rag.ingestion.model.entity.DocumentChunkEntity;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.elasticsearch.action.index.IndexRequest;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.client.IndicesClient;
import org.elasticsearch.client.RequestOptions;
import org.elasticsearch.client.RestHighLevelClient;
import org.elasticsearch.client.indices.CreateIndexRequest;
import org.elasticsearch.client.indices.CreateIndexResponse;
import org.elasticsearch.client.indices.GetIndexRequest;
import org.elasticsearch.common.bytes.BytesArray;
import org.elasticsearch.index.reindex.BulkByScrollResponse;
import org.elasticsearch.index.reindex.DeleteByQueryRequest;
import org.elasticsearch.search.SearchHit;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import org.mockito.Answers;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * {@link ElasticsearchService} 单元测试。
 * <p>
 * 通过 Mock {@link RestHighLevelClient}，用 {@link ArgumentCaptor} 捕获实际请求，
 * 验证重构后的 HLRC 版本与原 HttpClient 版本发送给 ES 的请求结构一致。
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("ElasticsearchService 单元测试")
class ElasticsearchServiceTest {

    private static final String INDEX_NAME = "dd_rag_document_chunks";

    @Mock
    private RestHighLevelClient client;

    @Mock
    private IndicesClient indicesClient;

    private ElasticsearchService service;

    @BeforeEach
    void setUp() throws IOException {
        lenient().when(client.indices()).thenReturn(indicesClient);
        // 默认索引已存在，跳过创建
        lenient().when(indicesClient.exists(any(GetIndexRequest.class), any())).thenReturn(true);

        service = new ElasticsearchService(client, INDEX_NAME);
    }

    // ──────────────────────────────────────────────
    // 辅助方法
    // ──────────────────────────────────────────────

    private static DocumentChunkEntity buildChunk(Long id, Long groupId, Long documentId, int chunkIndex, String text) {
        DocumentChunkEntity chunk = new DocumentChunkEntity();
        chunk.setId(id);
        chunk.setGroupId(groupId);
        chunk.setDocumentId(documentId);
        chunk.setChunkIndex(chunkIndex);
        chunk.setChunkText(text);
        return chunk;
    }

    private static SearchHit mockSearchHit(float score, Map<String, Object> source) {
        SearchHit hit = mock(SearchHit.class);
        when(hit.getScore()).thenReturn(score);
        when(hit.getSourceAsMap()).thenReturn(source);
        return hit;
    }

    // ──────────────────────────────────────────────
    // 参数校验
    // ──────────────────────────────────────────────

    @Nested
    @DisplayName("参数校验")
    class GuardClauses {

        @Test
        @DisplayName("search: groupId 为 null 时返回空列表")
        void search_shouldReturnEmptyWhenGroupIdIsNull() {
            assertThat(service.search(null, "test", 5)).isEmpty();
            verifyNoInteractions(client);
        }

        @Test
        @DisplayName("search: groupId 为 0 时返回空列表")
        void search_shouldReturnEmptyWhenGroupIdIsZero() {
            assertThat(service.search(0L, "test", 5)).isEmpty();
            verifyNoInteractions(client);
        }

        @Test
        @DisplayName("search: question 为 null 时返回空列表")
        void search_shouldReturnEmptyWhenQuestionIsNull() {
            assertThat(service.search(1L, null, 5)).isEmpty();
            verifyNoInteractions(client);
        }

        @Test
        @DisplayName("search: question 为空白时返回空列表")
        void search_shouldReturnEmptyWhenQuestionIsBlank() {
            assertThat(service.search(1L, "   ", 5)).isEmpty();
            verifyNoInteractions(client);
        }

        @Test
        @DisplayName("search: topK 为 0 时返回空列表")
        void search_shouldReturnEmptyWhenTopKIsZero() {
            assertThat(service.search(1L, "test", 0)).isEmpty();
            verifyNoInteractions(client);
        }

        @Test
        @DisplayName("indexReadyChunks: fileName 为 null 时不调用 ES")
        void indexReadyChunks_shouldSkipWhenFileNameIsNull() {
            service.indexReadyChunks(null, List.of(buildChunk(1L, 1L, 1L, 0, "text")));
            verifyNoInteractions(client);
        }

        @Test
        @DisplayName("indexReadyChunks: chunks 为空时不调用 ES")
        void indexReadyChunks_shouldSkipWhenChunksEmpty() {
            service.indexReadyChunks("file.pdf", List.of());
            verifyNoInteractions(client);
        }

        @Test
        @DisplayName("indexReadyChunks: chunks 为 null 时不调用 ES")
        void indexReadyChunks_shouldSkipWhenChunksNull() {
            service.indexReadyChunks("file.pdf", null);
            verifyNoInteractions(client);
        }

        @Test
        @DisplayName("deleteDocumentChunks: documentId 为 null 时不调用 ES")
        void deleteDocumentChunks_shouldSkipWhenDocumentIdIsNull() throws IOException {
            service.deleteDocumentChunks(null);
            verify(client, never()).deleteByQuery(any(), any());
        }

        @Test
        @DisplayName("deleteDocumentChunks: documentId 为 0 时不调用 ES")
        void deleteDocumentChunks_shouldSkipWhenDocumentIdIsZero() throws IOException {
            service.deleteDocumentChunks(0L);
            verify(client, never()).deleteByQuery(any(), any());
        }
    }

    // ──────────────────────────────────────────────
    // 索引写入
    // ──────────────────────────────────────────────

    @Nested
    @DisplayName("索引写入")
    class IndexChunks {

        @BeforeEach
        void stubIndex() throws IOException {
            lenient().when(client.index(any(IndexRequest.class), any())).thenReturn(mock(IndexResponse.class));
        }

        @Test
        @DisplayName("应逐条写入 chunk，IndexRequest 包含正确字段")
        void indexReadyChunks_shouldIndexEachChunkWithCorrectFields() throws IOException {
            DocumentChunkEntity chunk = buildChunk(42L, 1L, 100L, 0, "测试文本");

            service.indexReadyChunks("文档.pdf", List.of(chunk));

            ArgumentCaptor<IndexRequest> captor = ArgumentCaptor.forClass(IndexRequest.class);
            verify(client).index(captor.capture(), eq(RequestOptions.DEFAULT));

            IndexRequest request = captor.getValue();
            assertThat(request.index()).isEqualTo(INDEX_NAME);
            assertThat(request.id()).isEqualTo("42");

            Map<String, Object> source = request.sourceAsMap();
            assertThat(source).containsKeys("chunkId", "groupId", "documentId", "chunkIndex", "fileName", "chunkText", "status", "deleted");
            assertThat(((Number) source.get("chunkId")).longValue()).isEqualTo(42L);
            assertThat(((Number) source.get("groupId")).longValue()).isEqualTo(1L);
            assertThat(((Number) source.get("documentId")).longValue()).isEqualTo(100L);
            assertThat(source.get("chunkIndex")).isEqualTo(0);
            assertThat(source.get("fileName")).isEqualTo("文档.pdf");
            assertThat(source.get("chunkText")).isEqualTo("测试文本");
            assertThat(source.get("status")).isEqualTo("READY");
            assertThat(source.get("deleted")).isEqualTo(false);
        }

        @Test
        @DisplayName("多个 chunk 应调用对应次数的 client.index")
        void indexReadyChunks_shouldCallIndexForEachChunk() throws IOException {
            List<DocumentChunkEntity> chunks = List.of(
                    buildChunk(1L, 1L, 100L, 0, "a"),
                    buildChunk(2L, 1L, 100L, 1, "b"),
                    buildChunk(3L, 1L, 100L, 2, "c")
            );

            service.indexReadyChunks("file.pdf", chunks);

            verify(client, times(3)).index(any(IndexRequest.class), eq(RequestOptions.DEFAULT));
        }

        @Test
        @DisplayName("chunk 缺少必要字段时应抛出 BusinessException")
        void indexReadyChunks_shouldThrowWhenChunkInvalid() {
            DocumentChunkEntity badChunk = new DocumentChunkEntity();
            badChunk.setId(1L);

            assertThatThrownBy(() -> service.indexReadyChunks("file.pdf", List.of(badChunk)))
                    .isInstanceOf(BusinessException.class)
                    .hasMessageContaining("ES 索引写入缺少必要 chunk 字段");
        }
    }

    // ──────────────────────────────────────────────
    // 文档删除
    // ──────────────────────────────────────────────

    @Nested
    @DisplayName("文档删除")
    class DeleteChunks {

        @Test
        @DisplayName("应发送正确的 DeleteByQueryRequest")
        void deleteDocumentChunks_shouldSendCorrectQuery() throws IOException {
            when(client.deleteByQuery(any(DeleteByQueryRequest.class), any()))
                    .thenReturn(mock(BulkByScrollResponse.class));

            service.deleteDocumentChunks(100L);

            ArgumentCaptor<DeleteByQueryRequest> captor = ArgumentCaptor.forClass(DeleteByQueryRequest.class);
            verify(client).deleteByQuery(captor.capture(), eq(RequestOptions.DEFAULT));

            DeleteByQueryRequest request = captor.getValue();
            // 验证目标索引
            assertThat(request.indices()).containsExactly(INDEX_NAME);
            // 验证 query 包含 documentId 条件
            String queryStr = request.getSearchRequest().source().query().toString();
            assertThat(queryStr).contains("documentId").contains("100");
        }

        @Test
        @DisplayName("ES 抛 IOException 时应静默降级不抛异常")
        void deleteDocumentChunks_shouldDegradeOnIOException() throws IOException {
            when(client.deleteByQuery(any(), any())).thenThrow(new IOException("连接超时"));

            service.deleteDocumentChunks(100L);

            verify(client).deleteByQuery(any(), any());
        }

        @Test
        @DisplayName("ES 抛 RuntimeException 时应静默降级不抛异常")
        void deleteDocumentChunks_shouldDegradeOnRuntimeException() throws IOException {
            when(client.deleteByQuery(any(), any())).thenThrow(new RuntimeException("ES 错误"));

            service.deleteDocumentChunks(100L);

            verify(client).deleteByQuery(any(), any());
        }
    }

    // ──────────────────────────────────────────────
    // 关键词检索
    // ──────────────────────────────────────────────

    @Nested
    @DisplayName("关键词检索")
    class Search {

        @Test
        @DisplayName("应发送正确的 SearchRequest，包含 bool query + rescore")
        void search_shouldSendCorrectSearchRequest() throws IOException {
            when(client.search(any(SearchRequest.class), any())).thenThrow(new IOException("stop"));

            try { service.search(1L, "上传流程", 10); } catch (Exception ignored) {}

            ArgumentCaptor<SearchRequest> captor = ArgumentCaptor.forClass(SearchRequest.class);
            verify(client).search(captor.capture(), eq(RequestOptions.DEFAULT));

            SearchRequest request = captor.getValue();
            assertThat(request.indices()).containsExactly(INDEX_NAME);

            String requestJson = request.source().toString();

            // 验证 bool query filter: groupId=1, status=READY, deleted=false
            assertThat(requestJson).contains("\"groupId\"").contains("1");
            assertThat(requestJson).contains("\"status\"").contains("READY");
            assertThat(requestJson).contains("\"deleted\"").contains("false");

            // 验证 should 子句（4个）
            assertThat(requestJson).contains("fileName");
            assertThat(requestJson).contains("chunkText");
            assertThat(requestJson).contains("match_phrase");
            assertThat(requestJson).contains("match");

        }

        @Test
        @DisplayName("HLRC 生成的 DSL 应与原始手写 JSON 结构完全一致")
        void search_dslShouldMatchOriginalHandCraftedJson() throws Exception {
            when(client.search(any(SearchRequest.class), any())).thenThrow(new IOException("stop"));
            try { service.search(1L, "上传流程", 10); } catch (Exception ignored) {}

            ArgumentCaptor<SearchRequest> captor = ArgumentCaptor.forClass(SearchRequest.class);
            verify(client).search(captor.capture(), eq(RequestOptions.DEFAULT));

            String dslJson = captor.getValue().source().toString();
            ObjectMapper mapper = new ObjectMapper();
            JsonNode dsl = mapper.readTree(dslJson);

            // ── 顶层字段 ──
            assertThat(dsl.path("size").asInt()).isEqualTo(10);
            // _source 可能是数组 ["field1", ...] 或对象 {"includes": [...]}，取决于序列化方式
            JsonNode source = dsl.path("_source");
            JsonNode sourceIncludes = source.isArray() ? source : source.path("includes");
            assertThat(mapper.convertValue(sourceIncludes, String[].class))
                    .containsExactlyInAnyOrder("groupId", "documentId", "chunkId", "chunkIndex", "fileName", "chunkText");

            // ── bool query ──
            JsonNode boolNode = dsl.path("query").path("bool");
            assertThat(boolNode.isMissingNode()).isFalse();

            // filter: 3 个 term query（顺序不固定，逐个检查存在性）
            JsonNode filter = boolNode.path("filter");
            assertThat(filter).hasSize(3);
            String filterJson = filter.toString();
            assertThat(filterJson).contains("\"groupId\"");
            assertThat(filterJson).contains("\"status\"");
            assertThat(filterJson).contains("\"READY\"");
            assertThat(filterJson).contains("\"deleted\"");
            assertThat(filterJson).contains("\"value\"");

            // should: 4 个 clause，含正确 boost
            JsonNode should = boolNode.path("should");
            assertThat(should).hasSize(4);
            assertThat(should.get(0).path("match_phrase").path("fileName").path("query").asText()).isEqualTo("上传流程");
            assertThat(should.get(0).path("match_phrase").path("fileName").path("boost").asDouble()).isEqualTo(8.0);
            assertThat(should.get(1).path("match").path("fileName").path("query").asText()).isEqualTo("上传流程");
            assertThat(should.get(1).path("match").path("fileName").path("boost").asDouble()).isEqualTo(4.0);
            assertThat(should.get(2).path("match_phrase").path("chunkText").path("query").asText()).isEqualTo("上传流程");
            assertThat(should.get(2).path("match_phrase").path("chunkText").path("boost").asDouble()).isEqualTo(6.0);
            assertThat(should.get(3).path("match").path("chunkText").path("query").asText()).isEqualTo("上传流程");
            assertThat(should.get(3).path("match").path("chunkText").path("boost").asDouble()).isEqualTo(3.0);

            // minimum_should_match
            assertThat(boolNode.path("minimum_should_match").asInt()).isEqualTo(1);
        }

        @Test
        @DisplayName("应正确解析 SearchHit 为 KeywordHit")
        void parseSearchHits_shouldParseCorrectly() {
            SearchHit hit1 = mockSearchHit(8.5f, Map.of(
                    "documentId", 100L, "chunkId", 42L, "chunkIndex", 0,
                    "fileName", "文档.pdf", "chunkText", "这是测试文本"
            ));
            SearchHit hit2 = mockSearchHit(3.2f, Map.of(
                    "documentId", 100L, "chunkId", 43L, "chunkIndex", 1,
                    "fileName", "文档.pdf", "chunkText", "第二段文本"
            ));

            List<ElasticsearchService.KeywordHit> results =
                    service.parseSearchHits(new SearchHit[]{hit1, hit2}, 1L, 10, System.nanoTime());

            assertThat(results).hasSize(2);
            assertThat(results.get(0).documentId()).isEqualTo(100L);
            assertThat(results.get(0).chunkId()).isEqualTo(42L);
            assertThat(results.get(0).chunkIndex()).isEqualTo(0);
            assertThat(results.get(0).fileName()).isEqualTo("文档.pdf");
            assertThat(results.get(0).chunkText()).isEqualTo("这是测试文本");
            assertThat(results.get(0).rawScore()).isEqualTo(8.5f);
            assertThat(results.get(0).normalizedScore()).isBetween(0.0, 1.0);
            assertThat(results.get(1).chunkId()).isEqualTo(43L);
            assertThat(results.get(1).rawScore()).isEqualTo(3.2f);
        }

        @Test
        @DisplayName("空 hits 数组应返回空列表")
        void parseSearchHits_shouldReturnEmptyForNoHits() {
            assertThat(service.parseSearchHits(new SearchHit[0], 1L, 10, System.nanoTime())).isEmpty();
        }

        @Test
        @DisplayName("ES 抛 IOException 时应降级返回空列表")
        void search_shouldDegradeOnIOException() throws IOException {
            when(client.search(any(), any())).thenThrow(new IOException("连接超时"));
            assertThat(service.search(1L, "test", 10)).isEmpty();
        }

        @Test
        @DisplayName("ES 抛 RuntimeException 时应降级返回空列表")
        void search_shouldDegradeOnRuntimeException() throws IOException {
            when(client.search(any(), any())).thenThrow(new RuntimeException("ES 错误"));
            assertThat(service.search(1L, "test", 10)).isEmpty();
        }

        @Test
        @DisplayName("fetchSource 应限制返回字段为 6 个")
        void search_shouldLimitFetchSource() throws IOException {
            when(client.search(any(), any())).thenThrow(new IOException("stop"));
            try { service.search(1L, "test", 5); } catch (Exception ignored) {}

            ArgumentCaptor<SearchRequest> captor = ArgumentCaptor.forClass(SearchRequest.class);
            verify(client).search(captor.capture(), eq(RequestOptions.DEFAULT));

            String sourceJson = captor.getValue().source().toString();
            assertThat(sourceJson).contains("groupId", "documentId", "chunkId", "chunkIndex", "fileName", "chunkText");
        }
    }

    // ──────────────────────────────────────────────
    // 索引初始化
    // ──────────────────────────────────────────────

    @Nested
    @DisplayName("索引初始化")
    class IndexInitialization {

        @Test
        @DisplayName("索引已存在时不应调用 create")
        void shouldNotCreateWhenIndexExists() throws IOException {
            when(indicesClient.exists(any(GetIndexRequest.class), any())).thenReturn(true);
            when(client.index(any(IndexRequest.class), any())).thenReturn(mock(IndexResponse.class));

            service.indexReadyChunks("file.pdf", List.of(buildChunk(1L, 1L, 1L, 0, "text")));

            verify(indicesClient, never()).create(any(CreateIndexRequest.class), any());
        }

        @Test
        @DisplayName("索引不存在时应自动创建，settings 包含 IK 分词器配置")
        void shouldCreateIndexWhenNotExists() throws IOException {
            when(indicesClient.exists(any(GetIndexRequest.class), any())).thenReturn(false);
            when(indicesClient.create(any(CreateIndexRequest.class), any()))
                    .thenReturn(mock(CreateIndexResponse.class));
            when(client.index(any(IndexRequest.class), any())).thenReturn(mock(IndexResponse.class));

            service.indexReadyChunks("file.pdf", List.of(buildChunk(1L, 1L, 1L, 0, "text")));

            ArgumentCaptor<CreateIndexRequest> captor = ArgumentCaptor.forClass(CreateIndexRequest.class);
            verify(indicesClient).create(captor.capture(), eq(RequestOptions.DEFAULT));

            CreateIndexRequest request = captor.getValue();
            assertThat(request.index()).isEqualTo(INDEX_NAME);

            // 验证 settings 包含 IK 分词器
            String settingsJson = request.settings().toString();
            assertThat(settingsJson).contains("ddrag_ik_index", "ik_max_word", "ddrag_ik_search", "ik_smart");

            // 验证 mappings 包含字段定义
            String mappingsStr = request.mappings().utf8ToString();
            assertThat(mappingsStr).contains("groupId", "documentId", "chunkId", "chunkIndex");
            assertThat(mappingsStr).contains("fileName", "chunkText");
            assertThat(mappingsStr).contains("status", "deleted");
        }

        @Test
        @DisplayName("第二次调用不应再检查索引是否存在（双重检查锁定）")
        void shouldNotRecheckIndexOnSecondCall() throws IOException {
            when(indicesClient.exists(any(GetIndexRequest.class), any())).thenReturn(true);
            when(client.index(any(IndexRequest.class), any())).thenReturn(mock(IndexResponse.class));

            List<DocumentChunkEntity> chunks = List.of(buildChunk(1L, 1L, 1L, 0, "text"));
            service.indexReadyChunks("a.pdf", chunks);
            service.indexReadyChunks("b.pdf", chunks);

            // exists 只应被调用一次（第二次走 volatile 快速路径）
            verify(indicesClient, times(1)).exists(any(GetIndexRequest.class), any());
        }
    }

    // ──────────────────────────────────────────────
    // 分数归一化
    // ──────────────────────────────────────────────

    @Nested
    @DisplayName("分数归一化")
    class ScoreNormalization {

        @Test
        @DisplayName("rawScore=0 时归一化分数应为 0")
        void normalizeKeywordScore_shouldReturnZeroForZeroScore() {
            SearchHit hit = mockSearchHit(0f, Map.of(
                    "documentId", 1L, "chunkId", 1L, "chunkIndex", 0,
                    "fileName", "f", "chunkText", "t"
            ));
            List<ElasticsearchService.KeywordHit> results =
                    service.parseSearchHits(new SearchHit[]{hit}, 1L, 5, System.nanoTime());
            assertThat(results).hasSize(1);
            assertThat(results.get(0).normalizedScore()).isEqualTo(0.0);
        }

        @Test
        @DisplayName("rawScore=100 时归一化分数应约为 1.0")
        void normalizeKeywordScore_shouldReturnApproximatelyOneForScore100() {
            SearchHit hit = mockSearchHit(100f, Map.of(
                    "documentId", 1L, "chunkId", 1L, "chunkIndex", 0,
                    "fileName", "f", "chunkText", "t"
            ));
            List<ElasticsearchService.KeywordHit> results =
                    service.parseSearchHits(new SearchHit[]{hit}, 1L, 5, System.nanoTime());
            assertThat(results).hasSize(1);
            assertThat(results.get(0).normalizedScore()).isCloseTo(1.0, within(0.001));
        }

        @Test
        @DisplayName("rawScore=10 时归一化分数应约为 0.5")
        void normalizeKeywordScore_shouldReturnApproximatelyHalfForScore10() {
            SearchHit hit = mockSearchHit(10f, Map.of(
                    "documentId", 1L, "chunkId", 1L, "chunkIndex", 0,
                    "fileName", "f", "chunkText", "t"
            ));
            List<ElasticsearchService.KeywordHit> results =
                    service.parseSearchHits(new SearchHit[]{hit}, 1L, 5, System.nanoTime());
            assertThat(results).hasSize(1);
            assertThat(results.get(0).normalizedScore()).isCloseTo(0.520, within(0.01));
        }
    }
}
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.configs.chat_configs import HYBRID_ALPHA, NUM_RETURNED_HITS
from onyx.context.search.enums import QueryType
from onyx.context.search.models import (
    ChunkIndexRequest,
    IndexFilters,
    InferenceChunk,
    InferenceSection,
)
from onyx.context.search.utils import get_query_embedding, inference_section_from_chunks
from onyx.document_index.interfaces_new import DocumentIndex, DocumentSectionRequest
from onyx.federated_connectors.federated_retrieval import (
    FederatedRetrievalInfo,
    get_federated_retrieval_functions,
)
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel

logger = setup_logger()


def combine_retrieval_results(
    chunk_sets: list[list[InferenceChunk]],
) -> list[InferenceChunk]:
    all_chunks = [chunk for chunk_set in chunk_sets for chunk in chunk_set]

    unique_chunks: dict[tuple[str, int], InferenceChunk] = {}
    for chunk in all_chunks:
        key = (chunk.document_id, chunk.chunk_id)
        if key not in unique_chunks:
            unique_chunks[key] = chunk
            continue

        stored_chunk_score = unique_chunks[key].score or 0
        this_chunk_score = chunk.score or 0
        if stored_chunk_score < this_chunk_score:
            unique_chunks[key] = chunk

    sorted_chunks = sorted(
        unique_chunks.values(), key=lambda x: x.score or 0, reverse=True
    )

    return sorted_chunks


def _embed_and_hybrid_search(
    query_request: ChunkIndexRequest,
    document_index: DocumentIndex,
    db_session: Session | None = None,
    embedding_model: EmbeddingModel | None = None,
) -> list[InferenceChunk]:
    query_embedding = get_query_embedding(
        query_request.query,
        db_session=db_session,
        embedding_model=embedding_model,
    )

    hybrid_alpha = query_request.hybrid_alpha or HYBRID_ALPHA

    query_type = QueryType.KEYWORD if hybrid_alpha <= 0.2 else QueryType.SEMANTIC
    top_chunks = document_index.hybrid_retrieval(
        query=query_request.query,
        query_embedding=query_embedding,
        final_keywords=query_request.query_keywords,
        query_type=query_type,
        filters=query_request.filters,
        num_to_retrieve=query_request.limit or NUM_RETURNED_HITS,
    )

    return top_chunks


def _keyword_search(
    query_request: ChunkIndexRequest,
    document_index: DocumentIndex,
) -> list[InferenceChunk]:
    return document_index.keyword_retrieval(
        query=query_request.query,
        filters=query_request.filters,
        num_to_retrieve=query_request.limit or NUM_RETURNED_HITS,
    )


def search_chunks(
    query_request: ChunkIndexRequest,
    user_id: UUID | None,
    document_index: DocumentIndex,
    db_session: Session | None = None,
    embedding_model: EmbeddingModel | None = None,
    prefetched_federated_retrieval_infos: list[FederatedRetrievalInfo] | None = None,
) -> list[InferenceChunk]:
    """并行执行联邦检索和本地索引检索，合并返回文档片段。

    调用链路：search_pipeline 构建查询及权限过滤条件后调用本方法。
    本方法根据来源范围组织检索任务：
    - 联邦检索：调用已配置的外部来源检索函数。
    - 本地索引检索：hybrid_alpha 显式为 0 时调用 _keyword_search；
      否则调用 _embed_and_hybrid_search，生成查询向量后执行 hybrid_retrieval。
      使用 OpenSearch 时，后续进入 OpenSearchDocumentIndex.hybrid_retrieval
      → OpenSearchIndexClient.search → 底层客户端 search。

    各路结果由 combine_retrieval_results 按文档 ID 和片段 ID 去重，
    重复片段保留得分更高的版本，再按得分降序排列。这里不执行模型重排，
    也不对合并结果再次截取 limit；返回后由 search_pipeline 继续做字段权限处理。

    参数：
        query_request：包含查询文本、来源过滤、检索参数和召回数量的请求。
        user_id：获取当前用户可用的联邦检索配置时使用的用户 ID。
        document_index：本地文档索引接口，负责关键词或混合检索。
        db_session：查询联邦检索配置及生成查询向量时可用的数据库会话。
        embedding_model：可选的查询向量模型，传给混合检索流程。
        prefetched_federated_retrieval_infos：预取的联邦检索配置；空列表表示
            无联邦检索任务，None 表示需要通过 db_session 查询配置。

    异常：
        ValueError：既未传入预取的联邦检索配置，也未提供数据库会话。

    返回：
        各路检索合并、去重并按得分排序后的片段列表；无命中时返回空列表。
    """
    # 将指定来源转换为集合，便于判断是否还需要查询本地索引；None 表示不限制来源。
    source_filters = (
        set(query_request.filters.source_type)
        if query_request.filters.source_type
        else None
    )

    # 优先复用预取配置，避免重复查库；显式传入空列表时也不再查询。
    if prefetched_federated_retrieval_infos is not None:
        federated_retrieval_infos = prefetched_federated_retrieval_infos
    else:
        if db_session is None:
            raise ValueError(
                "Either db_session or prefetched_federated_retrieval_infos must be provided"
            )
        federated_retrieval_infos = get_federated_retrieval_functions(
            db_session=db_session,
            user_id=user_id,
            source_types=list(source_filters) if source_filters else None,
            document_set_names=query_request.filters.document_set,
        )

    # 将联邦来源转换为普通来源枚举，与请求中的来源过滤条件统一比较。
    federated_sources = {
        federated_retrieval_info.source.to_non_federated_source()
        for federated_retrieval_info in federated_retrieval_infos
    }
    # 先登记各联邦检索函数及参数，统一在后面并行执行。
    run_queries: list[tuple[Callable, tuple]] = [
        (federated_retrieval_info.retrieval_function, (query_request,))
        for federated_retrieval_info in federated_retrieval_infos
    ]

    # 未限制来源，或指定来源中仍有未被联邦检索覆盖的类型时，才查询本地索引。
    # 这里只决定是否添加本地任务，不会从 query_request 中移除联邦来源。
    normal_search_enabled = (source_filters is None) or (
        len(set(source_filters) - federated_sources) > 0
    )

    if normal_search_enabled:
        if query_request.hybrid_alpha is not None and query_request.hybrid_alpha == 0.0:
            # 显式指定 0 才走纯关键词检索，省去查询向量计算。
            # 当前由支持 OpenSearch 的调用方使用；Vespa 的 keyword_retrieval
            # 会抛出 NotImplementedError。
            run_queries.append(
                (
                    lambda: _keyword_search(query_request, document_index),
                    (),
                )
            )
        else:
            # 其他情况先计算查询向量，再调用索引的混合检索接口。
            run_queries.append(
                (
                    _embed_and_hybrid_search,
                    (query_request, document_index, db_session, embedding_model),
                )
            )

    # 并行执行已登记的任务，得到每一路返回的片段列表。
    parallel_search_results = run_functions_tuples_in_parallel(run_queries)
    # 按 (document_id, chunk_id) 去重并排序；缺失得分按 0 处理。
    top_chunks = combine_retrieval_results(parallel_search_results)

    if not top_chunks:
        logger.debug(
            "Search returned no results for query: %s with filters: %s.",
            query_request.query,
            query_request.filters,
        )

    return top_chunks


# TODO: This is unused code.
def inference_sections_from_ids(
    doc_identifiers: list[tuple[str, int]],
    document_index: DocumentIndex,
) -> list[InferenceSection]:
    # Currently only fetches whole docs
    doc_ids_set = {doc_id for doc_id, _ in doc_identifiers}

    chunk_requests: list[DocumentSectionRequest] = [
        DocumentSectionRequest(document_id=doc_id) for doc_id in doc_ids_set
    ]

    # No need for ACL here because the doc ids were validated beforehand
    filters = IndexFilters(access_control_list=None)

    retrieved_chunks = document_index.id_based_retrieval(
        chunk_requests=chunk_requests,
        filters=filters,
    )

    if not retrieved_chunks:
        return []

    # Group chunks by document ID
    chunks_by_doc_id: dict[str, list[InferenceChunk]] = {}
    for chunk in retrieved_chunks:
        chunks_by_doc_id.setdefault(chunk.document_id, []).append(chunk)

    inference_sections = [
        section
        for chunks in chunks_by_doc_id.values()
        if chunks
        and (
            section := inference_section_from_chunks(
                # The scores will always be 0 because the fetching by id gives back
                # no search scores. This is not needed though if the user is explicitly
                # selecting a document.
                center_chunk=chunks[0],
                chunks=chunks,
            )
        )
    ]

    return inference_sections

"""
An explanation of the search tool found below:

Step 1: Queries
- The LLM will generate some queries based on the chat history for what it thinks are the best things to search for.
This has a pretty generic prompt so it's not perfectly tuned for search but provides breadth and also the LLM can often break up
the query into multiple searches which the other flows do not do. Exp: Compare the sales process between company X and Y can be
broken up into "sales process company X" and "sales process company Y".
- A specifial prompt and history is used to generate another query which is best tuned for a semantic/hybrid search pipeline.
- A small set of keyword emphasized queries are also generated to cover additional breadth. This is important for cases where
the query is short, keyword heavy, or has a lot of model unseen terminology.

Step 2: Recombination
We use a weighted RRF to combine the search results from the queries above. Each query will have a list of search results with
some scores however these are downstream of a normalization step so they cannot easily be compared with one another on an
absolute scale. RRF is a good way to combine these and allows us to give some custom weightings. We also merge document chunks
that are adjacent to provide more continuous context to the LLM.

Step 3: Selection
We pass the recombined results (truncated set) to the LLM to select the most promising ones to read. This is to reduce noise and
reduce downstream chances of hallucination. The LLM at this point also has the entire set of document chunks so it has
information across documents not just per document. This also reduces the number of tokens required for the next step.

Step 4: Expansion
For the selected documents, we pass the main retrieved sections from above (this may be a single chunk or a section comprised of
several consecutive chunks) along with chunks above and below the section to the LLM. The LLM determines how much of the document
it wants to read. This is done in parallel for all selected documents. Reason being that the LLM would not be able to do a good
job of this with all of the documents in the prompt at once. Keeping every LLM decision step as simple as possible is key for
reliable performance.

Step 5: Prompt Building
We construct a response string back to the LLM as the result of the tool call. We also pass relevant richer objects back
so that the rest of the code can persist it, render it in the UI, etc. The response is a json that makes it easy for the LLM to
refer to by using matching keywords to other parts of the prompt and reminders.
"""

import time
from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.chat.emitter import Emitter
from onyx.configs.chat_configs import MAX_CHUNKS_FED_TO_CHAT
from onyx.configs.constants import DocumentSource, FederatedConnectorSource
from onyx.context.search.federated.slack_search import slack_retrieval
from onyx.context.search.models import (
    BaseFilters,
    ChunkIndexRequest,
    ChunkSearchRequest,
    IndexFilters,
    InferenceChunk,
    InferenceSection,
    PersonaSearchInfo,
    SearchDocsResponse,
)
from onyx.context.search.pipeline import merge_individual_chunks, search_pipeline
from onyx.context.search.preprocessing.access_filters import (
    build_access_filters_for_user,
)
from onyx.context.search.utils import (
    convert_inference_sections_to_search_docs,
    populate_file_ids_on_sections,
)
from onyx.db.connector import (
    check_connectors_exist,
    check_federated_connectors_exist,
    fetch_unique_document_sources,
)
from onyx.db.document_set import filter_document_set_names_by_user_access
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.federated import (
    get_federated_connector_document_set_mappings_by_document_set_names,
    list_federated_connector_oauth_tokens,
)
from onyx.db.models import SearchSettings, User
from onyx.db.search_settings import get_current_search_settings
from onyx.db.slack_bot import fetch_slack_bots
from onyx.document_index.interfaces_new import DocumentIndex
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.federated_connectors.federated_retrieval import (
    FederatedRetrievalInfo,
    get_federated_retrieval_functions,
)
from onyx.llm.factory import get_llm_token_counter
from onyx.llm.interfaces import LLM
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.onyxbot.slack.models import SlackContext
from onyx.secondary_llm_flows.document_filter import (
    select_chunks_for_relevance,
    select_sections_for_expansion,
)
from onyx.secondary_llm_flows.query_expansion import (
    keyword_query_expansion,
    semantic_query_rephrase,
)
from onyx.secondary_llm_flows.source_filter import SearchCycle, decide_search_scope
from onyx.secondary_llm_flows.time_filter import TimeFilter, decide_time_filter
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    Packet,
    SearchToolDocumentsDelta,
    SearchToolFilterDelta,
    SearchToolQueriesDelta,
    SearchToolStart,
)
from onyx.tools.interface import Tool
from onyx.tools.models import (
    ChatMinimalTextMessage,
    SearchToolOverrideKwargs,
    ToolCallException,
    ToolResponse,
)
from onyx.tools.tool_implementations.search.constants import (
    KEYWORD_QUERY_HYBRID_ALPHA,
    LLM_KEYWORD_QUERY_WEIGHT,
    LLM_NON_CUSTOM_QUERY_WEIGHT,
    LLM_SEMANTIC_QUERY_WEIGHT,
    MAX_CHUNKS_FOR_RELEVANCE,
    ORIGINAL_QUERY_WEIGHT,
    SELECTION_TOKEN_BUDGET_MULTIPLIER,
)
from onyx.tools.tool_implementations.search.search_utils import (
    expand_section_with_context,
    merge_overlapping_sections,
    weighted_reciprocal_rank_fusion,
)
from onyx.tools.tool_implementations.utils import (
    convert_inference_sections_to_llm_string,
)
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel
from onyx.utils.timing import log_function_time
from shared_configs.configs import (
    DOC_EMBEDDING_CONTEXT_SIZE,
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT,
)

logger = setup_logger()

QUERIES_FIELD = "queries"


class QueryExpansionAndScope(BaseModel):
    """Result of one search cycle's query expansion + source-scope decision."""

    semantic_query: str | None
    keyword_queries: list[str]
    plan_scope: list[DocumentSource] | None
    time_filter: TimeFilter | None = None


def _build_scope_note(
    scope: list[DocumentSource] | None, queries_run: list[str]
) -> str:
    """Note appended to a scoped search's response: which source(s) it covered
    and the queries that ran, so a repeat can vary terms. "" when unscoped."""
    if not scope:
        return ""
    searched = ", ".join(source.value for source in scope)
    queries_str = "; ".join(queries_run) or "(none)"
    return (
        f"(This internal search covered only: {searched}. Queries run: {queries_str}. "
        "Call internal_search again with different query terms to keep searching.)"
    )


def deduplicate_queries(
    queries_with_weights: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Deduplicate queries by case-insensitive comparison and sum weights.

    Args:
        queries_with_weights: List of (query, weight) tuples

    Returns:
        Deduplicated list of (query, weight) tuples with summed weights
    """
    query_map: dict[str, tuple[str, float]] = {}
    for query, weight in queries_with_weights:
        query_lower = query.lower()
        if query_lower in query_map:
            # Sum weights for duplicate queries
            existing_query, existing_weight = query_map[query_lower]
            query_map[query_lower] = (existing_query, existing_weight + weight)
        else:
            # Keep the first occurrence (preserves original casing)
            query_map[query_lower] = (query, weight)
    return list(query_map.values())


def _estimate_section_tokens(
    section: InferenceSection,
    token_counter: Callable[[str], int],
    max_chunks_per_section: int | None = None,
) -> int:
    """Estimate token count for a section using the LLM tokenizer.

    Args:
        section: InferenceSection to estimate tokens for
        token_counter: Function that counts tokens in text
        max_chunks_per_section: Maximum chunks to consider per section (None for all)

    Returns:
        Token count for the section
    """
    # Estimate for metadata (title, source_type, etc.)
    METADATA_TOKEN_ESTIMATE = 75

    # If max_chunks_per_section is specified, only count tokens for selected chunks
    if max_chunks_per_section is not None:
        selected_chunks = select_chunks_for_relevance(section, max_chunks_per_section)
        # Combine content from selected chunks
        combined_content = "\n".join(chunk.content for chunk in selected_chunks)
        content_tokens = token_counter(combined_content)
    else:
        content_tokens = token_counter(section.combined_content)

    return content_tokens + METADATA_TOKEN_ESTIMATE


@log_function_time(print_only=True)
def _trim_sections_by_tokens(
    sections: list[InferenceSection],
    max_tokens: int,
    token_counter: Callable[[str], int],
    max_chunks_per_section: int | None = None,
) -> list[InferenceSection]:
    """Trim sections to fit within a token budget using the LLM tokenizer.

    Args:
        sections: List of InferenceSection objects to trim
        max_tokens: Maximum token budget
        token_counter: Function that counts tokens in text
        max_chunks_per_section: Maximum chunks to consider per section (None for all)

    Returns:
        Trimmed list of sections that fit within the token budget
    """
    if not sections or max_tokens <= 0:
        return sections

    trimmed_sections = []
    total_tokens = 0

    for section in sections:
        section_tokens = _estimate_section_tokens(
            section, token_counter, max_chunks_per_section
        )
        if total_tokens + section_tokens <= max_tokens:
            trimmed_sections.append(section)
            total_tokens += section_tokens
        else:
            break

    logger.debug(
        "Trimmed sections from %s to %s (%s tokens, budget: %s)",
        len(sections),
        len(trimmed_sections),
        total_tokens,
        max_tokens,
    )

    return trimmed_sections


class SearchTool(Tool[SearchToolOverrideKwargs]):
    """内部搜索工具：检索已连接的数据源，筛选文档并生成回答所需的引用上下文。"""

    NAME = "internal_search"
    DISPLAY_NAME = "Internal Search"
    DESCRIPTION = "Search connected applications for information."

    def __init__(
        self,
        tool_id: int,
        emitter: Emitter,
        # 用于访问控制和联邦搜索；匿名用户只能访问公开文档。
        user: User,
        # 预先提取的智能体搜索配置。
        persona_search_info: PersonaSearchInfo,
        llm: LLM,
        document_index: DocumentIndex,
        # 用户选择的搜索筛选条件。
        user_selected_filters: BaseFilters | None,
        # 文件无法全部放入模型上下文时，用项目或智能体 ID 限定索引检索范围。
        # 这些是搜索过滤条件，不是无条件传入的当前项目或智能体 ID。
        project_id_filter: int | None,
        persona_id_filter: int | None = None,
        # Slack 联邦搜索的上下文；访问凭据在内部获取。
        slack_context: SlackContext | None = None,
        # 是否启用 Slack 联邦搜索。
        enable_slack_search: bool = True,
        # 是否从问题推断来源和时间范围；关闭时仅使用用户或智能体的筛选条件。
        auto_detect_filters: bool = True,
    ) -> None:
        """保存搜索依赖和配置，并初始化同一工具实例的搜索状态。

        初始化仅保存依赖，不执行检索。工具调度层匹配 internal_search 后，
        经 run_tool_calls → _safe_run_single_tool → run 触发实际搜索。
        """
        super().__init__(emitter=emitter)

        self.user = user
        self.persona_search_info = persona_search_info
        self.llm = llm
        self.document_index = document_index
        self.user_selected_filters = user_selected_filters
        self.project_id_filter = project_id_filter
        self.persona_id_filter = persona_id_filter
        self.slack_context = slack_context
        self.enable_slack_search = enable_slack_search
        self.auto_detect_filters = auto_detect_filters

        # 记录本实例已搜索的查询与来源，供重复搜索时决定下一步范围。
        self._search_cycles: list[SearchCycle] = []
        # 缓存查询扩展，切换来源时可复用。
        self._cached_expansion: tuple[str | None, list[str]] | None = None
        # 分别记录来源决策、时间范围及时间推断是否已完成。
        self._scope_decision_settled = False
        self._time_filter: TimeFilter | None = None
        self._time_filter_computed = False

        self._id = tool_id

    def _prefetch_slack_data(
        self, db_session: Session
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        """Pre-fetch Slack access token, bot token, and entity config from DB.

        All DB queries for Slack federated search are performed here in a
        single session, so the parallel search phase needs no DB access.

        Returns:
            (access_token, bot_token, entities) — access_token is None when
            Slack search should be skipped.
        """
        bot_token: str | None = None
        access_token: str | None = None
        entities: dict[str, Any] = {}

        # Case 1: Slack bot context — requires a Slack federated connector
        # linked via the persona's document sets
        if self.slack_context:
            document_set_names = self.persona_search_info.document_set_names
            if not document_set_names:
                logger.debug(
                    "Skipping Slack federated search: no document sets on persona"
                )
                return None, None, {}

            slack_federated_mappings = (
                get_federated_connector_document_set_mappings_by_document_set_names(
                    db_session, document_set_names
                )
            )
            found_slack_connector = False
            for mapping in slack_federated_mappings:
                if (
                    mapping.federated_connector is not None
                    and mapping.federated_connector.source
                    == FederatedConnectorSource.FEDERATED_SLACK
                ):
                    entities = mapping.federated_connector.config or {}
                    found_slack_connector = True
                    logger.debug("Found Slack federated connector config: %s", entities)
                    break

            if not found_slack_connector:
                logger.debug(
                    "Skipping Slack federated search: no Slack federated connector linked to document sets %s",
                    document_set_names,
                )
                return None, None, {}

            try:
                slack_bots = fetch_slack_bots(db_session)
                if not slack_bots:
                    return None, None, {}

                tenant_slack_bot = next(
                    (bot for bot in slack_bots if bot.enabled and bot.user_token),
                    None,
                )
                if not tenant_slack_bot:
                    tenant_slack_bot = next(
                        (bot for bot in slack_bots if bot.enabled), None
                    )

                if tenant_slack_bot:
                    bot_token = (
                        tenant_slack_bot.bot_token.get_value(apply_mask=False)
                        if tenant_slack_bot.bot_token
                        else None
                    )
                    user_token = (
                        tenant_slack_bot.user_token.get_value(apply_mask=False)
                        if tenant_slack_bot.user_token
                        else None
                    )
                    access_token = user_token or bot_token
            except Exception as e:
                logger.warning("Could not fetch Slack bot tokens: %s", e)

        # Case 2: Web user with federated OAuth (if bot context didn't yield a token)
        if not access_token and self.user:
            try:
                federated_oauth_tokens = list_federated_connector_oauth_tokens(
                    db_session, self.user.id
                )
                if not federated_oauth_tokens:
                    return access_token, bot_token, entities

                slack_oauth_token = next(
                    (
                        token
                        for token in federated_oauth_tokens
                        if token.federated_connector.source
                        == FederatedConnectorSource.FEDERATED_SLACK
                    ),
                    None,
                )
                if slack_oauth_token and slack_oauth_token.token:
                    access_token = slack_oauth_token.token.get_value(apply_mask=False)
                    entities = slack_oauth_token.federated_connector.config or {}
            except Exception as e:
                logger.warning("Could not fetch Slack OAuth token: %s", e)

        return access_token, bot_token, entities

    def _run_slack_search(
        self,
        query: str,
        access_token: str,
        bot_token: str | None,
        entities: dict[str, Any],
        search_settings: SearchSettings,
    ) -> list[InferenceChunk]:
        """Run Slack federated search using pre-fetched tokens and config.

        All DB data is pre-fetched in run() so this method needs no DB session.

        Args:
            query: The user's original search query
            access_token: Slack access token (user or bot)
            bot_token: Slack bot token (for enhanced permissions)
            entities: Federated connector entity config (channel filtering)
            search_settings: Pre-fetched SearchSettings for chunking config

        Returns:
            List of InferenceChunk results from Slack
        """
        try:
            chunk_request = ChunkIndexRequest(
                query=query,
                filters=IndexFilters(access_control_list=None),
            )

            chunks = slack_retrieval(
                query=chunk_request,
                access_token=access_token,
                connector=None,
                entities=entities,
                limit=None,
                slack_event_context=self.slack_context,
                bot_token=bot_token,
                team_id=None,
                search_settings=search_settings,
                llm=self.llm,
            )

            logger.info("Slack federated search returned %s chunks", len(chunks))
            return chunks

        except Exception as e:
            logger.error("Slack federated search error: %s", e, exc_info=True)
            return []

    def _run_search_for_query(
        self,
        query: str,
        hybrid_alpha: float | None,
        num_hits: int,
        acl_filters: list[str],
        embedding_model: EmbeddingModel,
        federated_retrieval_infos: list[FederatedRetrievalInfo],
        effective_filters: BaseFilters | None,
    ) -> list[InferenceChunk]:
        """Run search pipeline for a single query using pre-fetched data.

        All DB data (ACL filters, embedding model, federated retrieval info)
        is pre-fetched in run() so this method needs no DB session.

        Args:
            query: The search query string
            hybrid_alpha: Hybrid search alpha parameter (None for default)
            num_hits: Maximum number of hits to return
            acl_filters: Pre-fetched ACL filters for the acting user
            embedding_model: Pre-fetched embedding model
            federated_retrieval_infos: Pre-fetched federated retrieval functions
            effective_filters: Filters for THIS search, with the per-call source
                scope already applied (computed once in run()).

        Returns:
            List of InferenceChunk results
        """
        logger.info("[RAG_TRACE] search_tool_query -> search_pipeline")
        return search_pipeline(
            chunk_search_request=ChunkSearchRequest(
                query=query,
                hybrid_alpha=hybrid_alpha,
                # For projects, the search scope is the project and has no other limits
                user_selected_filters=(
                    effective_filters if self.project_id_filter is None else None
                ),
                limit=num_hits,
            ),
            project_id_filter=self.project_id_filter,
            persona_id_filter=self.persona_id_filter,
            document_index=self.document_index,
            user=self.user,
            persona_search_info=self.persona_search_info,
            acl_filters=acl_filters,
            embedding_model=embedding_model,
            prefetched_federated_retrieval_infos=federated_retrieval_infos,
        )

    @classmethod
    def is_available(cls, db_session: Session) -> bool:
        """Check if search tool is available.

        Returns False when the vector DB is disabled (search cannot function
        without it). Otherwise, available if ANY of the following exist:
        - Regular connectors (team knowledge)
        - Federated connectors (e.g., Slack)
        - User files (User Knowledge mode)
        """
        from onyx.configs.app_configs import DISABLE_VECTOR_DB
        from onyx.db.connector import check_user_files_exist

        if DISABLE_VECTOR_DB:
            return False

        return (
            check_connectors_exist(db_session)
            or check_federated_connectors_exist(db_session)
            or check_user_files_exist(db_session)
        )

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def description(self) -> str:
        return self.DESCRIPTION

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAME

    """For explicit tool calling"""

    def tool_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        QUERIES_FIELD: {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of search queries to execute, typically a single query. "
                                "Query expansion and filter extraction steps will be run "
                                "automatically downstream, do not include time or source type "
                                "scoping details in your query."
                            ),
                        },
                    },
                    "required": [QUERIES_FIELD],
                },
            },
        }

    def emit_start(self, placement: Placement) -> None:
        self.emitter.emit(
            Packet(
                placement=placement,
                obj=SearchToolStart(),
            )
        )

    @log_function_time(
        func_name="Search tool - query expansion + scope decision",
        print_only=True,
        debug_only=True,
    )
    def _expand_queries_and_decide_scope(
        self,
        skip_query_expansion: bool,
        message_history: list[ChatMinimalTextMessage],
        user_info: str | None,
        memories: list[str],
        decide_args: tuple[Any, ...],
    ) -> QueryExpansionAndScope:
        """Expand the query and decide the source/time scope, in parallel when each
        applies.

        Repeat calls reuse the cached expansion instead of re-expanding. Once the
        scope decision finds no source directive it latches off for the rest of the
        turn, since the conversation cannot introduce one mid-turn. The time-window
        decision is computed once per turn and cached. Both auto decisions are
        gated by ``auto_detect_filters``.
        """
        expand_queries = not skip_query_expansion
        decide_scope = self.auto_detect_filters and not self._scope_decision_settled
        decide_time = self.auto_detect_filters and not self._time_filter_computed

        jobs: list[tuple[Callable, tuple]] = []
        scope_job_index: int | None = None
        time_job_index: int | None = None
        if expand_queries:
            expansion_args = (message_history, self.llm, user_info, memories)
            jobs.append((semantic_query_rephrase, expansion_args))
            jobs.append((keyword_query_expansion, expansion_args))
        if decide_scope:
            scope_job_index = len(jobs)
            jobs.append((decide_search_scope, decide_args))
        if decide_time:
            time_job_index = len(jobs)
            jobs.append((decide_time_filter, (message_history, self.llm)))

        results = run_functions_tuples_in_parallel(jobs) if jobs else []

        semantic_query: str | None = None
        keyword_queries: list[str] = []
        if expand_queries:
            semantic_query = results[0]
            keyword_queries = results[1] or []
            self._cached_expansion = (semantic_query, keyword_queries)

        plan_scope: list[DocumentSource] | None = None
        if scope_job_index is not None:
            plan_scope = results[scope_job_index]
            self._scope_decision_settled = plan_scope is None

        if time_job_index is not None:
            self._time_filter = results[time_job_index]
            self._time_filter_computed = True

        return QueryExpansionAndScope(
            semantic_query=semantic_query,
            keyword_queries=keyword_queries,
            plan_scope=plan_scope,
            time_filter=self._time_filter,
        )

    @log_function_time(print_only=True)
    def run(
        self,
        placement: Placement,
        override_kwargs: SearchToolOverrideKwargs,
        **llm_kwargs: Any,
    ) -> ToolResponse:
        """执行内部知识库搜索，返回模型可读的文档文本及结构化引用。

        调用链路：
            run_llm_loop → run_tool_calls → _safe_run_single_tool
            → SearchTool.run（本方法）
               → 预取 ACL、嵌入模型与联邦检索配置
               → _expand_queries_and_decide_scope：扩展查询，确定来源和时间范围
               → 并行执行 _run_search_for_query
                  → search_pipeline：构建检索请求与过滤条件
                  → search_chunks：分发索引检索和联邦检索
                     → _embed_and_hybrid_search（启用普通混合检索时）
                        → get_query_embedding：生成查询向量
                        → document_index.hybrid_retrieval
                           → OpenSearchDocumentIndex.hybrid_retrieval（OpenSearch 实现）
                           → DocumentQuery.get_hybrid_search_query
                           → OpenSearchIndexClient.search → OpenSearch
               → weighted_reciprocal_rank_fusion：融合多查询排名
               → merge_individual_chunks：合并片段，限制候选数量
               → select_sections_for_expansion：LLM 筛选相关内容
               → expand_section_with_context：并行扩展文档上下文
               → merge_overlapping_sections：合并重叠内容
               → convert_inference_sections_to_llm_string：生成文本和引用映射
               → ToolResponse → run_llm_loop：将结果加入模型上下文

        实际分支取决于来源、过滤条件和索引配置；不是每次都执行向量搜索。
        Slack 搜索在条件满足时单独并行执行；无候选结果时提前返回空响应。
        本方法返回检索材料，最终面向用户的回答由后续模型调用生成。

        参数：
            placement：本次工具事件在响应流中的位置。
            override_kwargs：调度层传入的原始问题、历史、记忆、引用起点及数量限制。
            llm_kwargs：模型生成的工具参数，必须包含 queries。

        返回：
            ToolResponse，包含文档、引用映射、展示文档及供模型读取的文本。

        异常：
            ToolCallException：缺少 queries 参数。
            OnyxError：用户无权访问指定文档集。
            RuntimeError：未配置搜索设置。
            检索等步骤的其他异常交由上层工具执行包装函数处理。
        """
        logger.info("[RAG_TRACE] search_tool_run")
        # 先校验必填参数，再处理空来源等提前返回分支，避免掩盖错误调用。
        if QUERIES_FIELD not in llm_kwargs:
            raise ToolCallException(
                message=f"Missing required '{QUERIES_FIELD}' parameter in internal_search tool call",
                llm_facing_message=(
                    f"The internal_search tool requires a '{QUERIES_FIELD}' parameter "
                    f"containing an array of search queries. Please provide the queries "
                    f'like: {{"queries": ["your search query here"]}}'
                ),
            )

        # 显式空来源列表表示不搜索任何来源；None 表示不限制来源。
        # 项目模式忽略用户筛选条件，因此不受此空列表判断影响。
        if (
            self.user_selected_filters is not None
            and self.project_id_filter is None
            and self.user_selected_filters.source_type is not None
            and len(self.user_selected_filters.source_type) == 0
        ):
            empty_response, _ = convert_inference_sections_to_llm_string(
                top_sections=[],
                note=None,
            )
            return ToolResponse(
                rich_response=SearchDocsResponse(
                    search_docs=[],
                    citation_mapping={},
                    displayed_docs=None,
                ),
                llm_facing_response=empty_response,
            )

        # 记录整个搜索工具的执行耗时。
        overall_start_time = time.time()

        # 初始化筛选与扩展耗时。
        document_selection_elapsed = 0.0
        document_expansion_elapsed = 0.0

        connected_sources: list[DocumentSource] = []

        # 在短生命周期会话中预取权限、模型和联邦搜索配置，供并行检索使用。
        with get_session_with_current_tenant() as db_session:
            # 构建当前用户的文档访问控制条件。
            acl_filters: list[str] = build_access_filters_for_user(
                self.user, db_session
            )

            # 校验用户指定的文档集权限，拒绝访问未授权的文档集。
            if (
                self.user_selected_filters
                and self.user_selected_filters.document_set
                and self.user
                and not self.user.is_anonymous
            ):
                requested = self.user_selected_filters.document_set
                accessible = filter_document_set_names_by_user_access(
                    db_session=db_session,
                    document_set_names=requested,
                    user=self.user,
                )
                unauthorized = sorted(
                    name for name in requested if name not in accessible
                )
                if unauthorized:
                    raise OnyxError(
                        OnyxErrorCode.INSUFFICIENT_PERMISSIONS,
                        f"User does not have access to document sets: {unauthorized}",
                    )

            # 在会话关闭前构建嵌入模型，完成云提供方属性的延迟加载。
            search_settings = get_current_search_settings(db_session)
            if not search_settings:
                raise RuntimeError(
                    "No search settings configured — cannot run internal search"
                )

            embedding_model = EmbeddingModel.from_db_model(
                search_settings=search_settings,
                server_host=MODEL_SERVER_HOST,
                server_port=MODEL_SERVER_PORT,
            )

            # 预取非 Slack 的联邦检索函数；Slack 单独处理。
            if self.project_id_filter is not None:
                # 项目模式不使用用户指定的来源条件进行预取。
                prefetch_source_types = None
            else:
                prefetch_source_types = (
                    list(self.user_selected_filters.source_type)
                    if self.user_selected_filters
                    and self.user_selected_filters.source_type
                    else None
                )
            federated_retrieval_infos = (
                get_federated_retrieval_functions(
                    db_session=db_session,
                    user_id=self.user.id if self.user else None,
                    source_types=prefetch_source_types,
                    document_set_names=self.persona_search_info.document_set_names,
                )
                or []
            )

            # 非项目模式读取已连接来源，供后续搜索范围决策使用。
            if self.project_id_filter is None:
                connected_sources = fetch_unique_document_sources(db_session)

            # 启用 Slack 搜索或存在 Slack 机器人上下文时，才预取凭据和实体配置。
            if self.enable_slack_search or self.slack_context:
                slack_access_token, slack_bot_token, slack_entities = (
                    self._prefetch_slack_data(db_session)
                )
            else:
                slack_access_token, slack_bot_token, slack_entities = (
                    None,
                    None,
                    {},
                )
        # 预取会话在此关闭，后续检索使用已提取的数据。

        llm_queries = cast(list[str], llm_kwargs[QUERIES_FIELD])

        # 从运行时参数中提取历史、记忆和用户信息，供查询扩展使用。
        message_history = override_kwargs.message_history or []
        memories = (
            override_kwargs.user_memory_context.as_formatted_list()
            if override_kwargs.user_memory_context
            else []
        )
        user_info = override_kwargs.user_info

        # 来源决策必须在用户选择的来源范围内进行。
        user_source_restriction: list[DocumentSource] | None = (
            list(self.user_selected_filters.source_type)
            if self.user_selected_filters and self.user_selected_filters.source_type
            else None
        )
        if user_source_restriction is not None:
            allowed = set(user_source_restriction)
            candidate_sources = [s for s in connected_sources if s in allowed]
        else:
            candidate_sources = connected_sources

        decide_args = (
            message_history,
            self.llm,
            candidate_sources,
            list(self._search_cycles),
            llm_queries,
        )
        expansion = self._expand_queries_and_decide_scope(
            skip_query_expansion=override_kwargs.skip_query_expansion,
            message_history=message_history,
            user_info=user_info,
            memories=memories,
            decide_args=decide_args,
        )
        semantic_query = expansion.semantic_query
        keyword_queries = expansion.keyword_queries
        plan_scope = expansion.plan_scope

        resolved_scope = (
            plan_scope if plan_scope is not None else user_source_restriction
        )

        logger.info(
            "Internal search - source scope: %s",
            [s.value for s in resolved_scope] if resolved_scope else "all sources",
        )

        # 重复搜索切换到新来源时，复用与来源无关的查询扩展缓存。
        searched_sources = {
            value for cycle in self._search_cycles for value in cycle.searched_sources
        }
        is_new_filter = bool(resolved_scope) and any(
            source.value not in searched_sources for source in resolved_scope
        )
        if (
            override_kwargs.skip_query_expansion
            and is_new_filter
            and self._cached_expansion is not None
        ):
            semantic_query, keyword_queries = self._cached_expansion

        self._search_cycles.append(
            SearchCycle(
                cycle_number=len(self._search_cycles) + 1,
                queries=list(llm_queries),
                searched_sources=(
                    [source.value for source in resolved_scope]
                    if resolved_scope
                    else []
                ),
            )
        )

        # 向前端发送来源与时间筛选条件；覆盖全部来源时不显示来源收窄。
        scopes_all_sources = bool(connected_sources) and set(
            connected_sources
        ).issubset(resolved_scope or [])
        emitted_sources = (
            [source.value for source in resolved_scope]
            if resolved_scope and not scopes_all_sources
            else []
        )
        time_filter = expansion.time_filter
        if emitted_sources or time_filter is not None:
            self.emitter.emit(
                Packet(
                    placement=placement,
                    obj=SearchToolFilterDelta(
                        sources=emitted_sources,
                        time_filter_start=time_filter.start if time_filter else None,
                        time_filter_end=time_filter.end if time_filter else None,
                    ),
                )
            )

        queries_run = list(
            dict.fromkeys(
                llm_queries
                + ([semantic_query] if semantic_query else [])
                + keyword_queries
            )
        )
        scope_note = _build_scope_note(resolved_scope, queries_run)

        effective_filters = self.user_selected_filters
        if resolved_scope is not None:
            effective_filters = (
                self.user_selected_filters or BaseFilters()
            ).model_copy(update={"source_type": resolved_scope})
            federated_retrieval_infos = [
                info
                for info in federated_retrieval_infos
                if info.source.to_non_federated_source() in resolved_scope
            ]
            # Slack 不在本次搜索范围内时，禁用它的联邦搜索。
            if DocumentSource.SLACK not in resolved_scope:
                slack_access_token = None

        # 应用推断的时间范围；检索管线还会结合智能体配置的时间下限。
        if time_filter is not None:
            effective_filters = time_filter.apply_to(effective_filters or BaseFilters())
            logger.info(
                "Internal search - time window (%s): %s to %s",
                time_filter.field.value,
                time_filter.start.isoformat() if time_filter.start else "any",
                time_filter.end.isoformat() if time_filter.end else "any",
            )

        # 准备关键词查询及融合权重；检索参数使用 KEYWORD_QUERY_HYBRID_ALPHA。
        keyword_queries_with_weights = [
            (kw_query, LLM_KEYWORD_QUERY_WEIGHT) for kw_query in keyword_queries
        ]
        deduplicated_keyword_queries = deduplicate_queries(keyword_queries_with_weights)

        # 准备语义改写、模型查询和原始问题；这些查询使用默认混合检索参数。
        semantic_queries_with_weights = (
            [
                (semantic_query, LLM_SEMANTIC_QUERY_WEIGHT),
            ]
            if semantic_query
            else []
        )
        # 忽略模型偶尔返回的空查询。
        semantic_queries_with_weights.extend(
            (llm_query, LLM_NON_CUSTOM_QUERY_WEIGHT)
            for llm_query in llm_queries
            if llm_query
        )
        if override_kwargs.original_query:
            semantic_queries_with_weights.append(
                (override_kwargs.original_query, ORIGINAL_QUERY_WEIGHT)
            )
        deduplicated_semantic_queries = deduplicate_queries(
            semantic_queries_with_weights
        )

        # 合并两组查询，按权重降序生成前端展示列表。
        all_queries_with_weights = (
            deduplicated_semantic_queries + deduplicated_keyword_queries
        )
        all_queries_with_weights.sort(key=lambda x: x[1], reverse=True)

        # 展示列表按大小写不敏感去重；实际检索任务仍按各自分组构建。
        all_queries = []
        seen_lower = set()
        for query, _ in all_queries_with_weights:
            query_lower = query.lower()
            if query_lower not in seen_lower:
                all_queries.append(query)
                seen_lower.add(query_lower)

        logger.debug(
            "All Queries (sorted by weight): %s, Keyword queries: %s",
            all_queries,
            [q for q, _ in deduplicated_keyword_queries],
        )

        # 检索开始前发送查询列表，让前端及时显示搜索内容。
        self.emitter.emit(
            Packet(
                placement=placement,
                obj=SearchToolQueriesDelta(
                    queries=all_queries,
                ),
            )
        )

        # 构建并行任务及对应的融合权重；每个查询分别执行检索管线。
        search_functions: list[tuple[Callable, tuple]] = []
        search_weights: list[float] = []

        # 语义查询传入 None，使用下游默认的混合检索配置。
        for query, weight in deduplicated_semantic_queries:
            search_functions.append(
                (
                    self._run_search_for_query,
                    (
                        query,
                        None,
                        override_kwargs.num_hits,
                        acl_filters,
                        embedding_model,
                        federated_retrieval_infos,
                        effective_filters,
                    ),
                )
            )
            search_weights.append(weight)

        # 关键词查询传入配置的混合检索参数；具体行为由下游索引实现决定。
        for query, weight in deduplicated_keyword_queries:
            search_functions.append(
                (
                    self._run_search_for_query,
                    (
                        query,
                        KEYWORD_QUERY_HYBRID_ALPHA,
                        override_kwargs.num_hits,
                        acl_filters,
                        embedding_model,
                        federated_retrieval_infos,
                        effective_filters,
                    ),
                )
            )
            search_weights.append(weight)

        # 有有效凭据和原始问题时，额外执行一次 Slack 联邦搜索。
        # 它与索引检索并行，避免每条扩展查询都重复触发 Slack 搜索。
        if slack_access_token and override_kwargs.original_query:
            search_functions.append(
                (
                    self._run_slack_search,
                    (
                        override_kwargs.original_query,
                        slack_access_token,
                        slack_bot_token,
                        slack_entities,
                        search_settings,
                    ),
                )
            )
            # Slack 结果使用与原始问题相同的融合权重。
            search_weights.append(ORIGINAL_QUERY_WEIGHT)

        # 并行执行索引检索与可用的 Slack 联邦搜索。
        all_search_results = run_functions_tuples_in_parallel(search_functions)
        if not all_search_results:
            all_search_results = []

        # 按查询权重执行倒数排名融合（RRF），以文档 ID 与片段 ID 标识结果。
        top_chunks = weighted_reciprocal_rank_fusion(
            ranked_results=all_search_results,
            weights=search_weights,
            id_extractor=lambda chunk: f"{chunk.document_id}_{chunk.chunk_id}",
        )

        # 将片段合并为 section，并按 num_hits 截断，限制后续处理的候选范围。
        top_sections = merge_individual_chunks(top_chunks)[: override_kwargs.num_hits]

        if not top_sections:
            logger.info("Search tool - no results found, returning empty response")
            empty_response, _ = convert_inference_sections_to_llm_string(
                top_sections=[],
                note=scope_note or None,
            )
            return ToolResponse(
                rich_response=SearchDocsResponse(
                    search_docs=[],
                    citation_mapping={},
                    displayed_docs=None,
                ),
                llm_facing_response=empty_response,
            )

        # 从 PostgreSQL 补充 Document.file_id；该元数据不保存在搜索索引中。
        with get_session_with_current_tenant() as enrichment_session:
            populate_file_ids_on_sections(top_sections, enrichment_session)

        # 将候选 section 转为结构化文档，供工具响应使用。
        search_docs = convert_inference_sections_to_search_docs(
            top_sections, is_internet=False
        )

        secondary_flows_user_query = (
            override_kwargs.original_query
            or semantic_query
            or (llm_queries[0] if llm_queries else "")
        )

        token_counter = get_llm_token_counter(self.llm)

        # 在 LLM 筛选前按 token 预算裁剪，并限制每个 section 的片段数量。
        max_tokens_for_selection = (
            (override_kwargs.max_llm_chunks or MAX_CHUNKS_FED_TO_CHAT)
            * DOC_EMBEDDING_CONTEXT_SIZE
            * SELECTION_TOKEN_BUDGET_MULTIPLIER
        )

        # 这是近似预算，未构造实际提示词，元数据等 token 数可能被低估。
        sections_for_selection = _trim_sections_by_tokens(
            sections=top_sections,
            max_tokens=max_tokens_for_selection,
            token_counter=token_counter,
            max_chunks_per_section=MAX_CHUNKS_FOR_RELEVANCE,
        )

        # 记录 LLM 文档筛选的开始时间。
        document_selection_start_time = time.time()

        # 让 LLM 选出相关 section，以及需要重点扩展的文档。
        selected_sections, best_doc_ids = select_sections_for_expansion(
            sections=sections_for_selection,
            user_query=secondary_flows_user_query,
            llm=self.llm,
            max_chunks_per_section=MAX_CHUNKS_FOR_RELEVANCE,
        )

        # 记录 LLM 筛选耗时与选中数量。
        document_selection_elapsed = time.time() - document_selection_start_time
        logger.debug(
            "Search tool - LLM picking documents took %s seconds (selected %s sections)",
            format(document_selection_elapsed, ".3f"),
            len(selected_sections),
        )

        # 用集合快速判断 section 是否属于重点扩展文档。
        best_doc_ids_set = set(best_doc_ids) if best_doc_ids else set()

        # 向前端展示 LLM 筛选后的文档。
        final_ui_docs = convert_inference_sections_to_search_docs(
            selected_sections, is_internet=False
        )

        self.emitter.emit(
            Packet(
                placement=placement,
                obj=SearchToolDocumentsDelta(
                    documents=final_ui_docs,
                ),
            )
        )

        # 封装上下文扩展；单个 section 扩展失败时保留原文。
        def expand_section_safe(
            section: InferenceSection,
            user_query: str,
            llm: LLM,
            document_index: DocumentIndex,
            expand_override: bool,
        ) -> InferenceSection:
            """扩展 section 的上下文；出现异常时返回原始 section。"""
            try:
                expanded_section = expand_section_with_context(
                    section=section,
                    user_query=user_query,
                    llm=llm,
                    document_index=document_index,
                    expand_override=expand_override,
                )
                # 扩展未返回结果时，回退到原始 section。
                return expanded_section if expanded_section is not None else section
            except Exception as e:
                logger.warning(
                    "Error processing section context expansion: %s. Using original section.",
                    e,
                )
                return section

        # 为每个选中的 section 构建上下文扩展任务。
        expansion_functions: list[tuple[Callable, tuple]] = [
            (
                expand_section_safe,
                (
                    section,
                    secondary_flows_user_query,
                    self.llm,
                    self.document_index,
                    section.center_chunk.document_id in best_doc_ids_set,
                ),
            )
            for section in selected_sections
        ]

        # 记录上下文扩展的开始时间。
        document_expansion_start_time = time.time()

        # 并行扩展选中的 section。
        expanded_sections = run_functions_tuples_in_parallel(expansion_functions)

        # 记录扩展耗时和返回数量。
        document_expansion_elapsed = time.time() - document_expansion_start_time
        logger.debug(
            "Search tool - Expansion of selected documents took %s seconds (expanded %s sections)",
            format(document_expansion_elapsed, ".3f"),
            len(expanded_sections),
        )

        if not expanded_sections:
            expanded_sections = selected_sections

        # 合并同一文档中相邻或重叠的 section，减少重复内容和 token 消耗。
        merged_sections = merge_overlapping_sections(expanded_sections)

        docs_str, citation_mapping = convert_inference_sections_to_llm_string(
            top_sections=merged_sections,
            citation_start=override_kwargs.starting_citation_num,
            limit=override_kwargs.max_llm_chunks,
            include_document_id=False,
            include_link=override_kwargs.include_link,
            note=scope_note or None,
        )

        # 记录整个搜索工具的耗时。
        overall_elapsed = time.time() - overall_start_time
        logger.debug(
            "Search tool - Total execution time: %s seconds (document selection: %ss, document expansion: %ss)",
            format(overall_elapsed, ".3f"),
            format(document_selection_elapsed, ".3f"),
            format(document_expansion_elapsed, ".3f"),
        )

        llm_facing_response = docs_str

        return ToolResponse(
            # 保留候选文档、引用映射和实际展示文档，供调用方处理。
            rich_response=SearchDocsResponse(
                search_docs=search_docs,
                citation_mapping=citation_mapping,
                displayed_docs=final_ui_docs,
            ),
            # 返回经过数量限制的文档文本，供模型生成回答。
            llm_facing_response=llm_facing_response,
        )

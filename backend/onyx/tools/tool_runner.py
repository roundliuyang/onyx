import traceback
from collections import defaultdict
from typing import Any

import onyx.tracing.framework._error_tracing as _error_tracing
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import MessageType
from onyx.context.search.models import SearchDocsResponse
from onyx.db.memory import UserMemoryContext
from onyx.server.query_and_chat.streaming_models import (
    Packet,
    PacketException,
    SectionEnd,
)
from onyx.tools.interface import Tool
from onyx.tools.models import (
    ChatFile,
    ChatMinimalTextMessage,
    OpenURLToolOverrideKwargs,
    ParallelToolCallResponse,
    PythonToolOverrideKwargs,
    SearchToolOverrideKwargs,
    ToolCallException,
    ToolCallKickoff,
    ToolExecutionException,
    ToolResponse,
    WebSearchToolOverrideKwargs,
)
from onyx.tools.tool_implementations.coding_agent.coding_agent_tool import (
    CodingAgentTool,
    CodingAgentToolOverrideKwargs,
)
from onyx.tools.tool_implementations.memory.memory_tool import (
    MemoryTool,
    MemoryToolOverrideKwargs,
)
from onyx.tools.tool_implementations.open_url.open_url_tool import OpenURLTool
from onyx.tools.tool_implementations.python.python_tool import PythonTool
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.tool_implementations.web_search.web_search_tool import WebSearchTool
from onyx.tracing.framework.create import function_span
from onyx.tracing.framework.spans import SpanError
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel

logger = setup_logger()

QUERIES_FIELD = "queries"
URLS_FIELD = "urls"
GENERIC_TOOL_ERROR_MESSAGE = "Tool failed with error: {error}"

# 10 minute timeout for tool execution to prevent indefinite hangs
TOOL_EXECUTION_TIMEOUT_SECONDS = 10 * 60

# Mapping of tool name to the field that should be merged when multiple calls exist
MERGEABLE_TOOL_FIELDS: dict[str, str] = {
    SearchTool.NAME: QUERIES_FIELD,
    WebSearchTool.NAME: QUERIES_FIELD,
    OpenURLTool.NAME: URLS_FIELD,
}


def _merge_tool_calls(tool_calls: list[ToolCallKickoff]) -> list[ToolCallKickoff]:
    """Merge multiple tool calls for SearchTool, WebSearchTool, or OpenURLTool into a single call.

    For SearchTool (internal_search) and WebSearchTool (web_search), if there are
    multiple calls, their queries are merged into a single tool call.
    For OpenURLTool (open_url), multiple calls have their urls merged.
    Other tool calls are left unchanged.

    Args:
        tool_calls: List of tool calls to potentially merge

    Returns:
        List of merged tool calls
    """
    # Group tool calls by tool name
    tool_calls_by_name: dict[str, list[ToolCallKickoff]] = defaultdict(list)
    merged_calls: list[ToolCallKickoff] = []

    for tool_call in tool_calls:
        tool_calls_by_name[tool_call.tool_name].append(tool_call)

    # Process each tool name group
    for tool_name, calls in tool_calls_by_name.items():
        if tool_name in MERGEABLE_TOOL_FIELDS and len(calls) > 1:
            merge_field = MERGEABLE_TOOL_FIELDS[tool_name]

            # Merge field values from all calls
            all_values: list[str] = []
            for call in calls:
                values = call.tool_args.get(merge_field, [])
                if isinstance(values, list):
                    all_values.extend(values)
                elif values:
                    # Handle case where it might be a single string
                    all_values.append(str(values))

            # Create a merged tool call using the first call's ID and merging the field
            merged_args = calls[0].tool_args.copy()
            merged_args[merge_field] = all_values

            merged_call = ToolCallKickoff(
                tool_call_id=calls[0].tool_call_id,  # Use first call's ID
                tool_name=tool_name,
                tool_args=merged_args,
                # Use first call's placement since merged calls become a single call
                placement=calls[0].placement,
            )
            merged_calls.append(merged_call)
        else:
            # No merging needed, add all calls as-is
            merged_calls.extend(calls)

    return merged_calls


def _safe_run_single_tool(
    tool: Tool,
    tool_call: ToolCallKickoff,
    override_kwargs: Any,
) -> ToolResponse:
    """Execute a single tool and return its response.

    This function is designed to be run in parallel via run_functions_tuples_in_parallel.

    Exception handling:
    - ToolCallException: Expected errors from tool execution (e.g., invalid input,
      API failures). Uses the exception's llm_facing_message for LLM consumption.
    - Other exceptions: Unexpected errors. Uses a generic error message.

    In all cases (success or failure):
    - SectionEnd packet is emitted to signal tool completion
    - tool_call is set on the response for downstream processing
    """
    tool_response: ToolResponse | None = None

    with function_span(tool.name) as span_fn:
        span_fn.span_data.input = str(tool_call.tool_args)
        try:
            tool_response = tool.run(
                placement=tool_call.placement,
                override_kwargs=override_kwargs,
                **tool_call.tool_args,
            )
            span_fn.span_data.output = tool_response.llm_facing_response
        except ToolCallException as e:
            # ToolCallException is an expected error from tool execution
            # Use llm_facing_message which is specifically designed for LLM consumption
            logger.error("Tool call error for %s: %s", tool.name, e)
            tool_response = ToolResponse(
                rich_response=None,
                llm_facing_response=GENERIC_TOOL_ERROR_MESSAGE.format(
                    error=e.llm_facing_message
                ),
            )
            _error_tracing.attach_error_to_current_span(
                SpanError(
                    message="Tool call error (expected)",
                    data={
                        "tool_name": tool.name,
                        "tool_call_id": tool_call.tool_call_id,
                        "tool_args": tool_call.tool_args,
                        "error": str(e),
                        "llm_facing_message": e.llm_facing_message,
                        "stack_trace": traceback.format_exc(),
                        "error_type": "ToolCallException",
                    },
                )
            )
        except ToolExecutionException as e:
            # Unexpected error during tool execution
            logger.error("Unexpected error running tool %s: %s", tool.name, e)
            tool_response = ToolResponse(
                rich_response=None,
                llm_facing_response=GENERIC_TOOL_ERROR_MESSAGE.format(error=str(e)),
            )
            _error_tracing.attach_error_to_current_span(
                SpanError(
                    message="Tool execution error (unexpected)",
                    data={
                        "tool_name": tool.name,
                        "tool_call_id": tool_call.tool_call_id,
                        "tool_args": tool_call.tool_args,
                        "error": str(e),
                        "stack_trace": traceback.format_exc(),
                        "error_type": type(e).__name__,
                    },
                )
            )
            if e.emit_error_packet:
                tool.emitter.emit(
                    Packet(
                        placement=tool_call.placement,
                        obj=PacketException(exception=e),
                    )
                )
        except Exception as e:
            # Unexpected error during tool execution
            logger.error("Unexpected error running tool %s: %s", tool.name, e)
            tool_response = ToolResponse(
                rich_response=None,
                llm_facing_response=GENERIC_TOOL_ERROR_MESSAGE.format(error=str(e)),
            )
            _error_tracing.attach_error_to_current_span(
                SpanError(
                    message="Tool execution error (unexpected)",
                    data={
                        "tool_name": tool.name,
                        "tool_call_id": tool_call.tool_call_id,
                        "tool_args": tool_call.tool_args,
                        "error": str(e),
                        "stack_trace": traceback.format_exc(),
                        "error_type": type(e).__name__,
                    },
                )
            )

    # Emit SectionEnd after tool completes (success or failure)
    tool.emitter.emit(
        Packet(
            placement=tool_call.placement,
            obj=SectionEnd(),
        )
    )

    # Set tool_call on the response for downstream processing
    tool_response.tool_call = tool_call
    return tool_response


def run_tool_calls(
    tool_calls: list[ToolCallKickoff],
    tools: list[Tool],
    # 以下上下文参数供不同的内置工具使用。
    message_history: list[ChatMessageSimple],
    user_memory_context: UserMemoryContext | None,
    user_info: str | None,
    citation_mapping: dict[int, str],
    next_citation_num: int,
    # 同时限制本批次的调用总数和并发数；超出上限的调用直接丢弃，不排队。
    max_concurrent_tools: int | None = None,
    # 重复调用搜索工具时，可跳过查询扩展。
    skip_search_query_expansion: bool = False,
    # 会话文件，供 PythonTool 等工具读取。
    chat_files: list[ChatFile] | None = None,
    # URL 到摘要的映射，将网页搜索结果传给打开网页的工具。
    url_snippet_map: dict[str, str] | None = None,
    # 为 False 时，搜索查询扩展不使用记忆；记忆工具仍接收完整记忆上下文。
    inject_memories_in_prompt: bool = True,
) -> ParallelToolCallResponse:
    """合并并并行执行一批工具调用，汇总工具响应和引用映射。

    聊天主流程中的调用链路：
        run_llm_loop
        → run_llm_step：获取模型生成的工具调用
        → run_tool_calls（本方法）
           → _merge_tool_calls：合并搜索查询或待打开的 URL
           → 过滤未知工具，按批次上限截断调用列表
           → 准备各工具的上下文参数与引用起始编号
           → run_functions_tuples_in_parallel：在线程池中执行
              → _safe_run_single_tool
                 → tool.run：执行具体工具
                    └─ SearchTool.run：本地知识库搜索入口
                 → 返回 ToolResponse，并发送工具结束事件
           → 合并搜索结果的引用映射，返回 ParallelToolCallResponse
        → run_llm_loop：将工具结果加入历史，继续调用模型生成回答或选择工具

    本方法负责工具调度；实际检索由 SearchTool 等具体工具完成。
    SearchTool 和 WebSearchTool 合并 queries，OpenURLTool 合并 urls。
    引用类工具的起始编号按 100 递增，预留引用空间；这不是结果数量限制。
    citation_mapping 会被原地更新，调用方可以继续使用同一份映射。

    参数：
        tool_calls：模型请求执行的工具调用。
        tools：本轮可用的工具实例，按工具名称匹配。
        message_history：聊天历史，用于提取最近的用户问题及工具所需上下文。
        user_memory_context：用户信息与记忆，供搜索和记忆工具使用。
        user_info：传给搜索工具的用户信息文本。
        citation_mapping：已有的引用编号到 URL 的映射。
        next_citation_num：可分配的下一个引用编号。
        max_concurrent_tools：本批次调用数量与线程数上限；超出部分直接丢弃。
        skip_search_query_expansion：是否跳过搜索查询扩展，供重复搜索时使用。
        chat_files：传给 PythonTool 等工具的会话文件。
        url_snippet_map：URL 到摘要的映射，供 OpenURLTool 使用。
        inject_memories_in_prompt：是否向搜索工具传入记忆；不影响记忆工具的上下文。

    返回：
        工具响应列表与更新后的引用映射。工具内部异常通常转换为错误响应；
        在线程池层执行失败并返回 None 的条目会被排除。

    异常：
        ValueError：调用 SearchTool 时，聊天历史中没有用户消息。
    """
    # 合并同类搜索与网页打开调用，减少重复调度。
    if url_snippet_map is None:
        url_snippet_map = {}
    merged_tool_calls = _merge_tool_calls(tool_calls)

    if not merged_tool_calls:
        return ParallelToolCallResponse(
            tool_responses=[],
            updated_citation_mapping=citation_mapping,
        )

    tools_by_name = {tool.name: tool for tool in tools}

    # 先丢弃未知工具，避免它们占用本批次的调用名额。
    filtered_tool_calls: list[ToolCallKickoff] = []
    for tool_call in merged_tool_calls:
        if tool_call.tool_name not in tools_by_name:
            logger.warning("Tool %s not found in tools list", tool_call.tool_name)
            continue
        filtered_tool_calls.append(tool_call)

    # 按上限截断有效调用；非正数表示本批次不执行任何工具。
    if max_concurrent_tools is not None:
        if max_concurrent_tools <= 0:
            return ParallelToolCallResponse(
                tool_responses=[],
                updated_citation_mapping=citation_mapping,
            )
        filtered_tool_calls = filtered_tool_calls[:max_concurrent_tools]

    # 使用调用方提供的引用起点，避开已有项目文件的引用编号。
    starting_citation_num = next_citation_num

    # 仅保留消息文本与角色，供搜索和记忆工具共享。
    minimal_history = [
        ChatMinimalTextMessage(message=msg.message, message_type=msg.message_type)
        for msg in message_history
    ]
    last_user_message = None
    for i in range(len(minimal_history) - 1, -1, -1):
        if minimal_history[i].message_type == MessageType.USER:
            last_user_message = minimal_history[i].message
            break

    # 反转已有引用映射，让 OpenURLTool 按 URL 查找引用编号。
    url_to_citation: dict[str, int] = {
        url: citation_num for citation_num, url in citation_mapping.items()
    }

    # 为各工具准备运行时上下文；引用类工具使用不同的引用起点。
    tool_run_params: list[tuple[Tool, ToolCallKickoff, Any]] = []

    for tool_call in filtered_tool_calls:
        tool = tools_by_name[tool_call.tool_name]

        # 在线程池执行前发送开始事件，供前端显示工具步骤。
        tool.emit_start(placement=tool_call.placement)

        override_kwargs: (
            SearchToolOverrideKwargs
            | WebSearchToolOverrideKwargs
            | OpenURLToolOverrideKwargs
            | PythonToolOverrideKwargs
            | MemoryToolOverrideKwargs
            | CodingAgentToolOverrideKwargs
            | None
        ) = None

        if isinstance(tool, SearchTool):
            if last_user_message is None:
                raise ValueError("No user message found in message history")

            # 关闭记忆注入时，仅移除记忆内容，保留其余用户上下文。
            search_memory_context = (
                user_memory_context
                if inject_memories_in_prompt
                else (
                    user_memory_context.without_memories()
                    if user_memory_context
                    else None
                )
            )
            override_kwargs = SearchToolOverrideKwargs(
                starting_citation_num=starting_citation_num,
                original_query=last_user_message,
                message_history=minimal_history,
                user_memory_context=search_memory_context,
                user_info=user_info,
                skip_query_expansion=skip_search_query_expansion,
            )
            # 为本次搜索预留 100 个引用编号，再分配下一个工具的起点。
            starting_citation_num += 100

        elif isinstance(tool, WebSearchTool):
            override_kwargs = WebSearchToolOverrideKwargs(
                starting_citation_num=starting_citation_num,
            )
            # 为网页搜索预留 100 个引用编号。
            starting_citation_num += 100

        elif isinstance(tool, OpenURLTool):
            override_kwargs = OpenURLToolOverrideKwargs(
                starting_citation_num=starting_citation_num,
                citation_mapping=url_to_citation,
                url_snippet_map=url_snippet_map,
            )
            starting_citation_num += 100

        elif isinstance(tool, PythonTool):
            override_kwargs = PythonToolOverrideKwargs(
                chat_files=chat_files or [],
            )
        elif isinstance(tool, CodingAgentTool):
            override_kwargs = CodingAgentToolOverrideKwargs()
        elif isinstance(tool, MemoryTool):
            # 记忆工具始终接收已有记忆，以便处理记忆更新。
            override_kwargs = MemoryToolOverrideKwargs(
                user_name=(
                    user_memory_context.user_info.name if user_memory_context else None
                ),
                user_email=(
                    user_memory_context.user_info.email if user_memory_context else None
                ),
                user_role=(
                    user_memory_context.user_info.role if user_memory_context else None
                ),
                existing_memories=(
                    list(user_memory_context.memories) if user_memory_context else []
                ),
                chat_history=minimal_history,
            )

        tool_run_params.append((tool, tool_call, override_kwargs))

    # 将工具实例、模型参数和上下文交给安全包装函数，并在线程池执行。
    functions_with_args = [
        (_safe_run_single_tool, (tool, tool_call, override_kwargs))
        for tool, tool_call, override_kwargs in tool_run_params
    ]

    tool_run_results: list[ToolResponse | None] = run_functions_tuples_in_parallel(
        functions_with_args,
        allow_failures=True,  # 单个任务失败时，仍收集其他任务的结果。
        max_workers=max_concurrent_tools,
        timeout=TOOL_EXECUTION_TIMEOUT_SECONDS,
    )

    # 线程池完成后统一合并引用，跳过未返回响应的任务。
    for result in tool_run_results:
        if result is None:
            continue

        if result and isinstance(result.rich_response, SearchDocsResponse):
            new_citations = result.rich_response.citation_mapping
            if new_citations:
                # 原地加入新引用，供后续回答中的引用解析使用。
                citation_mapping.update(new_citations)

    tool_responses = [result for result in tool_run_results if result is not None]
    return ParallelToolCallResponse(
        tool_responses=tool_responses,
        updated_citation_mapping=citation_mapping,
    )

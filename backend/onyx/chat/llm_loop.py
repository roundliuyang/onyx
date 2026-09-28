import json
import time
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.chat_utils import (
    build_python_chat_files_from_search_docs,
    create_tool_call_failure_messages,
)
from onyx.chat.citation_processor import (
    CitationMapping,
    CitationMode,
    DynamicCitationProcessor,
)
from onyx.chat.citation_utils import update_citation_processor_from_tool_response
from onyx.chat.emitter import Emitter
from onyx.chat.llm_step import (
    _looks_like_xml_tool_call_payload,
    extract_tool_calls_from_response_text,
    run_llm_step,
)
from onyx.chat.models import (
    ChatMessageSimple,
    ContextFileMetadata,
    ExtractedContextFiles,
    FileToolMetadata,
    LlmStepResult,
    ToolCallSimple,
)
from onyx.chat.prompt_utils import (
    build_reminder_message,
    build_system_prompt,
    get_default_base_system_prompt,
    process_prompt_template,
)
from onyx.chat.token_budget import resolve_chat_token_budget
from onyx.configs.app_configs import INTEGRATION_TESTS_MODE
from onyx.configs.chat_configs import MAX_LLM_CYCLES
from onyx.configs.constants import DocumentSource, MessageType
from onyx.context.search.models import SearchDoc, SearchDocsResponse
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.memory import UserMemoryContext, add_memory, update_memory_at_index
from onyx.db.models import Persona
from onyx.file_store.models import ChatFileType
from onyx.llm.constants import LlmProviderNames
from onyx.llm.exceptions import ClassifiedLLMError
from onyx.llm.interfaces import LLM, LLMUserIdentity, ToolChoiceOptions
from onyx.llm.model_capabilities import is_true_openai_model
from onyx.llm.models import ReasoningEffort
from onyx.llm.utils import model_supports_image_input
from onyx.prompts.chat_prompts import (
    IMAGE_GEN_REMINDER,
    NON_VISION_IMAGE_MARKER,
    OPEN_URL_REMINDER,
)
from onyx.prompts.prompt_utils import substitute_user_placeholders
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    OverallStop,
    Packet,
    ToolCallDebug,
    TopLevelBranching,
)
from onyx.tools.built_in_tools import CITEABLE_TOOLS_NAMES, STOPPING_TOOLS_NAMES
from onyx.tools.constants import FILE_READER_TOOL_NAME
from onyx.tools.interface import Tool
from onyx.tools.models import (
    ChatFile,
    CustomToolCallSummary,
    CustomToolUserFileSnapshot,
    MemoryToolResponseSnapshot,
    PythonToolRichResponse,
    ToolCallInfo,
    ToolCallKickoff,
    ToolResponse,
)
from onyx.tools.tool_implementations.images.models import FinalImageGenerationResponse
from onyx.tools.tool_implementations.memory.models import MemoryToolResponse
from onyx.tools.tool_implementations.open_url.open_url_tool import OpenURLTool
from onyx.tools.tool_implementations.python.python_tool import PythonTool
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.tool_implementations.web_search.utils import extract_url_snippet_map
from onyx.tools.tool_implementations.web_search.web_search_tool import WebSearchTool
from onyx.tools.tool_runner import run_tool_calls
from onyx.tools.utils import compute_all_tool_tokens
from onyx.tracing.framework.create import ChatTraceMetadata, trace
from onyx.utils.logger import setup_logger
from shared_configs.contextvars import get_current_incognito_record_mode

logger = setup_logger()

# Used when no token_counter is available to measure the non-vision image
# marker; intentionally generous so budgeting stays conservative.
_NON_VISION_MARKER_TOKEN_FALLBACK = 40


class EmptyLLMResponseError(ClassifiedLLMError):
    """Raised when the streamed LLM response completes without a usable answer."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        tool_choice: ToolChoiceOptions,
        client_error_msg: str,
        error_code: str = "EMPTY_LLM_RESPONSE",
        is_retryable: bool = True,
        finish_reason: str | None = None,
    ) -> None:
        super().__init__(
            client_error_msg=client_error_msg,
            error_code=error_code,
            is_retryable=is_retryable,
        )
        self.provider = provider
        self.model = model
        self.tool_choice = tool_choice
        self.finish_reason = finish_reason


# LiteLLM maps these native policy blocks to content_filter, but gateways may
# forward the provider value unchanged.
_REFUSAL_FINISH_REASONS = {
    "BLOCKLIST",
    "CONTENT_BLOCKED",
    "ERROR_TOXIC",
    "IMAGE_OTHER",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
    "IMAGE_SAFETY",
    "LANGUAGE",
    "MODEL_ARMOR",
    "OTHER",
    "PROHIBITED_CONTENT",
    "RECITATION",
    "SAFETY",
    "SPII",
    "content_filter",
    "content_filtered",
    "guardrail_intervened",
    "refusal",
    "sensitive",
}


def _build_empty_llm_response_error(
    llm: LLM,
    llm_step_result: LlmStepResult,
    tool_choice: ToolChoiceOptions,
) -> EmptyLLMResponseError:
    provider = llm.config.model_provider
    model = llm.config.model_name
    finish_reason = llm_step_result.finish_reason

    # A refusal/content-filter stop is a deliberate model decision (HTTP 200
    # with no content), not a transport failure — retrying the same request
    # against the same model will not help.
    if finish_reason in _REFUSAL_FINISH_REASONS:
        model_suggestion = (
            " (e.g. Claude Opus 4.8)" if provider == LlmProviderNames.ANTHROPIC else ""
        )
        return EmptyLLMResponseError(
            provider=provider,
            model=model,
            tool_choice=tool_choice,
            client_error_msg=(
                "The selected model declined to respond to this request and "
                f"returned no content (finish_reason={finish_reason}). Try "
                "rephrasing the request or switching to a different model"
                f"{model_suggestion}."
            ),
            error_code="MODEL_REFUSAL",
            is_retryable=False,
            finish_reason=finish_reason,
        )

    # OpenAI quota exhaustion has reached us as a streamed "stop" with zero content.
    # When the stream is completely empty and there is no reasoning/tool output, surface
    # the likely account-level cause instead of a generic tool-calling error.
    if (
        not llm_step_result.reasoning
        and provider == LlmProviderNames.OPENAI
        and is_true_openai_model(provider, model)
    ):
        return EmptyLLMResponseError(
            provider=provider,
            model=model,
            tool_choice=tool_choice,
            client_error_msg=(
                "The selected OpenAI model returned an empty streamed response "
                "before producing any tokens. This commonly happens when the API "
                "key or project has no remaining quota or billing is not enabled. "
                "Verify quota and billing for this key and try again."
            ),
            error_code="BUDGET_EXCEEDED",
            is_retryable=False,
            finish_reason=finish_reason,
        )

    return EmptyLLMResponseError(
        provider=provider,
        model=model,
        tool_choice=tool_choice,
        client_error_msg=(
            "The selected model returned no final answer before the stream "
            "completed. No text or tool calls were received from the upstream "
            "provider."
        ),
        finish_reason=finish_reason,
    )


def _try_fallback_tool_extraction(
    llm_step_result: LlmStepResult,
    tool_choice: ToolChoiceOptions,
    fallback_extraction_attempted: bool,
    tool_defs: list[dict],
    turn_index: int,
) -> tuple[LlmStepResult, bool]:
    """Attempt to extract tool calls from response text as a fallback.

    This is a last resort fallback for low quality LLMs or those that don't have
    tool calling from the serving layer. Also triggers if there's reasoning but
    no answer and no tool calls.

    Args:
        llm_step_result: The result from the LLM step
        tool_choice: The tool choice option used for this step
        fallback_extraction_attempted: Whether fallback extraction was already attempted
        tool_defs: List of tool definitions
        turn_index: The current turn index for placement

    Returns:
        Tuple of (possibly updated LlmStepResult, whether fallback was attempted this call)
    """
    if fallback_extraction_attempted:
        return llm_step_result, False

    no_tool_calls = (
        not llm_step_result.tool_calls or len(llm_step_result.tool_calls) == 0
    )
    reasoning_but_no_answer_or_tools = (
        llm_step_result.reasoning and not llm_step_result.answer and no_tool_calls
    )
    xml_tool_call_text_detected = no_tool_calls and (
        _looks_like_xml_tool_call_payload(llm_step_result.answer)
        or _looks_like_xml_tool_call_payload(llm_step_result.raw_answer)
        or _looks_like_xml_tool_call_payload(llm_step_result.reasoning)
    )
    should_try_fallback = (
        (tool_choice == ToolChoiceOptions.REQUIRED and no_tool_calls)
        or reasoning_but_no_answer_or_tools
        or xml_tool_call_text_detected
    )

    if not should_try_fallback:
        return llm_step_result, False

    # Try to extract from answer first, then fall back to reasoning
    extracted_tool_calls: list[ToolCallKickoff] = []

    if llm_step_result.answer:
        extracted_tool_calls = extract_tool_calls_from_response_text(
            response_text=llm_step_result.answer,
            tool_definitions=tool_defs,
            placement=Placement(turn_index=turn_index),
        )
    if (
        not extracted_tool_calls
        and llm_step_result.raw_answer
        and llm_step_result.raw_answer != llm_step_result.answer
    ):
        extracted_tool_calls = extract_tool_calls_from_response_text(
            response_text=llm_step_result.raw_answer,
            tool_definitions=tool_defs,
            placement=Placement(turn_index=turn_index),
        )
    if not extracted_tool_calls and llm_step_result.reasoning:
        extracted_tool_calls = extract_tool_calls_from_response_text(
            response_text=llm_step_result.reasoning,
            tool_definitions=tool_defs,
            placement=Placement(turn_index=turn_index),
        )
    if extracted_tool_calls:
        logger.info(
            "Extracted %s tool call(s) from response text as fallback",
            len(extracted_tool_calls),
        )
        return (
            LlmStepResult(
                reasoning=llm_step_result.reasoning,
                answer=llm_step_result.answer,
                tool_calls=extracted_tool_calls,
                raw_answer=llm_step_result.raw_answer,
                finish_reason=llm_step_result.finish_reason,
            ),
            True,
        )

    return llm_step_result, True


# Default 6 covers the common search → open_url pattern:
# Cycle 1: Calls web_search for something
# Cycle 2: Calls open_url for some results
# Cycle 3: Calls web_search for some other aspect of the question
# Cycle 4: Calls open_url for some results
# Cycle 5: Maybe call open_url for some additional results or because last set failed
# Cycle 6: No more tools available, forced to answer
# Override via the MAX_LLM_CYCLES env var when running with tool-heavy MCPs
# that legitimately need more turns. Imported from chat_configs.


def _build_context_file_citation_mapping(
    file_metadata: list[ContextFileMetadata],
    starting_citation_num: int = 1,
) -> CitationMapping:
    """Build citation mapping for context files.

    Converts context file metadata into SearchDoc objects that can be cited.
    Citation numbers start from the provided starting number.

    Args:
        file_metadata: List of context file metadata
        starting_citation_num: Starting citation number (default: 1)

    Returns:
        Dictionary mapping citation numbers to SearchDoc objects
    """
    citation_mapping: CitationMapping = {}

    for idx, file_meta in enumerate(file_metadata, start=starting_citation_num):
        search_doc = SearchDoc(
            document_id=file_meta.file_id,
            chunk_ind=0,
            semantic_identifier=file_meta.filename,
            link=None,
            blurb=file_meta.file_content,
            source_type=DocumentSource.FILE,
            boost=1,
            hidden=False,
            metadata={},
            score=0.0,
            match_highlights=[file_meta.file_content],
        )
        citation_mapping[idx] = search_doc

    return citation_mapping


def _build_project_message(
    context_files: ExtractedContextFiles | None,
    token_counter: Callable[[str], int] | None,
    available_tool_names: set[str] | None = None,
) -> list[ChatMessageSimple]:
    """Build messages for context-injected / tool-backed files.

    Returns up to two messages:
    1. The full-text files message (if file_texts is populated).
    2. A lightweight metadata message for oversized files, naming whichever
       retrieval tool this request actually received.
    """
    if not context_files:
        return []

    messages: list[ChatMessageSimple] = []
    if context_files.file_texts:
        messages.append(
            _create_context_files_message(context_files, token_counter=None)
        )
    if context_files.file_metadata_for_tool and token_counter:
        messages.append(
            _create_file_tool_metadata_message(
                context_files.file_metadata_for_tool,
                token_counter,
                available_tool_names,
            )
        )
    return messages


def count_message_replay_tokens(
    msg: ChatMessageSimple,
    *,
    image_files_replayed_as_markers: bool = False,
    token_counter: Callable[[str], int] | None = None,
) -> int:
    if not image_files_replayed_as_markers:
        return msg.token_count
    # Include images whose stored cost is zero, such as project images.
    num_images = sum(
        1 for f in msg.image_files or [] if f.file_type == ChatFileType.IMAGE
    )
    if not num_images:
        return msg.token_count
    sample_marker = NON_VISION_IMAGE_MARKER.format(file_id="0" * 36)
    marker_tokens = (
        token_counter(sample_marker)
        if token_counter
        else _NON_VISION_MARKER_TOKEN_FALLBACK
    )
    return max(0, msg.token_count - msg.image_token_count) + num_images * marker_tokens


def construct_message_history(
    system_prompt: ChatMessageSimple | None,
    custom_agent_prompt: ChatMessageSimple | None,
    simple_chat_history: list[ChatMessageSimple],
    reminder_message: ChatMessageSimple | None,
    context_files: ExtractedContextFiles | None,
    available_tokens: int,
    last_n_user_messages: int | None = None,
    token_counter: Callable[[str], int] | None = None,
    all_injected_file_metadata: dict[str, FileToolMetadata] | None = None,
    image_files_replayed_as_markers: bool = False,
    # Tool names this step offers the model. Only the retrieval tools
    # (read_file, internal_search) are consulted, so the out-of-context file
    # notice never names one the model cannot call. Steps exposing neither pass
    # an empty set; leaving it unset also names no tool.
    available_tool_names: set[str] | None = None,
) -> list[ChatMessageSimple]:
    if last_n_user_messages is not None:
        if last_n_user_messages <= 0:
            raise ValueError(
                "filtering chat history by last N user messages must be a value greater than 0"
            )

    _replay_token_count = partial(
        count_message_replay_tokens,
        image_files_replayed_as_markers=image_files_replayed_as_markers,
        token_counter=token_counter,
    )

    # Build the project / file-metadata messages up front so we can use their
    # actual token counts for the budget.
    project_messages = _build_project_message(
        context_files, token_counter, available_tool_names
    )
    project_messages_tokens = sum(m.token_count for m in project_messages)

    history_token_budget = available_tokens
    history_token_budget -= system_prompt.token_count if system_prompt else 0
    history_token_budget -= (
        custom_agent_prompt.token_count if custom_agent_prompt else 0
    )
    history_token_budget -= project_messages_tokens
    history_token_budget -= reminder_message.token_count if reminder_message else 0

    if history_token_budget < 0:
        raise ValueError("Not enough tokens available to construct message history")

    if system_prompt:
        system_prompt.should_cache = True

    # If no history, build minimal context
    if not simple_chat_history:
        result = [system_prompt] if system_prompt else []
        if custom_agent_prompt:
            result.append(custom_agent_prompt)
        result.extend(project_messages)
        if reminder_message:
            result.append(reminder_message)
        return result

    # If last_n_user_messages is set, filter history to only include the last n user messages
    if last_n_user_messages is not None:
        # Find all user message indices
        user_msg_indices = [
            i
            for i, msg in enumerate(simple_chat_history)
            if msg.message_type == MessageType.USER
        ]

        if not user_msg_indices:
            raise ValueError("No user message found in simple_chat_history")

        # If we have more than n user messages, keep only the last n
        if len(user_msg_indices) > last_n_user_messages:
            # Find the index of the n-th user message from the end
            # For example, if last_n_user_messages=2, we want the 2nd-to-last user message
            nth_user_msg_index = user_msg_indices[-(last_n_user_messages)]
            # Keep everything from that user message onwards
            simple_chat_history = simple_chat_history[nth_user_msg_index:]

    # Find the last USER message in the history
    # The history may contain tool calls and responses after the last user message
    last_user_msg_index = None
    for i in range(len(simple_chat_history) - 1, -1, -1):
        if simple_chat_history[i].message_type == MessageType.USER:
            last_user_msg_index = i
            break

    if last_user_msg_index is None:
        raise ValueError("No user message found in simple_chat_history")

    # Split history into three parts:
    # 1. History before the last user message
    # 2. The last user message
    # 3. Messages after the last user message (tool calls, responses, etc.)
    history_before_last_user = simple_chat_history[:last_user_msg_index]
    last_user_message = simple_chat_history[last_user_msg_index]
    messages_after_last_user = simple_chat_history[last_user_msg_index + 1 :]

    # Calculate tokens needed for the last user message and everything after it
    last_user_tokens = _replay_token_count(last_user_message)
    after_user_tokens = sum(
        _replay_token_count(msg) for msg in messages_after_last_user
    )

    # Check if we can fit at least the last user message and messages after it
    required_tokens = last_user_tokens + after_user_tokens
    if required_tokens > history_token_budget:
        raise ValueError(
            f"Not enough tokens to include the last user message and subsequent messages. "
            f"Required: {required_tokens}, Available: {history_token_budget}"
        )

    # Calculate remaining budget for history before the last user message
    remaining_budget = history_token_budget - required_tokens

    # Truncate history_before_last_user from the top to fit in remaining budget.
    # Track dropped file messages so we can provide their metadata to the
    # FileReaderTool instead.
    truncated_history_before: list[ChatMessageSimple] = []
    current_token_count = 0

    for msg in reversed(history_before_last_user):
        msg_tokens = _replay_token_count(msg)
        if current_token_count + msg_tokens <= remaining_budget:
            msg.should_cache = True
            truncated_history_before.insert(0, msg)
            current_token_count += msg_tokens
        else:
            # Can't fit this message, stop truncating.
            # This message and everything older is dropped.
            break

    # Collect file_ids from ALL dropped messages (those not in
    # truncated_history_before). The truncation loop above keeps the most
    # recent messages, so the dropped ones are at the start of the original
    # list up to (len(history) - len(kept)).
    num_kept = len(truncated_history_before)
    dropped_file_ids: list[str] = [
        msg.file_id
        for msg in history_before_last_user[: len(history_before_last_user) - num_kept]
        if msg.file_id is not None
    ]

    # Also treat "orphaned" metadata entries as dropped -- these are files
    # from messages removed by summary truncation (before convert_chat_history
    # ran), so no ChatMessageSimple was ever tagged with their file_id.
    if all_injected_file_metadata:
        surviving_file_ids = {
            msg.file_id for msg in simple_chat_history if msg.file_id is not None
        }
        for fid in all_injected_file_metadata:
            if fid not in surviving_file_ids and fid not in dropped_file_ids:
                dropped_file_ids.append(fid)

    # Build a forgotten-files metadata message if any file messages were
    # dropped AND we have metadata for them (meaning the FileReaderTool is
    # available). Reserve tokens for this message in the budget.
    forgotten_files_message: ChatMessageSimple | None = None
    if dropped_file_ids and all_injected_file_metadata and token_counter:
        forgotten_meta = [
            all_injected_file_metadata[fid]
            for fid in dropped_file_ids
            if fid in all_injected_file_metadata
        ]
        if forgotten_meta:
            logger.debug(
                "FileReader: building forgotten-files message for %s",
                [(m.file_id, m.filename) for m in forgotten_meta],
            )
            forgotten_files_message = _create_file_tool_metadata_message(
                forgotten_meta, token_counter, available_tool_names
            )
            # Shrink the remaining budget. If the metadata message doesn't
            # fit we may need to drop more history messages.
            remaining_budget -= forgotten_files_message.token_count
            while truncated_history_before and current_token_count > remaining_budget:
                evicted = truncated_history_before.pop(0)
                current_token_count -= _replay_token_count(evicted)
                # If the evicted message is itself a file, add it to the
                # forgotten metadata (it's now dropped too).
                if (
                    evicted.file_id is not None
                    and evicted.file_id in all_injected_file_metadata
                    and evicted.file_id not in {m.file_id for m in forgotten_meta}
                ):
                    forgotten_meta.append(all_injected_file_metadata[evicted.file_id])
                    # Rebuild the message with the new entry
                    forgotten_files_message = _create_file_tool_metadata_message(
                        forgotten_meta, token_counter, available_tool_names
                    )

    # Build the final message list according to README ordering:
    # [system], [history_before_last_user], [custom_agent], [context_files],
    # [forgotten_files], [last_user_message], [messages_after_last_user], [reminder]
    result = [system_prompt] if system_prompt else []

    # 1. Add truncated history before last user message
    result.extend(truncated_history_before)

    # 2. Add custom agent prompt (inserted before last user message)
    if custom_agent_prompt:
        result.append(custom_agent_prompt)

    # 3. Add context files / file-metadata messages (inserted before last user message)
    result.extend(project_messages)

    # 4. Add forgotten-files metadata (right before the user's question)
    if forgotten_files_message:
        result.append(forgotten_files_message)

    # 5. Add last user message (with context images attached)
    result.append(last_user_message)

    # 6. Add messages after last user message (tool calls, responses, etc.)
    result.extend(messages_after_last_user)

    # 7. Add reminder message at the very end
    if reminder_message:
        result.append(reminder_message)

    return _drop_orphaned_tool_call_responses(result)


def _drop_orphaned_tool_call_responses(
    messages: list[ChatMessageSimple],
) -> list[ChatMessageSimple]:
    """Drop tool response messages whose tool_call_id is not in prior assistant tool calls.

    This can happen when history truncation drops an ASSISTANT tool-call message but
    leaves a later TOOL_CALL_RESPONSE message in context. Some providers (e.g. Ollama)
    reject such history with an "unexpected tool call id" error.
    """
    known_tool_call_ids: set[str] = set()
    sanitized: list[ChatMessageSimple] = []

    for msg in messages:
        if msg.message_type == MessageType.ASSISTANT and msg.tool_calls:
            for tool_call in msg.tool_calls:
                known_tool_call_ids.add(tool_call.tool_call_id)
            sanitized.append(msg)
            continue

        if msg.message_type == MessageType.TOOL_CALL_RESPONSE:
            if msg.tool_call_id and msg.tool_call_id in known_tool_call_ids:
                sanitized.append(msg)
            else:
                logger.debug(
                    "Dropping orphaned tool response with tool_call_id=%s while constructing message history",
                    msg.tool_call_id,
                )
            continue

        sanitized.append(msg)

    return sanitized


def _create_file_tool_metadata_message(
    file_metadata: list[FileToolMetadata],
    token_counter: Callable[[str], int],
    available_tool_names: set[str] | None = None,
) -> ChatMessageSimple:
    """Build a lightweight metadata-only message listing files not held in context.

    Name only a tool this step actually received. FileReaderTool is attached
    only when the vector DB is disabled, and internal search can be absent even
    when it is enabled (persona, ``allowed_tool_ids``, or a disabled search
    usage setting). Naming a tool the model was never given makes it invent
    workarounds — it searches the web for the document or guesses the contents.

    Preference order is read_file, then internal search, then the python tool.
    read_file pages through a file directly; search retrieves from the indexed
    copy; the python tool is handed the files themselves, so prompt truncation
    does not take them away from it.

    The python tier applies only when every listed file actually reached
    ``chat_files_for_tools`` (see ``FileToolMetadata.staged_for_tools``) —
    summary-truncated files are listed for the LLM but never staged, so naming
    python for them would send the model after bytes it does not have. The
    notice also stops short of promising a path, because PythonTool normalizes
    and de-duplicates filenames at staging time and applies its own count and
    byte caps.

    An unreported tool set names no tool. Steps that offer none are common (a
    deep-research final report runs with no tools), and under-promising is the
    safe direction to fail in.
    """
    offered: set[str] = available_tool_names or set()
    if FILE_READER_TOOL_NAME in offered:
        lines: list[str] = [
            "You have access to the following files. Use the read_file tool to "
            "read sections of any file. You MUST pass the file_id UUID (not the "
            "filename) to read_file:"
        ]
        # The UUID is only meaningful to read_file, so it is listed only here.
        lines.extend(
            f'- file_id="{meta.file_id}" filename="{meta.filename}" (~{meta.approx_char_count:,} chars)'
            for meta in file_metadata
        )
        return _finalize_file_metadata_message(lines, token_counter)

    if SearchTool.NAME in offered:
        lines = [
            "These files are attached but too large to include in full. Their "
            "contents are indexed — use internal search to find the relevant "
            "passages. Do not guess them or search the web for them:"
        ]
    elif PythonTool.NAME in offered and all(
        meta.staged_for_tools for meta in file_metadata
    ):
        lines = [
            "These files are attached but too large to include in full. The "
            "python tool receives them — read them there, listing the working "
            "directory if a name does not resolve. Do not guess their contents "
            "or search the web for them:"
        ]
    else:
        lines = [
            "These files are attached but too large to include in full, and no "
            "tool here can read them. Do not guess their contents or search the "
            "web for them — say they are too large to read in this conversation:"
        ]
    lines.extend(
        f'- filename="{meta.filename}" (~{meta.approx_char_count:,} chars)'
        for meta in file_metadata
    )
    return _finalize_file_metadata_message(lines, token_counter)


def _finalize_file_metadata_message(
    lines: list[str],
    token_counter: Callable[[str], int],
) -> ChatMessageSimple:
    message_content = "\n".join(lines)
    return ChatMessageSimple(
        message=message_content,
        token_count=token_counter(message_content),
        message_type=MessageType.USER,
    )


def _create_context_files_message(
    context_files: ExtractedContextFiles,
    token_counter: Callable[[str], int] | None,  # noqa: ARG001
) -> ChatMessageSimple:
    """Convert context files to a ChatMessageSimple message.

    Format follows the README specification for document representation.
    """
    import json

    # Format as documents JSON as described in README
    documents_list = []
    for idx, file_text in enumerate(context_files.file_texts, start=1):
        title = (
            context_files.file_metadata[idx - 1].filename
            if idx - 1 < len(context_files.file_metadata)
            else None
        )
        entry: dict[str, Any] = {"document": idx}
        if title:
            entry["title"] = title
        entry["contents"] = file_text
        documents_list.append(entry)

    documents_json = json.dumps({"documents": documents_list}, indent=2)
    message_content = f"Here are some documents provided for context, they may not all be relevant:\n{documents_json}"

    # Use pre-calculated token count from context_files
    return ChatMessageSimple(
        message=message_content,
        token_count=context_files.total_token_count,
        message_type=MessageType.USER,
    )


def select_reminder_text(
    *,
    ran_image_gen: bool,
    just_ran_web_search: bool,
    has_open_url_tool: bool,
    out_of_cycles: bool,
    persona_task_prompt: str | None,
    include_citation_reminder: bool,
    include_file_reminder: bool,
) -> str | None:
    """Choose the reminder appended after a tool cycle.

    The open_url nudge is gated on the tool actually being available; otherwise
    the model is told to call a tool it doesn't have and leaks confusing
    "open_url is not available" replies.
    """
    if ran_image_gen:
        return IMAGE_GEN_REMINDER
    if just_ran_web_search and has_open_url_tool and not out_of_cycles:
        return OPEN_URL_REMINDER
    return build_reminder_message(
        reminder_text=persona_task_prompt,
        include_citation_reminder=include_citation_reminder,
        include_file_reminder=include_file_reminder,
        is_last_cycle=out_of_cycles,
    )


def run_llm_loop(
    emitter: Emitter,
    state_container: ChatStateContainer,
    simple_chat_history: list[ChatMessageSimple],
    tools: list[Tool],
    custom_agent_prompt: str | None,
    context_files: ExtractedContextFiles,
    persona: Persona | None,
    user_memory_context: UserMemoryContext | None,
    llm: LLM,
    token_counter: Callable[[str], int],
    forced_tool_id: int | None = None,
    user_identity: LLMUserIdentity | None = None,
    chat_session_id: str | None = None,
    chat_files: list[ChatFile] | None = None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
    include_citations: bool = True,
    all_injected_file_metadata: dict[str, FileToolMetadata] | None = None,
    inject_memories_in_prompt: bool = True,
) -> None:
    """编排一轮聊天中的多步模型生成和工具执行，通过事件通道输出结果。

    普通聊天调用链路：
    handle_send_chat_message → handle_stream_message_objects → _stream_chat_turn
    → build_chat_turn（准备会话、模型、工具和历史）
    → _run_models → run_llm_loop（本方法）
       → 初始化引用处理器、token 预算和循环状态
       → 每步选择工具、构建提示词、裁剪消息历史
       → run_llm_step → run_llm_step_pkt_generator → llm.stream
          └─ 流式事件通过 emitter 输出，汇总结果返回本方法
       ├─ 模型直接回答：没有工具调用，退出循环
       └─ 模型请求工具：run_tool_calls → _safe_run_single_tool → tool.run
          ├─ internal_search：SearchTool.run → search_pipeline → search_chunks
          │  → _embed_and_hybrid_search → document_index.hybrid_retrieval
          │  └─ OpenSearch 后端执行混合检索，结果经筛选和扩展后返回
          └─ 其他工具：由对应工具实现处理
          → 收集工具结果、文档、文件和引用映射
          → 追加 ASSISTANT 工具调用消息和 TOOL_CALL_RESPONSE 结果消息
          → 下一步模型读取工具结果，继续回答或请求工具
       → 检查最终回答，发送 OverallStop 结束事件

    搜索不是必经步骤：AUTO 允许模型自行选择工具；搜索内部还可能走
    纯关键词或联邦检索分支。上述向量检索路径只适用于本地混合检索。
    普通循环最多执行 MAX_LLM_CYCLES 步；强制工具优先，只在首个适用步生效。
    非强制工具分支在最后一步或终止类工具执行后禁用工具，要求模型收尾。

    参数：
        emitter：输出推理、回答、工具和结束事件的通道。
        state_container：收集工具调用、文档和引用状态，供上层保存。
        simple_chat_history：可变消息历史，工具调用及响应会追加到此列表。
        tools / forced_tool_id：可用工具及可选的强制工具 ID。
        custom_agent_prompt / persona：Agent 提示词与行为配置。
        context_files / chat_files：上下文文件及工具可访问的附件。
        user_memory_context / inject_memories_in_prompt：用户信息及记忆注入配置。
        llm / token_counter：模型实例及 token 计数函数。
        user_identity / chat_session_id：模型调用身份与链路追踪会话标识。
        reasoning_effort / include_citations：推理强度与引用输出开关。
        all_injected_file_metadata：注入文件的元数据，传给消息历史构建逻辑。

    返回：
        None。回答通过 emitter 输出，相关状态写入 state_container。

    异常：
        工具不存在或响应缺少关联调用时抛出 ValueError；最终没有有效回答时
        抛出相应错误。工具有调用却无响应时，先写入失败消息供下一步恢复。
    """
    with trace(
        "run_llm_loop",
        group_id=chat_session_id,
        metadata=ChatTraceMetadata(
            chat_session_id=chat_session_id,
            user_id=user_identity.user_id if user_identity else None,
        ).model_dump(),
    ):
        # 延迟加载 LiteLLM；初始化逻辑每个进程只执行一次。
        from onyx.llm.litellm_singleton.config import initialize_litellm

        initialize_litellm()

        # 复制为可变列表，便于将搜索命中文档的附件加入后续 Python 工具可用文件。
        chat_files = list(chat_files or [])

        # 记录循环起点，用于计算回答开始前的等待时间。
        loop_start_time = time.monotonic()

        # 按配置处理引用：启用时生成链接，关闭时从输出中移除引用标记。
        citation_processor = DynamicCitationProcessor(
            citation_mode=(
                CitationMode.HYPERLINK if include_citations else CitationMode.REMOVE
            )
        )

        # 为直接提供给模型的项目文件建立引用映射。
        project_citation_mapping: CitationMapping = {}
        if context_files.file_metadata:
            project_citation_mapping = _build_context_file_citation_mapping(
                context_files.file_metadata
            )
            citation_processor.update_citation_mapping(project_citation_mapping)

        llm_step_result = LlmStepResult(
            reasoning=None,
            answer=None,
            tool_calls=None,
            raw_answer=None,
            finish_reason=None,
        )

        token_budget = resolve_chat_token_budget(llm)
        available_tokens = token_budget.input_tokens
        # 不支持图片输入的模型会将历史图片转换为文本标记，预算也按标记计算。
        image_files_replayed_as_markers = any(
            msg.message_type == MessageType.USER and msg.image_files
            for msg in simple_chat_history
        ) and not model_supports_image_input(
            llm.config.model_name, llm.config.model_provider, llm.config.deployment_name
        )
        tool_choice: ToolChoiceOptions = ToolChoiceOptions.AUTO
        # 将项目文件加入已收集文档，供引用处理和前端展示。
        gathered_documents: list[SearchDoc] | None = (
            list(project_citation_mapping.values())
            if project_citation_mapping
            else None
        )
        # TODO：支持项目图片引用；可考虑把图片作为带引用信息的独立消息。
        always_cite_documents: bool = bool(
            context_files.use_as_search_filter or context_files.file_texts
        )
        should_cite_documents: bool = False
        ran_image_gen: bool = False
        just_ran_web_search: bool = False
        has_open_url_tool: bool = any(isinstance(tool, OpenURLTool) for tool in tools)
        has_called_search_tool: bool = False
        code_interpreter_file_generated: bool = False
        fallback_extraction_attempted: bool = False
        citation_mapping: dict[int, str] = {}  # 引用编号 → 文档 ID/URL

        # 使用短生命周期会话读取提示词，避免流式生成期间一直占用数据库连接。
        with get_session_with_current_tenant() as prompt_db_session:
            default_base_system_prompt: str = get_default_base_system_prompt(
                prompt_db_session
            )
        system_prompt = None
        custom_agent_prompt_msg = None

        # 循环前替换提示词中的 {{user.<key>}}，确保各分支和 token 计算使用相同文本。
        # 不修改共享的 persona 对象。
        placeholder_values = (
            user_memory_context.user_info.placeholder_values
            if user_memory_context
            else {}
        )
        custom_agent_prompt = (
            substitute_user_placeholders(custom_agent_prompt, placeholder_values)
            if custom_agent_prompt
            else custom_agent_prompt
        )
        persona_system_prompt = (
            substitute_user_placeholders(persona.system_prompt, placeholder_values)
            if persona and persona.system_prompt
            else None
        )
        persona_task_prompt = (
            substitute_user_placeholders(persona.task_prompt, placeholder_values)
            if persona and persona.task_prompt
            else None
        )

        reasoning_cycles = 0
        for llm_cycle_count in range(MAX_LLM_CYCLES):
            # 根据循环次数、强制工具及前序工具结果，选择本步可用工具。
            out_of_cycles = llm_cycle_count == MAX_LLM_CYCLES - 1
            if forced_tool_id:
                # REQUIRED 只要求调用工具，不能直接指定名称，因此只保留目标工具。
                final_tools = [tool for tool in tools if tool.id == forced_tool_id]
                if not final_tools:
                    raise ValueError(f"Tool {forced_tool_id} not found in tools")
                tool_choice = ToolChoiceOptions.REQUIRED
                forced_tool_id = None
            elif out_of_cycles or ran_image_gen:
                # 达到最后一轮或已调用终止类工具时，禁用工具，要求模型生成最终回答。
                tool_choice = ToolChoiceOptions.NONE
                final_tools = []
            else:
                tool_choice = ToolChoiceOptions.AUTO
                final_tools = tools

            # 构建系统提示词和自定义 Agent 提示词，纳入时间、记忆及引用要求。
            persona_datetime_aware = persona.datetime_aware if persona else True
            cite_documents = should_cite_documents or always_cite_documents
            if persona and persona.replace_base_system_prompt:
                # 用户选择替换基础系统提示词时，仅使用 persona 提供的系统提示词。
                processed_system_prompt = (
                    process_prompt_template(
                        persona_system_prompt,
                        datetime_aware=persona_datetime_aware,
                        append_datetime_if_aware=True,
                        should_cite_documents=cite_documents,
                    )
                    if persona_system_prompt
                    else None
                )
                system_prompt = (
                    ChatMessageSimple(
                        message=processed_system_prompt,
                        token_count=token_counter(processed_system_prompt),
                        message_type=MessageType.SYSTEM,
                    )
                    if processed_system_prompt
                    else None
                )
                custom_agent_prompt_msg = None
            else:
                # 基础提示词为空时不创建空的 SYSTEM 消息。
                if default_base_system_prompt:
                    prompt_memory_context = (
                        user_memory_context
                        if inject_memories_in_prompt
                        else (
                            user_memory_context.without_memories()
                            if user_memory_context
                            else None
                        )
                    )
                    system_prompt_str = build_system_prompt(
                        base_system_prompt=default_base_system_prompt,
                        datetime_aware=persona_datetime_aware,
                        user_memory_context=prompt_memory_context,
                        tools=tools,
                        should_cite_documents=cite_documents,
                    )
                    system_prompt = ChatMessageSimple(
                        message=system_prompt_str,
                        token_count=token_counter(system_prompt_str),
                        message_type=MessageType.SYSTEM,
                    )
                    processed_custom_agent_prompt = (
                        process_prompt_template(
                            custom_agent_prompt,
                            datetime_aware=persona_datetime_aware,
                            append_datetime_if_aware=False,
                            should_cite_documents=cite_documents,
                        )
                        if custom_agent_prompt
                        else None
                    )
                    custom_agent_prompt_msg = (
                        ChatMessageSimple(
                            message=processed_custom_agent_prompt,
                            token_count=token_counter(processed_custom_agent_prompt),
                            message_type=MessageType.USER,
                        )
                        if processed_custom_agent_prompt
                        else None
                    )
                else:
                    # 没有基础系统提示词时，将自定义 Agent 提示词用作系统提示词。
                    processed_custom_agent_prompt = (
                        process_prompt_template(
                            custom_agent_prompt,
                            datetime_aware=persona_datetime_aware,
                            append_datetime_if_aware=True,
                            should_cite_documents=cite_documents,
                        )
                        if custom_agent_prompt
                        else None
                    )
                    system_prompt = (
                        ChatMessageSimple(
                            message=processed_custom_agent_prompt,
                            token_count=token_counter(processed_custom_agent_prompt),
                            message_type=MessageType.SYSTEM,
                        )
                        if processed_custom_agent_prompt
                        else None
                    )
                    custom_agent_prompt_msg = None

            processed_task_prompt = (
                process_prompt_template(
                    persona_task_prompt,
                    datetime_aware=persona_datetime_aware,
                    append_datetime_if_aware=False,
                    should_cite_documents=cite_documents,
                )
                if persona_task_prompt
                else None
            )
            reminder_message_text = select_reminder_text(
                ran_image_gen=ran_image_gen,
                just_ran_web_search=just_ran_web_search,
                has_open_url_tool=has_open_url_tool,
                out_of_cycles=out_of_cycles,
                persona_task_prompt=processed_task_prompt,
                include_citation_reminder=should_cite_documents
                or always_cite_documents,
                include_file_reminder=code_interpreter_file_generated,
            )

            reminder_msg = (
                ChatMessageSimple(
                    message=reminder_message_text,
                    token_count=token_counter(reminder_message_text),
                    message_type=MessageType.USER_REMINDER,
                )
                if reminder_message_text
                else None
            )

            # 为工具定义预留 token，再按剩余额度构建模型消息历史。
            tool_token_budget = compute_all_tool_tokens(final_tools, token_counter)
            truncated_message_history = construct_message_history(
                system_prompt=system_prompt,
                custom_agent_prompt=custom_agent_prompt_msg,
                simple_chat_history=simple_chat_history,
                reminder_message=reminder_msg,
                context_files=context_files,
                available_tokens=max(0, available_tokens - tool_token_budget),
                token_counter=token_counter,
                all_injected_file_metadata=all_injected_file_metadata,
                image_files_replayed_as_markers=image_files_replayed_as_markers,
                available_tool_names={tool.name for tool in final_tools},
            )

            max_output_tokens = token_budget.output_allowance(
                estimated_input_tokens=tool_token_budget
                + sum(
                    count_message_replay_tokens(
                        msg,
                        image_files_replayed_as_markers=image_files_replayed_as_markers,
                        token_counter=token_counter,
                    )
                    for msg in truncated_message_history
                ),
            )

            # 把工具转换为模型可识别的定义；实际调用由后面的 run_llm_step 发起。
            tool_defs = [tool.tool_definition() for tool in final_tools]

            # 计算从循环开始到本步调用前的累计耗时，传给回答事件。
            pre_answer_processing_time = time.monotonic() - loop_start_time

            llm_step_result, has_reasoned = run_llm_step(
                emitter=emitter,
                history=truncated_message_history,
                tool_definitions=tool_defs,
                tool_choice=tool_choice,
                llm=llm,
                placement=Placement(turn_index=llm_cycle_count + reasoning_cycles),
                citation_processor=citation_processor,
                state_container=state_container,
                # 传入已收集的完整文档，使回答流能够同时提供引用文档信息。
                final_documents=gathered_documents,
                user_identity=user_identity,
                pre_answer_processing_time=pre_answer_processing_time,
                reasoning_effort=reasoning_effort,
                max_tokens=max_output_tokens,
            )
            if has_reasoned:
                reasoning_cycles += 1

            # 兼容未正确使用原生工具调用的模型，尝试从其他输出通道提取工具指令。
            llm_step_result, attempted = _try_fallback_tool_extraction(
                llm_step_result=llm_step_result,
                tool_choice=tool_choice,
                fallback_extraction_attempted=fallback_extraction_attempted,
                tool_defs=tool_defs,
                turn_index=llm_cycle_count + reasoning_cycles,
            )
            if attempted:
                # 整个循环只允许一次兜底提取，避免反复尝试。
                fallback_extraction_attempted = True

            # 每步模型调用后保存引用映射，支持增量状态更新。
            state_container.set_citation_mapping(citation_processor.citation_to_doc)

            # 获取本步模型选择的工具调用；没有调用时使用空列表。
            tool_responses: list[ToolResponse] = []
            tool_calls = llm_step_result.tool_calls or []

            if INTEGRATION_TESTS_MODE and tool_calls:
                for tool_call in tool_calls:
                    emitter.emit(
                        Packet(
                            placement=tool_call.placement,
                            obj=ToolCallDebug(
                                tool_call_id=tool_call.tool_call_id,
                                tool_name=tool_call.tool_name,
                                tool_args=tool_call.tool_args,
                            ),
                        )
                    )

            if len(tool_calls) > 1:
                emitter.emit(
                    Packet(
                        placement=Placement(
                            turn_index=tool_calls[0].placement.turn_index
                        ),
                        obj=TopLevelBranching(num_parallel_branches=len(tool_calls)),
                    )
                )

            # 工具返回“引用编号 → 文档 ID/URL”的轻量映射；本层再处理 SearchDoc。
            # 引用处理器负责输出引用，两种映射承担不同职责。
            just_ran_web_search = False
            parallel_tool_call_results = run_tool_calls(
                tool_calls=tool_calls,
                tools=final_tools,
                message_history=truncated_message_history,
                user_memory_context=user_memory_context,
                user_info=None,  # TODO：用户信息目前包含在记忆上下文中，后续可独立传递
                citation_mapping=citation_mapping,
                next_citation_num=citation_processor.get_next_citation_number(),
                max_concurrent_tools=None,
                skip_search_query_expansion=has_called_search_tool,
                chat_files=chat_files,
                url_snippet_map=extract_url_snippet_map(gathered_documents or []),
                inject_memories_in_prompt=inject_memories_in_prompt,
            )
            tool_responses = parallel_tool_call_results.tool_responses
            citation_mapping = parallel_tool_call_results.updated_citation_mapping

            # 有工具调用却没有工具响应时，补入失败消息，让下一轮模型尝试恢复。
            if tool_calls and not tool_responses:
                failure_messages = create_tool_call_failure_messages(
                    tool_calls, token_counter
                )
                simple_chat_history.extend(failure_messages)
                continue

            for tool_response in tool_responses:
                # 工具调度器必须为响应关联原始工具调用，供结果配对和持久化。
                if tool_response.tool_call is None:
                    raise ValueError("Tool response missing tool_call reference")

                tool_call = tool_response.tool_call
                tab_index = tool_call.placement.tab_index

                # 记录搜索工具已执行，后续搜索跳过重复的查询扩展。
                if tool_call.tool_name == SearchTool.NAME:
                    has_called_search_tool = True

                # 记录代码解释器是否生成文件，以便下一轮提示模型提供下载链接。
                if (
                    tool_call.tool_name == PythonTool.NAME
                    and not code_interpreter_file_generated
                ):
                    try:
                        parsed = json.loads(tool_response.llm_facing_response)
                        if parsed.get("generated_files"):
                            code_interpreter_file_generated = True
                    except (json.JSONDecodeError, AttributeError):
                        pass

                tools_by_name = {tool.name: tool for tool in final_tools}

                # 查找工具实例以取得 tool_id；并行执行的结果在后面按线性消息序列写入历史。
                tool = tools_by_name.get(tool_call.tool_name)
                if not tool:
                    raise ValueError(
                        f"Tool '{tool_call.tool_name}' not found in tools list"
                    )

                # 提取搜索类工具返回的文档和展示文档。
                search_docs = None
                displayed_docs = None
                if isinstance(tool_response.rich_response, SearchDocsResponse):
                    search_docs = tool_response.rich_response.search_docs
                    displayed_docs = tool_response.rich_response.displayed_docs

                    # 将全部搜索文档加入状态容器，供后续持久化。
                    if search_docs:
                        state_container.add_search_docs(search_docs)

                    if gathered_documents:
                        gathered_documents.extend(search_docs)
                    else:
                        gathered_documents = search_docs

                    # 仅在网页搜索命中文档时，启用下一轮的打开 URL 提示。
                    if search_docs and tool_call.tool_name == WebSearchTool.NAME:
                        just_ran_web_search = True

                    # 将命中文档对应的原始附件加入 chat_files，供后续 Python 工具使用。
                    if search_docs:
                        staged = build_python_chat_files_from_search_docs(
                            search_docs=search_docs,
                        )
                        if staged:
                            existing_filenames = {cf.filename for cf in chat_files}
                            chat_files.extend(
                                cf
                                for cf in staged
                                if cf.filename not in existing_filenames
                            )

                # 提取图片生成工具的输出。
                generated_images = None
                if isinstance(
                    tool_response.rich_response, FinalImageGenerationResponse
                ):
                    generated_images = tool_response.rich_response.generated_images

                # 提取代码解释器生成的文件。
                generated_files = None
                if isinstance(tool_response.rich_response, PythonToolRichResponse):
                    generated_files = (
                        tool_response.rich_response.generated_files or None
                    )

                # 自定义工具保存图片或 CSV 等文件，并返回文件 ID。
                generated_file_ids = None
                if isinstance(
                    tool_response.rich_response, CustomToolCallSummary
                ) and isinstance(
                    tool_response.rich_response.tool_result, CustomToolUserFileSnapshot
                ):
                    generated_file_ids = (
                        tool_response.rich_response.tool_result.file_ids or None
                    )

                # 处理记忆工具响应，按当前会话模式决定是否保存。
                memory_snapshot: MemoryToolResponseSnapshot | None = None
                incognito_memory_refusal: str | None = None
                if isinstance(tool_response.rich_response, MemoryToolResponse):
                    # 无痕模式明确拒绝记忆写入，避免向模型或用户声称已保存。
                    if get_current_incognito_record_mode() is not None:
                        incognito_memory_refusal = (
                            "Error: memories cannot be saved from an incognito "
                            "chat. Tell the user their request was not saved."
                        )
                    else:
                        persisted_memory_id: int | None = None
                        if user_memory_context and user_memory_context.user_id:
                            if tool_response.rich_response.index_to_replace is not None:
                                persisted_memory_id = update_memory_at_index(
                                    user_id=user_memory_context.user_id,
                                    index=tool_response.rich_response.index_to_replace,
                                    new_text=tool_response.rich_response.memory_text,
                                )
                            else:
                                persisted_memory_id = add_memory(
                                    user_id=user_memory_context.user_id,
                                    memory_text=tool_response.rich_response.memory_text,
                                )
                        operation: Literal["add", "update"] = (
                            "update"
                            if tool_response.rich_response.index_to_replace is not None
                            else "add"
                        )
                        memory_snapshot = MemoryToolResponseSnapshot(
                            memory_text=tool_response.rich_response.memory_text,
                            operation=operation,
                            memory_id=persisted_memory_id,
                            index=tool_response.rich_response.index_to_replace,
                        )

                if incognito_memory_refusal:
                    saved_response = incognito_memory_refusal
                    # 把拒绝信息也返回给下一轮模型。
                    tool_response.llm_facing_response = incognito_memory_refusal
                elif memory_snapshot:
                    saved_response = json.dumps(memory_snapshot.model_dump())
                elif isinstance(tool_response.rich_response, CustomToolCallSummary):
                    saved_response = json.dumps(
                        tool_response.rich_response.model_dump()
                    )
                elif isinstance(tool_response.rich_response, str):
                    saved_response = tool_response.rich_response
                else:
                    saved_response = tool_response.llm_facing_response

                tool_call_info = ToolCallInfo(
                    parent_tool_call_id=None,  # 顶层工具调用关联到聊天消息
                    turn_index=llm_cycle_count + reasoning_cycles,
                    tab_index=tab_index,
                    tool_name=tool_call.tool_name,
                    tool_call_id=tool_call.tool_call_id,
                    tool_id=tool.id,
                    reasoning_tokens=llm_step_result.reasoning,  # 本步工具调用共享同一份推理文本
                    tool_call_arguments=tool_call.tool_args,
                    tool_call_response=saved_response,
                    search_docs=displayed_docs or search_docs,
                    generated_images=generated_images,
                    generated_files=generated_files,
                    generated_file_ids=generated_file_ids,
                )
                # 将工具调用结果加入状态容器，支持部分结果保存。
                state_container.add_tool_call(tool_call_info)

                # 根据工具响应更新引用处理器。
                update_citation_processor_from_tool_response(
                    tool_response, citation_processor
                )

            # 按并行工具调用协议写入历史：一条 ASSISTANT 消息包含所有工具调用，
            # 随后每个调用对应一条 TOOL_CALL_RESPONSE 消息。
            if tool_responses:
                # 仅保留已关联原始工具调用的响应。
                valid_tool_responses = [
                    tr for tr in tool_responses if tr.tool_call is not None
                ]

                # 构建本步所有工具调用的简化表示，并计算 token 数。
                tool_calls_simple: list[ToolCallSimple] = []
                for tool_response in valid_tool_responses:
                    tc = tool_response.tool_call
                    assert (
                        tc is not None
                    )  # 上面已过滤，此断言用于类型收窄

                    tool_call_message = tc.to_msg_str()
                    tool_call_token_count = token_counter(tool_call_message)

                    tool_calls_simple.append(
                        ToolCallSimple(
                            tool_call_id=tc.tool_call_id,
                            tool_name=tc.tool_name,
                            tool_arguments=tc.tool_args,
                            token_count=tool_call_token_count,
                        )
                    )

                # 用一条 ASSISTANT 消息记录本步全部工具调用。
                total_tool_call_tokens = sum(tc.token_count for tc in tool_calls_simple)
                assistant_with_tools = ChatMessageSimple(
                    message="",  # 该消息仅记录工具调用
                    token_count=total_tool_call_tokens,
                    message_type=MessageType.ASSISTANT,
                    tool_calls=tool_calls_simple,
                    image_files=None,
                )
                simple_chat_history.append(assistant_with_tools)

                # 逐条追加工具响应，并通过 tool_call_id 与调用关联。
                for tool_response in valid_tool_responses:
                    tc = tool_response.tool_call
                    assert tc is not None  # 上面已过滤空引用

                    tool_response_message = tool_response.llm_facing_response
                    tool_response_token_count = token_counter(tool_response_message)

                    tool_response_msg = ChatMessageSimple(
                        message=tool_response_message,
                        token_count=tool_response_token_count,
                        message_type=MessageType.TOOL_CALL_RESPONSE,
                        tool_call_id=tc.tool_call_id,
                        image_files=None,
                    )
                    simple_chat_history.append(tool_response_msg)

            # 没有后续工具调用时退出循环；最终回答是否有效由循环后的检查确认。
            if not llm_step_result.tool_calls or len(llm_step_result.tool_calls) == 0:
                break

            # 部分工具要求下一轮直接收尾，不允许继续调用工具。
            if any(
                tool.tool_name in STOPPING_TOOLS_NAMES
                for tool in llm_step_result.tool_calls
            ):
                ran_image_gen = True

            if llm_step_result.tool_calls and any(
                tool.tool_name in CITEABLE_TOOLS_NAMES
                for tool in llm_step_result.tool_calls
            ):
                # 调用过可引用文档的工具后，后续提示词加入引用要求。
                should_cite_documents = True

        # 循环退出不代表生成成功；分别检查空输出和只有工具调用的情况。
        if not llm_step_result.answer and not llm_step_result.tool_calls:
            raise _build_empty_llm_response_error(
                llm=llm,
                llm_step_result=llm_step_result,
                tool_choice=tool_choice,
            )

        if not llm_step_result.answer:
            raise RuntimeError(
                "The LLM did not return a final answer after tool execution. "
                "Typically this indicates invalid tool-call output, a model/provider mismatch, "
                "or serving API misconfiguration."
            )

        emitter.emit(
            Packet(
                placement=Placement(
                    turn_index=llm_cycle_count  # ty: ignore[possibly-unresolved-reference]
                    + reasoning_cycles
                ),
                obj=OverallStop(type="stop"),
            )
        )

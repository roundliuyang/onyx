from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, model_validator

from onyx.configs.constants import DocumentSource, MessageType, SessionType
from onyx.context.search.models import BaseFilters, SavedSearchDoc, SearchDoc, Tag
from onyx.db.enums import ChatSessionSharedStatus
from onyx.db.models import ChatSession
from onyx.file_store.models import FileDescriptor
from onyx.llm.override_models import LLMOverride
from onyx.server.query_and_chat.streaming_models import Packet

AUTO_PLACE_AFTER_LATEST_MESSAGE = -1


class MessageOrigin(str, Enum):
    """Origin of a chat message for telemetry tracking."""

    WEBAPP = "webapp"
    CHROME_EXTENSION = "chrome_extension"
    API = "api"
    SLACKBOT = "slackbot"
    WIDGET = "widget"
    DISCORDBOT = "discordbot"
    MOBILE = "mobile"
    UNKNOWN = "unknown"
    UNSET = "unset"


class MessageResponseIDInfo(BaseModel):
    user_message_id: int | None
    reserved_assistant_message_id: int


class ModelResponseSlot(BaseModel):
    """Pairs a reserved assistant message ID with its model display name."""

    message_id: int
    model_name: str


class MultiModelMessageResponseIDInfo(BaseModel):
    """Sent at the start of a multi-model streaming response.
    Contains the user message ID and one slot per model being run in parallel."""

    user_message_id: int | None
    responses: list[ModelResponseSlot]


class SourceTag(Tag):
    source: DocumentSource


class TagResponse(BaseModel):
    tags: list[SourceTag]


class UpdateChatSessionThreadRequest(BaseModel):
    # If not specified, use Onyx default persona
    chat_session_id: UUID
    new_alternate_model: str


class UpdateChatSessionTemperatureRequest(BaseModel):
    chat_session_id: UUID
    temperature_override: float


class UpdateChatSessionReasoningRequest(BaseModel):
    chat_session_id: UUID
    reasoning_effort_override: str | None = None


class ChatSessionCreationRequest(BaseModel):
    # If not specified, use Onyx default persona
    persona_id: int = 0
    description: str | None = None
    project_id: int | None = None
    # Start the session incognito. Refused with an error when incognito is
    # unavailable, never silently downgraded to an ordinary chat.
    incognito: bool = False
    # The id the client already used when uploading, so this session owns those
    # files. Ignored unless incognito.
    incognito_session_id: UUID | None = None


class ChatFeedbackRequest(BaseModel):
    chat_message_id: int
    is_positive: bool | None = None
    feedback_text: str | None = None
    predefined_feedback: str | None = None

    @model_validator(mode="after")
    def check_is_positive_or_feedback_text(self) -> "ChatFeedbackRequest":
        if self.is_positive is None and self.feedback_text is None:
            raise ValueError("Empty feedback received.")
        return self


# 注意：此模型用于 Onyx 的核心流程，修改需由经验丰富的团队成员审核批准。
# 应避免字段和逻辑膨胀，并保持跨版本的向后兼容性。
class SendMessageRequest(BaseModel):
    # 用户发送的消息正文。
    message: str

    # 覆盖本次请求使用的单模型配置。
    llm_override: LLMOverride | None = None
    # 多模型配置：最多并行调用 3 个模型。
    # 提供超过 1 项配置时，进入多模型流式处理流程。
    llm_overrides: list[LLMOverride] | None = None
    # 仅用于测试，为 LiteLLM 指定固定的模拟响应。
    mock_llm_response: str | None = None

    # 限定本次请求可用的工具 ID。
    allowed_tool_ids: list[int] | None = None
    # 指定本次请求要强制调用的工具，具体执行仍受工具可用性检查约束。
    forced_tool_id: int | None = None

    # 本次消息附带的文件描述信息，不直接存放文件内容。
    file_descriptors: list[FileDescriptor] = []

    # 内部文档检索使用的筛选条件。
    internal_search_filters: BaseFilters | None = None

    # 是否启用深度研究模式。
    deep_research: bool = False

    # 透传给 MCP 工具调用的请求头，例如用户 JWT 令牌和用户 ID。
    # 示例：{"Authorization": "Bearer <user_jwt>", "X-User-ID": "user123"}
    mcp_headers: dict[str, str] | None = None

    # 消息来源，用于遥测统计。
    origin: MessageOrigin = MessageOrigin.UNSET

    # 指定消息在对话树中的位置：
    # - -1：自动接在当前消息链的最新消息之后。
    # - null：从根节点重新生成，即从第一条消息开始。
    # - 正整数：接在指定的父消息之后。
    # 重新生成是目前唯一会在用户消息处产生分支的场景。
    # 如果父消息是用户消息，则忽略本次请求的 message，使用原用户消息重新生成。
    parent_message_id: int | None = AUTO_PLACE_AFTER_LATEST_MESSAGE
    # 继续现有会话时提供其 ID，与 chat_session_info 互斥。
    chat_session_id: UUID | None = None
    # 创建新会话时使用的配置；两个会话字段均未提供时，使用默认配置。
    chat_session_info: ChatSessionCreationRequest | None = None

    # True（默认）：返回 StreamingResponse 流式响应。
    # False：返回包含完整结果的 ChatFullResponse。
    stream: bool = True

    # False 表示禁用引用生成：
    # - 从响应正文中移除 [1]、[2] 等引用标记。
    # - 流式响应中不再发送 CitationInfo 数据包。
    include_citations: bool = True

    # 注入模型调用的额外上下文，不写入数据库，也不显示在聊天历史中。
    # 例如：Chrome 扩展启用“读取此标签页”时，可用此字段传入当前标签页的 URL。
    additional_context: str | None = None

    # 在字段校验完成后，检查会话 ID 与新建会话配置的组合。
    @model_validator(mode="after")
    def check_chat_session_id_or_info(self) -> "SendMessageRequest":
        # 两者均未提供时，返回带有默认新建会话配置的模型副本。
        if self.chat_session_id is None and self.chat_session_info is None:
            return self.model_copy(
                update={"chat_session_info": ChatSessionCreationRequest()}
            )
        # 不能同时指定现有会话和新建会话配置。
        if self.chat_session_id is not None and self.chat_session_info is not None:
            raise ValueError(
                "Only one of chat_session_id or chat_session_info should be provided, not both."
            )
        return self


class ChatMessageIdentifier(BaseModel):
    message_id: int


class ChatRenameRequest(BaseModel):
    chat_session_id: UUID
    name: str | None = None


class ChatSessionUpdateRequest(BaseModel):
    sharing_status: ChatSessionSharedStatus


class DeleteAllSessionsRequest(BaseModel):
    session_type: SessionType


class RenameChatSessionResponse(BaseModel):
    new_name: str  # This is only really useful if the name is generated


class ChatSessionDetails(BaseModel):
    id: UUID
    name: str | None
    persona_id: int | None = None
    time_created: str
    time_updated: str
    shared_status: ChatSessionSharedStatus
    current_alternate_model: str | None = None
    current_temperature_override: float | None = None
    current_reasoning_effort_override: str | None = None

    @classmethod
    def from_model(cls, model: ChatSession) -> "ChatSessionDetails":
        return cls(
            id=model.id,
            name=model.description,
            persona_id=model.persona_id,
            time_created=model.time_created.isoformat(),
            time_updated=model.time_updated.isoformat(),
            shared_status=model.shared_status,
            current_alternate_model=model.current_alternate_model,
            current_temperature_override=model.temperature_override,
            current_reasoning_effort_override=model.reasoning_effort_override,
        )


class ChatSessionsResponse(BaseModel):
    sessions: list[ChatSessionDetails]
    has_more: bool = False


class ChatMessageDetail(BaseModel):
    chat_session_id: UUID | None = None
    message_id: int
    parent_message: int | None = None
    latest_child_message: int | None = None
    message: str
    reasoning_tokens: str | None = None
    message_type: MessageType
    context_docs: list[SavedSearchDoc] | None = None
    # Dict mapping citation number to document_id
    citations: dict[int, str] | None = None
    time_sent: datetime
    files: list[FileDescriptor]
    error: str | None = None
    current_feedback: str | None = None  # "like" | "dislike" | null
    processing_duration_seconds: float | None = None
    preferred_response_id: int | None = None
    model_display_name: str | None = None
    # Absent on messages written before this was captured.
    request_params: dict[str, Any] | None = None

    def model_dump(  # ty: ignore[invalid-method-override]
        self, *args: list, **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        initial_dict = super().model_dump(
            *args,
            mode="json",
            **kwargs,  # ty: ignore[invalid-argument-type]
        )
        initial_dict["time_sent"] = self.time_sent.isoformat()
        return initial_dict


class SetPreferredResponseRequest(BaseModel):
    user_message_id: int
    preferred_response_id: int


class CurrentRunInfo(BaseModel):
    """In-flight run whose stream buffer can be replayed/tailed."""

    run_id: int


class ChatSessionDetailResponse(BaseModel):
    chat_session_id: UUID
    description: str | None
    persona_id: int | None = None
    persona_name: str | None
    personal_icon_name: str | None
    messages: list[ChatMessageDetail]
    time_created: datetime
    shared_status: ChatSessionSharedStatus
    current_alternate_model: str | None
    current_temperature_override: float | None
    current_reasoning_effort_override: str | None
    deleted: bool = False
    owner_name: str | None = None
    packets: list[list[Packet]]
    # Set while a run is in flight and resumable: cursor-0 replay+tail is
    # available at /chat-session/{id}/resume-stream.
    current_run: CurrentRunInfo | None = None
    # True for sessions pinned to an incognito record mode, so a reload can
    # restore the incognito UI state.
    incognito: bool = False


class AdminSearchRequest(BaseModel):
    query: str
    filters: BaseFilters


class AdminSearchResponse(BaseModel):
    documents: list[SearchDoc]


class ChatSessionSummary(BaseModel):
    id: UUID
    name: str | None = None
    persona_id: int | None = None
    time_created: datetime
    shared_status: ChatSessionSharedStatus
    current_alternate_model: str | None = None
    current_temperature_override: float | None = None
    current_reasoning_effort_override: str | None = None


class ChatSessionGroup(BaseModel):
    title: str
    chats: list[ChatSessionSummary]


class ChatSearchResponse(BaseModel):
    groups: list[ChatSessionGroup]
    has_more: bool
    next_page: int | None = None

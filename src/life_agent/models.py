"""Core domain models for Life Agent (v2: Two-Phase Forward/Backward Architecture)."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now().astimezone()


def new_id(prefix: str = "") -> str:
    short = uuid.uuid4().hex[:12]
    return f"{prefix}_{short}" if prefix else short


class ChannelType(str, Enum):
    PRIVATE = "private"
    GROUP = "group"
    NEWS = "news"
    PAPER = "paper"
    SYSTEM = "system"
    WEBHOOK = "webhook"


class TrustTier(str, Enum):
    """Three-level information source trust firewall (prevents domain/source crosstalk)."""

    HIGH = "high"  # Owner chat, schedule import, trusted senders/channels -> may write Timetable
    MEDIUM = "medium"  # Ordinary groups / external DMs -> suspected schedule changes need confirmation
    LOW = "low"  # RSS / arXiv / public feeds -> HARD-BLOCKED from mutating Timetable or body state


class StateTier(str, Enum):
    """Four-tier Scene State TTL lifecycle (T0–T3)."""

    T0_EPHEMERAL = "t0_ephemeral"  # Event-bound or 2–3h (current activity, temporary ambient note)
    T1_DAILY = "t1_daily"  # Auto-resets daily at 04:00 (awake, meals today, campus run today)
    T2_MULTIDAY = "t2_multiday"  # Default 3–7d (illness/injury, exam week); gentle renewal check before clear
    T3_PERSISTENT = "t3_persistent"  # Durable living profile & stable baseline preferences


class ActivityMode(str, Enum):
    SLEEPING = "sleeping"
    IN_CLASS = "in_class"
    IN_CLASS_CHILL = "in_class_chill"
    BUSY_EVENT = "busy_event"
    BREAK = "break"
    FREE_SHORT = "free_short"
    FREE_LONG = "free_long"
    DND = "dnd"


class Interruptibility(str, Enum):
    URGENT_ONLY = "urgent_only"
    LOW_FRICTION = "low_friction"
    OPEN = "open"


class BufferZone(str, Enum):
    """Box 1 in architecture sketch: message / URTI / news."""

    MESSAGE = "message"
    URTI = "urti"  # User Real-Time Info (实时信息区)
    NEWS = "news"


class BufferStatus(str, Enum):
    ACTIVE = "active"  # Waiting in buffer / eligible for window or re-evaluation
    PUSHED = "pushed"  # Already promoted to Active Push
    MERGED_SCENE = "merged_scene"  # Merged into Scene State
    DISMISSED = "dismissed"  # Manually deleted, archived, or moved to Slow Loop


class TimetableStatus(str, Enum):
    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class GatewayAction(str, Enum):
    ACTIVE_PUSH = "ACTIVE_PUSH"
    SCENE_STATE = "SCENE_STATE"
    DEFER_TO_WINDOW = "DEFER_TO_WINDOW"
    ARCHIVE_OR_SLOW = "ARCHIVE_OR_SLOW"
    # Legacy alias kept for full backward compatibility
    RETURN_TO_BUFFER = "RETURN_TO_BUFFER"


class PushStatus(str, Enum):
    WAITING_QQ_REPLY = "waiting_qq_reply"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED_WECHAT = "escalated_wechat"
    FAILED = "failed"


class SlowMemoryKind(str, Enum):
    """Structured categories stored in the BM25 Slow Loop."""

    IDEA = "idea"  # Undated ideas / free-time activities / wishlist
    FUTURE_EVENT = "future_event"  # Scheduled events > 7 days away (classes, shifts, exams)
    FUTURE_TASK = "future_task"  # Tasks / deadlines > 7 days away
    PLAN = "plan"  # Multi-week or long-term plans / goals
    NOTE = "note"  # Miscellaneous long-term structured notes


class SlowMemoryStatus(str, Enum):
    ACTIVE = "active"
    PROMOTED = "promoted"  # Entered the 7-day fast loop window and moved into Timetable
    COMPLETED = "completed"
    ARCHIVED = "archived"


class GatingRuleStatus(str, Enum):
    ACTIVE = "active"
    CANDIDATE = "candidate"  # System-inferred rule awaiting user confirmation
    DISABLED = "disabled"


class FollowUpStatus(str, Enum):
    OPEN = "open"
    DONE = "done"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class RawIncomingEvent(BaseModel):
    """Normalized event emitted by any Information Router plugin."""

    id: str = Field(default_factory=lambda: new_id("evt"))
    source_plugin: str
    platform_type: str
    platform_id: str
    channel_type: ChannelType
    channel_id: str
    channel_name: str = ""
    sender_id: str = ""
    sender_name: str = ""
    text: str
    timestamp: datetime = Field(default_factory=utc_now)
    is_from_owner: bool = False
    raw_meta: dict[str, Any] = Field(default_factory=dict)


class GroupChatBatch(BaseModel):
    """Cleaned batch of group chat messages flushed after silence window."""

    id: str = Field(default_factory=lambda: new_id("grp"))
    source_plugin: str
    platform_type: str
    platform_id: str
    group_id: str
    group_name: str = ""
    messages: list[RawIncomingEvent] = Field(default_factory=list)
    filtered_count: int = 0
    jielong_summaries: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utc_now)
    flushed_at: datetime = Field(default_factory=utc_now)

    def render_transcript(self) -> str:
        lines: list[str] = []
        if self.jielong_summaries:
            lines.append("[已折叠/过滤的接龙摘要]")
            for s in self.jielong_summaries:
                lines.append(f"- {s}")
        for m in self.messages:
            ts = m.timestamp.strftime("%H:%M")
            who = m.sender_name or m.sender_id or "群成员"
            lines.append(f"[{ts}] {who}: {m.text}")
        if self.filtered_count > 0:
            lines.append(f"(另有 {self.filtered_count} 条重复/刷屏/接龙中间消息已过滤)")
        return "\n".join(lines)


class TimetableEntry(BaseModel):
    """Global single-source-of-truth Timetable entry (events within the active 7-day horizon)."""

    id: str = Field(default_factory=lambda: new_id("tt"))
    title: str
    start_time: str
    end_time: str | None = None
    location: str = ""
    category: str = "other"  # class / shift / deadline / meeting / life / exercise / other
    is_mandatory: bool = True
    status: TimetableStatus = TimetableStatus.SCHEDULED
    notes: str = ""
    source: str = "manual"
    reminded: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ScheduledPassiveTrigger(BaseModel):
    """Explicit dated passive trigger written for fast-loop timetable events."""

    id: str = Field(default_factory=lambda: new_id("strg"))
    trigger_name: str
    trigger_at: str
    directive: str
    category: str = "schedule_reminder"
    priority: float = 9.0
    related_timetable_id: str | None = None
    fired: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class SlowMemoryItem(BaseModel):
    """Structured item stored in the Slow Loop and indexed via BM25 (ideas & >7-day items)."""

    id: str = Field(default_factory=lambda: new_id("slow"))
    kind: SlowMemoryKind = SlowMemoryKind.IDEA
    title: str
    summary: str
    details: str = ""
    tags: list[str] = Field(default_factory=list)
    planned_start: str | None = None
    planned_end: str | None = None
    location: str = ""
    category: str = "other"
    estimated_minutes: int | None = None
    energy_level: str = ""
    is_mandatory: bool = False
    reminder_lead_minutes: int = 20
    source: str = "manual"
    status: SlowMemoryStatus = SlowMemoryStatus.ACTIVE
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ScheduleImportResult(BaseModel):
    """Structured outcome report returned after importing a timetable or shift Excel/Image."""

    source_filename: str
    input_kind: str
    schedule_type: str
    total_extracted_events: int = 0
    fast_loop_timetable_count: int = 0
    passive_triggers_written: int = 0
    slow_loop_bm25_count: int = 0
    skipped_past_count: int = 0
    fast_loop_entries: list[TimetableEntry] = Field(default_factory=list)
    passive_triggers: list[ScheduledPassiveTrigger] = Field(default_factory=list)
    slow_loop_entries: list[SlowMemoryItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BufferItem(BaseModel):
    """Structured content stored in Box 1 (message / URTI / news) with dual-axis scoring."""

    id: str = Field(default_factory=lambda: new_id("buf"))
    zone: BufferZone
    source: str
    title: str
    summary: str
    details: str = ""
    tags: list[str] = Field(default_factory=list)
    weight: float = 4.0  # Effective display/priority score (synced with intrinsic_value + starvation_boost)
    intrinsic_value: float = 4.0  # Inherent value/interest (1.0..10.0)
    urgency: float = 4.0  # Time sensitivity (1.0..10.0); never inflated blindly by starvation
    starvation_boost: float = 0.0  # Wait-time compensation, strictly capped
    trust_tier: TrustTier = TrustTier.MEDIUM
    needs_confirmation: bool = False
    return_count: int = 0
    has_explicit_time: bool = False
    is_mandatory_task: bool = False
    status: BufferStatus = BufferStatus.ACTIVE
    next_review_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class TieredSceneEntry(BaseModel):
    """Scoped Scene State entry obeying T0–T3 TTL rules."""

    key: str
    tier: StateTier = StateTier.T0_EPHEMERAL
    value: Any
    source: str = "system"
    expires_at: datetime | None = None
    renewal_prompted: bool = False
    updated_at: datetime = Field(default_factory=utc_now)


class GatingRule(BaseModel):
    """Atomic, scoped, auditable rule composing the dynamic `grade` classifier."""

    id: str = Field(default_factory=lambda: new_id("rule"))
    scope_type: str = "topic"  # source / topic / course_or_time / routine / temporary_override
    scope_key: str = ""
    effect: str = "boost"  # always_push / boost / suppress / scene_only / timing_policy
    instruction: str
    score_delta: float = 0.0
    confidence: float = 1.0
    origin: str = "default"  # default / explicit_user / implicit_stats / nightly_reflection
    expires_at: datetime | None = None
    status: GatingRuleStatus = GatingRuleStatus.ACTIVE
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class GradeSnapshot(BaseModel):
    """Immutable versioned snapshot of Gateway's dynamic `grade` block (updated atomically nightly)."""

    version: int = 1
    grade_text: str
    rule_ids: list[str] = Field(default_factory=list)
    summary_note: str = ""
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)


class FeedbackLogEntry(BaseModel):
    """Append-only daytime Forward Pass feedback ledger consumed by Nightly Backward Pass."""

    id: str = Field(default_factory=lambda: new_id("fb"))
    event_type: str  # push_engaged / push_ignored / push_rejected / explicit_preference / candidate_decision
    source_key: str = ""
    tags: list[str] = Field(default_factory=list)
    category: str = ""
    activity_mode: str = ""
    hour_bucket: int = 0
    sentiment: str = "neutral"  # positive / neutral / negative
    signal_value: float = 0.0  # +1.0 engaged, -0.6 ignored, -2.0 rejected
    raw_text: str = ""
    is_explicit: bool = False
    processed_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)


class SourceAffinity(BaseModel):
    """Smoothed EMA affinity score per source/tag key (bounded in [-3.0, +3.0])."""

    key: str
    affinity: float = 0.0
    sample_count: int = 0
    updated_at: datetime = Field(default_factory=utc_now)


class FollowUpIntent(BaseModel):
    """Dynamic follow-up commitment (no separate timetable; always arbitrated by Message Gateway)."""

    id: str = Field(default_factory=lambda: new_id("fu"))
    topic: str
    context_hint: str = ""
    intent_type: str = "deadline_backtrack"  # event_anchored / deadline_backtrack / delay
    deadline_at: str | None = None
    anchor_event_id: str | None = None
    safety_margin_minutes: int = 60
    computed_check_at: str | None = None
    priority: float = 8.0
    status: FollowUpStatus = FollowUpStatus.OPEN
    check_count: int = 0
    max_checks: int = 2
    source: str = "conversation"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class PassiveTriggerPayload(BaseModel):
    """Fixed structured directive emitted by Passive Trigger / Follow-up / T2 Renewal to Gateway."""

    id: str = Field(default_factory=lambda: new_id("trg"))
    trigger_name: str
    directive: str
    category: str = "daily_routine"
    priority: float = 8.0
    urgency: float = 8.0
    related_timetable_id: str | None = None
    related_followup_id: str | None = None
    related_scene_key: str | None = None
    fired_at: datetime = Field(default_factory=utc_now)


class GatewayDecision(BaseModel):
    """Structured decision from the Message Gateway LLM."""

    action: GatewayAction
    reason: str = ""
    push_instruction: str = ""
    scene_updates: dict[str, Any] = Field(default_factory=dict)
    scene_tier: StateTier = StateTier.T0_EPHEMERAL
    weight_delta: float | None = None
    archive_to_slow: bool = False


class ChatTurn(BaseModel):
    """A single turn inside Working State Context."""

    id: int | None = None
    role: str
    content: str
    channel: str = "qq"
    parsed_by_rtir: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class PendingPush(BaseModel):
    """Tracks an active push sent to QQ awaiting user reply before WeChat failover."""

    id: str = Field(default_factory=lambda: new_id("push"))
    content: str
    trigger_source: str = ""
    source_key: str = ""
    tags: list[str] = Field(default_factory=list)
    category: str = "general"
    urgency: float = 8.0
    qq_target: str = ""
    weixin_target: str = ""
    status: PushStatus = PushStatus.WAITING_QQ_REPLY
    sent_at: datetime = Field(default_factory=utc_now)
    failover_at: datetime
    resolved_at: datetime | None = None
    error_message: str = ""

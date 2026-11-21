"""Dedicated Structurer Sub-Agent (Step 1 & 1.1 + Fast/Slow Split + 3-Tier Trust Firewall).

Trust Firewall:
  - TRUST_HIGH   (owner, schedule import, trusted senders/channels): may auto-write global Timetable.
  - TRUST_MEDIUM (ordinary groups / external DMs): suspected schedule changes do NOT silently
                 overwrite Timetable; marked `needs_confirmation=True` for Gateway to ask owner.
  - TRUST_LOW    (RSS news, arXiv papers, public feeds): HARD-BLOCKED in code from mutating
                 Timetable or body state; can only enter `news` buffer or Slow Loop BM25.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ..config import AppConfig
from ..llm import LLMClient
from ..models import (
    BufferItem,
    BufferStatus,
    BufferZone,
    ChannelType,
    GroupChatBatch,
    RawIncomingEvent,
    SlowMemoryItem,
    SlowMemoryKind,
    SlowMemoryStatus,
    TimetableEntry,
    TimetableStatus,
    TrustTier,
    utc_now,
)
from ..slow_loop.bm25_store import SlowLoopBM25Store, is_beyond_horizon
from ..storage import Storage

logger = logging.getLogger(__name__)

GatewayForwardCallback = Callable[[BufferItem], Awaitable[Any]]


STRUCTURER_SYSTEM_PROMPT = """你是信息结构化专用子代理（Structurer Sub-Agent）。
你的任务是读取来自信息路由（多平台私聊、群聊记录、新闻、新论文或外部插件）的原始信息，过滤掉纯无意义寒暄，将其提炼为标准化的结构化 JSON 列表。

输出必须严格为 JSON 对象，格式如下：
{
  "items": [
    {
      "zone": "message" | "urti" | "news",
      "title": "简短清晰标题（<=25字）",
      "summary": "1-3句话说清楚核心内容、结论或讨论焦点",
      "details": "重要细节（人物、教学楼/地点、链接、截止时间、讨论观点等）",
      "tags": ["标签1", "标签2"],
      "intrinsic_value": 5.0,
      "urgency": 4.0,
      "initial_weight": 5.0,
      "has_explicit_time": false,
      "is_mandatory_task": false,
      "route_to_slow_loop": false,
      "slow_kind": "idea" | "future_event" | "future_task" | "plan" | "note",
      "estimated_minutes": null,
      "timetable_entry": null
    }
  ]
}

规则说明：
1. zone 选择：
   - 来自群聊/私聊/通知消息 -> "message"
   - 来自新闻/RSS/arXiv新论文/资讯 -> "news"
   - 明确反映用户个人实时状态/行程变更的信息 -> "urti"
2. 双轴评分（1.0~10.0）：
   - `intrinsic_value`（固有价值/兴趣度）：对用户的长期价值或有趣程度；
   - `urgency`（时间紧迫度）：9-10=立刻要处理的调课/紧急通知/临近DDL；6-8=当天有效热点讨论；1-4=无时效性的论文/资讯；
   - 若完全是无意义灌水/寒暄，直接返回 {"items": []}。
3. 快慢回路分流（7天时间视野）：
   - 一周之内（<= 7天）且同时满足 `has_explicit_time = true` 与 `is_mandatory_task = true` 的日程/任务 -> 填写 `timetable_entry`。
   - 一周之外（> 7天）的远期通知/考试/排班/DDL，或没有确定时间的长期想法/未来可选活动 -> 设置 `route_to_slow_loop = true`。
4. `timetable_entry` 格式（有明确时间时填写）：
   {
     "title": "事件名称",
     "start_time": "YYYY-MM-DDTHH:MM:SS+08:00",
     "end_time": "YYYY-MM-DDTHH:MM:SS+08:00 或 null",
     "location": "教学楼/教室/地点（若无则空字符串）",
     "category": "class" | "shift" | "deadline" | "meeting" | "life" | "exercise" | "other",
     "notes": "补充说明"
   }
"""


class StructurerSubAgent:
    """Dedicated sub-agent with 3-tier trust firewall and dual-axis scoring."""

    def __init__(
        self,
        config: AppConfig,
        storage: Storage,
        llm: LLMClient,
        on_structured_item: GatewayForwardCallback | None = None,
        slow_store: SlowLoopBM25Store | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.llm = llm
        self.on_structured_item = on_structured_item
        self.slow_store = slow_store

    def resolve_trust_tier(
        self,
        *,
        source_plugin: str,
        channel_type: ChannelType,
        channel_name: str = "",
        channel_id: str = "",
        sender_names: list[str] | None = None,
        is_from_owner: bool = False,
    ) -> TrustTier:
        """Deterministically assign HIGH / MEDIUM / LOW trust tier before LLM extraction."""
        if is_from_owner:
            return TrustTier.HIGH
        if channel_type in (ChannelType.NEWS, ChannelType.PAPER) or source_plugin in (
            "rss_news",
            "arxiv_papers",
        ):
            return TrustTier.LOW

        high_channels = {c.lower() for c in self.config.gating_evolution.high_trust_channels}
        if (channel_name and channel_name.lower() in high_channels) or (
            channel_id and channel_id.lower() in high_channels
        ):
            return TrustTier.HIGH

        high_senders = [s.lower() for s in self.config.gating_evolution.high_trust_senders if s.strip()]
        for name in sender_names or []:
            low = name.lower()
            if any(hs in low for hs in high_senders):
                return TrustTier.HIGH

        return TrustTier.MEDIUM

    def _build_context_header(self, trust_tier: TrustTier) -> str:
        now = utc_now()
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        wd = weekdays[now.weekday()]
        periods = "\n".join(
            f"  - {k}: {v}" for k, v in self.config.owner.school_periods.items()
        )
        return (
            f"【当前时间】{now.isoformat()} ({wd})\n"
            f"【信息源信任等级】{trust_tier.value}\n"
            f"【快/慢回路时间分界】{self.config.slow_loop.horizon_days} 天之内走快回路，超过 {self.config.slow_loop.horizon_days} 天或无确定时间的想法走慢回路 BM25\n"
            f"【用户背景】{self.config.owner.persona_notes}\n"
            f"【课节时间表】\n{periods}\n"
        )

    async def process_event(self, event: RawIncomingEvent) -> list[BufferItem]:
        """Structure a single incoming event (private non-owner msg, news, paper, webhook)."""
        trust_tier = self.resolve_trust_tier(
            source_plugin=event.source_plugin,
            channel_type=event.channel_type,
            channel_name=event.channel_name,
            channel_id=event.channel_id,
            sender_names=[event.sender_name, event.sender_id],
            is_from_owner=event.is_from_owner,
        )
        default_zone = (
            BufferZone.NEWS
            if event.channel_type in (ChannelType.NEWS, ChannelType.PAPER)
            else BufferZone.MESSAGE
        )
        source_label = f"{event.platform_type}/{event.channel_name or event.channel_id}"
        user_prompt = (
            f"{self._build_context_header(trust_tier)}\n"
            f"【信息来源】插件={event.source_plugin}, 平台={event.platform_type}, "
            f"频道类型={event.channel_type.value}, 频道={event.channel_name or event.channel_id}, "
            f"发送者={event.sender_name or event.sender_id}\n"
            f"【原始内容】\n{event.text}"
        )
        return await self._run_structuring(
            user_prompt=user_prompt,
            source_label=source_label,
            default_zone=default_zone,
            trust_tier=trust_tier,
        )

    async def process_group_batch(self, batch: GroupChatBatch) -> list[BufferItem]:
        """Structure a cleaned group chat batch flushed after 5 minutes of silence."""
        sender_names = [m.sender_name or m.sender_id for m in batch.messages]
        trust_tier = self.resolve_trust_tier(
            source_plugin=batch.source_plugin,
            channel_type=ChannelType.GROUP,
            channel_name=batch.group_name,
            channel_id=batch.group_id,
            sender_names=sender_names,
            is_from_owner=False,
        )
        source_label = f"{batch.platform_type}/群:{batch.group_name or batch.group_id}"
        transcript = batch.render_transcript()
        user_prompt = (
            f"{self._build_context_header(trust_tier)}\n"
            f"【信息来源】群聊记录（已连续安静达到阈值自动汇总），平台={batch.platform_type}, "
            f"群名={batch.group_name or batch.group_id} (ID={batch.group_id})\n"
            f"【群聊内容】\n{transcript}"
        )
        return await self._run_structuring(
            user_prompt=user_prompt,
            source_label=source_label,
            default_zone=BufferZone.MESSAGE,
            trust_tier=trust_tier,
        )

    async def _run_structuring(
        self,
        *,
        user_prompt: str,
        source_label: str,
        default_zone: BufferZone,
        trust_tier: TrustTier,
    ) -> list[BufferItem]:
        data = await self.llm.complete_json(
            role="structurer",
            system_prompt=STRUCTURER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        raw_items = data.get("items")
        if not isinstance(raw_items, list):
            logger.debug("Structurer returned no items for %s", source_label)
            return []

        now = utc_now()
        horizon = self.config.slow_loop.horizon_days
        max_w = self.config.gateway.max_weight
        created_items: list[BufferItem] = []

        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            summary = str(raw.get("summary") or "").strip()
            if not title and not summary:
                continue

            zone_str = str(raw.get("zone") or default_zone.value).lower().strip()
            try:
                zone = BufferZone(zone_str)
            except ValueError:
                zone = default_zone
            # TRUST_LOW sources are strictly confined to news zone (never pollute URTI/message)
            if trust_tier == TrustTier.LOW:
                zone = BufferZone.NEWS

            has_explicit_time = bool(raw.get("has_explicit_time", False))
            is_mandatory_task = bool(raw.get("is_mandatory_task", False))
            explicit_slow = bool(raw.get("route_to_slow_loop", False))

            # Dual-axis scores (gracefully falls back to initial_weight for compatibility)
            raw_init = raw.get("initial_weight", 4.0)
            try:
                fallback_w = float(raw_init)
            except (TypeError, ValueError):
                fallback_w = 4.0
            try:
                intrinsic_value = float(raw.get("intrinsic_value", fallback_w))
            except (TypeError, ValueError):
                intrinsic_value = fallback_w
            try:
                urgency = float(raw.get("urgency", fallback_w if is_mandatory_task else min(fallback_w, 5.0)))
            except (TypeError, ValueError):
                urgency = fallback_w

            intrinsic_value = max(0.5, min(max_w, intrinsic_value))
            urgency = max(0.5, min(max_w, urgency))
            if trust_tier == TrustTier.LOW:
                # Public news/papers can never forge emergency urgency >= 8.5
                urgency = min(urgency, 6.0)

            tags = raw.get("tags")
            if not isinstance(tags, list):
                tags = []
            tags = [str(t).strip() for t in tags if str(t).strip()]

            tt_raw = raw.get("timetable_entry")
            tt_start = (
                str(tt_raw.get("start_time") or "").strip()
                if isinstance(tt_raw, dict)
                else ""
            )
            beyond_week = bool(
                tt_start and is_beyond_horizon(tt_start, now=now, horizon_days=horizon)
            )

            # Slow Loop routing: events > 7 days away (for trusted sources) OR explicitly undated ideas
            if (
                self.slow_store is not None
                and self.config.slow_loop.enabled
                and ((beyond_week and trust_tier != TrustTier.LOW) or explicit_slow)
            ):
                default_kind = (
                    SlowMemoryKind.FUTURE_EVENT
                    if (beyond_week and trust_tier != TrustTier.LOW)
                    else SlowMemoryKind.IDEA
                )
                try:
                    slow_kind = SlowMemoryKind(
                        str(raw.get("slow_kind") or default_kind.value).lower().strip()
                    )
                except ValueError:
                    slow_kind = default_kind

                est_min = raw.get("estimated_minutes")
                slow_item = SlowMemoryItem(
                    kind=slow_kind,
                    title=(
                        str(tt_raw.get("title") or title).strip()
                        if isinstance(tt_raw, dict) and trust_tier != TrustTier.LOW
                        else title
                    )
                    or summary[:24],
                    summary=summary or title,
                    details=str(
                        raw.get("details")
                        or (tt_raw.get("notes") if isinstance(tt_raw, dict) else "")
                        or ""
                    ).strip(),
                    tags=tags,
                    planned_start=(tt_start or None) if trust_tier != TrustTier.LOW else None,
                    planned_end=(
                        str(tt_raw["end_time"]).strip()
                        if isinstance(tt_raw, dict) and tt_raw.get("end_time") and trust_tier != TrustTier.LOW
                        else None
                    ),
                    location=(
                        str(tt_raw.get("location") or "").strip()
                        if isinstance(tt_raw, dict)
                        else ""
                    ),
                    category=(
                        str(tt_raw.get("category") or "other").strip()
                        if isinstance(tt_raw, dict) and trust_tier != TrustTier.LOW
                        else "idea"
                    ),
                    estimated_minutes=int(est_min) if est_min is not None else None,
                    is_mandatory=is_mandatory_task and trust_tier == TrustTier.HIGH,
                    source=f"structurer:{source_label}",
                    status=SlowMemoryStatus.ACTIVE,
                )
                await self.slow_store.add_or_update(slow_item)
                logger.info(
                    "Structurer -> Slow Loop BM25 [%s] kind=%s trust=%s: %s",
                    slow_item.id,
                    slow_item.kind.value,
                    trust_tier.value,
                    slow_item.title,
                )
                continue

            # Trust Firewall for Timetable writes (Step 1.1)
            needs_confirmation = False
            details = str(raw.get("details") or "").strip()
            if has_explicit_time and is_mandatory_task and isinstance(tt_raw, dict) and tt_start:
                if trust_tier == TrustTier.HIGH:
                    tt_title = str(tt_raw.get("title") or title).strip()
                    if tt_title:
                        entry = TimetableEntry(
                            title=tt_title,
                            start_time=tt_start,
                            end_time=str(tt_raw["end_time"]).strip()
                            if tt_raw.get("end_time")
                            else None,
                            location=str(tt_raw.get("location") or "").strip(),
                            category=str(tt_raw.get("category") or "other").strip(),
                            is_mandatory=True,
                            status=TimetableStatus.SCHEDULED,
                            notes=str(tt_raw.get("notes") or summary).strip(),
                            source=f"structurer:{source_label}",
                        )
                        await self.storage.upsert_timetable(entry)
                        logger.info(
                            "Step 1.1 [TRUST_HIGH] Auto-wrote mandatory timed task to Timetable: %s @ %s",
                            entry.title,
                            entry.start_time,
                        )
                elif trust_tier == TrustTier.MEDIUM:
                    # Do NOT silently overwrite Timetable from unverified group chatter!
                    needs_confirmation = True
                    urgency = max(urgency, 7.5)
                    loc = str(tt_raw.get("location") or "").strip()
                    proposal = f"[待主人确认的疑似日程] {tt_raw.get('title') or title} @ {tt_start}" + (
                        f" ({loc})" if loc else ""
                    )
                    details = f"{proposal}\n{details}".strip()
                    logger.info(
                        "Step 1.1 [TRUST_MEDIUM] Flagged suspected schedule change for confirmation: %s",
                        proposal,
                    )
                else:
                    # TRUST_LOW: HARD BLOCKED from touching Timetable!
                    logger.debug(
                        "Trust Firewall [TRUST_LOW] blocked Timetable write from %s: %s",
                        source_label,
                        title,
                    )

            item = BufferItem(
                zone=zone,
                source=source_label,
                title=title or summary[:24],
                summary=summary or title,
                details=details,
                tags=tags,
                weight=intrinsic_value,
                intrinsic_value=intrinsic_value,
                urgency=urgency,
                starvation_boost=0.0,
                trust_tier=trust_tier,
                needs_confirmation=needs_confirmation,
                has_explicit_time=has_explicit_time,
                is_mandatory_task=is_mandatory_task and trust_tier != TrustTier.LOW,
                status=BufferStatus.ACTIVE,
            )
            await self.storage.add_buffer_item(item)
            created_items.append(item)
            logger.info(
                "Structured item [%s] zone=%s trust=%s val=%.1f urg=%.1f: %s",
                item.id,
                item.zone.value,
                trust_tier.value,
                item.intrinsic_value,
                item.urgency,
                item.title,
            )

            if self.on_structured_item is not None:
                try:
                    await self.on_structured_item(item)
                except Exception:
                    logger.exception("Message Gateway failed while processing item %s", item.id)

        return created_items

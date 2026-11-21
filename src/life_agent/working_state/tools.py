"""Direct LLM Function Call Tools (Fast Loop 4 + Slow Loop 2 + v2 Planning/FollowUp/Grade 3).

Fast Loop Direct Tools:
  1. view_timetable         (查看全局 7 天内动态时间表)
  2. modify_timetable       (删改时间表内容，修改后自动联动重算跟进时机)
  3. view_buffer            (查看缓冲区 message/URTI/news)
  4. modify_buffer          (删改缓冲区)

Slow Loop BM25 Tools:
  5. search_slow_memory     (BM25 检索慢回路：无确定时间想法 / >7天安排)
  6. modify_slow_memory     (写入/修改/删除慢回路，或升入快回路时间表)

v2 Dynamic Planning, Follow-Up & Grade Governance Tools:
  7. plan_free_slots        (确定性计算真实空闲时间段 + 联动慢回路 BM25 想法与未结案跟进)
  8. manage_followups       (创建/完成/取消/查看动态跟进意图，100% 经网关仲裁)
  9. view_and_manage_grade  (查看冻结版 grade_vN、审批夜间候选规则、版本回滚、追加偏好反馈、临时勿扰)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from ..gateway.policy import GatingPolicyManager
from ..models import (
    BufferItem,
    BufferStatus,
    BufferZone,
    FollowUpIntent,
    FollowUpStatus,
    GatingRuleStatus,
    SlowMemoryItem,
    SlowMemoryKind,
    SlowMemoryStatus,
    TimetableEntry,
    TimetableStatus,
    TrustTier,
    utc_now,
)
from ..slow_loop.bm25_store import (
    SlowLoopBM25Store,
    build_passive_trigger_for_event,
    parse_iso_datetime,
)
from ..storage import Storage
from ..triggers.followup import FollowUpTimingCalculator

logger = logging.getLogger(__name__)


DIRECT_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "view_timetable",
            "description": (
                "查看全局唯一动态时间表（time table，通常为一周之内的近期安排）。当用户询问今天/明天/本周有什么安排、课表、"
                "排班、截止时间，或在修改/删除某条日程前需要查询其 entry_id 时调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from": {"type": "string", "description": "起始日期（YYYY-MM-DD 或 ISO），可选"},
                    "date_to": {"type": "string", "description": "结束日期（YYYY-MM-DD 或 ISO），可选"},
                    "status": {
                        "type": "string",
                        "enum": ["scheduled", "completed", "cancelled", "all"],
                        "description": "状态过滤，默认 'scheduled'",
                    },
                    "keyword": {"type": "string", "description": "关键词搜索，可选"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_timetable",
            "description": (
                "新增、修改或删除全局时间表中的条目（修改后会自动联动重算相关主动跟进时机）。"
                "注意：若用户未给出精确时间（如只说“把今天下午英语课改到晚上”），请先反问具体几点/第几节再调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "update", "delete"]},
                    "entry_id": {"type": "string", "description": "update/delete 时的条目 ID"},
                    "title": {"type": "string"},
                    "start_time": {"type": "string", "description": "精确开始时间（ISO-8601）"},
                    "end_time": {"type": "string", "description": "精确结束时间（ISO-8601），可选"},
                    "location": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["class", "shift", "deadline", "meeting", "life", "exercise", "other"],
                    },
                    "status": {"type": "string", "enum": ["scheduled", "completed", "cancelled"]},
                    "notes": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_buffer",
            "description": "查看缓冲区（message / urti / news）的内容与双轴分值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {"type": "string", "enum": ["all", "message", "urti", "news"]},
                    "status": {
                        "type": "string",
                        "enum": ["active", "pushed", "merged_scene", "dismissed", "all"],
                    },
                    "keyword": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_buffer",
            "description": "新增、修改或删除缓冲区（message / urti / news）条目。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "update", "delete"]},
                    "item_id": {"type": "string"},
                    "zone": {"type": "string", "enum": ["message", "urti", "news"]},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "details": {"type": "string"},
                    "weight": {"type": "number"},
                    "status": {"type": "string", "enum": ["active", "pushed", "merged_scene", "dismissed"]},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_slow_memory",
            "description": (
                "使用 BM25 算法检索慢回路中结构化存储的内容（无确定时间的想法/空闲活动清单、一周之外的课表/排班/任务/多周规划）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "BM25 检索关键词或自然语言"},
                    "kind": {
                        "type": "string",
                        "enum": ["all", "idea", "future_event", "future_task", "plan", "note"],
                    },
                    "category": {"type": "string"},
                    "max_minutes": {"type": "integer", "description": "空闲时间活动时长上限（分钟）"},
                    "date_from": {"type": "string"},
                    "date_to": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_slow_memory",
            "description": "向慢回路写入/更新/删除条目，或通过 promote_to_timetable 定下具体时间升入快回路时间表与被动触发器。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "update", "delete", "promote_to_timetable"],
                    },
                    "item_id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["idea", "future_event", "future_task", "plan", "note"]},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "details": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "planned_start": {"type": "string"},
                    "planned_end": {"type": "string"},
                    "location": {"type": "string"},
                    "category": {"type": "string"},
                    "estimated_minutes": {"type": "integer"},
                    "energy_level": {"type": "string", "enum": ["low", "medium", "high"]},
                    "status": {"type": "string", "enum": ["active", "promoted", "completed", "archived"]},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_free_slots",
            "description": (
                "确定性计算今天或指定日期范围内的真实空闲时间段（精确到分钟，自动排除已排课程/排班/活动与睡眠时段），"
                "并自动联合拉取适配空闲时长的慢回路 BM25 想法与未结案跟进事项，方便你在对话中直接帮用户规划工作与时间块。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "目标日期 YYYY-MM-DD（默认今天）"},
                    "days": {"type": "integer", "description": "向前规划天数（1~7，默认 1）"},
                    "min_slot_minutes": {"type": "integer", "description": "最短可用空档分钟数（默认 30）"},
                    "activity_query": {
                        "type": "string",
                        "description": "可选：用于从慢回路 BM25 匹配候选活动的主题词（如 '学习 Rust 论文 跑步'）",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_followups",
            "description": (
                "管理主动跟进意图（如“晚上8点前提醒取快递”“下课后确认交作业”“晚点问我跑步了吗”）。"
                "系统会依据全局唯一时间表动态测算最佳空档跟进时机，并 100% 经消息网关统一仲裁。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "complete", "cancel", "list"]},
                    "followup_id": {"type": "string", "description": "complete/cancel 时对应的 ID"},
                    "topic": {"type": "string", "description": "跟进主题"},
                    "context_hint": {"type": "string", "description": "背景补充"},
                    "intent_type": {
                        "type": "string",
                        "enum": ["deadline_backtrack", "event_anchored", "delay"],
                    },
                    "deadline_at": {"type": "string", "description": "截止时间（ISO-8601），可选"},
                    "anchor_event_id": {"type": "string", "description": "锚定时间表事件 ID，可选"},
                    "safety_margin_minutes": {"type": "integer", "description": "截止前提前安全余量（分钟，默认 60）"},
                    "priority": {"type": "number", "description": "优先级（1~10，默认 8.0）"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_and_manage_grade",
            "description": (
                "查看网关当前冻结生效的动态分级器 grade 版本与候选规则、批准/驳回夜间复盘生成的 candidate 规则、"
                "回滚历史版本、向 feedback_log 追加长期偏好反馈（每晚 23:30 集中演化，白天不乱改规则）、或设置带到期时间的临时勿扰开关。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "view_current",
                            "list_candidates",
                            "approve_candidate",
                            "reject_candidate",
                            "rollback_version",
                            "record_preference_feedback",
                            "set_temp_dnd",
                        ],
                    },
                    "rule_id": {"type": "string", "description": "approve_candidate / reject_candidate 时的规则 ID"},
                    "version": {"type": "integer", "description": "rollback_version 时的目标版本号"},
                    "preference_text": {"type": "string", "description": "用户的明确长期偏好描述（追加到 feedback_log）"},
                    "source_key": {"type": "string", "description": "关联来源或主题键（如 topic:CV论文）"},
                    "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                    "dnd_instruction": {"type": "string", "description": "set_temp_dnd 时的勿扰原因"},
                    "expires_at": {"type": "string", "description": "set_temp_dnd 时必填的精确到期时间 ISO-8601"},
                },
                "required": ["action"],
            },
        },
    },
]


class DirectToolExecutor:
    """Executes all 9 structured tools cleanly across Fast Loop, Slow Loop BM25, Follow-ups, and Grade."""

    def __init__(
        self,
        storage: Storage,
        slow_store: SlowLoopBM25Store | None = None,
        *,
        policy: GatingPolicyManager | None = None,
        followup_calc: FollowUpTimingCalculator | None = None,
        default_reminder_lead_minutes: int = 20,
        sleep_start_hour: int = 23,
        sleep_end_hour: int = 7,
    ) -> None:
        self.storage = storage
        self.slow_store = slow_store or SlowLoopBM25Store(storage)
        self.policy = policy
        self.followup_calc = followup_calc or FollowUpTimingCalculator(storage)
        self.default_reminder_lead_minutes = default_reminder_lead_minutes
        self.sleep_start_hour = sleep_start_hour
        self.sleep_end_hour = sleep_end_hour

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "view_timetable":
                result = await self._view_timetable(arguments)
            elif name == "modify_timetable":
                result = await self._modify_timetable(arguments)
            elif name == "view_buffer":
                result = await self._view_buffer(arguments)
            elif name == "modify_buffer":
                result = await self._modify_buffer(arguments)
            elif name == "search_slow_memory":
                result = await self._search_slow_memory(arguments)
            elif name == "modify_slow_memory":
                result = await self._modify_slow_memory(arguments)
            elif name == "plan_free_slots":
                result = await self._plan_free_slots(arguments)
            elif name == "manage_followups":
                result = await self._manage_followups(arguments)
            elif name == "view_and_manage_grade":
                result = await self._view_and_manage_grade(arguments)
            else:
                result = {"ok": False, "error": f"Unknown tool: {name}"}
        except Exception as exc:
            logger.exception("Direct tool %s failed", name)
            result = {"ok": False, "error": str(exc)}
        return json.dumps(result, ensure_ascii=False)

    async def _view_timetable(self, args: dict[str, Any]) -> dict[str, Any]:
        entries = await self.storage.list_timetable(
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
            status=args.get("status", "scheduled"),
            keyword=args.get("keyword"),
        )
        return {
            "ok": True,
            "count": len(entries),
            "entries": [
                {
                    "id": e.id,
                    "title": e.title,
                    "start_time": e.start_time,
                    "end_time": e.end_time,
                    "location": e.location,
                    "category": e.category,
                    "status": e.status.value,
                    "notes": e.notes,
                }
                for e in entries
            ],
        }

    async def _modify_timetable(self, args: dict[str, Any]) -> dict[str, Any]:
        action = str(args.get("action") or "").lower().strip()
        entry_id = str(args.get("entry_id") or "").strip()

        if action == "delete":
            if not entry_id:
                return {"ok": False, "error": "delete 需要提供 entry_id"}
            deleted = await self.storage.delete_timetable(entry_id)
            if deleted:
                await self.followup_calc.recalculate_all_open()
            return {"ok": deleted, "action": "delete", "entry_id": entry_id}

        if action == "update":
            if not entry_id:
                return {"ok": False, "error": "update 需要提供 entry_id"}
            existing = await self.storage.get_timetable_entry(entry_id)
            if existing is None:
                return {"ok": False, "error": f"未找到时间表条目 {entry_id}"}
            if args.get("title"):
                existing.title = str(args["title"]).strip()
            if args.get("start_time"):
                existing.start_time = str(args["start_time"]).strip()
                existing.reminded = False
            if "end_time" in args:
                existing.end_time = str(args["end_time"]).strip() if args["end_time"] else None
            if "location" in args and args["location"] is not None:
                existing.location = str(args["location"]).strip()
            if args.get("category"):
                existing.category = str(args["category"]).strip()
            if args.get("status"):
                existing.status = TimetableStatus(str(args["status"]).strip())
            if "notes" in args and args["notes"] is not None:
                existing.notes = str(args["notes"]).strip()
            existing.source = "user_direct_tool"
            updated = await self.storage.upsert_timetable(existing)
            await self.followup_calc.recalculate_all_open()
            return {
                "ok": True,
                "action": "update",
                "entry": {
                    "id": updated.id,
                    "title": updated.title,
                    "start_time": updated.start_time,
                    "end_time": updated.end_time,
                    "location": updated.location,
                    "status": updated.status.value,
                    "notes": updated.notes,
                },
            }

        if action == "add":
            title = str(args.get("title") or "").strip()
            start_time = str(args.get("start_time") or "").strip()
            if not title or not start_time:
                return {"ok": False, "error": "add 需要提供 title 和精确 start_time"}
            status_val = TimetableStatus(str(args.get("status") or "scheduled"))
            entry = TimetableEntry(
                title=title,
                start_time=start_time,
                end_time=str(args["end_time"]).strip() if args.get("end_time") else None,
                location=str(args.get("location") or "").strip(),
                category=str(args.get("category") or "other").strip(),
                status=status_val,
                notes=str(args.get("notes") or "").strip(),
                source="user_direct_tool",
            )
            saved = await self.storage.upsert_timetable(entry)
            await self.followup_calc.recalculate_all_open()
            return {
                "ok": True,
                "action": "add",
                "entry": {
                    "id": saved.id,
                    "title": saved.title,
                    "start_time": saved.start_time,
                    "end_time": saved.end_time,
                    "location": saved.location,
                    "status": saved.status.value,
                },
            }

        return {"ok": False, "error": f"不支持的 action: {action}"}

    async def _view_buffer(self, args: dict[str, Any]) -> dict[str, Any]:
        limit = int(args.get("limit") or 20)
        items = await self.storage.list_buffer_items(
            zone=args.get("zone", "all"),
            status=args.get("status", "active"),
            keyword=args.get("keyword"),
            limit=max(1, min(100, limit)),
        )
        return {
            "ok": True,
            "count": len(items),
            "items": [
                {
                    "id": b.id,
                    "zone": b.zone.value,
                    "source": b.source,
                    "title": b.title,
                    "summary": b.summary,
                    "details": b.details,
                    "weight": b.weight,
                    "intrinsic_value": b.intrinsic_value,
                    "urgency": b.urgency,
                    "starvation_boost": b.starvation_boost,
                    "trust_tier": b.trust_tier.value,
                    "needs_confirmation": b.needs_confirmation,
                    "return_count": b.return_count,
                    "status": b.status.value,
                    "created_at": b.created_at.isoformat(),
                }
                for b in items
            ],
        }

    async def _modify_buffer(self, args: dict[str, Any]) -> dict[str, Any]:
        action = str(args.get("action") or "").lower().strip()
        item_id = str(args.get("item_id") or "").strip()

        if action == "delete":
            if not item_id:
                return {"ok": False, "error": "delete 需要提供 item_id"}
            deleted = await self.storage.delete_buffer_item(item_id)
            return {"ok": deleted, "action": "delete", "item_id": item_id}

        if action == "update":
            if not item_id:
                return {"ok": False, "error": "update 需要提供 item_id"}
            existing = await self.storage.get_buffer_item(item_id)
            if existing is None:
                return {"ok": False, "error": f"未找到缓冲区条目 {item_id}"}
            if args.get("zone"):
                existing.zone = BufferZone(str(args["zone"]).strip().lower())
            if args.get("title"):
                existing.title = str(args["title"]).strip()
            if args.get("summary"):
                existing.summary = str(args["summary"]).strip()
            if "details" in args and args["details"] is not None:
                existing.details = str(args["details"]).strip()
            if "weight" in args and args["weight"] is not None:
                w = max(0.5, min(10.0, float(args["weight"])))
                existing.weight = w
                existing.intrinsic_value = w
            if args.get("status"):
                existing.status = BufferStatus(str(args["status"]).strip())
            updated = await self.storage.update_buffer_item(existing)
            return {
                "ok": True,
                "action": "update",
                "item": {
                    "id": updated.id,
                    "zone": updated.zone.value,
                    "title": updated.title,
                    "summary": updated.summary,
                    "weight": updated.weight,
                    "status": updated.status.value,
                },
            }

        if action == "add":
            title = str(args.get("title") or "").strip()
            summary = str(args.get("summary") or title).strip()
            if not title and not summary:
                return {"ok": False, "error": "add 需要提供 title 或 summary"}
            zone = BufferZone(str(args.get("zone") or "urti").strip().lower())
            weight = max(0.5, min(10.0, float(args.get("weight") if args.get("weight") is not None else 5.0)))
            item = BufferItem(
                zone=zone,
                source="user_direct_tool",
                title=title or summary[:24],
                summary=summary,
                details=str(args.get("details") or "").strip(),
                weight=weight,
                intrinsic_value=weight,
                urgency=min(weight, 7.0),
                trust_tier=TrustTier.HIGH,
                status=BufferStatus.ACTIVE,
                next_review_at=utc_now() + timedelta(minutes=5),
            )
            saved = await self.storage.add_buffer_item(item)
            return {
                "ok": True,
                "action": "add",
                "item": {
                    "id": saved.id,
                    "zone": saved.zone.value,
                    "title": saved.title,
                    "summary": saved.summary,
                    "weight": saved.weight,
                },
            }

        return {"ok": False, "error": f"不支持的 action: {action}"}

    async def _search_slow_memory(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = max(1, min(30, int(args.get("top_k") or 8)))
        max_minutes = args.get("max_minutes")
        hits = await self.slow_store.search(
            query=str(args.get("query") or ""),
            kind=str(args.get("kind") or "all"),
            category=args.get("category"),
            status="active",
            max_minutes=int(max_minutes) if max_minutes is not None else None,
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
            top_k=top_k,
        )
        return {"ok": True, "retriever": "bm25", "count": len(hits), "results": hits}

    async def _modify_slow_memory(self, args: dict[str, Any]) -> dict[str, Any]:
        action = str(args.get("action") or "").lower().strip()
        item_id = str(args.get("item_id") or "").strip()

        if action == "delete":
            if not item_id:
                return {"ok": False, "error": "delete 需要提供 item_id"}
            deleted = await self.slow_store.delete(item_id)
            return {"ok": deleted, "action": "delete", "item_id": item_id}

        if action == "promote_to_timetable":
            if not item_id:
                return {"ok": False, "error": "promote_to_timetable 需要提供 item_id"}
            existing = await self.slow_store.get(item_id)
            if existing is None:
                return {"ok": False, "error": f"未找到慢回路条目 {item_id}"}
            start_time = str(args.get("planned_start") or existing.planned_start or "").strip()
            if not start_time:
                return {
                    "ok": False,
                    "error": "该慢回路条目尚未指定具体时间，请先向用户确认具体开始时间再传入 planned_start",
                }
            end_time = str(args.get("planned_end") or existing.planned_end or "").strip() or None
            entry = TimetableEntry(
                title=str(args.get("title") or existing.title).strip(),
                start_time=start_time,
                end_time=end_time,
                location=str(args.get("location") if args.get("location") is not None else existing.location).strip(),
                category=str(args.get("category") or existing.category or "other").strip(),
                is_mandatory=existing.is_mandatory,
                status=TimetableStatus.SCHEDULED,
                notes=existing.summary if not existing.details else f"{existing.summary} | {existing.details}",
                source=f"promoted_by_user:{existing.id}",
            )
            saved_tt = await self.storage.upsert_timetable(entry)
            trig = build_passive_trigger_for_event(
                saved_tt,
                lead_minutes=existing.reminder_lead_minutes or self.default_reminder_lead_minutes,
            )
            if trig is not None:
                await self.storage.upsert_scheduled_trigger(trig)
            existing.status = SlowMemoryStatus.PROMOTED
            existing.planned_start = start_time
            existing.planned_end = end_time
            await self.slow_store.add_or_update(existing)
            await self.followup_calc.recalculate_all_open()
            return {
                "ok": True,
                "action": "promote_to_timetable",
                "timetable_entry": {
                    "id": saved_tt.id,
                    "title": saved_tt.title,
                    "start_time": saved_tt.start_time,
                    "end_time": saved_tt.end_time,
                    "location": saved_tt.location,
                },
                "passive_trigger_at": trig.trigger_at if trig else None,
            }

        if action == "update":
            if not item_id:
                return {"ok": False, "error": "update 需要提供 item_id"}
            existing = await self.slow_store.get(item_id)
            if existing is None:
                return {"ok": False, "error": f"未找到慢回路条目 {item_id}"}
            if args.get("kind"):
                existing.kind = SlowMemoryKind(str(args["kind"]).strip().lower())
            if args.get("title"):
                existing.title = str(args["title"]).strip()
            if args.get("summary"):
                existing.summary = str(args["summary"]).strip()
            if "details" in args and args["details"] is not None:
                existing.details = str(args["details"]).strip()
            if isinstance(args.get("tags"), list):
                existing.tags = [str(t).strip() for t in args["tags"] if str(t).strip()]
            if "planned_start" in args:
                existing.planned_start = str(args["planned_start"]).strip() if args["planned_start"] else None
            if "planned_end" in args:
                existing.planned_end = str(args["planned_end"]).strip() if args["planned_end"] else None
            if "location" in args and args["location"] is not None:
                existing.location = str(args["location"]).strip()
            if args.get("category"):
                existing.category = str(args["category"]).strip()
            if "estimated_minutes" in args:
                existing.estimated_minutes = (
                    int(args["estimated_minutes"]) if args["estimated_minutes"] is not None else None
                )
            if "energy_level" in args and args["energy_level"] is not None:
                existing.energy_level = str(args["energy_level"]).strip()
            if args.get("status"):
                existing.status = SlowMemoryStatus(str(args["status"]).strip())
            updated = await self.slow_store.add_or_update(existing)
            return {"ok": True, "action": "update", "item": updated.model_dump(mode="json")}

        if action == "add":
            title = str(args.get("title") or "").strip()
            summary = str(args.get("summary") or title).strip()
            if not title and not summary:
                return {"ok": False, "error": "add 需要提供 title 或 summary"}
            planned_start = str(args["planned_start"]).strip() if args.get("planned_start") else None
            default_kind = "future_event" if planned_start else "idea"
            kind = SlowMemoryKind(str(args.get("kind") or default_kind).strip().lower())
            tags = args.get("tags") if isinstance(args.get("tags"), list) else []
            item = SlowMemoryItem(
                kind=kind,
                title=title or summary[:24],
                summary=summary,
                details=str(args.get("details") or "").strip(),
                tags=[str(t).strip() for t in tags if str(t).strip()],
                planned_start=planned_start,
                planned_end=str(args["planned_end"]).strip() if args.get("planned_end") else None,
                location=str(args.get("location") or "").strip(),
                category=str(args.get("category") or ("idea" if kind == SlowMemoryKind.IDEA else "other")).strip(),
                estimated_minutes=(
                    int(args["estimated_minutes"]) if args.get("estimated_minutes") is not None else None
                ),
                energy_level=str(args.get("energy_level") or "").strip(),
                source="user_slow_tool",
                status=SlowMemoryStatus.ACTIVE,
            )
            saved = await self.slow_store.add_or_update(item)
            return {
                "ok": True,
                "action": "add",
                "item": {
                    "id": saved.id,
                    "kind": saved.kind.value,
                    "title": saved.title,
                    "summary": saved.summary,
                    "planned_start": saved.planned_start,
                    "tags": saved.tags,
                },
            }

        return {"ok": False, "error": f"不支持的 action: {action}"}

    # ------------------------------------------------------------------
    # v2 Tool 7: Live Free-Slot Calculation & Work Planning
    # ------------------------------------------------------------------

    async def _plan_free_slots(self, args: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        days = max(1, min(7, int(args.get("days") or 1)))
        min_slot = max(10, int(args.get("min_slot_minutes") or 30))
        date_arg = str(args.get("date") or "").strip()
        base_dt = parse_iso_datetime(date_arg, now) if date_arg else now
        if base_dt is None:
            base_dt = now

        start_day = base_dt.date()
        end_day = start_day + timedelta(days=days - 1)
        entries = await self.storage.list_timetable(
            date_from=start_day.isoformat(),
            date_to=end_day.isoformat(),
            status="scheduled",
            limit=120,
        )

        # Group busy intervals by day
        busy_by_day: dict[str, list[tuple[datetime, datetime]]] = {}
        for e in entries:
            s_dt = parse_iso_datetime(e.start_time, now)
            if s_dt is None:
                continue
            e_dt = parse_iso_datetime(e.end_time, now) if e.end_time else (s_dt + timedelta(minutes=45))
            d_key = s_dt.date().isoformat()
            busy_by_day.setdefault(d_key, []).append((s_dt, e_dt))

        free_slots: list[dict[str, Any]] = []
        longest_slot = 0
        for offset in range(days):
            cur_date = start_day + timedelta(days=offset)
            d_key = cur_date.isoformat()
            day_start = datetime(
                cur_date.year, cur_date.month, cur_date.day, self.sleep_end_hour, 30, tzinfo=now.tzinfo
            )
            day_end = datetime(
                cur_date.year, cur_date.month, cur_date.day, self.sleep_start_hour, 0, tzinfo=now.tzinfo
            )
            # For today, only return slots starting from now (if before day_end)
            if cur_date == now.date() and now > day_start:
                day_start = now.replace(second=0, microsecond=0)
            if day_start >= day_end:
                continue

            intervals = sorted(busy_by_day.get(d_key, []), key=lambda sp: sp[0])
            cursor_t = day_start
            for b_start, b_end in intervals:
                if b_end <= cursor_t:
                    continue
                if b_start > cursor_t:
                    gap_end = min(b_start, day_end)
                    mins = int((gap_end - cursor_t).total_seconds() // 60)
                    if mins >= min_slot:
                        longest_slot = max(longest_slot, mins)
                        free_slots.append(
                            {
                                "date": d_key,
                                "start": cursor_t.isoformat(),
                                "end": gap_end.isoformat(),
                                "duration_minutes": mins,
                            }
                        )
                cursor_t = max(cursor_t, min(b_end, day_end))
                if cursor_t >= day_end:
                    break

            if cursor_t < day_end:
                mins = int((day_end - cursor_t).total_seconds() // 60)
                if mins >= min_slot:
                    longest_slot = max(longest_slot, mins)
                    free_slots.append(
                        {
                            "date": d_key,
                            "start": cursor_t.isoformat(),
                            "end": day_end.isoformat(),
                            "duration_minutes": mins,
                        }
                    )

        query = str(args.get("activity_query") or "空闲 活动 任务 学习").strip()
        slow_candidates = await self.slow_store.search(
            query=query,
            kind="all",
            status="active",
            max_minutes=longest_slot if longest_slot > 0 else None,
            top_k=6,
        )
        open_followups = await self.storage.list_followups(status=FollowUpStatus.OPEN.value, limit=10)
        return {
            "ok": True,
            "date_range": f"{start_day.isoformat()} .. {end_day.isoformat()}",
            "free_slots": free_slots,
            "slow_loop_candidates": slow_candidates,
            "open_followups": [
                {
                    "id": f.id,
                    "topic": f.topic,
                    "deadline_at": f.deadline_at,
                    "computed_check_at": f.computed_check_at,
                }
                for f in open_followups
            ],
        }

    # ------------------------------------------------------------------
    # v2 Tool 8: Dynamic Follow-Up Management
    # ------------------------------------------------------------------

    async def _manage_followups(self, args: dict[str, Any]) -> dict[str, Any]:
        action = str(args.get("action") or "list").lower().strip()
        fu_id = str(args.get("followup_id") or "").strip()

        if action == "list":
            items = await self.storage.list_followups(status="open", limit=30)
            return {
                "ok": True,
                "count": len(items),
                "followups": [f.model_dump(mode="json") for f in items],
            }

        if action in ("complete", "cancel"):
            if not fu_id:
                return {"ok": False, "error": f"{action} 需要提供 followup_id"}
            existing = await self.storage.get_followup(fu_id)
            if existing is None:
                return {"ok": False, "error": f"未找到跟进项 {fu_id}"}
            existing.status = FollowUpStatus.DONE if action == "complete" else FollowUpStatus.CANCELLED
            await self.storage.upsert_followup(existing)
            return {"ok": True, "action": action, "followup_id": fu_id}

        if action == "add":
            topic = str(args.get("topic") or "").strip()
            if not topic:
                return {"ok": False, "error": "add 需要提供 topic"}
            fu = FollowUpIntent(
                topic=topic,
                context_hint=str(args.get("context_hint") or "").strip(),
                intent_type=str(args.get("intent_type") or "deadline_backtrack").strip(),
                deadline_at=str(args["deadline_at"]).strip() if args.get("deadline_at") else None,
                anchor_event_id=str(args["anchor_event_id"]).strip() if args.get("anchor_event_id") else None,
                safety_margin_minutes=int(args.get("safety_margin_minutes") or 60),
                priority=float(args.get("priority") or 8.0),
                source="user_direct_tool",
            )
            saved = await self.followup_calc.register_followup(fu)
            return {
                "ok": True,
                "action": "add",
                "followup": {
                    "id": saved.id,
                    "topic": saved.topic,
                    "deadline_at": saved.deadline_at,
                    "computed_check_at": saved.computed_check_at,
                },
            }

        return {"ok": False, "error": f"不支持的 action: {action}"}

    # ------------------------------------------------------------------
    # v2 Tool 9: Inspect / Govern Versioned Grade & Feedback Ledger
    # ------------------------------------------------------------------

    async def _view_and_manage_grade(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.policy is None:
            return {"ok": False, "error": "GatingPolicyManager 未挂载"}
        action = str(args.get("action") or "view_current").lower().strip()

        if action == "view_current":
            snap = await self.policy.get_frozen_grade()
            history = await self.storage.list_grade_snapshots(limit=5)
            candidates = await self.storage.list_gating_rules(status=GatingRuleStatus.CANDIDATE.value)
            return {
                "ok": True,
                "active_version": snap.version,
                "grade_text": snap.grade_text,
                "summary_note": snap.summary_note,
                "pending_candidates_count": len(candidates),
                "recent_versions": [
                    {"version": h.version, "is_active": h.is_active, "summary_note": h.summary_note}
                    for h in history
                ],
            }

        if action == "list_candidates":
            candidates = await self.storage.list_gating_rules(status=GatingRuleStatus.CANDIDATE.value)
            return {
                "ok": True,
                "count": len(candidates),
                "candidates": [c.model_dump(mode="json") for c in candidates],
            }

        if action == "approve_candidate":
            rid = str(args.get("rule_id") or "").strip()
            if not rid:
                return {"ok": False, "error": "需要提供 rule_id"}
            snap = await self.policy.approve_candidate_rule(rid)
            if snap is None:
                return {"ok": False, "error": f"未找到候选规则 {rid}"}
            return {"ok": True, "action": "approve_candidate", "published_version": snap.version}

        if action == "reject_candidate":
            rid = str(args.get("rule_id") or "").strip()
            if not rid:
                return {"ok": False, "error": "需要提供 rule_id"}
            ok = await self.policy.reject_candidate_rule(rid)
            return {"ok": ok, "action": "reject_candidate", "rule_id": rid}

        if action == "rollback_version":
            ver = args.get("version")
            if ver is None:
                return {"ok": False, "error": "需要提供 version"}
            snap = await self.storage.activate_grade_version(int(ver))
            if snap is None:
                return {"ok": False, "error": f"未找到历史版本 v{ver}"}
            return {"ok": True, "action": "rollback_version", "active_version": snap.version}

        if action == "record_preference_feedback":
            pref = str(args.get("preference_text") or "").strip()
            if not pref:
                return {"ok": False, "error": "需要提供 preference_text"}
            sent = str(args.get("sentiment") or "neutral").strip()
            sig = 1.5 if sent == "positive" else (-2.0 if sent == "negative" else 0.0)
            entry = await self.policy.append_feedback(
                event_type="explicit_preference",
                source_key=str(args.get("source_key") or "").strip(),
                sentiment=sent,
                signal_value=sig,
                raw_text=pref,
                is_explicit=True,
            )
            return {
                "ok": True,
                "action": "record_preference_feedback",
                "feedback_id": entry.id,
                "note": "已写入今日反馈账本，将在今晚 23:30 上行复盘时合入新版 grade 并在明天生效（白天不中途篡改当前规则）。",
            }

        if action == "set_temp_dnd":
            exp_raw = str(args.get("expires_at") or "").strip()
            exp_dt = parse_iso_datetime(exp_raw, utc_now())
            if exp_dt is None:
                return {"ok": False, "error": "set_temp_dnd 必须提供精确到期时间 expires_at (ISO-8601)"}
            rule = await self.policy.set_temporary_override(
                scope_key="override:temp_dnd",
                instruction=str(args.get("dnd_instruction") or "用户设置临时勿扰，仅紧急事项可打扰"),
                expires_at=exp_dt,
                effect="suppress",
            )
            return {
                "ok": True,
                "action": "set_temp_dnd",
                "rule_id": rule.id,
                "expires_at": exp_dt.isoformat(),
            }

        return {"ok": False, "error": f"不支持的 action: {action}"}

"""Schedule & Shift Auto-Importer (课表与排班表自动导入模块).

Accepts:
  - Excel files (.xlsx via openpyxl with merged-cell propagation, .xls via xlrd, .csv)
  - Images (.png, .jpg, .jpeg, .webp, .bmp, .gif) via multimodal LLM vision

Workflow:
  1. Parses spreadsheet grid or sends image to LLM for structured extraction
     (supporting both explicit `dated_events` and `recurring_rules` like "每周一 1-2节 第1-16周").
  2. Deterministically expands recurring rules into exact calendar dates in Python.
  3. Splits every extracted occurrence at the 7-day horizon (`slow_loop.horizon_days`):
     - Within 7 days  -> Fast Loop `timetable` + explicit `ScheduledPassiveTrigger`!
     - Beyond 7 days  -> Slow Loop BM25 Store (`SlowMemoryKind.FUTURE_EVENT`)!
"""

from __future__ import annotations

import csv
import io
import logging
import mimetypes
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import openpyxl
import xlrd

from ..config import AppConfig
from ..llm import LLMClient
from ..models import (
    ScheduleImportResult,
    SlowMemoryItem,
    SlowMemoryKind,
    SlowMemoryStatus,
    TimetableEntry,
    TimetableStatus,
    utc_now,
)
from ..slow_loop.bm25_store import (
    SlowLoopBM25Store,
    build_passive_trigger_for_event,
    parse_iso_datetime,
)
from ..storage import Storage

logger = logging.getLogger(__name__)

EXCEL_EXTENSIONS = {".xlsx", ".xlsm", ".xls", ".csv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


SCHEDULE_IMPORTER_SYSTEM_PROMPT = """你是课表与排班表自动结构化解析专家（Schedule & Shift Importer）。
你的任务是读取用户上传的课表/排班表（Excel 表格文本或截图照片），将其精准解析为标准 JSON。

必须严格输出如下 JSON 格式：
{
  "schedule_type": "timetable" | "shift" | "mixed",
  "dated_events": [
    {
      "title": "课程名或班次名（如 高等数学 / 早班值班）",
      "start_time": "YYYY-MM-DDTHH:MM:SS+08:00",
      "end_time": "YYYY-MM-DDTHH:MM:SS+08:00 或 null",
      "location": "教学楼/教室/值班地点（无则空字符串）",
      "category": "class" | "shift" | "exam" | "meeting" | "other",
      "notes": "任课老师/交接事项/备注",
      "reminder_lead_minutes": 20
    }
  ],
  "recurring_rules": [
    {
      "title": "课程名或班次名",
      "weekday": 0,
      "start_hhmm": "08:00",
      "end_hhmm": "09:40",
      "location": "三教201",
      "category": "class" | "shift" | "other",
      "week_start": 1,
      "week_end": 16,
      "week_parity": "all" | "odd" | "even",
      "notes": "教师/备注",
      "reminder_lead_minutes": 20
    }
  ],
  "warnings": []
}

关键解析规则：
1. 如果表格/图片中给出了**明确日期**（如“2026-11-23 早班”“12月2日 期末考”），直接写入 `dated_events`，确保 `start_time` / `end_time` 附带正确日期与时区。
2. 如果表格/图片是**按星期与周次循环的课表或固定排班**（如“周一 第1-2节 第1-16周”“每周三晚班 18:00-21:00”），写入 `recurring_rules`（`weekday`: 0=周一, 1=周二, 2=周三, 3=周四, 4=周五, 5=周六, 6=周日；`week_parity`: 全周="all"、单周="odd"、双周="even"）。程序会用精确日历算法展开所有日期，不要自己把同一门周期课手动重复写十几遍进 `dated_events`。
3. 如果课表中只写了课节（如“第1-2节”“3-4节”），请严格参考下方【课节时间对照表】换算 `start_hhmm` 和 `end_hhmm`。
4. 尽量提取完整地点（教学楼、教室号、科室/岗位）、教师或值班要求到 `location` 和 `notes`。
5. 若遇到无法辨认的模糊条目，在 `warnings` 中记录说明。
"""


def _read_xlsx_bytes(data: bytes, max_rows: int = 200) -> str:
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
    sections: list[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        # Build merged-cell lookup so multi-period merged cells propagate their value cleanly
        merged_values: dict[tuple[int, int], str] = {}
        for rng in ws.merged_cells.ranges:
            top_val = ws.cell(rng.min_row, rng.min_col).value
            if top_val is not None:
                text = " ".join(str(top_val).split())
                for r in range(rng.min_row, rng.max_row + 1):
                    for c in range(rng.min_col, rng.max_col + 1):
                        merged_values[(r, c)] = text

        rows_text: list[str] = [f"=== Sheet: {sheet_name} ==="]
        max_r = min(ws.max_row or 0, max_rows)
        max_c = min(ws.max_column or 0, 30)
        for r in range(1, max_r + 1):
            cells: list[str] = []
            has_content = False
            for c in range(1, max_c + 1):
                val = ws.cell(r, c).value
                if val is None and (r, c) in merged_values:
                    cell_str = merged_values[(r, c)]
                elif val is None:
                    cell_str = ""
                else:
                    cell_str = " ".join(str(val).split())
                if cell_str:
                    has_content = True
                cells.append(cell_str)
            if has_content:
                rows_text.append(" | ".join(cells))
        if len(rows_text) > 1:
            sections.append("\n".join(rows_text))
    return "\n\n".join(sections)


def _read_xls_bytes(data: bytes, max_rows: int = 200) -> str:
    try:
        book = xlrd.open_workbook(file_contents=data, formatting_info=False)
    except Exception:
        # Some university JWXT portals export HTML/XML tables renamed to .xls
        head = data[:512].lower()
        if b"<html" in head or b"<table" in head or b"<?xml" in head:
            import re

            text = data.decode("utf-8", errors="replace")
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.IGNORECASE | re.DOTALL)
            lines = ["=== HTML-XLS Schedule ==="]
            for r in rows[:max_rows]:
                cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.IGNORECASE | re.DOTALL)
                cleaned = [
                    " ".join(re.sub(r"<[^>]+>", " ", c).split())
                    for c in cells
                ]
                if any(cleaned):
                    lines.append(" | ".join(cleaned))
            return "\n".join(lines)
        raise

    sections: list[str] = []
    for sheet in book.sheets():
        rows_text: list[str] = [f"=== Sheet: {sheet.name} ==="]
        for r in range(min(sheet.nrows, max_rows)):
            cells: list[str] = []
            has_content = False
            for c in range(min(sheet.ncols, 30)):
                cell = sheet.cell(r, c)
                if cell.ctype == xlrd.XL_CELL_DATE:
                    dt_tuple = xlrd.xldate_as_tuple(cell.value, book.datemode)
                    val_str = f"{dt_tuple[0]:04d}-{dt_tuple[1]:02d}-{dt_tuple[2]:02d}"
                else:
                    val_str = " ".join(str(cell.value).split()) if cell.value != "" else ""
                if val_str:
                    has_content = True
                cells.append(val_str)
            if has_content:
                rows_text.append(" | ".join(cells))
        if len(rows_text) > 1:
            sections.append("\n".join(rows_text))
    return "\n\n".join(sections)


def _read_csv_bytes(data: bytes, max_rows: int = 200) -> str:
    decoded = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            decoded = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        decoded = data.decode("utf-8", errors="replace")

    reader = csv.reader(io.StringIO(decoded))
    lines: list[str] = ["=== CSV Schedule ==="]
    for idx, row in enumerate(reader):
        if idx >= max_rows:
            break
        cleaned = [" ".join(c.split()) for c in row]
        if any(cleaned):
            lines.append(" | ".join(cleaned))
    return "\n".join(lines)


def extract_spreadsheet_text(
    data: bytes,
    filename: str,
    *,
    max_rows: int = 200,
) -> tuple[str, str]:
    """Extract clean grid text from .xlsx, .xls, or .csv. Returns (input_kind, text)."""
    ext = Path(filename).suffix.lower()
    if ext in (".xlsx", ".xlsm"):
        return "excel", _read_xlsx_bytes(data, max_rows=max_rows)
    if ext == ".xls":
        return "excel", _read_xls_bytes(data, max_rows=max_rows)
    if ext == ".csv":
        return "csv", _read_csv_bytes(data, max_rows=max_rows)
    raise ValueError(f"Unsupported spreadsheet format: {ext}")


@dataclass
class _ResolvedOccurrence:
    title: str
    start_dt: datetime
    end_dt: datetime | None
    location: str
    category: str
    notes: str
    reminder_lead_minutes: int


class ScheduleImporter:
    """Parses Excel/CSV/Image timetables & shift schedules and splits them at the 7-day horizon."""

    def __init__(
        self,
        config: AppConfig,
        storage: Storage,
        llm: LLMClient,
        slow_store: SlowLoopBM25Store,
    ) -> None:
        self.config = config
        self.storage = storage
        self.llm = llm
        self.slow_store = slow_store

    async def import_file(
        self,
        file_path: str | Path,
        *,
        schedule_type: str = "auto",
        current_week: int | None = None,
        semester_start_date: str | None = None,
        extra_instructions: str = "",
        now: datetime | None = None,
    ) -> ScheduleImportResult:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Schedule file not found: {p}")
        return await self.import_bytes(
            data=p.read_bytes(),
            filename=p.name,
            schedule_type=schedule_type,
            current_week=current_week,
            semester_start_date=semester_start_date,
            extra_instructions=extra_instructions,
            now=now,
        )

    async def import_bytes(
        self,
        *,
        data: bytes,
        filename: str,
        schedule_type: str = "auto",
        current_week: int | None = None,
        semester_start_date: str | None = None,
        extra_instructions: str = "",
        now: datetime | None = None,
    ) -> ScheduleImportResult:
        ref_now = now or utc_now()
        ext = Path(filename).suffix.lower()
        user_context = self._build_user_prompt_header(
            ref_now=ref_now,
            schedule_type=schedule_type,
            current_week=current_week,
            semester_start_date=semester_start_date,
            extra_instructions=extra_instructions,
        )

        if ext in EXCEL_EXTENSIONS:
            input_kind, grid_text = extract_spreadsheet_text(
                data,
                filename,
                max_rows=self.config.schedule_importer.max_excel_rows_per_sheet,
            )
            if not grid_text.strip():
                return ScheduleImportResult(
                    source_filename=filename,
                    input_kind=input_kind,
                    schedule_type=schedule_type,
                    warnings=["表格内容为空，未解析到任何日程"],
                )
            parsed = await self.llm.complete_json(
                role="structurer",
                system_prompt=SCHEDULE_IMPORTER_SYSTEM_PROMPT,
                user_prompt=f"{user_context}\n\n【表格原始内容 ({filename})】\n{grid_text}",
            )
        elif ext in IMAGE_EXTENSIONS:
            input_kind = "image"
            mime = mimetypes.guess_type(filename)[0] or "image/png"
            parsed = await self.llm.complete_vision_json(
                role="vision",
                system_prompt=SCHEDULE_IMPORTER_SYSTEM_PROMPT,
                user_prompt=(
                    f"{user_context}\n\n"
                    f"请仔细识别附带图片（{filename}）中的每一条课程/排班信息、星期、周次、时间段与地点，并按 JSON 输出。"
                ),
                image_bytes=data,
                mime_type=mime,
            )
        else:
            raise ValueError(
                f"不支持的文件类型 '{ext}'。支持 Excel (.xlsx/.xls/.csv) 与图片 (.png/.jpg/.jpeg/.webp/.bmp/.gif)。"
            )

        detected_type = str(parsed.get("schedule_type") or schedule_type or "mixed")
        if detected_type == "auto":
            detected_type = "mixed"
        warnings = [str(w) for w in (parsed.get("warnings") or []) if str(w).strip()]

        occurrences = self._resolve_all_occurrences(
            parsed=parsed,
            ref_now=ref_now,
            current_week=current_week,
            semester_start_date=semester_start_date,
            warnings=warnings,
        )

        return await self._route_occurrences(
            occurrences=occurrences,
            filename=filename,
            input_kind=input_kind,
            schedule_type=detected_type,
            ref_now=ref_now,
            warnings=warnings,
        )

    def _build_user_prompt_header(
        self,
        *,
        ref_now: datetime,
        schedule_type: str,
        current_week: int | None,
        semester_start_date: str | None,
        extra_instructions: str,
    ) -> str:
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        periods = "\n".join(
            f"  - {k}: {v}" for k, v in self.config.owner.school_periods.items()
        )
        week_hint = ""
        if semester_start_date:
            week_hint += f"- 学期第1周周一日期：{semester_start_date}\n"
        if current_week is not None:
            week_hint += f"- 当前是第 {current_week} 周\n"

        return (
            f"【当前时间】{ref_now.isoformat()} ({weekdays[ref_now.weekday()]})\n"
            f"【导入类型提示】{schedule_type}\n"
            f"{week_hint}"
            f"【课节时间对照表】\n{periods}\n"
            + (f"【用户补充说明】{extra_instructions}\n" if extra_instructions else "")
        )

    def _resolve_week1_monday(
        self,
        *,
        ref_now: datetime,
        current_week: int | None,
        semester_start_date: str | None,
    ) -> date:
        """Compute Monday of Week 1 for expanding recurring weekly timetables/shifts."""
        if semester_start_date:
            dt = parse_iso_datetime(semester_start_date, ref_now)
            if dt is not None:
                d = dt.date()
                return d - timedelta(days=d.weekday())
        this_monday = ref_now.date() - timedelta(days=ref_now.weekday())
        if current_week is not None and current_week >= 1:
            return this_monday - timedelta(weeks=current_week - 1)
        return this_monday

    def _resolve_all_occurrences(
        self,
        *,
        parsed: dict[str, Any],
        ref_now: datetime,
        current_week: int | None,
        semester_start_date: str | None,
        warnings: list[str],
    ) -> list[_ResolvedOccurrence]:
        default_lead = self.config.schedule_importer.default_reminder_lead_minutes
        occurrences: list[_ResolvedOccurrence] = []

        # 1. Explicit dated events
        for raw in parsed.get("dated_events") or []:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            start_dt = parse_iso_datetime(str(raw.get("start_time") or ""), ref_now)
            if not title or start_dt is None:
                warnings.append(f"跳过缺少标题或合法开始时间的条目: {raw}")
                continue
            end_dt = parse_iso_datetime(str(raw.get("end_time") or ""), ref_now)
            lead = self._safe_int(raw.get("reminder_lead_minutes"), default_lead)
            occurrences.append(
                _ResolvedOccurrence(
                    title=title,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    location=str(raw.get("location") or "").strip(),
                    category=str(raw.get("category") or "other").strip(),
                    notes=str(raw.get("notes") or "").strip(),
                    reminder_lead_minutes=lead,
                )
            )

        # 2. Deterministically expand recurring weekly rules
        week1_monday = self._resolve_week1_monday(
            ref_now=ref_now,
            current_week=current_week,
            semester_start_date=semester_start_date,
        )
        default_weeks = self.config.schedule_importer.default_expand_weeks
        inferred_cur_week = (
            current_week
            if current_week is not None and current_week >= 1
            else max(1, ((ref_now.date() - week1_monday).days // 7) + 1)
        )

        for rule in parsed.get("recurring_rules") or []:
            if not isinstance(rule, dict):
                continue
            title = str(rule.get("title") or "").strip()
            weekday = self._safe_int(rule.get("weekday"), -1)
            start_hhmm = str(rule.get("start_hhmm") or "").strip()
            if not title or not (0 <= weekday <= 6) or ":" not in start_hhmm:
                warnings.append(f"跳过格式不完整的周期规则: {rule}")
                continue

            end_hhmm = str(rule.get("end_hhmm") or "").strip()
            w_start = self._safe_int(rule.get("week_start"), inferred_cur_week)
            w_end = self._safe_int(
                rule.get("week_end"),
                max(w_start, inferred_cur_week + default_weeks - 1),
            )
            parity = str(rule.get("week_parity") or "all").lower().strip()
            lead = self._safe_int(rule.get("reminder_lead_minutes"), default_lead)
            location = str(rule.get("location") or "").strip()
            category = str(rule.get("category") or "class").strip()
            notes = str(rule.get("notes") or "").strip()

            try:
                sh, sm = (int(x) for x in start_hhmm.split(":")[:2])
            except ValueError:
                warnings.append(f"无法解析开始时间 '{start_hhmm}': {title}")
                continue
            eh_em: tuple[int, int] | None = None
            if end_hhmm and ":" in end_hhmm:
                try:
                    eh, em = (int(x) for x in end_hhmm.split(":")[:2])
                    eh_em = (eh, em)
                except ValueError:
                    eh_em = None

            for wk in range(max(1, w_start), min(w_end, w_start + 30) + 1):
                if parity == "odd" and wk % 2 == 0:
                    continue
                if parity == "even" and wk % 2 != 0:
                    continue
                target_date = week1_monday + timedelta(weeks=wk - 1, days=weekday)
                start_dt = datetime(
                    target_date.year,
                    target_date.month,
                    target_date.day,
                    sh,
                    sm,
                    tzinfo=ref_now.tzinfo,
                )
                end_dt = (
                    datetime(
                        target_date.year,
                        target_date.month,
                        target_date.day,
                        eh_em[0],
                        eh_em[1],
                        tzinfo=ref_now.tzinfo,
                    )
                    if eh_em is not None
                    else None
                )
                wk_note = f"第{wk}周" + (f" | {notes}" if notes else "")
                occurrences.append(
                    _ResolvedOccurrence(
                        title=title,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        location=location,
                        category=category,
                        notes=wk_note,
                        reminder_lead_minutes=lead,
                    )
                )

        # Deduplicate identical (title, start_dt) occurrences
        seen: set[tuple[str, str]] = set()
        deduped: list[_ResolvedOccurrence] = []
        for occ in sorted(occurrences, key=lambda o: o.start_dt):
            key = (occ.title.lower(), occ.start_dt.isoformat())
            if key not in seen:
                seen.add(key)
                deduped.append(occ)
        return deduped

    async def _route_occurrences(
        self,
        *,
        occurrences: list[_ResolvedOccurrence],
        filename: str,
        input_kind: str,
        schedule_type: str,
        ref_now: datetime,
        warnings: list[str],
    ) -> ScheduleImportResult:
        horizon_days = self.config.slow_loop.horizon_days
        cutoff = ref_now + timedelta(days=horizon_days)
        past_floor = ref_now - timedelta(hours=2)

        fast_entries: list[TimetableEntry] = []
        triggers: list[Any] = []
        slow_entries: list[SlowMemoryItem] = []
        skipped_past = 0

        for occ in occurrences:
            # Skip already-ended past occurrences
            end_or_start = occ.end_dt or occ.start_dt
            if end_or_start < past_floor:
                skipped_past += 1
                continue

            if occ.start_dt <= cutoff:
                # <= 7 days: Fast Loop Timetable + Explicit Passive Trigger!
                tt_entry = TimetableEntry(
                    title=occ.title,
                    start_time=occ.start_dt.isoformat(),
                    end_time=occ.end_dt.isoformat() if occ.end_dt else None,
                    location=occ.location,
                    category=occ.category,
                    is_mandatory=True,
                    status=TimetableStatus.SCHEDULED,
                    notes=occ.notes,
                    source=f"schedule_import:{filename}",
                )
                saved_tt = await self.storage.upsert_timetable(tt_entry)
                fast_entries.append(saved_tt)

                trig = build_passive_trigger_for_event(
                    saved_tt,
                    lead_minutes=occ.reminder_lead_minutes,
                    priority=self.config.schedule_importer.default_priority,
                    now=ref_now,
                )
                if trig is not None:
                    saved_trig = await self.storage.upsert_scheduled_trigger(trig)
                    triggers.append(saved_trig)
            else:
                # > 7 days: Slow Loop BM25 Store!
                kind_tag = "课表" if occ.category == "class" else ("排班" if occ.category == "shift" else "未来日程")
                date_str = occ.start_dt.strftime("%Y-%m-%d %H:%M")
                loc_str = f" @ {occ.location}" if occ.location else ""
                slow_item = SlowMemoryItem(
                    kind=SlowMemoryKind.FUTURE_EVENT,
                    title=occ.title,
                    summary=f"{date_str}{loc_str}（{kind_tag}）",
                    details=occ.notes,
                    tags=[kind_tag, occ.category, occ.start_dt.strftime("%Y-%m")],
                    planned_start=occ.start_dt.isoformat(),
                    planned_end=occ.end_dt.isoformat() if occ.end_dt else None,
                    location=occ.location,
                    category=occ.category,
                    is_mandatory=True,
                    reminder_lead_minutes=occ.reminder_lead_minutes,
                    source=f"schedule_import:{filename}",
                    status=SlowMemoryStatus.ACTIVE,
                )
                saved_slow = await self.slow_store.add_or_update(slow_item)
                slow_entries.append(saved_slow)

        logger.info(
            "ScheduleImporter processed %s (%s): %d within %dd -> Timetable+PassiveTriggers, %d beyond %dd -> SlowLoop BM25 (%d past skipped).",
            filename,
            input_kind,
            len(fast_entries),
            horizon_days,
            len(slow_entries),
            horizon_days,
            skipped_past,
        )
        return ScheduleImportResult(
            source_filename=filename,
            input_kind=input_kind,
            schedule_type=schedule_type,
            total_extracted_events=len(occurrences),
            fast_loop_timetable_count=len(fast_entries),
            passive_triggers_written=len(triggers),
            slow_loop_bm25_count=len(slow_entries),
            skipped_past_count=skipped_past,
            fast_loop_entries=fast_entries,
            passive_triggers=triggers,
            slow_loop_entries=slow_entries,
            warnings=warnings,
        )

    @staticmethod
    def _safe_int(val: Any, default: int) -> int:
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

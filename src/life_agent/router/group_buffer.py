"""Group chat deduplication, jielong (接龙) filtering, and 5-minute silence debounce."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..config import GroupFilterConfig
from ..models import GroupChatBatch, RawIncomingEvent, utc_now

logger = logging.getLogger(__name__)

_NUMBERED_LINE_RE = re.compile(r"^\s*(?:第?\d+[.、)）\s:]|[①②③④⑤⑥⑦⑧⑨⑩])\s*\S+")
_JIELONG_HEADER_RE = re.compile(r"^\s*(?:#\s*接龙|接龙\b|【接龙】|\[接龙\])")
_WHITESPACE_RE = re.compile(r"\s+")
_PURE_PUNCT_RE = re.compile(r"^[\W_]+$")

FlushCallback = Callable[[GroupChatBatch], Awaitable[None]]


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip()).lower()


def extract_jielong_info(text: str, min_numbered_lines: int = 3) -> tuple[bool, str, int]:
    """Detect WeChat/QQ chain messages (接龙) and return (is_jielong, topic_key, entry_count)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False, "", 0

    has_explicit_tag = bool(_JIELONG_HEADER_RE.match(lines[0]))
    numbered_lines = [line for line in lines if _NUMBERED_LINE_RE.match(line)]

    if has_explicit_tag and (len(numbered_lines) >= 1 or len(lines) >= 2):
        # Extract topic from line 0 (stripped of #接龙) or line 1 if line 0 is solely #接龙
        header_remainder = _JIELONG_HEADER_RE.sub("", lines[0]).strip(" :：-—")
        if header_remainder:
            topic = header_remainder[:60]
        elif len(lines) >= 2 and not _NUMBERED_LINE_RE.match(lines[1]):
            topic = lines[1][:60]
        else:
            topic = "群接龙"
        return True, topic, max(len(numbered_lines), len(lines) - 1)

    if len(numbered_lines) >= min_numbered_lines:
        # Check if first non-numbered line looks like a topic header, or first numbered line is 1.
        non_numbered = [line for line in lines if not _NUMBERED_LINE_RE.match(line)]
        if non_numbered:
            topic = non_numbered[0][:60]
        else:
            topic = numbered_lines[0][:40]
        # Require at least 50% of lines to be numbered list items so normal paragraphs aren't flagged
        if len(numbered_lines) / len(lines) >= 0.5:
            return True, topic, len(numbered_lines)

    return False, "", 0


@dataclass
class _JielongTracker:
    topic: str
    updates: int = 1
    max_entries: int = 0
    latest_lines: set[str] = field(default_factory=set)


@dataclass
class _GroupBucket:
    key: str
    source_plugin: str
    platform_type: str
    platform_id: str
    group_id: str
    group_name: str
    messages: list[RawIncomingEvent] = field(default_factory=list)
    seen_texts: dict[str, datetime] = field(default_factory=dict)
    jielong_trackers: dict[str, _JielongTracker] = field(default_factory=dict)
    filtered_count: int = 0
    started_at: datetime = field(default_factory=utc_now)
    timer_task: asyncio.Task[None] | None = None


class GroupChatBufferManager:
    """Buffers group chats, strips duplicates/echoes/jielong spam, and flushes after silence."""

    def __init__(
        self,
        config: GroupFilterConfig,
        on_flush: FlushCallback,
    ) -> None:
        self.config = config
        self.on_flush = on_flush
        self._buckets: dict[str, _GroupBucket] = {}
        self._echo_set = {w.strip().lower() for w in config.echo_words}
        self._lock = asyncio.Lock()

    def _bucket_key(self, event: RawIncomingEvent) -> str:
        return f"{event.platform_id}:{event.channel_id}"

    def _is_echo_or_noise(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        norm = stripped.lower()
        if norm in self._echo_set:
            return True
        if _PURE_PUNCT_RE.match(stripped) and len(stripped) <= 6:
            return True
        # Repeated single character spam like "哈哈哈哈哈哈" or "111111"
        if len(set(norm)) == 1 and len(norm) >= 2:
            return True
        return False

    def _matches_existing_jielong(self, bucket: _GroupBucket, text: str) -> str | None:
        """Detect incremental copy-paste chain updates even if header was slightly edited."""
        lines = {ln.strip() for ln in text.splitlines() if ln.strip()}
        if len(lines) < 2:
            return None
        for topic_key, tracker in bucket.jielong_trackers.items():
            if not tracker.latest_lines:
                continue
            overlap = len(lines & tracker.latest_lines)
            smaller = min(len(lines), len(tracker.latest_lines))
            if smaller >= 2 and (overlap / smaller) >= 0.6:
                return topic_key
        return None

    async def ingest(self, event: RawIncomingEvent) -> bool:
        """Accept a group message. Returns True if kept, False if filtered as noise/dup/jielong."""
        key = self._bucket_key(event)
        should_force_flush = False

        async with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _GroupBucket(
                    key=key,
                    source_plugin=event.source_plugin,
                    platform_type=event.platform_type,
                    platform_id=event.platform_id,
                    group_id=event.channel_id,
                    group_name=event.channel_name or event.channel_id,
                    started_at=event.timestamp,
                )
                self._buckets[key] = bucket
            elif event.channel_name and bucket.group_name == bucket.group_id:
                bucket.group_name = event.channel_name

            text = event.text.strip()
            norm = normalize_text(text)

            # 1. Echo / spam filter
            if self._is_echo_or_noise(text):
                bucket.filtered_count += 1
                self._reset_timer_locked(bucket)
                return False

            # 2. Exact / normalized duplicate filter within rolling window
            prev_ts = bucket.seen_texts.get(norm)
            if prev_ts is not None:
                delta = abs((event.timestamp - prev_ts).total_seconds())
                if delta <= self.config.dedup_window_seconds:
                    bucket.filtered_count += 1
                    self._reset_timer_locked(bucket)
                    return False
            bucket.seen_texts[norm] = event.timestamp

            # 3. Chain message (接龙) detection & collapse
            is_jielong, topic, count = extract_jielong_info(
                text,
                min_numbered_lines=self.config.min_jielong_lines,
            )
            matched_topic = self._matches_existing_jielong(bucket, text) if not is_jielong else None
            if is_jielong or matched_topic is not None:
                topic_key = normalize_text(matched_topic or topic) or "群接龙"
                tracker = bucket.jielong_trackers.get(topic_key)
                lines_set = {ln.strip() for ln in text.splitlines() if ln.strip()}
                if tracker is None:
                    bucket.jielong_trackers[topic_key] = _JielongTracker(
                        topic=topic or matched_topic or "群接龙",
                        updates=1,
                        max_entries=max(count, len(lines_set)),
                        latest_lines=lines_set,
                    )
                else:
                    tracker.updates += 1
                    tracker.max_entries = max(tracker.max_entries, count, len(lines_set))
                    tracker.latest_lines = lines_set
                bucket.filtered_count += 1
                self._reset_timer_locked(bucket)
                return False

            # 4. Keep meaningful message & reset 5-minute silence timer
            bucket.messages.append(event)
            self._reset_timer_locked(bucket)
            if len(bucket.messages) >= self.config.max_buffer_messages:
                should_force_flush = True

        if should_force_flush:
            await self.flush_group(key)
        return True

    def _reset_timer_locked(self, bucket: _GroupBucket) -> None:
        if bucket.timer_task is not None and not bucket.timer_task.done():
            bucket.timer_task.cancel()
        bucket.timer_task = asyncio.create_task(
            self._silence_countdown(bucket.key, self.config.silence_seconds),
            name=f"group_silence_{bucket.key}",
        )

    async def _silence_countdown(self, key: str, seconds: float) -> None:
        try:
            await asyncio.sleep(seconds)
            await self.flush_group(key)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Error flushing group %s after silence window", key)

    async def flush_group(self, key: str) -> GroupChatBatch | None:
        """Immediately flush a group bucket to the Structurer Sub-Agent."""
        async with self._lock:
            bucket = self._buckets.pop(key, None)
            if bucket is None:
                return None
            if bucket.timer_task is not None and not bucket.timer_task.done():
                current = asyncio.current_task()
                if bucket.timer_task is not current:
                    bucket.timer_task.cancel()

        jielong_summaries: list[str] = []
        if self.config.jielong_mode == "keep_header_only":
            for tracker in bucket.jielong_trackers.values():
                jielong_summaries.append(
                    f"接龙主题「{tracker.topic}」(已折叠 {tracker.updates} 条更新，当前约 {tracker.max_entries} 项)"
                )

        if not bucket.messages and not jielong_summaries:
            logger.debug(
                "Group %s silence elapsed with only %d filtered noise messages; skipping Structurer.",
                key,
                bucket.filtered_count,
            )
            return None

        batch = GroupChatBatch(
            source_plugin=bucket.source_plugin,
            platform_type=bucket.platform_type,
            platform_id=bucket.platform_id,
            group_id=bucket.group_id,
            group_name=bucket.group_name,
            messages=bucket.messages,
            filtered_count=bucket.filtered_count,
            jielong_summaries=jielong_summaries,
            started_at=bucket.started_at,
            flushed_at=utc_now(),
        )
        logger.info(
            "Flushing group %s (%s): %d kept messages, %d filtered, %d jielong summaries",
            batch.group_id,
            batch.group_name,
            len(batch.messages),
            batch.filtered_count,
            len(batch.jielong_summaries),
        )
        await self.on_flush(batch)
        return batch

    async def flush_all(self) -> None:
        keys = list(self._buckets.keys())
        for k in keys:
            await self.flush_group(k)

    async def stop(self) -> None:
        async with self._lock:
            for bucket in self._buckets.values():
                if bucket.timer_task is not None and not bucket.timer_task.done():
                    bucket.timer_task.cancel()
            self._buckets.clear()

    def pending_summary(self) -> list[dict[str, object]]:
        return [
            {
                "key": b.key,
                "group_name": b.group_name,
                "kept_messages": len(b.messages),
                "filtered_count": b.filtered_count,
                "jielong_topics": [t.topic for t in b.jielong_trackers.values()],
                "started_at": b.started_at.isoformat(),
            }
            for b in self._buckets.values()
        ]

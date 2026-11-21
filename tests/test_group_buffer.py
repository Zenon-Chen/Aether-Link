"""Tests for GroupChatBufferManager: dedup, echo filter, jielong (接龙) filter, 5-min silence flush."""

from __future__ import annotations

import asyncio

import pytest

from life_agent.config import GroupFilterConfig
from life_agent.models import ChannelType, GroupChatBatch, RawIncomingEvent
from life_agent.router.group_buffer import GroupChatBufferManager, extract_jielong_info


def _group_msg(text: str, sender: str = "同学A", group_id: str = "g100") -> RawIncomingEvent:
    return RawIncomingEvent(
        source_plugin="astrbot",
        platform_type="aiocqhttp",
        platform_id="qq_onebot",
        channel_type=ChannelType.GROUP,
        channel_id=group_id,
        channel_name="计科2301通知群",
        sender_id=sender,
        sender_name=sender,
        text=text,
    )


def test_jielong_detection() -> None:
    msg1 = "#接龙\n周五聚餐报名\n1. 张三\n2. 李四"
    is_jl, topic, count = extract_jielong_info(msg1)
    assert is_jl is True
    assert "周五聚餐报名" in topic
    assert count >= 2

    msg2 = "周三实验分组：\n1. 小明\n2. 小红\n3. 小刚\n4. 小华"
    is_jl2, topic2, count2 = extract_jielong_info(msg2)
    assert is_jl2 is True
    assert "实验分组" in topic2
    assert count2 == 4

    normal = "大家注意明天高数课调到三教201，记得带作业本。"
    is_jl3, _, _ = extract_jielong_info(normal)
    assert is_jl3 is False


async def test_group_dedup_echo_jielong_and_silence_flush() -> None:
    flushed: list[GroupChatBatch] = []

    async def on_flush(batch: GroupChatBatch) -> None:
        flushed.append(batch)

    cfg = GroupFilterConfig(silence_seconds=0.15, jielong_mode="keep_header_only")
    mgr = GroupChatBufferManager(cfg, on_flush=on_flush)

    # Send echo spam, exact duplicates, growing jielong chain, and 2 meaningful messages
    assert await mgr.ingest(_group_msg("明天下午高数课改到三教201", sender="班长")) is True
    assert await mgr.ingest(_group_msg("收到", sender="同学1")) is False
    assert await mgr.ingest(_group_msg("+1", sender="同学2")) is False
    assert await mgr.ingest(_group_msg("哈哈哈哈", sender="同学3")) is False
    assert await mgr.ingest(_group_msg("明天下午高数课改到三教201", sender="复读机")) is False

    # Growing jielong chain (3 updates -> collapsed into 1 compact summary)
    assert await mgr.ingest(_group_msg("#接龙\n周六秋游统计\n1. 张三\n2. 李四", sender="张三")) is False
    assert await mgr.ingest(_group_msg("#接龙\n周六秋游统计\n1. 张三\n2. 李四\n3. 王五", sender="王五")) is False
    assert await mgr.ingest(_group_msg("#接龙\n周六秋游统计\n1. 张三\n2. 李四\n3. 王五\n4. 赵六", sender="赵六")) is False

    # Wait 0.08s (< 0.15s silence), send another meaningful message -> should reset silence timer!
    await asyncio.sleep(0.08)
    assert len(flushed) == 0
    assert await mgr.ingest(_group_msg("顺便提醒大家周四交操作系统实验报告", sender="学委")) is True

    # Wait 0.08s again -> still hasn't been silent for 0.15s since last message
    await asyncio.sleep(0.08)
    assert len(flushed) == 0

    # Wait another 0.15s -> silence window elapses -> auto flush!
    await asyncio.sleep(0.15)
    assert len(flushed) == 1

    batch = flushed[0]
    assert len(batch.messages) == 2
    assert batch.filtered_count == 7  # 3 echoes + 1 duplicate + 3 jielong copies
    assert len(batch.jielong_summaries) == 1
    assert "周六秋游统计" in batch.jielong_summaries[0]
    assert "3 条更新" in batch.jielong_summaries[0]

    transcript = batch.render_transcript()
    assert "三教201" in transcript
    assert "操作系统实验报告" in transcript
    assert "收到" not in transcript
    await mgr.stop()


async def test_pure_noise_group_does_not_call_structurer() -> None:
    flushed: list[GroupChatBatch] = []

    async def on_flush(batch: GroupChatBatch) -> None:
        flushed.append(batch)

    cfg = GroupFilterConfig(silence_seconds=0.1, jielong_mode="drop")
    mgr = GroupChatBufferManager(cfg, on_flush=on_flush)

    await mgr.ingest(_group_msg("收到"))
    await mgr.ingest(_group_msg("+1"))
    await mgr.ingest(_group_msg("#接龙\n签到\n1. A\n2. B"))
    await asyncio.sleep(0.2)

    assert len(flushed) == 0
    await mgr.stop()

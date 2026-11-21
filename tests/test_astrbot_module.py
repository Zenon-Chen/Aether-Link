"""Integration tests verifying AstrBot is mounted cleanly as an independent-process module."""

from __future__ import annotations

import asyncio
from pathlib import Path

from life_agent.config import AstrBotModuleConfig, OwnerConfig
from life_agent.models import ChannelType
from life_agent.router.plugins.astrbot_bridge import (
    AstrBotBridgePlugin,
    bootstrap_astrbot_environment,
)


def test_astrbot_module_isolation_and_adapter_instantiation(tmp_path: Path) -> None:
    """Verify AstrBot's PlatformManager, AiocqhttpAdapter, and WeixinOCAdapter load with zero RAG deps."""
    project_root = Path(__file__).resolve().parents[1]
    vendor_repo = project_root / "vendor" / "AstrBot"
    assert (vendor_repo / "astrbot").exists(), "vendor/AstrBot must be cloned"

    bootstrap_astrbot_environment(vendor_repo, tmp_path / "astrbot_data")

    from astrbot.api.platform import AstrBotMessage, MessageMember, MessageType, PlatformMetadata
    from astrbot.core.platform.astr_message_event import AstrMessageEvent
    from astrbot.core.platform.manager import PlatformManager
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_platform_adapter import AiocqhttpAdapter
    from astrbot.core.platform.sources.weixin_oc.weixin_oc_adapter import WeixinOCAdapter

    assert PlatformManager is not None

    q: asyncio.Queue = asyncio.Queue()
    qq_adapter = AiocqhttpAdapter(
        {
            "id": "qq_onebot",
            "type": "aiocqhttp",
            "ws_reverse_host": "127.0.0.1",
            "ws_reverse_port": 16199,
            "ws_reverse_token": "",
        },
        {"unique_session": True},
        q,
    )
    wx_adapter = WeixinOCAdapter(
        {
            "id": "weixin_personal",
            "type": "weixin_oc",
            "weixin_oc_base_url": "https://ilinkai.weixin.qq.com",
            "weixin_oc_token": "",
        },
        {"unique_session": True},
        q,
    )
    assert qq_adapter.meta().name == "aiocqhttp"
    assert wx_adapter.meta().name == "weixin_oc"

    # Verify AstrBotBridgePlugin converts AstrMessageEvent cleanly for both owner DM and group chat
    bridge = AstrBotBridgePlugin(
        config=AstrBotModuleConfig(
            enabled=True,
            astrbot_repo_path=str(vendor_repo),
            astrbot_data_root=str(tmp_path / "astrbot_data"),
        ),
        owner=OwnerConfig(qq_id="123456", qq_platform_id="qq_onebot"),
        base_dir=project_root,
    )

    # 1. Owner QQ private chat -> is_from_owner=True
    abm_owner = AstrBotMessage()
    abm_owner.self_id = "999999"
    abm_owner.sender = MessageMember(user_id="123456", nickname="主人")
    abm_owner.type = MessageType.FRIEND_MESSAGE
    abm_owner.session_id = "123456"
    abm_owner.message_str = "帮我查一下今天时间表"
    ev_owner = AstrMessageEvent(
        message_str="帮我查一下今天时间表",
        message_obj=abm_owner,
        platform_meta=PlatformMetadata(name="aiocqhttp", description="QQ", id="qq_onebot"),
        session_id="123456",
    )
    raw_owner = bridge._convert_event(ev_owner)
    assert raw_owner is not None
    assert raw_owner.is_from_owner is True
    assert raw_owner.channel_type == ChannelType.PRIVATE

    # 2. QQ Group chat -> channel_type=GROUP, is_from_owner=False
    abm_grp = AstrBotMessage()
    abm_grp.self_id = "999999"
    abm_grp.sender = MessageMember(user_id="555555", nickname="班长")
    abm_grp.type = MessageType.GROUP_MESSAGE
    abm_grp.group_id = "888888"
    abm_grp.session_id = "888888"
    abm_grp.message_str = "明天停课"
    ev_grp = AstrMessageEvent(
        message_str="明天停课",
        message_obj=abm_grp,
        platform_meta=PlatformMetadata(name="aiocqhttp", description="QQ", id="qq_onebot"),
        session_id="888888",
    )
    raw_grp = bridge._convert_event(ev_grp)
    assert raw_grp is not None
    assert raw_grp.is_from_owner is False
    assert raw_grp.channel_type == ChannelType.GROUP
    assert raw_grp.channel_id == "888888"

    # 3. Official personal WeChat (weixin_oc) -> auto-discovers owner weixin_user_id
    abm_wx = AstrBotMessage()
    abm_wx.self_id = "bot_ilink"
    abm_wx.sender = MessageMember(user_id="u_abc123@im.wechat", nickname="我")
    abm_wx.type = MessageType.FRIEND_MESSAGE
    abm_wx.session_id = "u_abc123@im.wechat"
    abm_wx.message_str = "收到啦"
    ev_wx = AstrMessageEvent(
        message_str="收到啦",
        message_obj=abm_wx,
        platform_meta=PlatformMetadata(name="weixin_oc", description="个人微信", id="weixin_personal"),
        session_id="u_abc123@im.wechat",
    )
    raw_wx = bridge._convert_event(ev_wx)
    assert raw_wx is not None
    assert raw_wx.is_from_owner is True
    assert bridge.get_effective_weixin_user_id() == "u_abc123@im.wechat"

"""Configuration schema and YAML loader for Life Agent (v2: Two-Phase Gating + Living Profile)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        def repl(match: re.Match[str]) -> str:
            var = match.group(1)
            default = match.group(2)
            val = os.environ.get(var)
            if val is not None:
                return val
            return default if default is not None else match.group(0)

        return _ENV_PATTERN.sub(repl, obj)
    if isinstance(obj, list):
        return [_expand_env(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


class ModelEndpointConfig(BaseModel):
    """OpenAI-compatible endpoint configuration."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = "${OPENAI_API_KEY:-}"
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout_seconds: float = 60.0


class LLMConfig(BaseModel):
    """Separate model configs per architectural role (falls back to default)."""

    default: ModelEndpointConfig = Field(default_factory=ModelEndpointConfig)
    structurer: ModelEndpointConfig | None = None
    gateway: ModelEndpointConfig | None = None
    chat: ModelEndpointConfig | None = None
    rtir: ModelEndpointConfig | None = None
    vision: ModelEndpointConfig | None = None
    reflector: ModelEndpointConfig | None = None  # Nightly Backward Pass grade evolution

    def for_role(self, role: str) -> ModelEndpointConfig:
        override = getattr(self, role, None)
        if override is not None:
            return override
        if role == "vision" and self.chat is not None:
            return self.chat
        if role == "reflector" and self.gateway is not None:
            return self.gateway
        return self.default


class LivingProfileConfig(BaseModel):
    """T3 persistent living baseline synced into Scene State (no external weather API required)."""

    wakeup_lead_minutes: int = 40  # Wake up N minutes before first morning class/event
    weekday_latest_wakeup: str = "08:30"  # Latest wake-up on weekdays without early class
    weekend_wakeup: str = "09:30"  # Weekend wake-up time if no morning events
    sleep_start_hour: int = 23  # Quiet/sleeping hours start (23:30)
    sleep_end_hour: int = 7  # Quiet hours end (07:00)
    home_min_gap_minutes: int = 150  # Midday gap >= 150 min -> recommend going home; else dorm nap
    weekly_campus_run_target: int = 3  # Target campus runs per week
    favorite_breakfasts: list[str] = Field(
        default_factory=lambda: ["豆浆油条", "肉包配小米粥", "鸡蛋灌饼", "三明治配热牛奶", "热干面"]
    )
    favorite_lunches: list[str] = Field(
        default_factory=lambda: ["二食堂麻辣香锅", "一食堂滑蛋盖饭", "轻食鸡胸肉沙拉", "兰州牛肉面", "木桶饭"]
    )
    favorite_dinners: list[str] = Field(
        default_factory=lambda: ["三食堂黄焖鸡", "番茄鸡蛋面", "烤肉拌饭", "清淡小炒套餐", "水饺"]
    )
    dorm_notes: str = "回寝室步行约 8 分钟，适合短午休"
    home_notes: str = "回家单程约 30 分钟，适合下午无课或午休间隔大于 2.5 小时"


class OwnerConfig(BaseModel):
    """Owner identity and personal preferences across platforms."""

    name: str = "主人"
    timezone: str = "Asia/Shanghai"
    qq_id: str = ""
    qq_platform_id: str = "qq_onebot"
    weixin_user_id: str = ""
    weixin_platform_id: str = "weixin_personal"
    extra_owner_identities: list[str] = Field(
        default_factory=list,
        description="Optional extra 'platform_id:user_id' strings treated as direct user chat",
    )
    school_periods: dict[str, str] = Field(
        default_factory=lambda: {
            "第一节": "08:00-08:45",
            "第二节": "08:55-09:40",
            "第三节": "10:00-10:45",
            "第四节": "10:55-11:40",
            "第五节": "14:00-14:45",
            "第六节": "14:55-15:40",
            "第七节": "16:00-16:45",
            "第八节": "16:55-17:40",
            "第九节": "19:00-19:45",
            "第十节": "19:55-20:40",
        }
    )
    persona_notes: str = (
        "用户是一名高校学生，希望你在合适的时间主动提醒起床、饮食、去哪栋教学楼上课、"
        "中午回家还是回寝室休息、晚上是否去校园跑，并在课间或空闲时推荐有趣新闻/论文和值得参与的群聊讨论。"
    )
    living_profile: LivingProfileConfig = Field(default_factory=LivingProfileConfig)


class GroupFilterConfig(BaseModel):
    """Group chat deduplication, jielong filtering, and silence debounce settings."""

    silence_seconds: float = 300.0
    dedup_window_seconds: float = 600.0
    jielong_mode: str = "keep_header_only"
    min_jielong_lines: int = 3
    echo_words: list[str] = Field(
        default_factory=lambda: [
            "+1",
            "1",
            "收到",
            "好的",
            "好",
            "ok",
            "OK",
            "同上",
            "附议",
            "谢谢",
            "感谢",
            "哈哈",
            "哈哈哈",
            "哈哈哈哈",
            "6",
            "666",
            "？",
            "?",
        ]
    )
    max_buffer_messages: int = 200


class SlowLoopConfig(BaseModel):
    """Slow Loop (慢回路) BM25 structured storage for undated ideas & >7-day schedules/tasks."""

    enabled: bool = True
    horizon_days: int = 7
    default_top_k: int = 8
    auto_promote_interval_seconds: float = 3600.0


class ScheduleImporterConfig(BaseModel):
    """Auto-importer for timetables (课表) & shift schedules (排班表) from Excel or images."""

    default_reminder_lead_minutes: int = 20
    default_expand_weeks: int = 8
    max_excel_rows_per_sheet: int = 200
    default_priority: float = 9.0


class GatingEvolutionConfig(BaseModel):
    """Two-phase gating evolution, anti-avalanche caps, and source trust firewall config."""

    nightly_reflection_time: str = "23:30"  # Long-beat Backward Pass schedule
    auto_publish_inferred_rules: bool = False  # Option A: inferred rules enter `candidate` awaiting approval
    max_active_rules: int = 25
    max_starvation_boost: float = 2.0  # Cap wait-time inflation so news never impersonates emergencies
    non_urgent_push_cooldown_minutes: float = 25.0
    digest_max_items: int = 3
    urgent_pierce_threshold: float = 8.5  # Urgency >= 8.5 pierces urgent_only hard guardrail
    t1_daily_reset_hour: int = 4  # T1 daily state resets cleanly at 04:00
    t2_default_ttl_days: int = 5  # Default T2 multi-day state TTL (e.g. illness / exam week)
    t2_renewal_lead_hours: float = 24.0  # Emit gentle renewal check within 24h of T2 expiry
    max_t0_ambient_notes: int = 8
    high_trust_senders: list[str] = Field(
        default_factory=lambda: ["班长", "学委", "辅导员", "班主任", "教务处", "助教", "导师"]
    )
    high_trust_channels: list[str] = Field(default_factory=list)


class GatewayConfig(BaseModel):
    """Message Gateway fixed system prompt, seed grading criteria, and buffer parameters."""

    weight_boost_on_return: float = 1.5
    max_weight: float = 10.0
    return_retry_minutes: float = 30.0
    max_returns: int = 8
    buffer_ttl_hours: float = 36.0
    buffer_sweep_interval_seconds: float = 900.0
    sweep_batch_size: int = 5
    # Permanently fixed system prompt (NEVER mutated by Backward Pass!)
    system_prompt: str = (
        "你是个人生活调度Agent的消息网关（Message Gateway）唯一主动出口决策核心。\n"
        "你的职责是严格基于【固定系统协议】+【当前只读版分级器 grade】+【全局唯一同步时间表 time table】，"
        "对每一条结构化缓冲条目、定时指令、动态跟进申请或状态续期请求做出四选一仲裁：\n"
        "1. ACTIVE_PUSH：立即推送给工作状态区开启主动对话。\n"
        "2. SCENE_STATE：不打扰用户，静默写入分级场景状态（注明 T0 瞬时态 / T1 当日态 / T2 跨日持续态）。\n"
        "3. DEFER_TO_WINDOW：当前时机不合适（如正在上课、有更紧急事务），暂存缓冲区并附带封顶饥饿补偿，等待下次空闲/下课跃迁窗口。\n"
        "4. ARCHIVE_OR_SLOW：低价值噪音直接归档；若属长尾优质资料/远期想法，可标记 archive_to_slow=true 转存慢回路 BM25。"
    )
    # Initial seed for grade_v1; evolves atomically ONLY during nightly Backward Pass!
    grading_criteria: str = (
        "【基础分级标准 Grade v1】\n"
        "- ACTIVE_PUSH（主动消息）：\n"
        "  1) 临近的硬性课程/排班/考试/DDL/调课/紧急通知与动态起床节点；\n"
        "  2) 到点的午休/晚间校园跑决策及经过时间表仲裁确认可发的主动跟进请求；\n"
        "  3) 处于课间/空闲窗口（open）且综合价值高的新闻、新论文或值得参与的群聊讨论；\n"
        "  4) 需要主人确认的疑似课表/日程变动（needs_confirmation=true）。\n"
        "- SCENE_STATE（状态更新）：\n"
        "  1) 非紧急背景信息、适合下次聊天顺带提及的状态更新。\n"
        "- DEFER_TO_WINDOW（退回等待空闲窗）：\n"
        "  1) 用户正在上课/考试/专注/有更紧急安排时的非紧急消息、新闻、论文或普通跟进。\n"
        "- ARCHIVE_OR_SLOW（归档或转慢回路）：\n"
        "  1) 毫无价值的水群/重复资讯直接丢弃；值得以后慢慢看的深度论文/长尾想法转慢回路 BM25。"
    )


class WorkingStateConfig(BaseModel):
    """Working State Area + RTIR timing config (Fast Loop plain context + Slow Loop BM25 tools)."""

    max_context_messages: int = 40
    rtir_silence_seconds: float = 180.0
    system_prompt: str = (
        "你是用户的贴身智能生活管家，风格自然、简洁、有温度、不啰嗦。"
        "你只需要在正确的时间告诉用户要干什么，帮用户把各种鸡毛蒜皮的小事整理妥当，主动提醒、主动跟进、主动规划。\n"
        "重要行为规范：\n"
        "1. 【快回路 4 条直通指令】：view_timetable（查看全局 7 天内时间表）、modify_timetable（删改时间表）、"
        "view_buffer（查看缓冲区）、modify_buffer（删改缓冲区）。\n"
        "2. 【慢回路 BM25 工具】：没有确定时间的想法/空闲活动清单与一周之外的未来安排存放在慢回路 BM25 中，不直接塞进明文上下文。"
        "当用户问空闲干点啥、回忆想法或规划未来几周时，调用 search_slow_memory 检索；写入或定下时间升入时间表调用 modify_slow_memory。\n"
        "3. 【实时规划与主动跟进工具】：当用户让你帮着规划今天/本周工作或找出空闲时段时，先调用 plan_free_slots 精准算出空档并结合慢回路待办排期；"
        "当用户约定待会/稍后/某截止前要做的事（如拿快递、交作业、去跑步），调用 manage_followups 建立动态跟进意图（系统会结合全局时间表自动测算最佳跟进时机并经网关仲裁）。\n"
        "4. 【动态门控分级器查看与偏好反馈】：用户询问当前推送规则、批准/驳回夜间候选规则、或明确提出推送偏好/临时勿扰时，调用 view_and_manage_grade。\n"
        "   注意：长期偏好反馈会记入反馈账本并于每晚 23:30 统一演化新版 grade（下行绝不中途乱改规则）；只有“今晚几小时内勿扰”等临时硬开关会即时生效并在到期后自动解除。\n"
        "5. 如果用户的日程修改缺少精确时间（如只说“改到晚上”），先礼貌反问具体几点/第几节再修改。\n"
        "6. 主动推送消息时直接切入重点，若是一次合并了多件事的课间简报，用清晰短句分点说出。"
    )


class DeliveryConfig(BaseModel):
    """Proactive QQ delivery + 5-min no-reply Official Personal WeChat direct failover."""

    failover_timeout_seconds: float = 300.0
    failover_prefix: str = "【QQ 5分钟未回复·微信提醒】\n"
    failover_immediately_on_qq_error: bool = True
    # Option B selected: 0.0 means ALL proactive pushes (after bundling) escalate to WeChat after 5m!
    failover_min_urgency: float = 0.0


class PassiveTriggerSlot(BaseModel):
    """A fixed daily time slot emitting a structured directive to Message Gateway."""

    name: str
    time: str
    weekdays: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    directive: str
    priority: float = 8.5
    enabled: bool = True


class PassiveTriggersConfig(BaseModel):
    enabled: bool = True
    tick_interval_seconds: float = 20.0
    timetable_reminder_lead_minutes: int = 15
    dynamic_wakeup_enabled: bool = True
    transition_digest_enabled: bool = True
    slots: list[PassiveTriggerSlot] = Field(
        default_factory=lambda: [
            PassiveTriggerSlot(
                name="lunch_and_nap",
                time="11:45",
                directive=(
                    "中午提醒：结合 T3 食堂偏好、T1 今日已吃餐食（避免重复）以及下午课程真实间隔，"
                    "建议中午吃什么，并明确建议回家还是回寝室午休。"
                ),
                priority=9.0,
            ),
            PassiveTriggerSlot(
                name="afternoon_class",
                time="13:40",
                directive="下午上课前提醒：检查下午全局时间表，告知要去哪栋教学楼上课及注意事项。",
                priority=9.0,
            ),
            PassiveTriggerSlot(
                name="evening_campus_run",
                time="19:30",
                directive=(
                    "晚间安排提醒：结合 T2 持续身体状态、T1 今日状态、T3 每周校园跑目标和今晚时间表，"
                    "建议今晚要不要去校园跑以及晚间待办。"
                ),
                priority=8.5,
            ),
            PassiveTriggerSlot(
                name="night_wrapup",
                time="22:45",
                directive="睡前梳理：简短总结明天全局时间表中的关键安排（尤其是早课或DDL），提醒早点休息。",
                priority=8.0,
            ),
        ]
    )


class RSSFeedItem(BaseModel):
    name: str
    url: str


class RSSPluginConfig(BaseModel):
    enabled: bool = False
    poll_interval_seconds: float = 1800.0
    max_items_per_feed: int = 5
    feeds: list[RSSFeedItem] = Field(
        default_factory=lambda: [
            RSSFeedItem(name="Hacker News", url="https://hnrss.org/frontpage"),
        ]
    )


class ArxivPluginConfig(BaseModel):
    enabled: bool = False
    poll_interval_seconds: float = 3600.0
    categories: list[str] = Field(default_factory=lambda: ["cs.AI", "cs.CL"])
    keywords: list[str] = Field(default_factory=list)
    max_results: int = 8


class AstrBotModuleConfig(BaseModel):
    """AstrBot platform adapter module config (embedded in-process, NOT an AstrBot plugin)."""

    enabled: bool = True
    astrbot_repo_path: str = "./vendor/AstrBot"
    astrbot_data_root: str = "./data/astrbot"
    platform_settings: dict[str, Any] = Field(default_factory=lambda: {"unique_session": True})
    platforms: list[dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "id": "qq_onebot",
                "type": "aiocqhttp",
                "enable": False,
                "ws_reverse_host": "0.0.0.0",
                "ws_reverse_port": 6199,
                "ws_reverse_token": "",
            },
            {
                "id": "weixin_personal",
                "type": "weixin_oc",
                "enable": False,
                "weixin_oc_base_url": "https://ilinkai.weixin.qq.com",
                "weixin_oc_cdn_base_url": "https://novac2c.cdn.weixin.qq.com/c2c",
                "weixin_oc_bot_type": "3",
                "weixin_oc_token": "",
                "weixin_oc_account_id": "",
                "weixin_oc_qr_poll_interval": 1,
                "weixin_oc_long_poll_timeout_ms": 35000,
                "weixin_oc_api_timeout_ms": 120000,
            },
        ]
    )


class WebConfig(BaseModel):
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765


class AppConfig(BaseModel):
    data_dir: str = "./data"
    plugins_dir: str = "./plugins"
    db_path: str = "./data/life_agent.db"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    owner: OwnerConfig = Field(default_factory=OwnerConfig)
    group_filter: GroupFilterConfig = Field(default_factory=GroupFilterConfig)
    slow_loop: SlowLoopConfig = Field(default_factory=SlowLoopConfig)
    schedule_importer: ScheduleImporterConfig = Field(default_factory=ScheduleImporterConfig)
    gating_evolution: GatingEvolutionConfig = Field(default_factory=GatingEvolutionConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    working_state: WorkingStateConfig = Field(default_factory=WorkingStateConfig)
    delivery: DeliveryConfig = Field(default_factory=DeliveryConfig)
    passive_triggers: PassiveTriggersConfig = Field(default_factory=PassiveTriggersConfig)
    astrbot: AstrBotModuleConfig = Field(default_factory=AstrBotModuleConfig)
    rss: RSSPluginConfig = Field(default_factory=RSSPluginConfig)
    arxiv: ArxivPluginConfig = Field(default_factory=ArxivPluginConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        expanded = _expand_env(raw)
        return cls.model_validate(expanded)

    def save_yaml(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(
                self.model_dump(mode="json"),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

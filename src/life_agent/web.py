"""Lightweight local FastAPI Control Panel (v2: Two-Phase Gating, Tiered Scene, FollowUps, BM25, Importer)."""

from __future__ import annotations

import base64
import io
from typing import Any

import qrcode
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .models import (
    ChannelType,
    FollowUpIntent,
    FollowUpStatus,
    PassiveTriggerPayload,
    SlowMemoryItem,
    SlowMemoryKind,
    SlowMemoryStatus,
)
from .orchestrator import LifeAgentApp


class ChatRequest(BaseModel):
    text: str
    channel: str = "web"


class IngestRequest(BaseModel):
    text: str
    platform_type: str = "webhook"
    platform_id: str = "webhook"
    channel_type: ChannelType = ChannelType.WEBHOOK
    channel_id: str = "default"
    channel_name: str = "外部Webhook"
    sender_id: str = "external"
    sender_name: str = "外部平台"
    is_from_owner: bool = False
    raw_meta: dict[str, Any] = {}


class FireTriggerRequest(BaseModel):
    slot_name: str | None = None
    directive: str | None = None
    priority: float = 9.0


class SlowMemoryCreateRequest(BaseModel):
    kind: SlowMemoryKind = SlowMemoryKind.IDEA
    title: str
    summary: str = ""
    details: str = ""
    tags: list[str] = []
    planned_start: str | None = None
    planned_end: str | None = None
    location: str = ""
    category: str = "idea"
    estimated_minutes: int | None = None
    energy_level: str = ""


class FollowUpCreateRequest(BaseModel):
    topic: str
    context_hint: str = ""
    intent_type: str = "deadline_backtrack"
    deadline_at: str | None = None
    anchor_event_id: str | None = None
    safety_margin_minutes: int = 60
    priority: float = 8.0


def _make_qr_data_url(content: str) -> str:
    img = qrcode.make(content)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def create_web_app(agent: LifeAgentApp) -> FastAPI:
    app = FastAPI(title="Life Agent v2 Control Panel", version="0.3.0")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _DASHBOARD_HTML

    @app.get("/api/state")
    async def get_state() -> dict[str, Any]:
        return await agent.get_state_snapshot()

    @app.post("/api/chat")
    async def post_chat(req: ChatRequest) -> dict[str, Any]:
        return await agent.chat_local(req.text, channel=req.channel)

    @app.post("/api/ingest")
    async def post_ingest(req: IngestRequest) -> dict[str, Any]:
        evt = await agent.webhook_plugin.push(
            text=req.text,
            platform_type=req.platform_type,
            platform_id=req.platform_id,
            channel_type=req.channel_type,
            channel_id=req.channel_id,
            channel_name=req.channel_name,
            sender_id=req.sender_id,
            sender_name=req.sender_name,
            is_from_owner=req.is_from_owner,
            raw_meta=req.raw_meta,
        )
        return {"ok": True, "event_id": evt.id}

    # --- v2 Two-Phase Gating & Backward Pass APIs ---
    @app.post("/api/backward/short-beat")
    async def post_short_beat() -> dict[str, Any]:
        res = await agent.passive_triggers.run_short_beat_backward()
        return {"ok": True, "result": res}

    @app.post("/api/backward/nightly-reflect")
    async def post_nightly_reflect() -> dict[str, Any]:
        res = await agent.policy.run_nightly_reflection()
        return {"ok": True, "result": res}

    @app.post("/api/grade/candidates/{rule_id}/approve")
    async def post_approve_candidate(rule_id: str) -> dict[str, Any]:
        snap = await agent.policy.approve_candidate_rule(rule_id)
        if snap is None:
            raise HTTPException(status_code=404, detail=f"Candidate rule not found: {rule_id}")
        return {"ok": True, "published_version": snap.version}

    @app.post("/api/grade/candidates/{rule_id}/reject")
    async def post_reject_candidate(rule_id: str) -> dict[str, Any]:
        ok = await agent.policy.reject_candidate_rule(rule_id)
        return {"ok": ok, "rule_id": rule_id}

    @app.post("/api/grade/rollback/{version}")
    async def post_rollback_grade(version: int) -> dict[str, Any]:
        snap = await agent.storage.activate_grade_version(version)
        if snap is None:
            raise HTTPException(status_code=404, detail=f"Grade version v{version} not found")
        return {"ok": True, "active_version": snap.version}

    # --- v2 Dynamic Follow-Ups APIs ---
    @app.post("/api/followups")
    async def post_followup(req: FollowUpCreateRequest) -> dict[str, Any]:
        fu = FollowUpIntent(
            topic=req.topic.strip(),
            context_hint=req.context_hint.strip(),
            intent_type=req.intent_type,
            deadline_at=req.deadline_at,
            anchor_event_id=req.anchor_event_id,
            safety_margin_minutes=req.safety_margin_minutes,
            priority=req.priority,
            source="web_ui",
        )
        saved = await agent.followup_calc.register_followup(fu)
        return {"ok": True, "followup": saved.model_dump(mode="json")}

    @app.post("/api/followups/{followup_id}/done")
    async def post_followup_done(followup_id: str) -> dict[str, Any]:
        fu = await agent.storage.get_followup(followup_id)
        if fu is None:
            raise HTTPException(status_code=404, detail=f"Follow-up not found: {followup_id}")
        fu.status = FollowUpStatus.DONE
        await agent.storage.upsert_followup(fu)
        return {"ok": True, "followup_id": followup_id}

    # --- Slow Loop BM25 APIs ---
    @app.get("/api/slow-memory/search")
    async def get_slow_search(
        q: str = "",
        kind: str = "all",
        category: str | None = None,
        max_minutes: int | None = None,
        top_k: int = 8,
    ) -> dict[str, Any]:
        hits = await agent.slow_store.search(
            query=q,
            kind=kind,
            category=category,
            max_minutes=max_minutes,
            top_k=top_k,
        )
        return {"ok": True, "retriever": "bm25", "count": len(hits), "results": hits}

    @app.post("/api/slow-memory")
    async def post_slow_memory(req: SlowMemoryCreateRequest) -> dict[str, Any]:
        item = SlowMemoryItem(
            kind=req.kind,
            title=req.title.strip(),
            summary=(req.summary or req.title).strip(),
            details=req.details.strip(),
            tags=[t.strip() for t in req.tags if t.strip()],
            planned_start=req.planned_start,
            planned_end=req.planned_end,
            location=req.location.strip(),
            category=req.category.strip(),
            estimated_minutes=req.estimated_minutes,
            energy_level=req.energy_level.strip(),
            source="web_ui",
            status=SlowMemoryStatus.ACTIVE,
        )
        saved = await agent.slow_store.add_or_update(item)
        return {"ok": True, "item": saved.model_dump(mode="json")}

    @app.delete("/api/slow-memory/{item_id}")
    async def delete_slow_memory(item_id: str) -> dict[str, Any]:
        ok = await agent.slow_store.delete(item_id)
        return {"ok": ok, "item_id": item_id}

    @app.post("/api/slow-memory/promote")
    async def post_slow_promote() -> dict[str, Any]:
        promoted = await agent.slow_store.promote_due_items(
            horizon_days=agent.config.slow_loop.horizon_days,
            default_lead_minutes=agent.config.schedule_importer.default_reminder_lead_minutes,
            default_priority=agent.config.schedule_importer.default_priority,
        )
        return {
            "ok": True,
            "promoted_count": len(promoted),
            "entries": [e.model_dump(mode="json") for e in promoted],
        }

    # --- Schedule & Shift Auto-Importer (Excel / CSV / Image) ---
    @app.post("/api/import/schedule")
    async def post_import_schedule(
        file: UploadFile = File(...),
        schedule_type: str = Form("auto"),
        current_week: str = Form(""),
        semester_start_date: str = Form(""),
        extra_instructions: str = Form(""),
    ) -> dict[str, Any]:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="上传文件为空")
        cur_week_int: int | None = None
        if current_week.strip():
            try:
                cur_week_int = int(current_week.strip())
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="current_week 必须为整数") from exc
        try:
            result = await agent.schedule_importer.import_bytes(
                data=raw,
                filename=file.filename or "upload.xlsx",
                schedule_type=schedule_type,
                current_week=cur_week_int,
                semester_start_date=semester_start_date.strip() or None,
                extra_instructions=extra_instructions.strip(),
            )
            await agent.followup_calc.recalculate_all_open()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "result": result.model_dump(mode="json")}

    @app.post("/api/triggers/fire")
    async def post_fire_trigger(req: FireTriggerRequest) -> dict[str, Any]:
        if req.slot_name:
            slot = next(
                (s for s in agent.config.passive_triggers.slots if s.name == req.slot_name),
                None,
            )
            if slot is None:
                raise HTTPException(status_code=404, detail=f"Slot not found: {req.slot_name}")
            await agent.passive_triggers.fire_slot(slot)
            return {"ok": True, "fired": slot.name}
        if req.directive:
            payload = PassiveTriggerPayload(
                trigger_name="manual_trigger",
                directive=req.directive,
                priority=req.priority,
                urgency=req.priority,
            )
            decision = await agent.gateway.evaluate_passive_trigger(payload)
            return {"ok": True, "decision": decision.model_dump(mode="json")}
        raise HTTPException(status_code=400, detail="Provide slot_name or directive")

    @app.post("/api/groups/flush")
    async def post_flush_groups() -> dict[str, Any]:
        await agent.router.group_buffer.flush_all()
        return {"ok": True}

    @app.post("/api/rtir/flush")
    async def post_flush_rtir() -> dict[str, Any]:
        res = await agent.working_state.flush_rtir_now()
        return {"ok": True, "result": res}

    @app.post("/api/gateway/sweep")
    async def post_gateway_sweep() -> dict[str, Any]:
        count = await agent.gateway.sweep_due_buffer_items()
        return {"ok": True, "evaluated": count}

    @app.post("/api/weixin/qr")
    async def post_weixin_qr() -> dict[str, Any]:
        data = await agent.astrbot_bridge.request_weixin_qr()
        data["qr_data_url"] = _make_qr_data_url(data["qrcode_img_content"])
        return data

    @app.get("/api/weixin/qr/poll")
    async def get_weixin_qr_poll(qrcode: str) -> dict[str, Any]:
        return await agent.astrbot_bridge.poll_weixin_qr(qrcode=qrcode)

    return app


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Life Agent v2 控制台</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --muted:#94a3b8; --text:#f1f5f9; --accent:#38bdf8; --ok:#22c55e; --warn:#f59e0b; --purple:#a855f7; --border:#334155; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,sans-serif; background:var(--bg); color:var(--text); }
  header { padding:14px 20px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; }
  header h1 { margin:0; font-size:17px; }
  .toolbar { display:flex; gap:8px; flex-wrap:wrap; }
  button { background:#0284c7; color:#fff; border:none; border-radius:6px; padding:6px 11px; cursor:pointer; font-size:12px; }
  button.sec { background:#334155; }
  button.pur { background:#7e22ce; }
  button.ok { background:#15803d; }
  button:hover { opacity:0.9; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(370px,1fr)); gap:14px; padding:16px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px; display:flex; flex-direction:column; gap:10px; }
  .card h2 { margin:0; font-size:14px; color:var(--accent); display:flex; justify-content:space-between; align-items:center; }
  .list { max-height:260px; overflow:auto; display:flex; flex-direction:column; gap:8px; }
  .item { background:#0f172a; border:1px solid var(--border); border-radius:8px; padding:9px; font-size:12.5px; }
  .meta { color:var(--muted); font-size:11px; margin-top:4px; }
  .badge { display:inline-block; padding:2px 6px; border-radius:4px; font-size:11px; background:#334155; margin-right:4px; }
  .badge.ok { background:#14532d; color:#86efac; }
  .badge.warn { background:#78350f; color:#fde68a; }
  .badge.pur { background:#581c87; color:#e9d5ff; }
  textarea, input, select { width:100%; background:#0f172a; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:7px; font-size:13px; }
  pre { margin:0; background:#0f172a; padding:10px; border-radius:8px; font-size:11.5px; overflow:auto; max-height:220px; white-space:pre-wrap; }
  .row { display:flex; gap:8px; align-items:center; }
</style>
</head>
<body>
<header>
  <h1>Life Agent v2 控制台（上下行解耦 · T0–T3 状态 · 夜间演化 grade · 动态跟进 · 慢回路 BM25）</h1>
  <div class="toolbar">
    <button onclick="refresh()">刷新状态</button>
    <button class="sec" onclick="apiPost('/api/groups/flush')">汇总群聊</button>
    <button class="sec" onclick="apiPost('/api/rtir/flush')">执行 RTIR</button>
    <button class="sec" onclick="apiPost('/api/backward/short-beat')">短节拍上行校准</button>
    <button class="pur" onclick="apiPost('/api/backward/nightly-reflect')">模拟 23:30 夜间演化 grade</button>
    <button class="sec" onclick="startWeixinQR()">官方个人微信扫码</button>
  </div>
</header>

<div id="qrBox" style="display:none; padding:14px 20px; background:#1e293b; border-bottom:1px solid #334155; align-items:center; gap:16px;">
  <img id="qrImg" style="width:160px;height:160px;background:#fff;padding:6px;border-radius:8px;" />
  <div>
    <div id="qrStatus" style="font-weight:600;">请使用手机微信扫码确认登录官方个人微信 (weixin_oc / iLink)</div>
    <div class="meta">扫码成功后，请给微信机器人发一条任意消息以初始化 context_token。</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>工作状态区 · 直接对话（9 个结构化工具）</h2>
    <div id="chatList" class="list"></div>
    <div class="row">
      <input id="chatInput" placeholder="例如：帮我规划下午空档 / 今晚8点前提醒取快递 / 看看当前 grade 分级规则" onkeydown="if(event.key==='Enter')sendChat()" />
      <button onclick="sendChat()" style="white-space:nowrap;">发送</button>
    </div>
  </div>

  <div class="card">
    <h2>版本化分级器 grade（只读冻结）& 候选规则审批 <span id="gradeVer" class="badge pur">v1</span></h2>
    <pre id="gradeText"></pre>
    <div class="meta" style="font-weight:600;">夜间推测候选规则 (candidate)</div>
    <div id="candList" class="list" style="max-height:130px;"></div>
  </div>

  <div class="card">
    <h2>四级保质期场景状态 (T0 瞬时 / T1 当日04:00 / T2 跨日续期 / T3 基线)</h2>
    <pre id="sceneJson">{}</pre>
  </div>

  <div class="card">
    <h2>动态主动跟进队列 (Follow-Ups · 100% 过网关)</h2>
    <div id="fuList" class="list"></div>
    <div class="meta" style="font-weight:600;">下行反馈账本 (feedback_log · 只追加)</div>
    <div id="fbList" class="list" style="max-height:120px;"></div>
  </div>

  <div class="card">
    <h2>全局唯一时间表 (<=7天) & 被动触发器 <span id="ttCount" class="meta"></span></h2>
    <div id="ttList" class="list"></div>
    <div class="meta" style="font-weight:600;">显式定时被动触发器 (scheduled_triggers)</div>
    <div id="trigList" class="list" style="max-height:110px;"></div>
  </div>

  <div class="card">
    <h2>课表 / 排班表自动导入（Excel / 图片）</h2>
    <input type="file" id="schedFile" accept=".xlsx,.xls,.csv,.png,.jpg,.jpeg,.webp,.bmp,.gif" />
    <div class="row">
      <select id="schedType">
        <option value="auto">自动识别（课表/排班表）</option>
        <option value="timetable">课表 (Timetable)</option>
        <option value="shift">排班表 (Shift)</option>
      </select>
      <input id="schedWeek" placeholder="当前第几周（可选）" />
    </div>
    <div class="row">
      <input id="schedSemStart" placeholder="第1周周一日期（可选）" />
      <button onclick="uploadSchedule()" style="white-space:nowrap;">解析并 7 天分流</button>
    </div>
    <div id="importResult" class="meta">7 天内写入全局时间表 + 被动触发器；7 天外录入慢回路 BM25。</div>
  </div>

  <div class="card">
    <h2>慢回路 · BM25 结构化记忆 <span id="slowCount" class="meta"></span></h2>
    <div class="row">
      <input id="slowQuery" placeholder="BM25 关键词检索..." onkeydown="if(event.key==='Enter')searchSlow()" />
      <button class="pur" onclick="searchSlow()" style="white-space:nowrap;">BM25 检索</button>
    </div>
    <div id="slowList" class="list"></div>
  </div>

  <div class="card">
    <h2>缓冲区 Buffer (双轴分 + 信任分级) & 容灾队列</h2>
    <div id="bufList" class="list" style="max-height:150px;"></div>
    <div id="pushList" class="list" style="max-height:110px;"></div>
  </div>
</div>

<script>
async function apiPost(url, body={}) {
  const r = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) });
  const j = await r.json();
  await refresh();
  return j;
}

async function sendChat() {
  const inp = document.getElementById('chatInput');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  await apiPost('/api/chat', { text, channel: 'web' });
}

async function uploadSchedule() {
  const fileEl = document.getElementById('schedFile');
  if (!fileEl.files || !fileEl.files[0]) { alert('请先选择文件'); return; }
  const fd = new FormData();
  fd.append('file', fileEl.files[0]);
  fd.append('schedule_type', document.getElementById('schedType').value);
  fd.append('current_week', document.getElementById('schedWeek').value);
  fd.append('semester_start_date', document.getElementById('schedSemStart').value);
  document.getElementById('importResult').textContent = '解析中...';
  const resp = await fetch('/api/import/schedule', { method:'POST', body:fd });
  const data = await resp.json();
  if (!resp.ok) { document.getElementById('importResult').textContent = '失败: ' + data.detail; return; }
  const r = data.result;
  document.getElementById('importResult').textContent =
    `导入完成：7天内 ${r.fast_loop_timetable_count} 条入时间表（${r.passive_triggers_written} 条被动触发器），7天外 ${r.slow_loop_bm25_count} 条入慢回路 BM25。`;
  await refresh();
}

async function searchSlow() {
  const q = document.getElementById('slowQuery').value.trim();
  if (!q) { await refresh(); return; }
  const data = await (await fetch('/api/slow-memory/search?q=' + encodeURIComponent(q))).json();
  renderSlowItems(data.results, true);
}

function renderSlowItems(items, showScore=false) {
  document.getElementById('slowCount').textContent = items.length + ' 条';
  document.getElementById('slowList').innerHTML = items.map(s =>
    `<div class="item"><span class="badge pur">${s.kind}</span>${showScore?`<span class="badge ok">BM25=${s.bm25_score}</span>`:`<span class="badge">${s.status}</span>`}<b>${s.title}</b>
     <div>${s.summary||''}</div><div class="meta">${s.planned_start||'无确定时间'} ${s.estimated_minutes?'| ~'+s.estimated_minutes+'m':''}</div></div>`
  ).join('') || '<div class="meta">暂无慢回路条目</div>';
}

let qrPollTimer = null;
async function startWeixinQR() {
  document.getElementById('qrBox').style.display = 'flex';
  const data = await (await fetch('/api/weixin/qr', { method:'POST' })).json();
  document.getElementById('qrImg').src = data.qr_data_url;
  if (qrPollTimer) clearInterval(qrPollTimer);
  qrPollTimer = setInterval(async () => {
    const res = await (await fetch('/api/weixin/qr/poll?qrcode=' + encodeURIComponent(data.qrcode))).json();
    if (res.status === 'created') {
      clearInterval(qrPollTimer);
      document.getElementById('qrStatus').textContent = '微信官方直连登录成功！User ID: ' + (res.weixin_oc_user_id || '');
      refresh();
    }
  }, 2000);
}

async function refresh() {
  const data = await (await fetch('/api/state')).json();

  document.getElementById('gradeVer').textContent = 'grade_v' + data.grade.active.version;
  document.getElementById('gradeText').textContent = data.grade.active.grade_text;
  document.getElementById('candList').innerHTML = (data.grade.candidates || []).map(c =>
    `<div class="item"><span class="badge warn">candidate</span><b>[${c.scope_type}:${c.scope_key}]</b> ${c.instruction}
     <div class="row" style="margin-top:6px;">
       <button class="ok" onclick="apiPost('/api/grade/candidates/${c.id}/approve')">批准启用</button>
       <button class="sec" onclick="apiPost('/api/grade/candidates/${c.id}/reject')">驳回</button>
     </div></div>`
  ).join('') || '<div class="meta">暂无待确认候选规则</div>';

  document.getElementById('sceneJson').textContent = JSON.stringify(data.scene_state, null, 2);

  document.getElementById('fuList').innerHTML = (data.follow_ups || []).map(f =>
    `<div class="item"><span class="badge ${f.status==='done'?'ok':'warn'}">${f.status}</span><b>${f.topic}</b>
     <div class="meta">测算时机: ${f.computed_check_at||'-'} | 截止: ${f.deadline_at||'-'} | 回访: ${f.check_count}/${f.max_checks}</div>
     ${f.status==='open'?`<button class="ok" style="margin-top:4px;" onclick="apiPost('/api/followups/${f.id}/done')">标记完成</button>`:''}</div>`
  ).join('') || '<div class="meta">暂无跟进意图</div>';

  document.getElementById('fbList').innerHTML = (data.feedback_log || []).map(fb =>
    `<div class="item"><span class="badge">${fb.event_type}</span><span class="badge ${fb.processed_at?'ok':'warn'}">${fb.processed_at?'已复盘':'待23:30复盘'}</span> ${fb.raw_text||fb.source_key}</div>`
  ).join('') || '<div class="meta">反馈账本为空</div>';

  document.getElementById('ttCount').textContent = data.timetable.length + ' 条';
  document.getElementById('ttList').innerHTML = data.timetable.map(e =>
    `<div class="item"><b>${e.title}</b> <span class="badge">${e.category}</span><div class="meta">${e.start_time}${e.end_time?' ~ '+e.end_time:''} ${e.location?'@ '+e.location:''}</div></div>`
  ).join('') || '<div class="meta">暂无时间表条目</div>';

  document.getElementById('trigList').innerHTML = (data.scheduled_triggers || []).map(t =>
    `<div class="item"><span class="badge ${t.fired?'ok':'warn'}">${t.fired?'已触发':'待触发'}</span><b>${t.trigger_at}</b><div class="meta">${t.directive}</div></div>`
  ).join('') || '<div class="meta">暂无定时被动触发器</div>';

  if (!document.getElementById('slowQuery').value.trim()) renderSlowItems(data.slow_memory || [], false);

  const allBuf = [...data.buffer.urti, ...data.buffer.message, ...data.buffer.news].sort((a,b) => b.weight - a.weight);
  document.getElementById('bufList').innerHTML = allBuf.map(b =>
    `<div class="item"><span class="badge">${b.zone.toUpperCase()}</span><span class="badge warn">v=${b.intrinsic_value}/u=${b.urgency}</span><span class="badge">${b.trust_tier}</span><b>${b.title}</b><div>${b.summary}</div></div>`
  ).join('') || '<div class="meta">缓冲区为空</div>';

  document.getElementById('pushList').innerHTML = data.pending_pushes.map(p =>
    `<div class="item"><span class="badge ${p.status==='acknowledged'?'ok':'warn'}">${p.status}</span> ${p.content.slice(0,80)}</div>`
  ).join('') || '<div class="meta">暂无主动推送记录</div>';

  document.getElementById('chatList').innerHTML = data.context.map(c =>
    `<div class="item"><b>${c.role==='user'?'你':'Agent'}</b> <span class="badge">${c.channel}</span> ${c.parsed_by_rtir?'<span class="badge ok">RTIR</span>':''}<div style="white-space:pre-wrap;margin-top:4px;">${c.content}</div></div>`
  ).join('') || '<div class="meta">暂无对话记录</div>';
}
refresh();
</script>
</body>
</html>
"""

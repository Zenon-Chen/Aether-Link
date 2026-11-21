"""CLI entry point for Life Agent (independent process)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import click
import qrcode
import uvicorn
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .config import AppConfig
from .orchestrator import LifeAgentApp
from .web import create_web_app

console = Console()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
def main() -> None:
    """Life Agent — 独立进程主动生活调度 Agent（快慢双回路 + 课表/排班自动导入）."""


@main.command()
@click.option("--config", "config_path", default="config.yaml", help="Target YAML config path.")
@click.option("--force", is_flag=True, help="Overwrite existing config file.")
def init(config_path: str, force: bool) -> None:
    """Generate a starter config.yaml."""
    target = Path(config_path)
    if target.exists() and not force:
        console.print(f"[yellow]{target} already exists. Use --force to overwrite.[/yellow]")
        return
    cfg = AppConfig()
    cfg.save_yaml(target)
    console.print(f"[green]Generated starter config at {target.resolve()}[/green]")


@main.command()
@click.option("--config", "config_path", default="config.yaml", help="Path to config.yaml.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def run(config_path: str, verbose: bool) -> None:
    """Run Life Agent as an independent daemon (+ optional local web dashboard)."""
    _configure_logging(verbose)
    cfg_file = Path(config_path).resolve()
    cfg = AppConfig.load(cfg_file)
    base_dir = cfg_file.parent if cfg_file.exists() else Path.cwd()

    async def _runner() -> None:
        agent = LifeAgentApp(cfg, base_dir=base_dir)
        await agent.start()
        try:
            if cfg.web.enabled:
                web_app = create_web_app(agent)
                uv_cfg = uvicorn.Config(
                    web_app,
                    host=cfg.web.host,
                    port=cfg.web.port,
                    log_level="info" if verbose else "warning",
                )
                server = uvicorn.Server(uv_cfg)
                console.print(
                    f"[bold green]Life Agent running![/bold green] "
                    f"Control Panel: http://{cfg.web.host}:{cfg.web.port}"
                )
                await server.serve()
            else:
                console.print("[bold green]Life Agent running (headless mode). Ctrl+C to stop.[/bold green]")
                await asyncio.Event().wait()
        finally:
            await agent.stop()

    try:
        asyncio.run(_runner())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down Life Agent...[/yellow]")


@main.command("import-schedule")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--config", "config_path", default="config.yaml", help="Path to config.yaml.")
@click.option(
    "--type",
    "schedule_type",
    type=click.Choice(["auto", "timetable", "shift"]),
    default="auto",
    help="Schedule kind hint.",
)
@click.option("--current-week", type=int, default=None, help="Current semester week number (e.g. 12).")
@click.option("--semester-start", default=None, help="Week 1 Monday date (YYYY-MM-DD).")
@click.option("--note", default="", help="Extra instructions for the LLM parser.")
def import_schedule(
    file_path: str,
    config_path: str,
    schedule_type: str,
    current_week: int | None,
    semester_start: str | None,
    note: str,
) -> None:
    """Import a timetable (课表) or shift schedule (排班表) from Excel (.xlsx/.xls/.csv) or image."""
    _configure_logging(False)
    cfg_file = Path(config_path).resolve()
    cfg = AppConfig.load(cfg_file)
    cfg.astrbot.enabled = False
    cfg.passive_triggers.enabled = False
    cfg.rss.enabled = False
    cfg.arxiv.enabled = False
    base_dir = cfg_file.parent if cfg_file.exists() else Path.cwd()

    async def _run_import() -> None:
        agent = LifeAgentApp(cfg, base_dir=base_dir)
        await agent.storage.init()
        try:
            res = await agent.import_schedule_file(
                file_path,
                schedule_type=schedule_type,
                current_week=current_week,
                semester_start_date=semester_start,
                extra_instructions=note,
            )
            console.print(f"[bold green]导入成功：{res.source_filename} ({res.input_kind})[/bold green]")
            console.print(
                f"• 总计解析：{res.total_extracted_events} 条（已跳过过去 {res.skipped_past_count} 条）\n"
                f"• 一周之内 → 快回路时间表：[cyan]{res.fast_loop_timetable_count}[/cyan] 条，"
                f"写入被动触发器：[cyan]{res.passive_triggers_written}[/cyan] 条\n"
                f"• 一周之外 → 慢回路 BM25：[magenta]{res.slow_loop_bm25_count}[/magenta] 条"
            )
            if res.warnings:
                console.print("[yellow]解析警告：[/yellow]")
                for w in res.warnings:
                    console.print(f"  - {w}")
        finally:
            await agent.llm.close()
            await agent.storage.close()

    asyncio.run(_run_import())


@main.command("search-slow")
@click.argument("query")
@click.option("--config", "config_path", default="config.yaml", help="Path to config.yaml.")
@click.option("--kind", default="all", help="Filter by kind: all|idea|future_event|future_task|plan|note.")
@click.option("--top-k", default=8, type=int, help="Max results.")
def search_slow(query: str, config_path: str, kind: str, top_k: int) -> None:
    """Test BM25 retrieval over Slow Loop memory from the CLI."""
    cfg_file = Path(config_path).resolve()
    cfg = AppConfig.load(cfg_file)
    base_dir = cfg_file.parent if cfg_file.exists() else Path.cwd()

    async def _search() -> None:
        agent = LifeAgentApp(cfg, base_dir=base_dir)
        await agent.storage.init()
        try:
            hits = await agent.slow_store.search(query=query, kind=kind, top_k=top_k)
            table = Table(title=f"慢回路 BM25 检索结果: {query}")
            table.add_column("BM25", justify="right")
            table.add_column("类型")
            table.add_column("标题")
            table.add_column("计划时间")
            table.add_column("摘要")
            for h in hits:
                table.add_row(
                    f"{h['bm25_score']:.2f}",
                    h["kind"],
                    h["title"],
                    h["planned_start"] or "-",
                    h["summary"][:60],
                )
            console.print(table)
        finally:
            await agent.storage.close()

    asyncio.run(_search())


@main.command("weixin-login")
@click.option("--config", "config_path", default="config.yaml", help="Path to config.yaml.")
@click.option("--save-png", default="data/weixin_qr.png", help="Save QR code image to this path.")
def weixin_login(config_path: str, save_png: str) -> None:
    """Log into Official Personal WeChat (weixin_oc / ilinkai.weixin.qq.com) via QR scan."""
    _configure_logging(False)
    cfg_file = Path(config_path).resolve()
    cfg = AppConfig.load(cfg_file)
    base_dir = cfg_file.parent if cfg_file.exists() else Path.cwd()

    async def _login() -> None:
        agent = LifeAgentApp(cfg, base_dir=base_dir)
        from .router.plugins.astrbot_bridge import bootstrap_astrbot_environment

        repo = agent.astrbot_bridge._resolve_path(cfg.astrbot.astrbot_repo_path)
        root = agent.astrbot_bridge._resolve_path(cfg.astrbot.astrbot_data_root)
        bootstrap_astrbot_environment(repo, root)
        from astrbot.core import astrbot_config

        agent.astrbot_bridge._astrbot_config = astrbot_config
        existing = astrbot_config.get("platform", [])
        if not isinstance(existing, list):
            existing = []
        astrbot_config["platform"] = agent.astrbot_bridge._merge_platform_configs(
            cfg.astrbot.platforms,
            existing,
        )
        astrbot_config.save_config()

        console.print("[cyan]Requesting Official Personal WeChat (iLink) login QR code...[/cyan]")
        qr_info = await agent.astrbot_bridge.request_weixin_qr()
        qr_content = qr_info["qrcode_img_content"]
        qr_obj = qrcode.QRCode(border=1)
        qr_obj.add_data(qr_content)
        qr_obj.make(fit=True)
        qr_obj.print_ascii(invert=True)

        png_path = (base_dir / save_png).resolve()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        qr_obj.make_image(fill_color="black", back_color="white").save(png_path)
        console.print(f"[green]QR code also saved to {png_path}[/green]")
        console.print("请打开手机微信扫一扫并在手机端确认登录...")

        while True:
            res = await agent.astrbot_bridge.poll_weixin_qr(qrcode=qr_info["qrcode"])
            status = res.get("status")
            if status == "created":
                console.print("[bold green]官方个人微信直连登录成功！[/bold green]")
                console.print(f"Account ID: {res.get('weixin_oc_account_id')}")
                console.print(f"User ID:    {res.get('weixin_oc_user_id')}")
                console.print(
                    "[yellow]提示：请现在用微信给刚绑定的机器人发送一条任意消息（例如“你好”），"
                    "以便缓存 context_token，确保 5 分钟 QQ 超时后微信主动提醒正常送达。[/yellow]"
                )
                break
            if status in ("expired", "denied", "error"):
                console.print(f"[red]登录未完成：{res.get('message') or status}[/red]")
                break
            await asyncio.sleep(max(1, int(qr_info.get("interval", 1))))

    asyncio.run(_login())


@main.command()
@click.option("--config", "config_path", default="config.yaml", help="Path to config.yaml.")
def chat(config_path: str) -> None:
    """Interactive local terminal chat against Life Agent Working State (for testing)."""
    _configure_logging(False)
    cfg_file = Path(config_path).resolve()
    cfg = AppConfig.load(cfg_file)
    cfg.astrbot.enabled = False
    cfg.passive_triggers.enabled = False
    cfg.web.enabled = False
    base_dir = cfg_file.parent if cfg_file.exists() else Path.cwd()

    async def _repl() -> None:
        agent = LifeAgentApp(cfg, base_dir=base_dir)
        await agent.start()
        console.print("[bold cyan]Life Agent 本地对话测试模式（输入 /rtir 立即触发解析，/quit 退出）[/bold cyan]")
        try:
            while True:
                user_text = await asyncio.to_thread(input, "\n你 > ")
                user_text = user_text.strip()
                if not user_text:
                    continue
                if user_text in ("/quit", "/exit"):
                    break
                if user_text == "/rtir":
                    res = await agent.working_state.flush_rtir_now()
                    console.print(f"[green]RTIR 解析结果：{res}[/green]")
                    continue
                out = await agent.chat_local(user_text, channel="cli")
                if out["tools_used"]:
                    names = ", ".join(t["name"] for t in out["tools_used"])
                    console.print(f"[dim]调用工具: {names}[/dim]")
                console.print(f"[bold green]Agent >[/bold green] {out['reply']}")
        finally:
            await agent.stop()

    asyncio.run(_repl())


if __name__ == "__main__":
    main()

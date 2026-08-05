#!/usr/bin/env python3
"""SGLang 自适应限流控制器。

配合 sglang_start.sh --adaptive-limit 使用：
定时读取后端 SGLang 的 /metrics（sglang:num_queue_reqs / sglang:token_usage），
按"排队深但 KV 有余量 → 放宽 tool_call；排队浅或 KV 吃紧 → 收紧"的规则
自动调整前置代理 /admin/limits，替代人工盯曲线手调。

只调整 tool_call（主力场景）；thinking 是保护阀，默认不动，除非 --adaptive-thinking 显式开启。
"""

import argparse
import asyncio
import logging
import re
import signal
import sys

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [adaptive-limits] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("adaptive-limits")

# sglang:num_queue_reqs{priority="10"} 5.0
_GAUGE_RE = re.compile(r'^sglang:(num_queue_reqs|token_usage)(?:\{[^}]*\})?\s+([0-9.]+)', re.M)


def parse_metrics(text: str) -> dict:
    """从 Prometheus 文本里提取关注的 gauge（同名多标签取最大值）。"""
    out: dict = {}
    for name, value in _GAUGE_RE.findall(text):
        v = float(value)
        if name not in out or v > out[name]:
            out[name] = v
    return out


def decide(queue: int, token_usage: float, tool_call: int, args) -> int:
    """核心决策：返回调整后的 tool_call（可能等于当前值）。纯函数，便于测试。"""
    busy = token_usage >= args.token_high
    idle = token_usage <= args.token_low

    if queue > args.queue_high and not busy:
        # 排队深 + KV 有余量 → 后端还有容量，放宽 tool call
        return min(args.max_tool_call, tool_call + args.step)
    if queue < args.queue_low and (idle or tool_call > args.max_tool_call - args.step):
        # 排队浅（空闲或已近上限）→ 收紧，避免挤占 thinking 的 GPU/KV 份额
        return max(args.min_tool_call, tool_call - args.step)
    if queue >= args.queue_low and busy:
        # KV 吃紧但仍有排队 → 说明闸门放太多，收紧
        return max(args.min_tool_call, tool_call - args.step)
    return tool_call


async def fetch_backend_metrics(backend: str, client: httpx.AsyncClient) -> dict:
    resp = await client.get(f"{backend}/metrics", timeout=5.0)
    resp.raise_for_status()
    return parse_metrics(resp.text)


async def adjust_limits(
    proxy: str,
    client: httpx.AsyncClient,
    tool_call: int,
    thinking: int,
) -> None:
    payload = {"tool_call": tool_call}
    if thinking is not None:
        payload["thinking"] = thinking
    await client.post(f"{proxy}/admin/limits", json=payload, timeout=5.0)


async def run(args) -> None:
    async with httpx.AsyncClient() as client:
        # 首次读取当前 limits 作为基线，避免把人工热调的值覆盖掉
        try:
            resp = await client.get(f"{args.proxy}/admin/limits", timeout=5.0)
            cur = resp.json().get("limits", {})
            tool_call = int(cur.get("tool_call", args.tool_call))
            thinking = int(cur.get("thinking", args.thinking))
            log.info("基线 limits: tool_call=%d thinking=%d", tool_call, thinking)
        except Exception:  # noqa: BLE001 代理未就绪时用启动参数兜底
            tool_call = args.tool_call
            thinking = args.thinking
            log.warning("读取代理 limits 失败，用启动参数兜底 tool_call=%d", tool_call)

        failures = 0
        while True:
            try:
                m = await fetch_backend_metrics(args.backend, client)
            except Exception as e:  # noqa: BLE001
                failures += 1
                log.warning("读取后端 /metrics 失败(%d): %s", failures, e)
                if failures >= args.max_failures:
                    log.error("后端持续不可达，自适应限流退出")
                    return
                await asyncio.sleep(args.interval)
                continue
            failures = 0

            queue = int(m.get("num_queue_reqs", 0))
            token_usage = m.get("token_usage", 0.0)
            new_tool = decide(queue, token_usage, tool_call, args)

            if new_tool != tool_call:
                await adjust_limits(
                    args.proxy, client, new_tool,
                    new_tool if args.adaptive_thinking else None,
                )
                log.info(
                    "调整 limits: queue=%d token_usage=%.2f → tool_call %d→%d",
                    queue, token_usage, tool_call, new_tool,
                )
                tool_call = new_tool
            else:
                log.debug("维持 limits: queue=%d token_usage=%.2f tool_call=%d", queue, token_usage, tool_call)

            await asyncio.sleep(args.interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="SGLang 自适应限流控制器")
    ap.add_argument("--backend", default="http://127.0.0.1:8000", help="SGLang /metrics 地址")
    ap.add_argument("--proxy", default="http://127.0.0.1:8080", help="代理 admin 地址")
    ap.add_argument("--tool-call", type=int, default=16, help="tool_call 初始值（读取代理当前值优先）")
    ap.add_argument("--thinking", type=int, default=12, help="thinking 初始值")
    ap.add_argument("--min-tool-call", type=int, default=4, help="tool_call 下限")
    ap.add_argument("--max-tool-call", type=int, default=24, help="tool_call 上限")
    ap.add_argument("--queue-high", type=int, default=6, help="排队数高于此值且 KV 有余量则放宽")
    ap.add_argument("--queue-low", type=int, default=2, help="排队数低于此值则收紧")
    ap.add_argument("--token-high", type=float, default=0.92, help="token_usage 高于此值视为 KV 吃紧")
    ap.add_argument("--token-low", type=float, default=0.60, help="token_usage 低于此值视为空闲")
    ap.add_argument("--step", type=int, default=4, help="每次调整步长")
    ap.add_argument("--interval", type=float, default=15.0, help="轮询间隔（秒）")
    ap.add_argument("--max-failures", type=int, default=12, help="后端连续不可达退出阈值（约 3 分钟）")
    ap.add_argument("--adaptive-thinking", action="store_true", help="允许同步调整 thinking（默认只调 tool_call）")
    args = ap.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop = asyncio.Event()

    def _stop(*_):
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)

    async def _main():
        task = asyncio.create_task(run(args))
        await stop.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    try:
        loop.run_until_complete(_main())
    finally:
        loop.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

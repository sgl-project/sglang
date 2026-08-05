#!/usr/bin/env python3
"""SGLang 生产监控采集器（无外部依赖，输出 CSV）。

配合 sglang_start.sh --monitor 使用：
定时从后端 /metrics 抓取关键指标，追加写入 CSV，供基线对比与自适应限流效果评估。
如需正式可视化，直接让 Prometheus 抓取后端 /metrics 即可（脚本覆盖同样的指标）。
"""

import argparse
import asyncio
import bisect
import csv
import logging
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [monitor] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("monitor")

# 关注的 gauge（同名多标签取最大值）
_GAUGES = (
    "num_queue_reqs",
    "num_running_reqs",
    "gen_throughput",
    "cache_hit_rate",
    "token_usage",
    "kv_available_tokens",
    "spec_accept_rate",
    "spec_accept_length",
    "num_aborted_requests_total",
)
_GAUGE_RE = re.compile(
    r"^sglang:(" + "|".join(_GAUGES) + r")(?:\{[^}]*\})?\s+([0-9.]+)", re.M
)

def parse_gauges(text: str) -> dict:
    out: dict = {}
    for name, value in _GAUGE_RE.findall(text):
        v = float(value)
        if name not in out or v > out[name]:
            out[name] = v
    return out


def parse_histograms(text: str) -> dict:
    """解析直方图（stream=true 系列）：返回 {name: {"count": int, "sum": float, "buckets": [(le, cum)]}}"""
    out: dict = {}
    for line in text.splitlines():
        if not line.startswith("sglang:") or "_bucket" not in line:
            continue
        # 只取 stream="true" 的桶（tool call 走流式的占比大）
        m = re.match(
            r'^sglang:(time_to_first_token_seconds|inter_token_latency_seconds|e2e_request_latency_seconds)_bucket\{([^}]*)\}',
            line,
        )
        if not m or 'stream="true"' not in m.group(2):
            continue
        name = m.group(1)
        le_m = re.search(r'le="([^"]*)"', line)
        if not le_m:
            continue
        le, val = le_m.group(1), float(line.rsplit(" ", 1)[1])
        out.setdefault(name, {"buckets": []})
        out[name]["buckets"].append((float("inf") if le == "+Inf" else float(le), val))
    return out


def quantile(buckets: list, count: float, q: float) -> float:
    """标准直方图分位数估算：找到累计计数越过 q*count 的最小桶上界。"""
    if count <= 0 or not buckets:
        return float("nan")
    target = q * count
    buckets = sorted(buckets)
    last_finite = float("nan")
    for upper, cum in buckets:
        if upper != float("inf"):
            last_finite = upper
        if cum >= target:
            # +Inf 桶越过目标（数据超出有限桶覆盖）→ 用最后一个有限上界兜底，避免出 inf 尖峰
            return last_finite if upper == float("inf") else upper
    return last_finite


def summarize_histograms(hists: dict) -> dict:
    """把直方图转成 {metric: {"p50":.., "p90":.., "p99":.., "count":.., "avg":..}}"""
    out = {}
    for name, h in hists.items():
        count = h["buckets"][-1][1] if h["buckets"] else 0.0
        out[name] = {
            "count": count,
            "p50": quantile(h["buckets"], count, 0.50),
            "p90": quantile(h["buckets"], count, 0.90),
            "p99": quantile(h["buckets"], count, 0.99),
        }
    return out


def format_row(ts: str, g: dict, h: dict) -> dict:
    row = {"ts": ts}
    for k in _GAUGES:
        row[k] = g.get(k)
    for name, v in h.items():
        row[f"{name}:p50"] = v["p50"]
        row[f"{name}:p90"] = v["p90"]
        row[f"{name}:p99"] = v["p99"]
        row[f"{name}:count"] = v["count"]
    return row


CSV_FIELDS = [
    "ts",
    *_GAUGES,
    "time_to_first_token_seconds:p50",
    "time_to_first_token_seconds:p90",
    "time_to_first_token_seconds:p99",
    "time_to_first_token_seconds:count",
    "inter_token_latency_seconds:p50",
    "inter_token_latency_seconds:p90",
    "inter_token_latency_seconds:p99",
    "inter_token_latency_seconds:count",
    "e2e_request_latency_seconds:p50",
    "e2e_request_latency_seconds:p90",
    "e2e_request_latency_seconds:p99",
    "e2e_request_latency_seconds:count",
]


async def run(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "sglang_monitor.csv")
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    async with httpx.AsyncClient() as client:
        failures = 0
        while True:
            try:
                resp = await client.get(f"{args.backend}/metrics", timeout=5.0)
                resp.raise_for_status()
                text = resp.text
            except Exception as e:  # noqa: BLE001
                failures += 1
                log.warning("读取 /metrics 失败(%d): %s", failures, e)
                if failures >= args.max_failures:
                    log.error("后端持续不可达，监控采集退出")
                    return
                await asyncio.sleep(args.interval)
                continue
            failures = 0

            row = format_row(
                datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                parse_gauges(text),
                summarize_histograms(parse_histograms(text)),
            )
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if need_header:
                    writer.writeheader()
                    need_header = False
                writer.writerow(row)
            log.info("采样写入 %s: queue=%s ttft_p50=%s", csv_path, row["num_queue_reqs"], row["time_to_first_token_seconds:p50"])
            await asyncio.sleep(args.interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="SGLang 监控采集器")
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    ap.add_argument("--out-dir", default="/tmp/sglang_monitor")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--max-failures", type=int, default=12)
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

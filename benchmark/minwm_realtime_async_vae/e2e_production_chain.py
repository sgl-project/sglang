#!/usr/bin/env python3
"""Production-chain acceptance runner for MinWM realtime inference."""

from __future__ import annotations

import argparse
import asyncio
import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import httpx

from load_test import derive_trace_http_url, run_level


REQUIRED_TRACE_EVENTS = {
    "gateway.ws_accepted",
    "gateway.coordinator_admit_complete",
    "coordinator.admit_complete",
    "server.scheduler_forward_start",
    "server.model_denoise_complete",
    "server.vae_decode_complete",
    "server.vae_frame_batch_sent",
    "server.chunk_complete",
    "gateway.session_closed",
}


class ProductionGateError(RuntimeError):
    pass


def _p95(summary: dict | None) -> float | None:
    value = (summary or {}).get("p95")
    return None if value is None else float(value)


def validate_production_result(
    result: dict,
    *,
    required_concurrency: tuple[int, ...] = (1, 4),
    display_lag_p95_limit_ms: float = 250.0,
) -> None:
    runs = {int(run["concurrency"]): run for run in result.get("runs", [])}
    for concurrency in required_concurrency:
        run = runs.get(concurrency)
        if run is None:
            raise ProductionGateError(f"missing {concurrency}-concurrency run")
        if int(run.get("successful_sessions") or 0) != concurrency:
            raise ProductionGateError(
                f"{concurrency}-concurrency run did not complete every Session"
            )
        if run.get("errors") or float(run.get("error_rate") or 0) > 0:
            raise ProductionGateError(
                f"{concurrency}-concurrency run contains request failures"
            )
        observed = set(run.get("trace_event_names") or [])
        missing = sorted(REQUIRED_TRACE_EVENTS - observed)
        if missing:
            raise ProductionGateError(
                "missing production Trace events: " + ", ".join(missing)
            )
        if int(run.get("direct_vae_frame_batches") or 0) < 1:
            raise ProductionGateError("direct VAE-to-Gateway media route was not observed")

    display_p95 = _p95((result.get("browser") or {}).get("display_lag_ms"))
    if display_p95 is None:
        raise ProductionGateError("browser display lag evidence is required")
    if display_p95 > display_lag_p95_limit_ms:
        raise ProductionGateError(
            f"browser display lag p95 {display_p95:.1f}ms exceeds "
            f"{display_lag_p95_limit_ms:.1f}ms"
        )


def _metric(summary: dict | None, field: str = "p95") -> str:
    value = (summary or {}).get(field)
    return "-" if value is None else f"{float(value):.1f} ms"


def render_chinese_report(result: dict, *, run_id: str) -> str:
    hardware = result.get("hardware") or {}
    denoiser = hardware.get("denoiser") or {}
    vae = hardware.get("vae") or {}
    browser = result.get("browser") or {}
    lines = [
        "# MinWM 生产链路端到端验收报告",
        "",
        f"- Run ID：`{run_id}`",
        "- 入口：NLB -> Gateway -> Coordinator -> Denoiser -> VAE -> Gateway -> Browser",
        "- Trace：独立 HTTP 查询；视频 WebSocket 未携带 Trace 数据",
        "",
        "## 硬件",
        "",
        "| 模块 | 实例 | GPU | 数量 |",
        "| --- | --- | --- | ---: |",
        f"| Denoiser | {denoiser.get('instance_type', '-')} | {denoiser.get('gpu', '-')} | {denoiser.get('count', '-')} |",
        f"| VAE | {vae.get('instance_type', '-')} | {vae.get('gpu', '-')} | {vae.get('count', '-')} |",
        "",
        "## 并发与耗时",
        "",
        "| 场景 | 成功 Session | Chunk P95 | Action 到首帧 P95 | Denoise P95 | VAE Decode P95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in sorted(result.get("runs", []), key=lambda item: item["concurrency"]):
        stage = run.get("stage_ms") or {}
        lines.append(
            f"| {run['concurrency']} 并发 | {run.get('successful_sessions', 0)} | "
            f"{_metric(run.get('chunk_total_ms'))} | "
            f"{_metric(run.get('action_to_first_frame_ms'))} | "
            f"{_metric(stage.get('denoise_ms'))} | "
            f"{_metric(stage.get('vae_decode_ms'))} |"
        )
    lines.extend(
        [
            "",
            "## 浏览器播放",
            "",
            f"- Display Lag P50：{_metric(browser.get('display_lag_ms'), 'p50')}",
            f"- Display Lag P95：{_metric(browser.get('display_lag_ms'), 'p95')}",
            "",
            "## 发布门禁",
            "",
            "- Coordinator 原子配对、H100 Denoiser、低成本 VAE、VAE 直传 Gateway 均由同一 Trace 证明。",
            "- 1/4 并发无错误，且每个 Session 均完成。",
            "- Display Lag P95 不高于 250ms。",
            "",
        ]
    )
    return "\n".join(lines)


async def _preflight(http_origin: str) -> dict:
    async with httpx.AsyncClient(base_url=http_origin, timeout=15.0) as client:
        ready, models, index = await asyncio.gather(
            client.get("/readyz"),
            client.get("/v1/models"),
            client.get("/"),
        )
        for name, response in (("readyz", ready), ("models", models), ("index", index)):
            if response.status_code != 200:
                raise ProductionGateError(
                    f"Gateway {name} preflight failed with HTTP {response.status_code}"
                )
        return {"ready": ready.json(), "models": models.json()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hardware-json", type=Path, required=True)
    parser.add_argument("--browser-metrics-json", type=Path, required=True)
    parser.add_argument("--model", default="/work/model")
    parser.add_argument("--prompt", default="A cinematic forward-moving landscape")
    parser.add_argument("--size", default="832x480")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--warmup-chunks", type=int, default=2)
    parser.add_argument("--measured-chunks", type=int, default=6)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--trace-timeout-s", type=float, default=90.0)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> dict:
    http_origin = derive_trace_http_url(args.ws_url)
    preflight = await _preflight(http_origin)
    load_args = Namespace(
        ws_url=args.ws_url,
        trace_http_url=http_origin,
        trace_timeout_s=args.trace_timeout_s,
        skip_trace_query=False,
        profile="async",
        warmup_chunks=args.warmup_chunks,
        measured_chunks=args.measured_chunks,
        model=args.model,
        prompt=args.prompt,
        size=args.size,
        fps=args.fps,
        timeout_s=args.timeout_s,
    )
    runs = [await run_level(load_args, concurrency) for concurrency in (1, 4)]
    return {
        "schema_version": "minwm-production-chain/v1",
        "preflight": preflight,
        "hardware": json.loads(args.hardware_json.read_text()),
        "browser": json.loads(args.browser_metrics_json.read_text()),
        "runs": runs,
    }


def main() -> None:
    args = _parse_args()
    result = asyncio.run(_run(args))
    validate_production_result(result)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "report.zh-CN.md").write_text(
        render_chinese_report(result, run_id=run_id)
    )
    print(f"production chain accepted: {args.output_dir}")


if __name__ == "__main__":
    main()

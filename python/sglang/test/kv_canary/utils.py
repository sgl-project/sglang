from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.kv_canary.mode_config import _ModeConfig


def build_canary_server_args(
    *,
    kv_canary_mode: CanaryMode,
    mode_cfg: _ModeConfig,
    extra_server_args: tuple[str, ...] = (),
) -> list[str]:
    args = [
        "--kv-canary",
        kv_canary_mode.value,
        "--disable-piecewise-cuda-graph",
        "--context-length",
        "16384",
        *extra_server_args,
    ]
    if mode_cfg.json_model_override_args is not None:
        args.extend(["--json-model-override-args", mode_cfg.json_model_override_args])
    return args


def post_parallel_generate(
    *,
    url: str,
    prompts: list[str],
    max_new_tokens: int,
    timeout: float,
) -> list[dict]:
    def _send(prompt: str) -> dict:
        try:
            resp = requests.post(
                url,
                json={
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": 0.0,
                    },
                },
                timeout=timeout,
            )
            return {"status_code": resp.status_code, "body": resp.text}
        except requests.RequestException as exc:
            return {"status_code": -1, "error": repr(exc)}

    with ThreadPoolExecutor(max_workers=max(1, len(prompts))) as pool:
        return list(pool.map(_send, prompts))

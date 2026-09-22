"""Device-neutral pieces the DWDP served-model tests share.

DWDP runs on CUDA and on Intel XPU, so those tests resolve the backend at import
time rather than hardcoding one; ``test/registered`` is not an importable
package, so the part they share lives here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

from sglang.srt.utils import is_xpu
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import popen_launch_server, write_github_step_summary

# The XPU MoE path lives in the intel_xpu attention backend, and overlap
# scheduling plus the radix cache wedged the Level Zero driver at TP>=2.
_XPU_SERVER_ARGS = [
    "--device",
    "xpu",
    "--attention-backend",
    "intel_xpu",
    "--dtype",
    "bfloat16",
    "--trust-remote-code",
    "--disable-overlap-schedule",
    "--disable-radix-cache",
]
# Matches test_disaggregation_dwdp_gpt_oss.py, the CUDA DWDP gate; the attention
# backend is left to the default, which differs by SM version.
_CUDA_SERVER_ARGS = [
    "--disable-flashinfer-autotune",
]


def dwdp_server_args() -> List[str]:
    """Backend flags a DWDP model test needs ahead of its own arguments."""
    return list(_XPU_SERVER_ARGS if is_xpu() else _CUDA_SERVER_ARGS)


def launch_dwdp_server(
    model: str,
    base_url: str,
    *,
    extra_args: List[str],
    timeout: float,
    tp_size: Optional[int] = None,
):
    other_args = dwdp_server_args() + list(extra_args)
    if tp_size is not None:
        other_args += ["--tp-size", str(tp_size)]
    try:
        return popen_launch_server(
            model, base_url, timeout=timeout, other_args=other_args
        )
    except Exception as error:
        write_github_step_summary(f"Failed to launch server for {model}: {error}")
        raise


def assert_gsm8k_accuracy(
    test_case,
    *,
    model: str,
    base_url: str,
    accuracy: float,
    num_examples: int,
    num_threads: int,
    max_tokens: int = 512,
) -> dict:
    """Score a launched server on GSM8K and assert the floor.

    Calls ``run_eval`` directly rather than through ``GSM8KMixin``, which pins
    ``api="completion"``; the thresholds here were measured over the chat API
    that ``run_eval`` defaults to, and the two score differently.
    """
    metrics = run_eval(
        SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="gsm8k",
            num_examples=num_examples,
            num_threads=num_threads,
            max_tokens=max_tokens,
        )
    )
    write_github_step_summary(
        f"### {model} gsm8k\nscore={metrics['score']:.4f} (floor {accuracy})\n"
    )
    test_case.assertGreaterEqual(
        metrics["score"],
        accuracy,
        f"GSM8K score of {model} is {metrics['score']}, below {accuracy}",
    )
    return metrics

"""
Step-03 coverage: full-server E2E tests for all attention backend × runner combinations.

These are Python equivalents of the bash smoke tests in test/registered/step03_coverage/.
Each test launches a server with --load-format dummy, sends one /generate request,
and verifies the response is well-formed (non-empty text, no errors, no NaN markers).

Coverage completes the matrix from the bash tests and adds missing combinations:
  Runner × backend:
    eager (--disable-cuda-graph):
      triton, flashinfer, fa3, trtllm_mha, torch_native
    full CUDA graph (--disable-piecewise-cuda-graph):
      triton, flashinfer, fa3, trtllm_mha,
      flashinfer_mla, flashmla, cutlass_mla, trtllm_mla,
      dsv4, dsa, hybrid_mamba, breakable_cg
    piecewise CUDA graph (--enforce-piecewise-cuda-graph):
      triton, flashinfer, fa3, trtllm_mha
  Spec decoding:
      triton + EAGLE
      fa3 + EAGLE3
      dsv4 + EAGLE

Models are taken from environment variables or defaults that exist in the
cluster's shared HF cache at /root/.cache/huggingface.  Set MODEL_PATH
in the environment to override any individual test's default.

Run a single test on the cluster:
  sudo srun ... python test_step03_e2e_runners.py TestTritonEager

Run the full suite (use multiple nodes in parallel):
  sudo srun ... python test_step03_e2e_runners.py

NOTE: Tests that need hardware-specific kernels (flashmla=Hopper, cutlass_mla=B200)
      or large models (dsv4, dsa, hybrid_mamba) skip automatically when the
      required model or GPU arch is not available.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Optional

import requests
import torch

sys.path.insert(0, str(Path(__file__).parent))
from step03_test_utils import _model_exists, gpu_arch_sm

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

_SM = gpu_arch_sm()
_CUDA = torch.cuda.is_available()

# Default small model — always assumed present in the HF cache.
_SMALL_MODEL = os.environ.get("SMALL_MODEL", "Qwen/Qwen3-0.6B")
# TP size constants — must be defined before _DSV3_TP_SIZE below.
_TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
_TP_SIZE_LARGE = int(os.environ.get("TP_SIZE_LARGE", "4"))
# DSV3 MLA test model. lmsys/sglang-ci-dsv3-test is a private tiny CI model;
# fall back to deepseek-ai/DeepSeek-V3.2 (same MLA architecture, dummy-load only).
_DSV3_MODEL = os.environ.get(
    "DSV3_MODEL",
    (
        "lmsys/sglang-ci-dsv3-test"
        if _model_exists("lmsys/sglang-ci-dsv3-test")
        else "deepseek-ai/DeepSeek-V3.2"
    ),
)
# TP size for the DSV3 model: tiny CI model fits on 1 GPU; full DeepSeek-V3.2
# needs TP=4 (too large for 1 GPU even with dummy weights).
_DSV3_TP_SIZE = (
    "1" if _DSV3_MODEL == "lmsys/sglang-ci-dsv3-test" else str(_TP_SIZE_LARGE)
)
# DSV4 model (large; on cluster at /flash_model or via HF)
_DSV4_MODEL = os.environ.get("DSV4_MODEL", "/flash_model")
# DSA / DeepSeek-V3.2
_DSA_MODEL = os.environ.get("DSA_MODEL", "deepseek-ai/DeepSeek-V3.2")
# Hybrid Mamba
_HYBRID_MODEL = os.environ.get("HYBRID_MODEL", "nvidia/Nemotron-H-8B-Base-8K")
# SWA model
_SWA_MODEL = os.environ.get("SWA_MODEL", "openai/gpt-oss-20b")
# EAGLE target
_EAGLE_TARGET = os.environ.get("EAGLE_TARGET", "meta-llama/Llama-2-7b-chat-hf")
_EAGLE_DRAFT = os.environ.get("EAGLE_DRAFT", "lmsys/sglang-EAGLE-llama2-chat-7B")
# EAGLE3 target
_EAGLE3_TARGET = os.environ.get("EAGLE3_TARGET", "meta-llama/Llama-3.1-8B-Instruct")

_DEFAULT_PORT = int(os.environ.get("TEST_PORT", "30000"))


def _check_response(resp: requests.Response) -> dict:
    """Assert a /generate response is well-formed and return the parsed body."""
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:200]}"
    body = resp.json()
    if isinstance(body, list):
        body = body[0]
    text = body.get("text", "")
    assert isinstance(text, str) and len(text) > 0, f"empty text in: {body}"
    assert "error" not in str(body).lower() or "Error" not in body.get(
        "text", ""
    ), f"error in response: {body}"
    return body


# ---------------------------------------------------------------------------
# Base class for server smoke tests
# ---------------------------------------------------------------------------


class _ServerSmokeBase(CustomTestCase):
    """Launches a server in setUpClass, tears it down in tearDownClass."""

    model: str = _SMALL_MODEL
    launch_args: list = []
    port: int = _DEFAULT_PORT
    timeout: int = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    # Set to a human-readable reason to skip the whole class
    skip_reason: Optional[str] = None

    @classmethod
    def setUpClass(cls):
        if cls.skip_reason:
            raise unittest.SkipTest(cls.skip_reason)
        if not _CUDA:
            raise unittest.SkipTest("CUDA required")
        if not _model_exists(cls.model):
            raise unittest.SkipTest(f"model not found: {cls.model}")

        base_url = f"http://127.0.0.1:{cls.port}"
        cls.base_url = base_url
        cls.process = popen_launch_server(
            cls.model,
            base_url,
            timeout=cls.timeout,
            other_args=cls.launch_args
            + ["--load-format", "dummy", "--trust-remote-code"],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_generate(self):
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"max_new_tokens": 16, "temperature": 0.0},
            },
            timeout=120,
        )
        _check_response(resp)


# ===========================================================================
# MHA backends × eager runner
# ===========================================================================


class TestTritonEager(_ServerSmokeBase):
    """triton backend + no CUDA graph — baseline eager path."""

    model = _SMALL_MODEL
    launch_args = [
        "--attention-backend",
        "triton",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--max-running-requests",
        "8",
        "--tp-size",
        "1",
    ]


class TestFlashInferEager(_ServerSmokeBase):
    """flashinfer backend + eager (no CUDA graph)."""

    model = _SMALL_MODEL
    launch_args = [
        "--attention-backend",
        "flashinfer",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--max-running-requests",
        "8",
        "--tp-size",
        "1",
    ]


class TestFA3Eager(_ServerSmokeBase):
    """fa3 backend + eager — skip on Blackwell (trtllm_mha is default there)."""

    model = _SMALL_MODEL
    skip_reason = (
        None
        if (_SM is not None and 80 <= _SM < 100)
        else "FA3 requires SM 80-90 (Ampere/Hopper)"
    )
    launch_args = [
        "--attention-backend",
        "fa3",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--max-running-requests",
        "8",
        "--tp-size",
        "1",
    ]


class TestTRTLLMMHAEager(_ServerSmokeBase):
    """trtllm_mha backend + eager."""

    model = _SMALL_MODEL
    launch_args = [
        "--attention-backend",
        "trtllm_mha",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--max-running-requests",
        "8",
        "--tp-size",
        "1",
    ]


# ===========================================================================
# MHA backends × full CUDA graph
# ===========================================================================


class TestTritonCudaGraph(_ServerSmokeBase):
    """triton + full CUDA graph decode."""

    model = _SMALL_MODEL
    launch_args = [
        "--attention-backend",
        "triton",
        "--disable-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--tp-size",
        "1",
    ]


class TestFlashInferCudaGraph(_ServerSmokeBase):
    """flashinfer + full CUDA graph decode."""

    model = _SMALL_MODEL
    launch_args = [
        "--attention-backend",
        "flashinfer",
        "--disable-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--tp-size",
        "1",
    ]


class TestFA3CudaGraph(_ServerSmokeBase):
    """FA3 + full CUDA graph."""

    model = _SMALL_MODEL
    skip_reason = (
        None
        if (_SM is not None and 80 <= _SM < 100)
        else "FA3 requires SM 80-90 (Ampere/Hopper)"
    )
    launch_args = [
        "--attention-backend",
        "fa3",
        "--disable-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--tp-size",
        "1",
    ]


class TestTRTLLMMHACudaGraph(_ServerSmokeBase):
    """trtllm_mha + full CUDA graph (default B200 backend)."""

    model = _SMALL_MODEL
    launch_args = [
        "--attention-backend",
        "trtllm_mha",
        "--disable-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--tp-size",
        "1",
    ]


# ===========================================================================
# MHA backends × PCG (piecewise CUDA graph)  — PCG exercises the EXTEND path
# ===========================================================================


class TestTritonPCG(_ServerSmokeBase):
    """triton + PCG (piecewise CUDA graph) — exercises extend-capture path."""

    model = _SMALL_MODEL
    launch_args = [
        "--attention-backend",
        "triton",
        "--enforce-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--tp-size",
        "1",
    ]


class TestFlashInferPCG(_ServerSmokeBase):
    """flashinfer + PCG — exercises EXTEND-mode PCG capture (step-03 regression guard)."""

    model = _SMALL_MODEL
    launch_args = [
        "--attention-backend",
        "flashinfer",
        "--enforce-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--tp-size",
        "1",
    ]


class TestFA3PCG(_ServerSmokeBase):
    """FA3 + PCG."""

    model = _SMALL_MODEL
    skip_reason = (
        None if (_SM is not None and 80 <= _SM < 100) else "FA3 requires SM 80-90"
    )
    launch_args = [
        "--attention-backend",
        "fa3",
        "--enforce-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--tp-size",
        "1",
    ]


class TestTRTLLMMHAPCG(_ServerSmokeBase):
    """trtllm_mha + PCG — step-03 regression guard for PCG asymmetry fix."""

    model = _SMALL_MODEL
    launch_args = [
        "--attention-backend",
        "trtllm_mha",
        "--enforce-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--tp-size",
        "1",
    ]


# ===========================================================================
# Breakable CUDA graph runner
# ===========================================================================


class TestBreakableCudaGraph(_ServerSmokeBase):
    """triton + breakable CUDA graph runner."""

    model = _SMALL_MODEL
    launch_args = [
        "--attention-backend",
        "triton",
        "--enable-breakable-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--tp-size",
        "1",
    ]


# ===========================================================================
# MLA backends × full CUDA graph
# ===========================================================================


class TestFlashInferMLACudaGraph(_ServerSmokeBase):
    """flashinfer MLA backend + full CUDA graph (DSV3 model)."""

    model = _DSV3_MODEL
    launch_args = [
        "--attention-backend",
        "flashinfer",
        "--disable-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--tp-size",
        _DSV3_TP_SIZE,
    ]


class TestFlashMLACudaGraph(_ServerSmokeBase):
    """FlashMLA backend + full CUDA graph (Hopper only)."""

    model = _DSV3_MODEL
    skip_reason = (
        None if (_SM is not None and _SM == 90) else "FlashMLA requires Hopper (SM90)"
    )
    launch_args = [
        "--attention-backend",
        "flashmla",
        "--disable-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--tp-size",
        _DSV3_TP_SIZE,
    ]


class TestCutlassMLACudaGraph(_ServerSmokeBase):
    """CutlassMLA backend + full CUDA graph (Blackwell only)."""

    model = _DSV3_MODEL
    skip_reason = (
        None
        if (_SM is not None and _SM == 100)
        else "CutlassMLA requires Blackwell (SM100)"
    )
    launch_args = [
        "--attention-backend",
        "cutlass_mla",
        "--disable-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--tp-size",
        _DSV3_TP_SIZE,
    ]


class TestTRTLLMMLACudaGraph(_ServerSmokeBase):
    """TRTLLM MLA backend + full CUDA graph."""

    model = _DSV3_MODEL
    launch_args = [
        "--attention-backend",
        "trtllm_mla",
        "--disable-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--tp-size",
        _DSV3_TP_SIZE,
    ]


# ===========================================================================
# Special backends: DSV4, DSA, Hybrid Mamba, SWA
# ===========================================================================


class TestDSV4CudaGraph(_ServerSmokeBase):
    """DeepSeek-V4-Flash with DSV4 backend + full CUDA graph.

    Requires the DSV4-Flash checkpoint at $DSV4_MODEL (default: /flash_model).
    Requires PR #26024 cherry-picked onto the branch (NVFP4 routing fix).
    """

    model = _DSV4_MODEL
    timeout = 2400
    launch_args = [
        "--attention-backend",
        "dsv4",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--tp-size",
        str(_TP_SIZE_LARGE),
        "--mem-fraction-static",
        "0.8",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--chunked-prefill-size",
        "4096",
        "--disable-flashinfer-autotune",
        "--disable-piecewise-cuda-graph",
    ]


class TestDSACudaGraph(_ServerSmokeBase):
    """DeepSeek-V3.2 with DSA (native sparse attention) + full CUDA graph."""

    model = _DSA_MODEL
    timeout = 2400
    launch_args = [
        "--attention-backend",
        "dsa",
        "--tp-size",
        str(_TP_SIZE_LARGE),
        "--mem-fraction-static",
        "0.8",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--disable-piecewise-cuda-graph",
    ]


class TestHybridMambaCudaGraph(_ServerSmokeBase):
    """Nemotron-H hybrid SSM+attention model + full CUDA graph."""

    model = _HYBRID_MODEL
    timeout = 1800
    launch_args = [
        "--mem-fraction-static",
        "0.7",
        "--disable-piecewise-cuda-graph",
        "--tp-size",
        "1",
    ]


class TestSWACudaGraph(_ServerSmokeBase):
    """GPT-OSS-20B sliding-window attention + full CUDA graph."""

    model = _SWA_MODEL
    timeout = 1800
    launch_args = [
        "--mem-fraction-static",
        "0.7",
        "--disable-piecewise-cuda-graph",
        "--tp-size",
        str(_TP_SIZE_LARGE),
    ]


# ===========================================================================
# Spec decoding (EAGLE / EAGLE3)
# ===========================================================================


class TestTritonEAGLE(_ServerSmokeBase):
    """triton + full CUDA graph + EAGLE speculative decoding (Llama-2-7b)."""

    model = _EAGLE_TARGET
    timeout = 1800
    launch_args = [
        "--attention-backend",
        "triton",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-draft-model-path",
        _EAGLE_DRAFT,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "4",
        "--speculative-num-draft-tokens",
        "8",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--disable-piecewise-cuda-graph",
        "--tp-size",
        "1",
    ]


class TestDSV4EAGLE(_ServerSmokeBase):
    """DSV4 + full CUDA graph + EAGLE speculative decoding (motivating workload)."""

    model = _DSV4_MODEL
    timeout = 2400
    launch_args = [
        "--attention-backend",
        "dsv4",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "1",
        "--speculative-num-draft-tokens",
        "1",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--tp-size",
        str(_TP_SIZE_LARGE),
        "--mem-fraction-static",
        "0.8",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--disable-piecewise-cuda-graph",
    ]


class TestFA3EAGLE3(_ServerSmokeBase):
    """FA3 (or triton fallback on Blackwell) + full CG + EAGLE3 multi-layer draft."""

    model = _EAGLE3_TARGET
    timeout = 1800
    # On Blackwell (SM100+), FA3 is unavailable so we fall back to triton
    launch_args = [
        "--mem-fraction-static",
        "0.7",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-num-steps",
        "3",
        "--speculative-num-draft-tokens",
        "6",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
        "--disable-piecewise-cuda-graph",
        "--tp-size",
        "1",
    ] + (
        ["--attention-backend", "fa3", "--dtype", "float16"]
        if (_SM is not None and 80 <= _SM < 100)
        else []  # let sglang auto-select on Blackwell
    )


# ===========================================================================
# DP attention + idle rank
# ===========================================================================


class TestDPIdleRank(_ServerSmokeBase):
    """DP-attention with idle DP rank — exercises IDLE forward mode.

    Matches dp_idle.sh: --tp-size 4 --dp-size 4 --enable-dp-attention.
    With DP-attention, tp_size GPUs act as both TP shards and DP ranks
    (dp_size must equal tp_size). Only some DP groups receive tokens for
    a given request, exercising the IDLE forward-mode path.
    """

    model = os.environ.get("DP_MODEL", "Qwen/Qwen3-30B-A3B")
    timeout = 2400
    launch_args = [
        "--attention-backend",
        "triton",
        "--tp-size",
        str(_TP_SIZE_LARGE),
        "--dp-size",
        str(_TP_SIZE_LARGE),
        "--enable-dp-attention",
        "--disable-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "4",
    ]


if __name__ == "__main__":
    unittest.main(verbosity=2)

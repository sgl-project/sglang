"""Qwen3.5-4B unified-memory quality check with AITER on AMD."""

import os
import sys
import unittest
from pathlib import Path
from typing import ClassVar

# CI executes each file directly; reuse the recipe without collecting its
# Triton/FA3/FlashInfer classes as additional tests in this module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from registered.page_major import test_page_major_qwen_hybrid as _qwen_recipe

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    envs.SGLANG_ENABLE_AITER_UNIFIED_MEMORY.get(),
    "requires SGLANG_ENABLE_AITER_UNIFIED_MEMORY=1 and AITER with ROCm/aiter#6661",
)
@unittest.skipUnless(is_hip(), "AMD ROCm required")
class TestUnifiedQwenHybridAiter(_qwen_recipe.TestUnifiedQwenHybridTriton):
    # Exercise automatic unified-attention dispatch for the arena.
    server_env: ClassVar[dict[str, str]] = {
        **{
            key: value
            for key, value in os.environ.items()
            if key != "SGLANG_USE_AITER_UNIFIED_ATTN"
        },
        "SGLANG_USE_AITER": "1",
        "SGLANG_ROCM_GDN_ONLINE_MXFP4": "0",
    }
    other_args = _qwen_recipe._UNIFIED_COMMON_ARGS + [
        "--attention-backend",
        "aiter",
        "--tp",
        "1",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bf16",
        "--mamba-ssm-dtype",
        "float32",
        "--page-size",
        "1",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "full",
        "--random-seed",
        "123",
        "--context-length",
        "4096",
        "--chunked-prefill-size",
        "128",
        "--max-running-requests",
        "32",
        "--max-total-tokens",
        "131072",
        "--max-mamba-cache-size",
        "160",
        "--cuda-graph-bs-decode",
        "1",
        "2",
        "4",
        "8",
        "16",
        "32",
    ]


if __name__ == "__main__":
    unittest.main()

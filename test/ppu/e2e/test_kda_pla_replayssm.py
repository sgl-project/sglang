"""Real-model ReplaySSM integration smoke tests (one TP8 server at a time).

SGLANG_SAIL_PLA_CUDA=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m pytest -s test/ppu/e2e/test_kda_pla_replayssm.py -k glm

Use -k kimi for Kimi-K3 + DSpark. MODEL_PATH and DSPARK_PATH override the
defaults; TP_SIZE may be increased to fit the checkpoint. Kimi-K3 requires
enough memory for the full checkpoint and its draft. GLM-5.3-Flash requires
the Glm5Next model-support patch in the serving environment. SERVER_ARGS
(JSON list) can supply site-specific launch options. STARTUP_TIMEOUT overrides
the 3600-second startup limit for slow checkpoint storage. These are test inputs,
not new runtime switches. Launches retain CUDA Graph and verify actual PLA
dispatch after generation, speculative progress, and completed outputs.
"""

import io
import json
import os
from contextlib import contextmanager

import pytest
import requests

from sglang.srt.utils import is_ppu, kill_process_tree
from sglang.test.test_utils import popen_launch_server

pytestmark = pytest.mark.skipif(not is_ppu(), reason="requires PPU and model weights")


def model_configuration(family):
    common = [
        "--tp-size",
        os.getenv("TP_SIZE", "8"),
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.88",
        "--max-running-requests",
        "8",
        "--cuda-graph-max-bs",
        "8",
        "--random-seed",
        "37",
        "--enable-linear-replayssm-spec",
        "--watchdog-timeout",
        "1200",
    ]
    env = {
        **os.environ,
        "SGLANG_SAIL_PLA_CUDA": "1",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
    }
    if family == "kimi":
        model = os.getenv(
            "MODEL_PATH", "/ppusw/datasets/checkpoints/LLM/moonshotai/K3/Kimi-K3"
        )
        common += [
            "--prefill-attention-backend",
            "fa3",
            "--decode-attention-backend",
            "flashmla",
            "--linear-attn-backend",
            "flashkda",
            "--mamba-radix-cache-strategy",
            "extra_buffer",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            os.getenv(
                "DSPARK_PATH",
                "/ppusw/datasets/checkpoints/LLM/RadixArk/K3/Kimi-K3-DSpark",
            ),
            "--speculative-draft-attention-backend",
            "fa3",
            "--speculative-dspark-block-size",
            "7",
            "--moe-runner-backend",
            "deep_gemm",
        ]
        env.update(
            DG_USE_MOE_DYNAMIC_TILE="1",
            SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA="1",
        )
    else:
        assert family == "glm"
        model = os.getenv(
            "MODEL_PATH", "/ppusw/datasets/checkpoints/LLM/ZhipuAI/v5.3/GLM-5.3-Flash"
        )
        common += [
            "--max-total-tokens",
            "65536",
            "--ep-size",
            os.getenv("TP_SIZE", "8"),
            "--attention-backend",
            "nsa",
            "--dsa-prefill-backend",
            "flashmla_sparse",
            "--dsa-decode-backend",
            "flashmla_sparse",
            "--nsa-decode-backend",
            "flashmla_sparse",
            "--kv-cache-dtype",
            "bfloat16",
            "--disable-radix-cache",
            "--disable-overlap-schedule",
            "--disable-piecewise-cuda-graph",
            "--disable-shared-experts-fusion",
            "--disable-custom-all-reduce",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "5",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "6",
        ]
        env.update(
            SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8="0",
            SGLANG_NSA_DUAL_STREAM="0",
            SGLANG_DISABLE_DSA_INDEXER_FUSION="1",
            SGLANG_NSA_USE_DSV4_BF16_TOPK="1",
            SGLANG_INT8_INDEXER_ENABLE="1",
            # GLM KPool's INT8 path requires 132 bytes/token, not FP4's 68.
            SGLANG_OPT_USE_FP4_INDEXER_CACHE="0",
            SGLANG_SAIL_DEEPGEMM_DENSE="1",
        )
    return model, common + json.loads(os.getenv("SERVER_ARGS", "[]")), env


@contextmanager
def launch_model(family):
    model, args, env = model_configuration(family)
    stdout, stderr = io.StringIO(), io.StringIO()
    url = os.getenv("BASE_URL", "http://127.0.0.1:31316")
    startup_timeout = int(os.getenv("STARTUP_TIMEOUT", "3600"))
    assert startup_timeout > 0, "STARTUP_TIMEOUT must be positive"
    process = popen_launch_server(
        model,
        url,
        timeout=startup_timeout,
        other_args=args,
        env=env,
        return_stdout_stderr=(stdout, stderr),
    )
    try:
        yield url, stdout, stderr
    finally:
        kill_process_tree(process.pid)


@pytest.mark.parametrize("family", ["kimi", "glm"])
def test_model_replayssm(family):
    prompts = [
        "Explain how a compiler translates a short Python program into instructions.",
        "Describe the water cycle and why rain forms.",
        "Write a Python function that returns the first ten Fibonacci numbers.",
        "What are the differences between a stack and a queue? Give examples.",
    ]
    with launch_model(family) as (url, stdout, stderr):
        # A second batch also exercises committed state reuse and recycled slots.
        for batch in (prompts, prompts[:2]):
            response = requests.post(
                url + "/generate",
                json={
                    "text": batch,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 96,
                        "ignore_eos": True,
                    },
                },
                timeout=1200,
            )
            response.raise_for_status()
            results = response.json()
            assert len(results) == len(batch)
            for result in results:
                assert result["text"].strip()
                meta = result["meta_info"]
                assert meta["completion_tokens"] == 96
                assert meta["spec_verify_ct"] > 0, meta
                print(
                    json.dumps(
                        {
                            "model": family,
                            "completion_tokens": 96,
                            "spec_verify_ct": meta["spec_verify_ct"],
                            "text": result["text"],
                        }
                    )
                )
        logs = stdout.getvalue() + stderr.getvalue()
        for api in ("fused_kda_decode_mtp_dspark", "commit_kda_replayssm_after_verify"):
            assert f"USE PPU SAIL CUDA PLA kernel: {api} (direct layout)" in logs, logs[
                -8000:
            ]

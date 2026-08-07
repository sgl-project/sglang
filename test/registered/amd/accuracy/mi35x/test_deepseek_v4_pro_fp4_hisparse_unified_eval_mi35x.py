"""MI35x DeepSeek-V4-Pro FP4 unified-KV + HiSparse GSM8K evaluation test (8-GPU).

Exercises the unified_kv attention backend (``SGLANG_HACK_FLASHMLA_BACKEND=
unified_kv_triton``) together with HiSparse on ROCm. Unlike the separate-KV
path, the compressed C4 KV lives inside the unified pool's ``rows[swa_pages:]``
and the HiSparse hot device buffer is a bf16 view into that region with a linear
host cold pool; swap-in runs outside CUDA/HIP graph capture (the hot buffer
shape is fixed), so decode replays a captured graph while compressed C4 tokens
are streamed in from host on demand.

``FP4`` here refers to the model checkpoint's MXFP4 routed-expert
weights (``deepseek-ai/DeepSeek-V4-Pro``); it is NOT an FP4 KV cache. The
KV / compressed-C4 cache is FP8, as on every DSA path.

Two complementary checks:
- ``test_a_gsm8k`` — accuracy/sparse-selection guard: GSM8K few-shot eval must
  align with the dense baseline. NOTE: GSM8K prompts (~1k tokens) stay fully
  device-resident, so this case validates HiSparse top-k selection numerics but
  does NOT exercise the host<->device swap data path.
- ``test_b_long_context_swap`` — swap-path guard: a ~19k-token request whose
  compressed C4 footprint overflows the GPU-resident hot buffer
  (``device_buffer_size``), forcing cold C4 tokens to be swapped in from the
  host pool on demand. A unique passcode buried at the top must be retrieved,
  so the swap kernel is validated end-to-end (correct cold->hot copy), not just
  sparse selection.

Registry: nightly-amd-8-gpu-mi35x-deepseek-v4-pro-hisparse-unified suite
"""

import os
import resource
import unittest
from types import SimpleNamespace

import requests

import sglang.test as sglang_test_pkg
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=7200,
    suite="nightly-amd-8-gpu-mi35x-deepseek-v4-pro-hisparse-unified",
    nightly=True,
)

DEEPSEEK_V4_PRO_FP4_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V4_PRO_MODEL_PATH_FP4", "deepseek-ai/DeepSeek-V4-Pro"
)
# Pro is 1.6T; weight load + warmup is much longer than Flash 285B.
SERVER_LAUNCH_TIMEOUT = 5400

# Common DeepSeek-V4 env vars (AMD ROCm 7.2 path: AITER indexer + ROCm700A),
# with the unified_kv attention backend instead of the plain triton backend.
COMMON_ENV_VARS = {
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "false",
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_ROCM700A": "1",
    "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
    "SGLANG_OPT_USE_FUSED_COMPRESS_TRITON": "true",
    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
    "SGLANG_OPT_FP8_WO_A_GEMM": "false",
    "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "false",
    "SGLANG_OPT_USE_TOPK_V2": "false",
    "SGLANG_OPT_USE_AITER_INDEXER": "true",
    "SGLANG_OPT_USE_TILELANG_INDEXER": "false",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "false",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "false",
    "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
    "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "false",
    "SGLANG_ROCM_USE_MULTI_STREAM": "false",
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "SGLANG_EAGER_INPUT_NO_COPY": "true",
}

# HiSparse config: top_k aligned to the model's index_topk (1024). The swap
# data path is triggered at runtime when a request's compressed C4 footprint
# exceeds device_buffer_size (2048) -- see test_b_long_context_swap. Setting
# host_to_device_ratio=2 (c4_shrink_factor) additionally shrinks the
# device-resident C4 pool by 2x and provisions a 2x host cold mirror, so the
# c4_shrink_factor>1 offload-provisioning path is exercised too (ratio=1 leaves
# it uncovered).
HISPARSE_CONFIG = (
    '{"top_k": 1024, "device_buffer_size": 2048, "host_to_device_ratio": 2}'
)


class TestDeepseekV4ProFp4HiSparseUnifiedMI35x(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # GSM8K eval opens `parallel` concurrent HTTP connections; raise the
        # open-file soft limit so a high parallelism does not hit EMFILE.
        _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(
            resource.RLIMIT_NOFILE, (max(_soft, min(_hard, 65536)), _hard)
        )

        cls.model = DEEPSEEK_V4_PRO_FP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        env = os.environ.copy()
        env.update(COMMON_ENV_VARS)

        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--disable-radix-cache",
            "--attention-backend",
            "dsv4",
            "--max-running-requests",
            "256",
            "--page-size",
            "256",
            "--mem-fraction-static",
            "0.90",
            "--swa-full-tokens-ratio",
            "0.1",
            "--chunked-prefill-size",
            "8192",
            "--disable-shared-experts-fusion",
            "--tool-call-parser",
            "deepseekv4",
            "--reasoning-parser",
            "deepseek-v4",
            "--enable-hisparse",
            "--hisparse-config",
            HISPARSE_CONFIG,
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        # `a` prefix to run first (alphabetical) and warm up the server.
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            parallel=1319,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v4-pro-fp4 unified_kv hisparse)\n"
                f'{metrics["accuracy"]=:.3f}\n'
            )
        # unified_kv + HiSparse must align with the dense DSV4-Pro baseline.
        self.assertGreater(metrics["accuracy"], 0.91)

    def test_b_long_context_swap(self):
        """Exercise the host<->device swap path end-to-end.

        GSM8K prompts (~1k tokens) stay fully device-resident and never swap.
        Here we build a ~19k-token prompt whose compressed C4 footprint clearly
        exceeds device_buffer_size (2048), so cold C4 tokens must be streamed in
        from the host pool on every decode step. A unique passcode is buried at
        the very top of the context; retrieving it correctly is only possible if
        the swap-in kernel copies the cold C4 KV into the hot buffer correctly,
        so this guards the swap data path (not just sparse selection).
        """
        # long_prompt.txt (~6.4k tokens) lives next to the sglang.test package.
        dirpath = os.path.dirname(sglang_test_pkg.__file__)
        with open(os.path.join(dirpath, "long_prompt.txt")) as f:
            filler = f.read()
        # Repeat 3x -> ~19k prompt tokens, well past device_buffer_size (2048),
        # so the compressed C4 footprint overflows the hot buffer with margin
        # and forces host->device swap (verified: ~4.9k tokens stay host-side).
        filler = filler * 3

        passcode = "7G-ZULU-4419"
        prompt = (
            f"Remember this passcode exactly: {passcode}. "
            "You will be asked to recall it at the end.\n\n"
            f"{filler}\n\n"
            "Question: What is the exact passcode given at the very beginning of "
            "this message? Reply with only the passcode, nothing else."
        )

        # Use the chat endpoint (applies the chat template + reasoning); the raw
        # /generate completion path degenerates into repetition on this reasoning
        # model and is not a reliable retrieval probe.
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 1024,
            },
            timeout=900,
        )
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        # The passcode may land in either the reasoning trace or the final answer.
        text = (
            (message.get("reasoning_content") or "")
            + "\n"
            + (message.get("content") or "")
        )
        print(f"long_context_swap completion={text!r}")

        retrieved = passcode in text
        if is_in_ci():
            write_github_step_summary(
                f"### test_long_context_swap (deepseek-v4-pro-fp4 unified_kv hisparse)\n"
                f"{retrieved=}\n"
            )
        # Correct retrieval from a swapped-in cold region proves the
        # host<->device swap path works end-to-end.
        self.assertTrue(
            retrieved,
            f"passcode {passcode!r} not found in completion: {text!r}",
        )


if __name__ == "__main__":
    import sys

    sys.argv = [a for a in sys.argv if a not in ("-f", "--failfast")]
    unittest.main()

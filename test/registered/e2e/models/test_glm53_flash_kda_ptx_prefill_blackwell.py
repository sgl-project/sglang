"""GLM-5.3-Flash accuracy parity for the PTX KDA prefill backend on Blackwell.

``--linear-attn-prefill-backend ptx_kda`` selects the hand-written tcgen05
chunked-prefill kernel, JIT-built for the running device's arch (sm_100a on
B200, sm_103a on GB300 / B300). GLM-5.3-Flash is the target: its 34 KDA layers
use the safe gate (``gate_lower_bound=-5``), the only gate mode the kernel
handles. Without a lower bound (Kimi-Linear) the per-chunk decay overflows and
the kernel returns NaN.

Its failure modes are silent: the dispatcher swaps in Triton when it rejects
the arch, and the wrapper hands individual batches to Triton (spec
draft-extend, track-state snapshots, unsupported shapes, a kernel exception).
None of that reaches the HTTP result, so this file

* reads the server log to prove the dispatcher picked ``PtxKDAKernel`` and
  that no batch fell back, and
* scores the kernel against the same server on the default Triton prefill:

  - GSM8K, 20-shot completion, greedy, 500 questions. The many-sequence
    batches take the kernel's builder + chain route, and every batch has a new
    shape, which is what exposes per-shape state in the kernel.
  - Prompt logprobs of one 2040-token prefill (the head of
    ``python/sglang/test/long_prompt.txt``), which takes the single-sequence
    fused route through the wrapper's padding path (2040 is not a multiple of
    the 64-token chunk). It stays below the DSA indexer's top-k (2048): past
    that the sparse key selection is not run-to-run reproducible, and the
    comparison would measure that instead of the KDA kernel.

Server args follow the cookbook's Blackwell TP4 recipe with two changes, both
made on the Triton and PTX servers alike:

* ``--disable-radix-cache``. With the radix cache on, the mamba cache
  strategy resolves to ``extra_buffer`` (``no_buffer`` needs page size 1, and
  DSA pins 64), whose tracked prefill batches carry a track-state buffer; the
  PTX wrapper routes exactly those batches to Triton, so the kernel under
  test would barely run. It also sends every GSM8K few-shot prefix through
  prefill instead of the cache.
* ``--chunked-prefill-size 16384``, the B200 default, pinned so GB300 runs the
  same batch sizes.

    python -m pytest \\
        test/registered/e2e/models/test_glm53_flash_kda_ptx_prefill_blackwell.py -v
"""

import math
import os
import tempfile
import unittest
from types import SimpleNamespace

import requests
import torch
from transformers import AutoTokenizer

import sglang.test
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _wait_for_gpu_idle_in_ci,
    is_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
    write_github_step_summary,
)

# Measured on B200: 822 s for both servers, including the PTX extension build.
register_cuda_ci(est_time=900, stage="nightly", runner_config="4-gpu-b200")
register_cuda_ci(est_time=900, stage="nightly", runner_config="4-gpu-gb300")

MODEL = os.environ.get("SGLANG_TEST_GLM53_FLASH_MODEL") or try_cached_model(
    "zai-org/GLM-5.3-Flash"
)
# Offline hosts can point this at a local copy of the GSM8K test.jsonl.
GSM8K_DATA_PATH = os.environ.get("SGLANG_TEST_GSM8K_DATA_PATH")
SERVER_LAUNCH_TIMEOUT = 3600
GSM8K_NUM_EXAMPLES = 500
GSM8K_NUM_SHOTS = 20
GSM8K_NUM_THREADS = 128
# Collapse guard only; test_glm53_flash_b200.py owns the recipe's 0.93 floor.
GSM8K_SCORE_FLOOR = 0.90
# Measured gap 0.000 at 500 questions (0.930 / 0.930); 1 sigma is ~0.011.
GSM8K_MAX_GAP = 0.03
PROBE_TOKENS = 2040
# Measured on B200 over the probe: mean |dlogprob| 0.080 and 0.2% of tokens
# past 1 nat, identical across fresh and loaded servers because both backends
# are bit-reproducible here. That is the kernels' numeric difference, amplified
# by MoE routing; a wrong state or a NaN lands far above both bars.
MAX_MEAN_ABS_DLOGPROB = 0.2
MAX_FRAC_ABOVE_1NAT = 0.02

LONG_PROMPT_PATH = os.path.join(
    os.path.dirname(sglang.test.__file__), "long_prompt.txt"
)

COMMON_ARGS = [
    "--tp-size",
    "4",
    "--dsa-prefill-backend",
    "trtllm",
    "--dsa-decode-backend",
    "trtllm",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--moe-runner-backend",
    "flashinfer_trtllm",
    "--chunked-prefill-size",
    "16384",
    "--disable-radix-cache",
]
PTX_ARGS = COMMON_ARGS + ["--linear-attn-prefill-backend", "ptx_kda"]

DISPATCHER_PTX = "extend=PtxKDAKernel"
DISPATCHER_TRITON = "extend=TritonKDAKernel"
PTX_LOADED = "Using PTX KDA chunked prefill"
FALLBACK_MARKERS = (
    "falling back to Triton extend",  # dispatcher rejected the arch
    "Falling back to Triton",  # wrapper: unsupported shape / dtype
    "fell back to Triton",  # wrapper: kernel raised, state restored
)


def _ptx_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    from sglang.kernels.ops.attention.linear.kda_ptx_prefill import SM_ARCHS

    return torch.cuda.get_device_capability() in SM_ARCHS


class TestGLM53FlashKdaPtxPrefill(CustomTestCase):
    """PTX KDA prefill vs Triton on the same weights: GSM8K and prompt logprobs."""

    process = None
    log_file = None

    @classmethod
    def setUpClass(cls):
        if not _ptx_supported():
            raise unittest.SkipTest("PTX KDA prefill requires SM100 or SM103")
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        with open(LONG_PROMPT_PATH) as f:
            text = f.read()
        tokenizer = AutoTokenizer.from_pretrained(cls.model, trust_remote_code=True)
        cls.probe_ids = tokenizer.encode(text)[:PROBE_TOKENS]

    @classmethod
    def tearDownClass(cls):
        cls._stop_server()

    @classmethod
    def _stop_server(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
            cls.process = None
            _wait_for_gpu_idle_in_ci()
        if cls.log_file is not None:
            cls.log_file.close()
            cls.log_file = None

    @classmethod
    def _start_server(cls, other_args):
        cls._stop_server()
        cls.log_file = tempfile.NamedTemporaryFile(
            "w", prefix="kda_ptx_prefill_", suffix=".log", delete=False
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            return_stdout_stderr=(cls.log_file, cls.log_file),
        )

    @classmethod
    def _server_log(cls) -> str:
        cls.log_file.flush()
        with open(cls.log_file.name) as f:
            return f.read()

    def _gsm8k(self) -> float:
        requests.get(self.base_url + "/flush_cache")
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=GSM8K_NUM_EXAMPLES,
            num_shots=GSM8K_NUM_SHOTS,
            num_threads=GSM8K_NUM_THREADS,
            gsm8k_data_path=GSM8K_DATA_PATH,
        )
        metrics = run_eval(args)
        print(f"gsm8k {metrics=}")
        return metrics["score"]

    def _probe_logprobs(self) -> list:
        requests.get(self.base_url + "/flush_cache")
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": self.probe_ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True,
                "logprob_start_len": 0,
            },
            timeout=300,
        )
        response.raise_for_status()
        # Entries are [logprob, token_id, text]. The first token has no logprob,
        # and the JSON encoder writes NaN as null, so count the nulls.
        entries = response.json()["meta_info"]["input_token_logprobs"][1:]
        logprobs = [lp for lp, *_ in entries]
        num_null = sum(lp is None for lp in logprobs)
        self.assertEqual(num_null, 0, f"{num_null} null (NaN) prompt logprobs")
        self.assertEqual(len(logprobs), PROBE_TOKENS - 1)
        return logprobs

    def _run_backend(self, other_args):
        self._start_server(other_args)
        score = self._gsm8k()
        logprobs = self._probe_logprobs()
        log = self._server_log()
        self._stop_server()
        return score, logprobs, log

    def test_ptx_prefill_matches_triton(self):
        triton_score, triton_lp, triton_log = self._run_backend(COMMON_ARGS)
        self.assertIn(DISPATCHER_TRITON, triton_log)

        ptx_score, ptx_lp, ptx_log = self._run_backend(PTX_ARGS)

        # The arch gate and the per-batch fallbacks only show in the log.
        self.assertIn(DISPATCHER_PTX, ptx_log, "dispatcher did not pick PtxKDAKernel")
        self.assertIn(PTX_LOADED, ptx_log, "PTX extension never ran a batch")
        for marker in FALLBACK_MARKERS:
            self.assertNotIn(marker, ptx_log, f"PTX prefill fell back: {marker}")

        diff = [p - t for p, t in zip(ptx_lp, triton_lp)]
        mean_abs = sum(abs(d) for d in diff) / len(diff)
        frac_above_1 = sum(abs(d) > 1 for d in diff) / len(diff)
        max_abs = max(abs(d) for d in diff)
        # k3 (the KL estimator of kl_test_utils) is carried by a few tail tokens;
        # reported for reference, not asserted.
        k3 = sum(math.exp(d) - 1 - d for d in diff) / len(diff)

        summary = (
            f"gsm8k triton={triton_score:.3f} ptx_kda={ptx_score:.3f} "
            f"gap={ptx_score - triton_score:+.3f} | {PROBE_TOKENS}-token prefill: "
            f"mean|dlogprob|={mean_abs:.4f} frac>1nat={frac_above_1:.4f} "
            f"max={max_abs:.3f} k3={k3:.2e}"
        )
        print(summary)
        if is_in_ci():
            write_github_step_summary(
                f"### test_glm53_flash_kda_ptx_prefill\n{summary}\n"
            )

        self.assertGreaterEqual(ptx_score, GSM8K_SCORE_FLOOR)
        self.assertGreaterEqual(ptx_score, triton_score - GSM8K_MAX_GAP)
        self.assertLess(mean_abs, MAX_MEAN_ABS_DLOGPROB)
        self.assertLess(frac_above_1, MAX_FRAC_ABOVE_1NAT)


if __name__ == "__main__":
    unittest.main()

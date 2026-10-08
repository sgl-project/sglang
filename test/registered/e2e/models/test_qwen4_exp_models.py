"""Qwen3.8-Flash-Next (Qwen4-Exp) E2E on B200; the plain-serving case is kept:
MTP's verify widths never exercise the QSA sparse-decode path."""

import time
import unittest

import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

CUDA_RUNNER_CONFIG = "4-gpu-b200"
TP_SIZE = CUDA_RUNNER_CONFIG.partition("-gpu-")[0]

register_cuda_ci(est_time=1500, stage="base-c", runner_config=CUDA_RUNNER_CONFIG)

MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"

SERVER_LAUNCH_TIMEOUT = 3600
GSM8K_SCORE_THRESHOLD = 0.94

BASE_ARGS = [
    "--tp-size",
    TP_SIZE,
    "--mem-fraction-static",
    "0.85",
    "--chunked-prefill-size",
    "8192",
    "--linear-attn-prefill-backend",
    "flashinfer",
    "--linear-attn-decode-backend",
    "flashinfer",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--reasoning-parser",
    "qwen3-thinking",
]

LAYER_NORM_SP_ARGS = [
    "--ep-size",
    TP_SIZE,
    "--disable-radix-cache",
    "--cuda-graph-backend-prefill=breakable",
]
CHUNKED_PREFILL_SIZE = int(BASE_ARGS[BASE_ARGS.index("--chunked-prefill-size") + 1])
SP_PROBE_REMAINDERS = (5, 7)
SP_PROBE_OUTPUT_TOKENS = 8
SP_PROBE_TEXT = "SGLang Qwen4Exp LayerNorm sequence parallel replay probe. "


def _post(path, payload=None, timeout=SERVER_LAUNCH_TIMEOUT):
    response = requests.post(
        DEFAULT_URL_FOR_TEST + path,
        json=payload or {},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json() if response.content else {}


def _output_token_ids(result):
    rows = result["meta_info"].get("output_token_logprobs")
    if rows is None:
        rows = result["meta_info"]["output_logprobs"]
    return tuple(
        int(row["token_id"] if isinstance(row, dict) else row[1]) for row in rows
    )


class _Qwen4ExpServer:
    speculative_args: list[str] = []
    model = try_cached_model(MODEL)
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_backend = "sgl_eval"
    gsm8k_thinking = True
    gsm8k_num_examples = 200
    gsm8k_num_threads = 32
    gsm8k_max_tokens = 16384
    gsm8k_score_threshold = GSM8K_SCORE_THRESHOLD

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=BASE_ARGS + cls.speculative_args,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestQwen4ExpBase(_Qwen4ExpServer, GSM8KMixin, CustomTestCase):
    """Normal autoregressive serving."""


class TestQwen4ExpMTP(_Qwen4ExpServer, GSM8KMixin, CustomTestCase):
    """NEXTN MTP serving (3 steps, topk 1, 4 draft tokens)."""

    # GSM8K accept length measured at 3.02-3.03 (max 4.0 with 3 steps);
    # 2.9 leaves noise margin while still failing on a real drop.
    gsm8k_accept_length_thres = 2.9
    speculative_args = [
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]


class TestQwen4ExpLayerNormSP(CustomTestCase):
    """LayerNorm-SP parity through padded PLE rows and prefill graph replay."""

    model = try_cached_model(MODEL)

    def _run_server(self, enable_sp):
        args = BASE_ARGS + LAYER_NORM_SP_ARGS
        if enable_sp:
            args += ["--enable-layernorm-sp"]
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=args,
        )
        try:
            seed = _post("/tokenize", {"prompt": SP_PROBE_TEXT}, timeout=120)
            seed_ids = seed.get("tokens") or seed.get("input_ids") or seed["token_ids"]
            local_physical_rows = CHUNKED_PREFILL_SIZE // int(TP_SIZE)
            self.assertTrue(
                all(
                    processed_rows < local_physical_rows
                    for processed_rows in SP_PROBE_REMAINDERS
                )
            )
            outputs = []
            # Both final chunks use the same padded PLE/prefill-graph bucket.
            # Changing its processed row count makes the second request replay it.
            for remainder in SP_PROBE_REMAINDERS:
                input_tokens = CHUNKED_PREFILL_SIZE + remainder
                repeats, tail = divmod(input_tokens, len(seed_ids))
                input_ids = seed_ids * repeats + seed_ids[:tail]
                result = _post(
                    "/generate",
                    {
                        "input_ids": input_ids,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": SP_PROBE_OUTPUT_TOKENS,
                            "ignore_eos": True,
                        },
                        "return_logprob": True,
                    },
                )
                outputs.append(_output_token_ids(result))
            return outputs
        finally:
            kill_process_tree(process.pid)
            time.sleep(5)

    def test_off_on_outputs_match_with_padded_ple_graph_replay(self):
        off_outputs = self._run_server(enable_sp=False)
        on_outputs = self._run_server(enable_sp=True)
        self.assertEqual(off_outputs, on_outputs)


if __name__ == "__main__":
    unittest.main()

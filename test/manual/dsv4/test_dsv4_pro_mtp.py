"""DSV4-Pro 1.6T MTP performance tests on B200 TP=8.

1. TestDSV4ProMTPSimulatedAcc — `SGLANG_SIMULATE_ACC_LEN=3` pins EAGLE accept
   length so latency comparisons are apples-to-apples. Runs `bench_one_batch_server`
   at bs=1 for isl=4096 and isl=900000 (osl=1024).

2. TestDSV4ProMTPHongloumeng — real EAGLE accept (no SIMULATE) on Chinese
   long-context input (`hongloumeng.txt`, ~627k DSV4 tokens). Builds a one-line
   custom JSONL dataset on the fly and drives `bench_serving --dataset-name custom`
   with one short slice (30k tokens) and the full long prompt.

Manual test (8× B200, 1.6T weights). Not registered in CI.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import requests

from sglang.bench_one_batch_server import BenchArgs as OneBatchBenchArgs
from sglang.bench_one_batch_server import run_benchmark as run_one_batch_benchmark
from sglang.bench_serving import run_benchmark as run_serving_benchmark
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DSV4_PRO_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Pro"

HONGLOUMENG_PATH = os.environ.get(
    "SGLANG_HONGLOUMENG_PATH",
    os.path.join(os.path.dirname(__file__), "hongloumeng.txt"),
)

DSV4_PRO_BASE_ENV = {
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_OPT_USE_TOPK_V2": "1",
    "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": "1",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
}

DSV4_PRO_SERVER_ARGS = [
    "--trust-remote-code",
    "--tp",
    "8",
    "--moe-runner-backend",
    "flashinfer_mxfp4",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--chunked-prefill-size",
    "4096",
    "--disable-flashinfer-autotune",
    "--mem-fraction-static",
    "0.82",
    "--max-running-requests",
    "8",
]


def _launch_dsv4_pro_server(extra_env=None):
    env = dict(DSV4_PRO_BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return popen_launch_server(
        DSV4_PRO_MODEL_PATH,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 4,
        other_args=DSV4_PRO_SERVER_ARGS,
        env=env,
    )


class TestDSV4ProMTPSimulatedAcc(CustomTestCase):
    """bs=1 latency at isl=4096 / 900000 with `SGLANG_SIMULATE_ACC_LEN=3`.

    Reference (B200 Pro TP8):
      - isl=4096   → output 194.6 tok/s, accept 2.96
      - isl=900000 → output 174.6 tok/s, accept 2.93
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = _launch_dsv4_pro_server(
            extra_env={"SGLANG_SIMULATE_ACC_LEN": "3"}
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _run_one_batch(self, input_len):
        requests.get(self.base_url + "/flush_cache")
        server_args = ServerArgs(model_path=DSV4_PRO_MODEL_PATH)
        bench_args = OneBatchBenchArgs(
            run_name=f"dsv4_pro_simacc_isl{input_len}",
            batch_size=(1,),
            input_len=(input_len,),
            output_len=(1024,),
            base_url=self.base_url,
            skip_warmup=True,
            result_filename=os.path.join(
                tempfile.gettempdir(), f"dsv4_pro_simacc_isl{input_len}.jsonl"
            ),
            append_to_github_summary=False,
        )
        results, _ = run_one_batch_benchmark(server_args, bench_args)
        self.assertTrue(results, "bench_one_batch_server returned no results")
        return results[0]

    def test_isl_4096(self):
        r = self._run_one_batch(4096)
        print(
            f"[pro simacc isl=4096] output_throughput={r.output_throughput:.2f} tok/s "
            f"latency={r.latency:.2f}s last_ttft={r.last_ttft:.2f}s "
            f"acc_length={r.acc_length:.2f}"
        )
        # Reference 194.6 tok/s / acc=2.96 — give 10% throughput margin and a
        # generous accept-length floor to absorb run-to-run jitter.
        self.assertGreater(r.output_throughput, 175.0)
        self.assertGreater(r.acc_length, 2.85)

    def test_isl_900k(self):
        r = self._run_one_batch(900_000)
        print(
            f"[pro simacc isl=900k] output_throughput={r.output_throughput:.2f} tok/s "
            f"latency={r.latency:.2f}s last_ttft={r.last_ttft:.2f}s "
            f"acc_length={r.acc_length:.2f}"
        )
        # Reference 174.6 tok/s / acc=2.93.
        self.assertGreater(r.output_throughput, 155.0)
        self.assertGreater(r.acc_length, 2.85)


def _build_hongloumeng_jsonl(num_tokens, tokenizer, out_path):
    """Slice the first `num_tokens` DSV4 tokens of hongloumeng.txt into a
    one-line CustomDataset JSONL. Pass num_tokens=None to keep the full text.
    """
    with open(HONGLOUMENG_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    if num_tokens is not None:
        ids = tokenizer.encode(text)
        text = tokenizer.decode(ids[:num_tokens])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"conversations": [{"value": text}, {"value": "x"}]},
                ensure_ascii=False,
            )
            + "\n"
        )
    return out_path


class TestDSV4ProMTPHongloumeng(CustomTestCase):
    """Real EAGLE accept on Chinese long-context (hongloumeng.txt).

    Reference (B200 Pro TP8, no SIMULATE):
      - isl=30000  → output 124.4 tok/s, decode peak 184 tok/s, accept 2.47
      - isl=627059 → output 125.7 tok/s, decode peak 179 tok/s, accept 2.52
    """

    SHORT_TOKENS = 30_000
    LONG_TOKENS = None  # full file (~627k DSV4 tokens)
    OUTPUT_TOKENS = 4096

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = _launch_dsv4_pro_server()

        # Resolve tokenizer once; the server reports its own tokenizer path so
        # on-the-fly token-level slicing matches what the server will see.
        info = requests.get(cls.base_url + "/server_info", timeout=60).json()
        tokenizer_path = info.get("tokenizer_path") or DSV4_PRO_MODEL_PATH
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        cls.tokenizer = get_tokenizer(tokenizer_path)

        cls.tmpdir = tempfile.mkdtemp(prefix="dsv4_hongloumeng_")
        cls.short_jsonl = _build_hongloumeng_jsonl(
            cls.SHORT_TOKENS,
            cls.tokenizer,
            os.path.join(cls.tmpdir, "hongloumeng_30k.jsonl"),
        )
        cls.long_jsonl = _build_hongloumeng_jsonl(
            cls.LONG_TOKENS,
            cls.tokenizer,
            os.path.join(cls.tmpdir, "hongloumeng_full.jsonl"),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _run_custom_bench(self, dataset_path):
        requests.get(self.base_url + "/flush_cache")
        args = SimpleNamespace(
            backend="sglang",
            base_url=self.base_url,
            host=None,
            port=None,
            dataset_name="custom",
            dataset_path=dataset_path,
            model=None,
            tokenizer=None,
            num_prompts=1,
            sharegpt_output_len=self.OUTPUT_TOKENS,
            sharegpt_context_len=None,
            random_input_len=4096,
            random_output_len=2048,
            random_range_ratio=0.0,
            request_rate=float("inf"),
            max_concurrency=1,
            warmup_requests=0,
            flush_cache=True,
            multi=None,
            output_file=None,
            disable_tqdm=False,
            disable_stream=False,
            return_logprob=False,
            return_routed_experts=False,
            seed=0,
            disable_ignore_eos=False,
            extra_request_body=None,
            apply_chat_template=False,
            profile=None,
            lora_name=None,
            lora_request_distribution="uniform",
            lora_zipf_alpha=1.5,
            prompt_suffix="",
            device="cuda",
            pd_separated=False,
            ready_check_timeout_sec=0,
        )
        return run_serving_benchmark(args)

    def test_short_30k(self):
        res = self._run_custom_bench(self.short_jsonl)
        print(
            f"[hongloumeng 30k] output_throughput={res['output_throughput']:.2f} tok/s "
            f"accept_length={res['accept_length']:.2f} "
            f"mean_ttft_ms={res['mean_ttft_ms']:.0f} "
            f"mean_tpot_ms={res['mean_tpot_ms']:.2f}"
        )
        # Reference 124 tok/s / accept 2.47.
        self.assertGreater(res["output_throughput"], 105.0)
        self.assertGreater(res["accept_length"], 2.30)

    def test_long_full(self):
        res = self._run_custom_bench(self.long_jsonl)
        print(
            f"[hongloumeng full] output_throughput={res['output_throughput']:.2f} tok/s "
            f"accept_length={res['accept_length']:.2f} "
            f"mean_ttft_ms={res['mean_ttft_ms']:.0f} "
            f"mean_tpot_ms={res['mean_tpot_ms']:.2f}"
        )
        # Reference 125 tok/s / accept 2.52. Cold prefill takes ~85s on 627k
        # tokens so the run is dominated by prefill, but decode steady-state
        # accept_length is the metric we care about.
        self.assertGreater(res["output_throughput"], 105.0)
        self.assertGreater(res["accept_length"], 2.30)


if __name__ == "__main__":
    unittest.main()

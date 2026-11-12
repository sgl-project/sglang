"""GLM-5.2 NVFP4: DSA prefill work inside the breakable prefill CUDA graph.

Each class serves one TP4 EAGLE configuration once per arm and sends every arm
the same token-id requests. The deterministic FlashInfer DSA top-k makes each
prefill reproducible, so the arms must agree bitwise on the input logprobs and
the greedy output tokens.
"""

import random
import re
import tempfile
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gb300_utils import GB300_NCCL_PORT
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=3300, stage="nightly", runner_config="4-gpu-gb300")

MODEL = "nvidia/GLM-5.2-NVFP4"
SERVER_ARGS = [
    "--trust-remote-code",
    "--tp",
    "4",
    "--quantization",
    "modelopt_fp4",
    "--moe-runner-backend",
    "flashinfer_trtllm",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--mem-fraction-static",
    "0.8",
    "--max-running-requests",
    "16",
    "--chunked-prefill-size",
    "8192",
    "--dsa-topk-backend",
    "flashinfer",
    "--speculative-dsa-topk-backend",
    "flashinfer",
    "--enable-cache-report",
    "--nccl-port",
    GB300_NCCL_PORT,
]
# Reproducible top-k for kv > top-k, so every prefill has a bitwise reference.
DETERMINISTIC_ENV = {"SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC": "1"}
# Plain-text ids of the GLM-5.x vocabulary (154880 entries).
_TOKEN_LO, _TOKEN_HI = 1000, 150000
_LOGPROB_TAIL = 256
_OUTPUT_LEN = 8

_GRAPH_ATTN_LOG = re.compile(
    r"DSA prefill attention runs inside the breakable prefill CUDA graph "
    r"for buckets of >= (\d+) rows"
)
_DRAFT_GRAPH_LOG = "Capture draft prefill CUDA graph end"
_LAYERWISE_LOAD_LOG = "a cache tier loads prefix KV layer by layer"


def _rand_ids(rng, n):
    return [rng.randrange(_TOKEN_LO, _TOKEN_HI) for _ in range(n)]


def _generate(base_url, input_ids, *, max_new_tokens=_OUTPUT_LEN, logprob=True):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    if logprob:
        lens = (
            [len(ids) for ids in input_ids] if isinstance(input_ids[0], list) else None
        )
        payload["return_logprob"] = True
        payload["logprob_start_len"] = (
            [max(0, n - _LOGPROB_TAIL) for n in lens]
            if lens is not None
            else max(0, len(input_ids) - _LOGPROB_TAIL)
        )
    resp = requests.post(base_url + "/generate", json=payload, timeout=1800)
    resp.raise_for_status()
    return resp.json()


def _signature(out):
    meta = out["meta_info"]
    return [entry[0] for entry in meta["input_token_logprobs"]], out["output_ids"]


def _host_hit_tokens(out):
    return (out["meta_info"].get("cached_tokens_details") or {}).get("host", 0)


def _serve(*, env, extra_args, run):
    """Launch one arm, return (run(base_url), server log)."""
    base_url = DEFAULT_URL_FOR_TEST
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        process = popen_launch_server(
            MODEL,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=SERVER_ARGS + extra_args,
            env={**DETERMINISTIC_ENV, **env},
            return_stdout_stderr=(out, err),
        )
        try:
            result = run(base_url)
        finally:
            terminate_and_kill_process_tree(process)
        out.seek(0)
        err.seek(0)
        return result, out.read() + err.read()


def _accept_length(base_url):
    items = ("apples", "pencils", "books", "tickets")
    prompts = [
        f"Question: {a} {item} cost {b} dollars. How much do {c} {item} cost?\n"
        "Answer: Let's think step by step."
        for item in items
        for a, b, c in ((3, 6, 7), (5, 10, 12), (4, 2, 9), (6, 9, 20), (7, 21, 11),
                        (8, 4, 30), (9, 27, 5), (2, 7, 13))
    ]  # fmt: skip
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompts,
            "sampling_params": {"temperature": 0, "max_new_tokens": 64},
        },
        timeout=1800,
    )
    resp.raise_for_status()
    metas = [out["meta_info"] for out in resp.json()]
    return sum(m["completion_tokens"] for m in metas) / sum(
        m["spec_verify_ct"] for m in metas
    )


class TestGLM52NVFP4DsaPrefillCudaGraph(CustomTestCase):
    """Captured DSA prefill attention + indexer prep and the NextN draft prefill
    graph (the default) against both as eager graph breaks, without HiCache
    (`graph` vs `eager`) and under HiCache with every prefix loaded back from
    host (`hicache` vs `hicache_eager`: the same pool, so the same KV slots)."""

    # Small device pool so the filler prompts evict every prefix to host.
    DEVICE_TOKENS = 98304
    EAGER_ENV = {
        "SGLANG_ENABLE_DSA_PREFILL_ATTN_IN_CUDA_GRAPH": "0",
        "SGLANG_ENABLE_NEXTN_DRAFT_PREFILL_CUDA_GRAPH": "0",
    }
    HICACHE_ARGS = [
        "--enable-hierarchical-cache",
        "--hicache-size",
        "135",
        "--hicache-write-policy",
        "write_back",
        "--hicache-io-backend",
        "direct",
        "--max-total-tokens",
        str(DEVICE_TOKENS),
    ]
    ARMS = {
        "eager": (EAGER_ENV, []),
        "graph": ({}, []),
        "hicache_eager": (EAGER_ENV, HICACHE_ARGS),
        "hicache": ({}, HICACHE_ARGS),
    }
    REFERENCE = {"graph": "eager", "hicache": "hicache_eager"}

    @classmethod
    def setUpClass(cls):
        cls.results, cls.logs = {}, {}
        for arm, (env, extra_args) in cls.ARMS.items():
            evict = arm.startswith("hicache")
            cls.results[arm], cls.logs[arm] = _serve(
                env=env,
                extra_args=extra_args,
                run=lambda url, evict=evict: cls._workload(url, evict=evict),
            )

    @classmethod
    def _evict(cls, base_url, rng):
        for _ in range(cls.DEVICE_TOKENS // 8192 + 2):
            _generate(base_url, _rand_ids(rng, 8192), max_new_tokens=1, logprob=False)

    @classmethod
    def _workload(cls, base_url, *, evict):
        rng = random.Random(4242)
        # Fillers draw from their own stream so every arm sends the same requests.
        filler_rng = random.Random(4444)
        signatures, host_hits = {}, {}
        # 40 rows stay below the bitwise bucket floor (an eager break in every arm).
        for n in (40, 300, 1000, 2600, 4000):
            signatures[f"fresh{n}"] = _signature(_generate(base_url, _rand_ids(rng, n)))
        prefixes = {p: _rand_ids(rng, p) for p in (8192, 32768)}
        for prefix_len, new_lens in ((8192, (128, 512, 1536)), (32768, (512, 3968))):
            prefix = prefixes[prefix_len]
            _generate(base_url, prefix, max_new_tokens=1, logprob=False)
            for n in new_lens:
                if evict:
                    cls._evict(base_url, filler_rng)
                out = _generate(base_url, prefix + _rand_ids(rng, n))
                name = f"hit{prefix_len}+{n}"
                signatures[name] = _signature(out)
                host_hits[name] = _host_hit_tokens(out)
        if evict:
            cls._evict(base_url, filler_rng)
        batch = [
            prefixes[8192] + _rand_ids(rng, 512),
            prefixes[32768] + _rand_ids(rng, 128),
            _rand_ids(rng, 3000),
        ]
        for name, out in zip(
            ("8192+512", "32768+128", "fresh3000"), _generate(base_url, batch)
        ):
            signatures[f"batch_{name}"] = _signature(out)
            host_hits[f"batch_{name}"] = _host_hit_tokens(out)
        return dict(
            signatures=signatures,
            host_hits=host_hits,
            accept_length=_accept_length(base_url),
        )

    def test_paths_engage(self):
        for arm, reference in self.REFERENCE.items():
            with self.subTest(arm):
                match = _GRAPH_ATTN_LOG.search(self.logs[arm])
                self.assertIsNotNone(match)
                # Smallest bucket whose real rows all exceed SMs / 2 (152 SMs).
                self.assertEqual(int(match.group(1)), 96)
                self.assertIn(_DRAFT_GRAPH_LOG, self.logs[arm])
                self.assertNotIn(_LAYERWISE_LOAD_LOG, self.logs[arm])
                self.assertIsNone(_GRAPH_ATTN_LOG.search(self.logs[reference]))
                self.assertNotIn(_DRAFT_GRAPH_LOG, self.logs[reference])

    def test_outputs_bitwise_equal_to_eager_breaks(self):
        for arm, reference in self.REFERENCE.items():
            want = self.results[reference]["signatures"]
            for name, signature in self.results[arm]["signatures"].items():
                with self.subTest(arm=arm, request=name):
                    self.assertEqual(signature, want[name])

    def test_hicache_prefixes_come_back_from_host(self):
        for arm in ("hicache", "hicache_eager"):
            for name, host in self.results[arm]["host_hits"].items():
                if name != "batch_fresh3000":
                    with self.subTest(arm=arm, request=name):
                        self.assertGreater(host, 0)

    def test_draft_prefill_graph_keeps_acceptance(self):
        # Padded draft buckets may round the draft's KV differently (one FP8 code),
        # which flips near-tie drafts; a broken draft state costs far more.
        self.assertGreaterEqual(
            self.results["graph"]["accept_length"],
            self.results["eager"]["accept_length"] - 0.1,
        )


if __name__ == "__main__":
    unittest.main()

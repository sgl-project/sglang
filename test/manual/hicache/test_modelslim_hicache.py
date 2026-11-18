"""L2 round-trip correctness for the manual Ascend ModelSlim KV tests.

Set SGLANG_TEST_HICACHE_ACCURACY_BASELINE=1 to also score a device-only run.
"""

import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import requests

from sglang.test.kits.cache_hit_kit import gen_payload
from sglang.test.kl_test_utils import _flush_cache
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# Both models cache this prefix on a whole-page boundary (128 / 512 tokens).
_PREFIX_TOKENS = 3072
_SUFFIX_TOKENS = 128


def _tokens(rng, length):
    return [rng.randrange(1000, 20000) for _ in range(length)]


def _generate(url, ids, output_len=1, *, logprobs=False):
    payload = gen_payload(ids, output_len, lora_path=None)
    payload.update(stream=False, return_logprob=logprobs)
    response = requests.post(f"{url}/generate", json=payload, timeout=600)
    response.raise_for_status()
    meta = response.json()["meta_info"]
    assert meta["completion_tokens"] == output_len, meta["finish_reason"]
    return meta


class ModelSlimHiCacheMixin:
    """Each subclass supplies its checkpoint and non-speculative launch profile."""

    small_pool_args = ()

    @contextmanager
    def server(self, *, hicache, small_pool):
        description = Path(self.model) / "quant_model_description.json"
        with description.open(encoding="utf-8") as f:
            kv_types = {v for k, v in json.load(f).items() if k.endswith(".quant_type")}
            self.assertIn(self.kv_quant_type, kv_types, str(description))
        args = [
            "--device",
            "npu",
            "--attention-backend",
            "ascend",
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--quantization",
            "modelslim",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--tp-size",
            self.tp_size,
            "--page-size",
            self.page_size,
            "--mem-fraction-static",
            "0.85",
            "--enable-cache-report",
            "--base-gpu-id",
            os.environ.get("BASE_GPU_ID", "0"),
            *self.model_args,
        ]
        if small_pool:
            args += ["--max-total-tokens", 8192, *self.small_pool_args]
        if hicache:
            args += [
                "--enable-hierarchical-cache",
                "--hicache-io-backend",
                "kernel_ascend",
                "--hicache-write-policy",
                "write_through",
            ]
            if small_pool:
                args += ["--hicache-ratio", 4]
        port = find_available_port(int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1]))
        self.base_url = f"http://127.0.0.1:{port}"
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=3600,
            device="npu",
            other_args=args,
            env={
                "ASCEND_USE_FIA": "1",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                # MLA otherwise defaults to ACL-registered host memory.
                "SGLANG_HICACHE_HOST_MEM_BACKEND": "",
                **self.model_env,
            },
        )
        try:
            response = requests.get(f"{self.base_url}/server_info", timeout=30)
            response.raise_for_status()
            info = response.json()
            self.assertIsNone(info["speculative_algorithm"])
            self.assertFalse(info["disable_radix_cache"])
            self.assertEqual(info["enable_hierarchical_cache"], hicache)
            self.assertIsNone(info["hicache_storage_backend"])
            yield
        finally:
            terminate_and_kill_process_tree(process)

    def accuracy_modes(self):
        # Optional same-card L2-off calibration before the normal L2-on run.
        return (
            (False, True)
            if os.getenv("SGLANG_TEST_HICACHE_ACCURACY_BASELINE") == "1"
            else (True,)
        )

    def test_l2_round_trip(self):
        rng = random.Random(1)
        prefix = _tokens(rng, _PREFIX_TOKENS)
        prime = prefix + _tokens(rng, _SUFFIX_TOKENS)
        probe = prefix + _tokens(rng, _SUFFIX_TOKENS)
        runs = []
        with self.server(hicache=True, small_pool=True):
            # Two resident runs measure numerical repeatability with the same prefill boundary.
            for evict in (False, False, True):
                _flush_cache(self.base_url)
                self.assertEqual(_generate(self.base_url, prime)["cached_tokens"], 0)
                if evict:
                    for _ in range(6):
                        _generate(self.base_url, _tokens(rng, len(prime)))
                meta = _generate(self.base_url, probe, 16, logprobs=True)
                details = meta["cached_tokens_details"] or {}
                self.assertEqual(details.get("storage", 0), 0, details)
                if evict:
                    self.assertGreaterEqual(
                        details.get("host", 0), self.page_size, details
                    )
                else:
                    self.assertGreaterEqual(
                        details.get("device", 0), self.page_size, details
                    )
                    self.assertEqual(details["host"], 0, details)
                runs.append(meta)

        resident, _, restored = runs
        reference_ids = [t[1] for t in resident["output_token_logprobs"]]
        logprobs = []
        for meta in runs:
            self.assertEqual(meta["cached_tokens"], resident["cached_tokens"])
            entries = meta["output_token_logprobs"]
            self.assertEqual(len(entries), 16)
            self.assertEqual([t[1] for t in entries], reference_ids)
            values = np.array([t[0] for t in entries])
            self.assertTrue(np.isfinite(values).all(), entries)
            logprobs.append(values)
        ref, repeated, reloaded = logprobs
        atol = max(1e-5, float(np.max(np.abs(ref - repeated))))
        np.testing.assert_allclose(reloaded, ref, rtol=0, atol=atol)
        print(
            f"L1 {resident['cached_tokens_details']}, L2 {restored['cached_tokens_details']}, "
            f"max logprob delta={np.max(np.abs(reloaded - ref)):.6g}, atol={atol:.6g}"
        )

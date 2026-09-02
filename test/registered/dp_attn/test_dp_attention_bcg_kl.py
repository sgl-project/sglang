from __future__ import annotations

import os
import random
import re
import shutil
import tempfile
import time
import unittest

import numpy as np
import requests

from sglang.srt.utils import get_device_capability, is_blackwell, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _get_input_logprobs,
    get_input_ids,
)
from sglang.test.test_utils import (
    DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=320, stage="base-b", runner_config="2-gpu-large")

PREFILL_GRAPH_REPLAY_PATTERN = re.compile(r"Prefill batch.*cuda graph: True")
CACHED_PREFIX_GRAPH_REPLAY_PATTERN = re.compile(
    r"Prefill batch.*#cached-token: [1-9][0-9]*.*cuda graph: True"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_input_ids(tokenizer_path, num_samples, max_prompt_tokens):
    try:
        return get_input_ids(
            tokenizer_path,
            max_prompt_tokens=max_prompt_tokens,
            num_samples=num_samples,
        )
    except (ValueError, OSError) as e:
        print(
            f"WARNING: Could not load LongBench inputs with tokenizer "
            f"'{tokenizer_path}': {e}"
        )
        print("Falling back to random token IDs")
        return [
            [
                random.randint(1, 32000 - 1)
                for _ in range(int(max_prompt_tokens * random.uniform(0.5, 1.5)))
            ]
            for _ in range(num_samples)
        ]


def _compute_kl(input_logprobs, output_logprobs):
    kl_divs = []
    for idx, (inp_lp, out_lp) in enumerate(zip(input_logprobs, output_logprobs)):
        inp_none = any(v is None for v in inp_lp)
        out_none = any(v is None for v in out_lp)
        if inp_none or out_none:
            src = "input" if inp_none else "output"
            if inp_none and out_none:
                src = "input and output"
            print(f"  WARNING: sample {idx}: skipping due to None in {src} logprobs")
            continue
        logr = np.array(inp_lp) - np.array(out_lp)
        kl_divs.append(float(np.mean((np.exp(logr) - 1) - logr)))
    avg = sum(kl_divs) / len(kl_divs)
    print(f"  per-sample KL: {kl_divs}")
    print(f"  avg KL: {avg:.6f}")
    return avg


def _device_only_hit(meta_info):
    details = meta_info.get("cached_tokens_details") or {}
    if (details.get("host", 0) or 0) > 0:
        return 0
    return details.get("device", 0) or 0


# ---------------------------------------------------------------------------
# Hit detection helpers
# ---------------------------------------------------------------------------


def _is_prefill_hit(result, is_hicache):
    if is_hicache:
        return _device_only_hit(result["meta_info"]) > 0
    return result["meta_info"]["cached_tokens"] > 0


def _is_decode_hit(result, first_turn_len, is_hicache):
    if result["meta_info"]["cached_tokens"] <= first_turn_len + 1:
        return False
    if is_hicache:
        return _device_only_hit(result["meta_info"]) > 0
    return True


def _hit_info(result, is_hicache):
    if is_hicache:
        return f"device_only={_device_only_hit(result['meta_info'])}"
    return f"cached_tokens={result['meta_info']['cached_tokens']}"


# ---------------------------------------------------------------------------
# Prefill / decode cache hit tests
# ---------------------------------------------------------------------------


def test_prefill_cache_hit(base_url, input_ids, max_new_tokens, is_hicache=False):
    label = "device (L1) " if is_hicache else ""
    print(f"--- Prefill {label}cache hit KL test ---")
    _flush_cache(base_url)
    _generate(base_url, input_ids, max_new_tokens=0)

    results = _generate(base_url, input_ids, max_new_tokens, return_logprob=True)
    new_input_ids, output_logprobs = [], []
    for i, r in enumerate(results):
        hit = _is_prefill_hit(r, is_hicache)
        info = _hit_info(r, is_hicache)
        print(f"  [{i}] prefix_len={len(input_ids[i])} {info} hit={hit}")
        if not hit:
            continue
        new_input_ids.append(input_ids[i] + r["output_ids"])
        output_logprobs.append(_extract_output_logprobs(r))

    hit_label = "L1 hits" if is_hicache else "cache hits"
    print(f"  {hit_label}: {len(new_input_ids)}/{len(input_ids)}")
    assert (
        len(new_input_ids) > len(input_ids) // 2
    ), f"too few {hit_label}: {len(new_input_ids)}/{len(input_ids)}"

    input_logprobs = _get_input_logprobs(base_url, new_input_ids, output_logprobs)
    return _compute_kl(input_logprobs, output_logprobs)


def test_decode_cache_hit(base_url, input_ids, max_new_tokens, is_hicache=False):
    label = "device (L1) " if is_hicache else ""
    print(f"--- Decode {label}cache hit KL test ---")
    suffix_token = [1]

    _flush_cache(base_url)
    first = _generate(base_url, input_ids, max_new_tokens, return_logprob=True)
    turn2_ids = [
        input_ids[i] + r["output_ids"] + suffix_token for i, r in enumerate(first)
    ]

    results = _generate(base_url, turn2_ids, max_new_tokens, return_logprob=True)
    new_input_ids, output_logprobs = [], []
    for i, r in enumerate(results):
        hit = _is_decode_hit(r, len(input_ids[i]), is_hicache)
        info = _hit_info(r, is_hicache)
        print(f"  [{i}] prefix_len={len(turn2_ids[i])} {info} hit={hit}")
        if not hit:
            continue
        new_input_ids.append(turn2_ids[i] + r["output_ids"])
        output_logprobs.append(_extract_output_logprobs(r))

    hit_label = "L1 decode hits" if is_hicache else "cache hits"
    print(f"  {hit_label}: {len(new_input_ids)}/{len(turn2_ids)}")
    assert (
        len(new_input_ids) > len(turn2_ids) // 2
    ), f"too few {hit_label}: {len(new_input_ids)}/{len(turn2_ids)}"

    input_logprobs = _get_input_logprobs(base_url, new_input_ids, output_logprobs)
    return _compute_kl(input_logprobs, output_logprobs)


# ---------------------------------------------------------------------------
# Server test
# ---------------------------------------------------------------------------


def _select_attention_backend():
    major, minor = get_device_capability()
    if major == 9:
        return "fa3"
    if is_blackwell():
        return "fa4"
    raise NotImplementedError(
        f"DP attention BCG KL test only supports Hopper (fa3) and "
        f"Blackwell (fa4); got compute capability {major}.{minor}"
    )


class _DPAttentionPrefillCudaGraphKLMixin:
    num_samples = 48
    max_prompt_tokens = 1024
    max_new_tokens = 256
    prefill_backend: str

    @classmethod
    def setUpClass(cls):
        random.seed(42)
        cls.model = DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.attention_backend = _select_attention_backend()
        cls.log_dir = tempfile.mkdtemp(prefix=f"dp_attn_{cls.prefill_backend}_")
        cls.stdout_path = os.path.join(cls.log_dir, "server.out")
        cls.stderr_path = os.path.join(cls.log_dir, "server.err")
        cls.stdout = open(cls.stdout_path, "w")
        cls.stderr = open(cls.stderr_path, "w")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--enable-deterministic-inference",
                "--attention-backend",
                cls.attention_backend,
                "--moe-runner-backend",
                "triton",
                f"--cuda-graph-backend-prefill={cls.prefill_backend}",
                "--chunked-prefill-size",
                "2048",
                "--prefill-max-requests",
                "2",
                "--mem-fraction-static",
                "0.70",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

        server_info = requests.get(f"{cls.base_url}/server_info", timeout=30).json()
        tokenizer_path = (
            server_info.get("tokenizer_path")
            or server_info.get("model_path")
            or cls.model
        )
        cls.input_ids = _load_input_ids(
            tokenizer_path, cls.num_samples, cls.max_prompt_tokens
        )
        print(f"Built {len(cls.input_ids)} prompts\n")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        for attr in ("stdout", "stderr"):
            output = getattr(cls, attr, None)
            if output is not None and not output.closed:
                output.close()
        if hasattr(cls, "log_dir"):
            shutil.rmtree(cls.log_dir, ignore_errors=True)

    def _wait_for_prefill_graph_replay(
        self,
        offsets,
        pattern=PREFILL_GRAPH_REPLAY_PATTERN,
        case="a lone request",
    ):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            chunks = []
            for path, offset in zip(
                (self.stdout_path, self.stderr_path), offsets, strict=True
            ):
                with open(path, "rb") as log:
                    log.seek(offset)
                    chunks.append(log.read().decode(errors="replace"))
            logs = "\n".join(chunks)
            if pattern.search(logs):
                return
            time.sleep(0.5)
        self.fail(
            f"No {self.prefill_backend} prefill CUDA graph replay was logged "
            f"for {case}"
        )

    def test_prefill_and_decode_cache_hit_kl_is_zero(self):
        server_info = requests.get(self.base_url + "/server_info", timeout=30).json()
        self.assertFalse(server_info["disable_radix_cache"])
        self.assertTrue(server_info["enable_dp_attention"])
        self.assertTrue(server_info["enable_deterministic_inference"])
        self.assertEqual(server_info["attention_backend"], self.attention_backend)
        self.assertEqual(
            server_info["cuda_graph_config"]["prefill"]["backend"],
            self.prefill_backend,
        )

        print("=== Radix Cache KL Divergence Eval ===")
        print(f"Server: {self.base_url}  Samples: {self.num_samples}\n")

        offsets = [
            os.path.getsize(path) for path in (self.stdout_path, self.stderr_path)
        ]
        prefill_kl = test_prefill_cache_hit(
            self.base_url, self.input_ids, self.max_new_tokens
        )
        self._wait_for_prefill_graph_replay(
            offsets,
            CACHED_PREFIX_GRAPH_REPLAY_PATTERN,
            "a cached-prefix request",
        )
        decode_kl = test_decode_cache_hit(
            self.base_url, self.input_ids, self.max_new_tokens
        )

        self.assertEqual(prefill_kl, 0.0)
        self.assertEqual(decode_kl, 0.0)

    def test_lone_request_replays_prefill_cuda_graph(self):
        _flush_cache(self.base_url)
        offsets = [
            os.path.getsize(path) for path in (self.stdout_path, self.stderr_path)
        ]
        result = _generate(
            self.base_url,
            self.input_ids[0],
            max_new_tokens=1,
            return_logprob=True,
        )
        self.assertNotIn("error", result)
        self._wait_for_prefill_graph_replay(offsets)


class TestDPAttentionBreakablePrefillCudaGraphKL(
    _DPAttentionPrefillCudaGraphKLMixin, CustomTestCase
):
    prefill_backend = "breakable"


class TestDPAttentionFullPrefillCudaGraphKL(
    _DPAttentionPrefillCudaGraphKLMixin, CustomTestCase
):
    prefill_backend = "full"


if __name__ == "__main__":
    unittest.main()

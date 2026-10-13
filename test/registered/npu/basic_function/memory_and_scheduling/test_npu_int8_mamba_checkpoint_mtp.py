import os
import random
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=3600, suite="full-2-npu-a3", nightly=True)

MODEL = os.environ.get("SGLANG_TEST_MODEL", "/mnt/paas/weights/Qwen3.6-35B-A3B")
BASE_URL = f"http://127.0.0.1:{find_available_port(10000)}"
BASE_GPU_ID = os.environ.get("BASE_GPU_ID")

# The checkpoint pool has an explicit size (3x active slots), not the 2x default.
MAX_MAMBA_CACHE_SIZE = 64
INT8_MAMBA_CKPT_SIZE = 192
PREFIX_TOKENS = 1024
SUFFIX_TOKENS = 16
K_SCAN = [16, 32, 64, 128, 256]
PARALLEL = 8
COLLAPSE_THRESHOLD = 0.5
REUSE_GAP = 0.3

NPU_ENV = {
    "NO_PROXY": "127.0.0.1,localhost",
    "no_proxy": "127.0.0.1,localhost",
    "ASCEND_USE_FIA": "1",
    "SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "SGLANG_DEEPEP_BF16_DISPATCH": "1",
    "ENABLE_ASCEND_MOE_NZ": "1",
}


def _random_ids(length, seed):
    rng = random.Random(seed)
    return [rng.randint(1, 30000) for _ in range(length)]


def _make_capacity_requests():
    """Build identical cache-capacity probes for every server configuration."""
    plan = {}
    for K in K_SCAN:
        prefixes = [
            _random_ids(1, 1000 + i) + _random_ids(PREFIX_TOKENS, 2000 + i)
            for i in range(K)
        ]
        warm = [
            prefix + _random_ids(SUFFIX_TOKENS, 3000 + i)
            for i, prefix in enumerate(prefixes)
        ]
        probe = [
            prefix + _random_ids(SUFFIX_TOKENS, 4000 + i)
            for i, prefix in enumerate(prefixes)
        ]
        plan[K] = (warm, probe)
    return plan


def _generate(payload):
    response = requests.post(BASE_URL + "/generate", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def _flush_cache():
    response = requests.post(BASE_URL + "/flush_cache", timeout=60)
    response.raise_for_status()
    time.sleep(1.5)


def _capacity_payload(input_ids):
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1,
            "ignore_eos": True,
        },
    }


def _generate_capacity_request(input_ids):
    return _generate(_capacity_payload(input_ids))


def _measure_running_server(int8_enabled, mtp_enabled, plan, answer_prompt):
    info_response = requests.get(BASE_URL + "/server_info", timeout=30)
    info_response.raise_for_status()
    server_info = info_response.json()
    assert server_info["max_mamba_cache_size"] == MAX_MAMBA_CACHE_SIZE
    assert server_info["enable_int8_mamba_checkpoint"] is int8_enabled
    if int8_enabled:
        assert server_info["int8_mamba_ckpt_size"] == INT8_MAMBA_CKPT_SIZE
    has_draft_tokens = bool(server_info.get("speculative_num_draft_tokens"))
    assert has_draft_tokens is mtp_enabled

    _check_cached_output_matches_recompute(
        int8_enabled, mtp_enabled, answer_prompt
    )

    results = {}
    for K, (warm, probe) in plan.items():
        _flush_cache()
        with ThreadPoolExecutor(PARALLEL) as executor:
            list(executor.map(_generate_capacity_request, warm))
        with ThreadPoolExecutor(PARALLEL) as executor:
            metadata = [
                response["meta_info"]
                for response in executor.map(
                    lambda ids: _generate(
                        {
                            "input_ids": ids,
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": 1,
                                "ignore_eos": True,
                            },
                        }
                    ),
                    probe,
                )
            ]
        prompt_tokens = sum(item["prompt_tokens"] for item in metadata)
        cached_tokens = sum(item["cached_tokens"] for item in metadata)
        results[K] = cached_tokens / max(1, prompt_tokens)
        print(
            f"[int8={'on' if int8_enabled else 'off'}, mtp={'on' if mtp_enabled else 'off'}] "
            f"K={K}: reuse_frac={results[K]:.3f}",
            flush=True,
        )
    return results


def _make_answer_prompt(tokenizer):
    """Create a long, non-thinking chat prompt that should have a cache hit."""
    context = (
        "\n\n".join(
            [
                "You are preparing a field guide for a community science team. "
                "Explain each observation carefully, distinguish direct "
                "evidence from interpretation, and keep the instructions "
                "practical. The team records the weather, location, time, "
                "and equipment before collecting samples.",
                "For every sample, write down the label exactly as it "
                "appears, use a clean container, and note any unusual color, "
                "texture, or odor without guessing at its cause. If a "
                "measurement is uncertain, repeat it and record both values.",
                "At the end of the day, compare the field notes with the "
                "instrument log. Preserve the original readings, identify "
                "missing entries, and list questions for the next visit. "
                "Do not silently replace an observation with an estimate.",
            ]
        )
        * 8
    )
    messages = [
        {
            "role": "system",
            "content": "Reply with exactly one digit and no explanation.",
        },
        {
            "role": "user",
            "content": f"{context}\n\nQuestion: What is 2 + 2?",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
    assert token_count > 256, (
        f"correctness prompt must exceed the cache page size; got {token_count}"
    )
    print(f"[cache-hit answer prompt] token_count={token_count}", flush=True)
    return prompt


def _check_cached_output_matches_recompute(int8_enabled, mtp_enabled, prompt):
    """Check a short answer after reuse and observe runtime MTP acceptance."""
    answer_request = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 16,
        },
    }

    _flush_cache()
    recompute = _generate(answer_request)
    _flush_cache()
    _generate(answer_request)
    cached = _generate(answer_request)
    assert cached["meta_info"]["cached_tokens"] > 0, (
        "the repeated prompt must hit the cache"
    )
    assert recompute["text"].strip() == "4" and cached["text"].strip() == "4", (
        "the model must answer the stable arithmetic question with one digit "
        "both after recompute and after a cache hit: "
        f"recompute={recompute['text']!r}, cached={cached['text']!r}"
    )
    print(
        f"[int8={'on' if int8_enabled else 'off'}, "
        f"mtp={'on' if mtp_enabled else 'off'}] cache-hit answer: "
        f"recompute={recompute['text']!r}, cached={cached['text']!r}, "
        f"cached_tokens={cached['meta_info']['cached_tokens']}",
        flush=True,
    )

    if mtp_enabled:
        # Generate enough tokens to run the speculative verifier, reusing the
        # same cached prefix, without imposing an acceptance-rate threshold.
        long_request = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 64,
                "ignore_eos": True,
            },
        }
        mtp_cached = _generate(long_request)
        meta = mtp_cached["meta_info"]
        assert meta["cached_tokens"] > 0, "MTP request must also use the cached prefix"
        assert meta.get("spec_accept_length", 0) > 1.0, (
            "MTP was configured but the cache-hit generation accepted no draft tokens"
        )
        assert meta.get("spec_num_correct_drafts", 0) > 0, (
            "MTP was configured but the cache-hit generation accepted no draft tokens"
        )
        assert meta.get("spec_verify_ct", 0) > 0, (
            "MTP was configured but the cache-hit generation did not run verification"
        )
        print(
            f"[int8={'on' if int8_enabled else 'off'}, mtp=on] cache-hit "
            "speculation: "
            f"spec_accept_length={meta['spec_accept_length']}, "
            f"spec_num_correct_drafts={meta['spec_num_correct_drafts']}, "
            f"spec_verify_ct={meta['spec_verify_ct']}, "
            f"cached_tokens={meta['cached_tokens']}",
            flush=True,
        )


def _collapse_point(results):
    for K in K_SCAN:
        if results[K] < COLLAPSE_THRESHOLD:
            return K
    return None


class TestNpuInt8MambaCheckpointMtp(CustomTestCase):
    """Exercise int8 checkpoint reuse with NPU MTP enabled and disabled."""

    def test_checkpoint_capacity_and_cache_hit_correctness(self):
        plan = _make_capacity_requests()
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=True)
        answer_prompt = _make_answer_prompt(tokenizer)
        measurements = {}
        for mtp_enabled in (False, True):
            for int8_enabled in (False, True):
                # Start a single server for each feature combination, check its
                # resolved configuration, and keep one capacity sweep per server.
                if mtp_enabled:
                    args = [
                        "--speculative-algorithm",
                        "NEXTN",
                        "--speculative-num-steps",
                        "3",
                        "--speculative-eagle-topk",
                        "1",
                        "--speculative-num-draft-tokens",
                        "4",
                    ]
                else:
                    args = []
                args += [
                    # Exercise text requests directly; the generic multimodal
                    # warmup uses a fake image and is unrelated to this test.
                    "--skip-server-warmup",
                    "--trust-remote-code",
                    "--device",
                    "npu",
                    "--tp-size",
                    "2",
                    "--attention-backend",
                    "ascend",
                    "--mem-fraction-static",
                    "0.8",
                    "--mamba-radix-cache-strategy",
                    "extra_buffer",
                    "--mamba-ssm-dtype",
                    "bfloat16",
                    "--max-mamba-cache-size",
                    str(MAX_MAMBA_CACHE_SIZE),
                ]
                if BASE_GPU_ID is not None:
                    args += ["--base-gpu-id", BASE_GPU_ID]
                if int8_enabled:
                    args += [
                        "--enable-int8-mamba-checkpoint",
                        "--int8-mamba-ckpt-size",
                        str(INT8_MAMBA_CKPT_SIZE),
                    ]

                process = popen_launch_server(
                    MODEL,
                    BASE_URL,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=args,
                    env=NPU_ENV,
                    device="npu",
                )
                try:
                    measurements[(mtp_enabled, int8_enabled)] = _measure_running_server(
                        int8_enabled, mtp_enabled, plan, answer_prompt
                    )
                finally:
                    terminate_and_kill_process_tree(process, terminate_timeout=60)
                    time.sleep(2)

        for mtp_enabled in (False, True):
            off = measurements[(mtp_enabled, False)]
            on = measurements[(mtp_enabled, True)]
            for results, tag in ((off, "off"), (on, "on")):
                for a, b in zip(K_SCAN, K_SCAN[1:]):
                    self.assertGreaterEqual(
                        results[a] + 1e-6,
                        results[b],
                        f"[mtp={'on' if mtp_enabled else 'off'}, {tag}] reuse should "
                        f"not increase with K ({results[a]:.3f}@{a} -> {results[b]:.3f}@{b})",
                    )
            off_collapse = _collapse_point(off)
            on_collapse = _collapse_point(on)
            off_label = (
                f"K={off_collapse}"
                if off_collapse is not None
                else f"not observed through K={K_SCAN[-1]}"
            )
            on_label = (
                f"K={on_collapse}"
                if on_collapse is not None
                else f"not observed through K={K_SCAN[-1]}"
            )
            print(
                f"[mtp={'on' if mtp_enabled else 'off'}] collapse K: "
                f"off={off_label}, on={on_label}",
                flush=True,
            )
            if off_collapse is not None:
                self.assertTrue(
                    on_collapse is None or on_collapse >= off_collapse,
                    f"int8 checkpoint capacity collapsed earlier (mtp={mtp_enabled}): "
                    f"off={off_collapse}, on={on_collapse}",
                )
            gains = {K: on[K] - off[K] for K in K_SCAN}
            best_K = max(gains, key=gains.get)
            self.assertGreater(
                gains[best_K],
                REUSE_GAP,
                f"expected >{REUSE_GAP:.2f} reuse gain with MTP={mtp_enabled}; "
                f"max was {gains[best_K]:.3f} at K={best_K}",
            )


if __name__ == "__main__":
    unittest.main()

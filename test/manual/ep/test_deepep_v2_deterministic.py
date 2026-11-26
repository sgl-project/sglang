"""Check exact logprobs across batches against a running DeepEP v2 server.

Example on four NVIDIA GPUs with DeepEP v2 installed (test BF16 and FP8 models):

    EP_DISABLE_GIN=1 NCCL_CUMEM_ENABLE=1 EP_REUSE_NCCL_COMM=0 \
    SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=4096 \
    python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-FP8 \
        --tp-size 4 --ep-size 4 --moe-a2a-backend deepep_v2 \
        --deepep-v2-mode direct --attention-backend triton \
        --enable-deterministic-inference --disable-radix-cache \
        --chunked-prefill-size 4096 --cuda-graph-max-bs-decode 32

    python test/manual/ep/test_deepep_v2_deterministic.py

Radix caching must be disabled: otherwise repeated prompts can reuse the same
prefill and hide batch-dependent results. Keep DeepGEMM precompilation enabled.
"""

import argparse

import requests


def check_batch_invariance(base_url, batch_sizes, max_new_tokens):
    response = requests.get(f"{base_url}/get_server_info", timeout=30)
    response.raise_for_status()
    info = response.json()
    assert info["moe_a2a_backend"] == "deepep_v2", info["moe_a2a_backend"]
    assert info["enable_deterministic_inference"], "Enable deterministic inference"
    assert info["disable_radix_cache"], "Disable radix caching to test fresh prefill"

    prompts = [
        "Tell me about Richard Feynman: ",
        "The capital city of France is",
        "Explain why the sky is blue in three sentences.",
    ]
    for prompt in prompts:
        reference = None
        for mixed in (False, True):
            for batch_size in batch_sizes:
                texts = [prompt] * batch_size
                if mixed:
                    # Put the target last, after unrelated variable-length prompts,
                    # to move its tokens between TP shards as the batch changes.
                    texts = [
                        "Write a Python function that sorts a list. " * (i % 5 + 1)
                        for i in range(batch_size - 1)
                    ] + [prompt]
                response = requests.post(
                    f"{base_url}/generate",
                    json={
                        "text": texts,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": max_new_tokens,
                            "ignore_eos": True,
                        },
                        "return_logprob": True,
                    },
                    timeout=600,
                )
                response.raise_for_status()
                signatures = [
                    [(p[0], p[1]) for p in r["meta_info"]["output_token_logprobs"]]
                    for r in response.json()
                ]
                assert len(signatures) == batch_size
                target = signatures[-1]
                assert len(target) == max_new_tokens
                if reference is None:
                    reference = target
                compared = [target] if mixed else signatures
                assert all(s == reference for s in compared), (
                    f"Logprobs differ: {prompt=!r}, {batch_size=}, {mixed=}"
                )
                print(f"PASS {prompt=!r} {batch_size=} {mixed=}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 2, 3, 4, 8, 16, 32]
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or any(bs < 1 for bs in args.batch_sizes):
        parser.error("Token count and batch sizes must be positive")
    check_batch_invariance(
        args.base_url.rstrip("/"), args.batch_sizes, args.max_new_tokens
    )

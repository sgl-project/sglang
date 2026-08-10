#!/usr/bin/env python3
"""Minimal end-to-end test: load DSV4 Flash model, run one forward with MXFP4 KV cache."""

import os

os.environ["SGLANG_OPT_DSV4_MXFP4_KVCACHE"] = "1"


from sglang.srt.entrypoints.engine import Engine

MODEL_PATH = "/data/models/DeepSeek-V4-Flash"
TP_SIZE = 8  # Use all 8 GPUs to fit the 600GB model


def main():
    print("=== MXFP4 End-to-End Test ===")
    print(f"Model: {MODEL_PATH}")
    print(f"TP: {TP_SIZE}")
    print(f"MXFP4: {os.environ.get('SGLANG_OPT_DSV4_MXFP4_KVCACHE', 'NOT SET')}")

    # Start engine with minimal memory usage
    engine = Engine(
        model_path=MODEL_PATH,
        tp_size=TP_SIZE,
        mem_fraction_static=0.85,
        disable_radix_cache=True,
        max_running_requests=1,
        moe_runner_backend="marlin",
        log_level="info",
        disable_cuda_graph=True,
        trust_remote_code=True,
    )

    print("\n=== Forward test ===")
    result = engine.generate(
        prompt="Hello, my name is",
        sampling_params={"max_new_tokens": 3, "temperature": 0},
    )
    print(f"Output: {result['text']!r}")
    print(f"Tokens: {result.get('output_ids', 'N/A')}")
    print("\n✅ End-to-end test passed!")


if __name__ == "__main__":
    main()

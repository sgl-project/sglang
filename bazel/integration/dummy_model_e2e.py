"""Smallest real SGLang execution path for a Bazel GPU smoke test.

The Bazel target deliberately uses the runtime installed by SGLang's existing
hardware CI while native wheels are still being migrated into Bazel.
"""

import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

import sglang as sgl  # noqa: E402


def main() -> None:
    engine = sgl.Engine(
        model_path=os.environ.get("SGLANG_BAZEL_E2E_MODEL", "Qwen/Qwen3-0.6B"),
        load_format="dummy",
        skip_tokenizer_init=True,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        random_seed=0,
        log_level="error",
        mem_fraction_static=0.5,
        max_total_tokens=512,
    )
    try:
        output = engine.generate(
            input_ids=[464, 9345, 3958, 1752, 13],
            sampling_params={
                "temperature": 0.0,
                "max_new_tokens": 4,
                "ignore_eos": True,
            },
        )
        output_ids = output["output_ids"]
        if len(output_ids) != 4:
            raise AssertionError(f"expected 4 generated tokens, got {output_ids!r}")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()

"""Minimal offline smoke test for Dream-v0-Base-7B."""

import argparse
import json
import os

import sglang as sgl


DEFAULT_MODEL = "Dream-org/Dream-v0-Base-7B"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("MAX_NEW_TOKENS", "16")),
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=float(os.getenv("MEM_FRACTION_STATIC", "0.80")),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    engine = None

    try:
        print(f"Loading Dream model: {args.model_path}")
        engine = sgl.Engine(
            model_path=args.model_path,
            trust_remote_code=True,
            tp_size=1,
            pp_size=1,
            dllm_algorithm="Dream",
            dllm_fdfo=False,
            disable_radix_cache=True,
            disable_cuda_graph=True,
            mem_fraction_static=args.mem_fraction_static,
        )

        assert engine.server_args.tp_size == 1
        assert engine.server_args.disable_radix_cache
        assert engine.server_args.disable_cuda_graph

        outputs = engine.generate(
            [args.prompt],
            {
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        )
        output = outputs[0]
        assert isinstance(output.get("text"), str), output

        print("Dream offline smoke test passed")
        print(
            json.dumps(
                {
                    "prompt": args.prompt,
                    "generated_text": output["text"],
                    "meta_info": output.get("meta_info", {}),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()

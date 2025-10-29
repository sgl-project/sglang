import os

from utils import get_args

import sglang as sgl

args = get_args()


def main():
    llm = sgl.Engine(
        model_path=args.model_path,
        device=args.device,
        model_impl=args.model_impl,
        max_total_tokens=args.max_total_tokens,
        attention_backend=args.attention_backend,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        log_level=args.log_level,
        mem_fraction_static=args.mem_fraction_static,
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    if args.enable_greedy:
        sampling_params = {"temperature": 0.0, "top_k": 1.0}
    else:
        sampling_params = {"temperature": 0.01, "top_p": 0.9}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    main()

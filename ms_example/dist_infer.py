import argparse
import os

import sglang as sgl

parser = argparse.ArgumentParser("sglang-mindspore dist infer")

parser.add_argument(
    "--model_path",
    metavar="--model_path",
    dest="model_path",
    required=False,
    default="/home/ckpt/Qwen3-8B",
    help="the model path",
    type=str,
)

args = parser.parse_args()


def main():
    llm = sgl.Engine(
        model_path=args.model_path,
        device="npu",
        model_impl="mindspore",
        attention_backend="ascend",
        tp_size=2,
        dp_size=1,
        mem_fraction_static=0.8,
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {"temperature": 0.01, "top_p": 0.9}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    main()

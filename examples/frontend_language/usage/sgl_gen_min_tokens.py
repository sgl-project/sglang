"""
This example demonstrates how to use `min_tokens` to enforce sgl.gen to generate a longer sequence

Usage:
python3 sgl_gen_min_tokens.py
"""

import sglang as sgl


@sgl.function
def long_answer(s):
    s += sgl.user("What is the capital of the United States?")
    s += sgl.assistant(sgl.gen("answer", min_tokens=64, max_tokens=128))


@sgl.function
def short_answer(s):
    s += sgl.user("What is the capital of the United States?")
    s += sgl.assistant(sgl.gen("answer"))


if __name__ == "__main__":
    runtime = sgl.Runtime(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")
    sgl.set_default_backend(runtime)

    state = long_answer.run()
    print("=" * 20)
    print("Longer Answer", state["answer"])

    state = short_answer.run()
    print("=" * 20)
    print("Short Answer", state["answer"])

    runtime.shutdown()

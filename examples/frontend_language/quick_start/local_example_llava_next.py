"""
Usage: python3 local_example_llava_next.py
"""

import sglang as sgl
from sglang.lang.chat_template import get_chat_template


@sgl.function
def image_qa(s, image_path, question):
    s += sgl.user(sgl.image(image_path) + question)
    s += sgl.assistant(sgl.gen("answer"))


def single():
    state = image_qa.run(
        image_path="images/cat.jpeg", question="What is this?", max_new_tokens=128
    )
    print(state["answer"], "\n")


def stream():
    state = image_qa.run(
        image_path="images/cat.jpeg",
        question="What is this?",
        max_new_tokens=64,
        stream=True,
    )

    for out in state.text_iter("answer"):
        print(out, end="", flush=True)
    print()


def batch():
    states = image_qa.run_batch(
        [
            {"image_path": "images/cat.jpeg", "question": "What is this?"},
            {"image_path": "images/dog.jpeg", "question": "What is this?"},
        ],
        max_new_tokens=128,
    )
    for s in states:
        print(s["answer"], "\n")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    runtime = sgl.Runtime(model_path="lmms-lab/llama3-llava-next-8b")
    runtime.endpoint.chat_template = get_chat_template("llama-3-instruct-llava")

    # Or you can use the 72B model
    # runtime = sgl.Runtime(model_path="lmms-lab/llava-next-72b", tp_size=8)
    # runtime.endpoint.chat_template = get_chat_template("chatml-llava")

    sgl.set_default_backend(runtime)
    print(f"chat template: {runtime.endpoint.chat_template.name}")

    # Or you can use API models
    # sgl.set_default_backend(sgl.OpenAI("gpt-4-vision-preview"))
    # sgl.set_default_backend(sgl.VertexAI("gemini-pro-vision"))

    # Run a single request
    print("\n========== single ==========\n")
    single()

    # Stream output
    print("\n========== stream ==========\n")
    stream()

    # Run a batch of requests
    print("\n========== batch ==========\n")
    batch()

    runtime.shutdown()

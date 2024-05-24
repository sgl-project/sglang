"""
Usage: python3 srt_example_llava.py
"""

import sglang as sgl
from sglang.srt.utils import load_image
from sglang.lang.chat_template import get_chat_template

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow loading of truncated images

@sgl.function
def image_qa(s, image, question):
    s += sgl.user(sgl.image(image) + question)
    s += sgl.assistant(sgl.gen("answer"))


def single():
    image_url = "https://farm4.staticflickr.com/3175/2653711032_804ff86d81_z.jpg"
    pil_image, _ = load_image(image_url)
    state = image_qa.run(image=pil_image, question="What is this?", max_new_tokens=512)
    print(state["answer"], "\n")


def stream():
    image_url = "https://farm4.staticflickr.com/3175/2653711032_804ff86d81_z.jpg"
    pil_image, _ = load_image(image_url)
    state = image_qa.run(
        image=pil_image,
        question="Please generate short caption for this image.",
        max_new_tokens=512,
        temperature=0,
        stream=True,
    )

    for out in state.text_iter("answer"):
        print(out, end="", flush=True)
    print()


def batch():
    image_url = "https://farm4.staticflickr.com/3175/2653711032_804ff86d81_z.jpg"
    pil_image, _ = load_image(image_url)
    states = image_qa.run_batch(
        [
            {"image": pil_image, "question": "What is this?"},
            {"image": pil_image, "question": "What is this?"},
        ],
        max_new_tokens=512,
    )
    for s in states:
        print(s["answer"], "\n")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    runtime = sgl.Runtime(
        model_path="lmms-lab/llama3-llava-next-8b",
        tokenizer_path="lmms-lab/llama3-llava-next-8b-tokenizer",
    )
    runtime.endpoint.chat_template = get_chat_template("llama-3-instruct")
    # runtime = sgl.Runtime(
    #     model_path="lmms-lab/llava-next-72b",
    #     tokenizer_path="lmms-lab/llavanext-qwen-tokenizer",
    # )
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

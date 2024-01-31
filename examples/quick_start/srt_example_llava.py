"""
Usage: python3 srt_example_llava.py
"""
import sglang as sgl


@sgl.function
def image_qa(s, image_path, question):
    s += sgl.user(sgl.image(image_path) + question)
    s += sgl.assistant(sgl.gen("answer"))


def single():
    state = image_qa.run(
        image_path="images/cat.jpeg",
        question="What is this?",
        max_new_tokens=64)
    print(state["answer"], "\n")


def stream():
    state = image_qa.run(
        image_path="images/cat.jpeg",
        question="What is this?",
        max_new_tokens=64,
        stream=True)

    for out in state.text_iter("answer"):
        print(out, end="", flush=True)
    print()


def batch():
    states = image_qa.run_batch(
        [
            {"image_path": "images/cat.jpeg", "question":"What is this?"},
            {"image_path": "images/dog.jpeg", "question":"What is this?"},
        ],
        max_new_tokens=64,
    )
    for s in states:
        print(s["answer"], "\n")


if __name__ == "__main__":
    runtime = sgl.Runtime(model_path="liuhaotian/llava-v1.5-7b",
                          tokenizer_path="llava-hf/llava-1.5-7b-hf")
    sgl.set_default_backend(runtime)
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

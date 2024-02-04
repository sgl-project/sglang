"""
Usage: python3 srt_example_yi_vl.py
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
        max_new_tokens=64,
        stop="###")
    print(state["answer"], "\n")


def stream():
    state = image_qa.run(
        image_path="images/cat.jpeg",
        question="What is this?",
        max_new_tokens=64,
        stream=True,
        stop="###")

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
        stop="###"
    )
    for s in states:
        print(s["answer"], "\n")


if __name__ == "__main__":
    runtime = sgl.Runtime(model_path="BabyChou/Yi-VL-6B")
    # runtime = sgl.Runtime(model_path="BabyChou/Yi-VL-34B")
    sgl.set_default_backend(runtime)

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

"""
Usage:
python3 local_example_complete.py
"""

import sglang as sgl


@sgl.function
def few_shot_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", temperature=0, max_tokens=1024, min_tokens=512)


def single():
    state = few_shot_qa.run(question="What is the meaning of life?")
    answer = state["answer"].strip().lower()
    print(state.text())


def stream():
    state = few_shot_qa.run(
        question="What is the capital of the United States?", stream=True
    )

    for out in state.text_iter("answer"):
        print(out, end="", flush=True)
    print()


def batch():
    states = few_shot_qa.run_batch(
        [
            {"question": "What is the capital of the United States?"},
            {"question": "What is the capital of China?"},
        ]
    )

    for s in states:
        print(s["answer"])


if __name__ == "__main__":
    # runtime = sgl.Runtime(model_path="/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/")
    backend = sgl.RuntimeEndpoint("http://localhost:8080")
    sgl.set_default_backend(backend)

    # Run a single request
    print("\n========== single ==========\n")
    single()

    # # Stream output
    # print("\n========== stream ==========\n")
    # stream()

    # # Run a batch of requests
    # print("\n========== batch ==========\n")
    # batch()


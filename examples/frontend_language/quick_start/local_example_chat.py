"""
Usage:
python3 local_example_chat.py
"""

import sglang as sgl


@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))


def single():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
    )

    for m in state.messages():
        print(m["role"], ":", m["content"])

    print("\n-- answer_1 --\n", state["answer_1"])


def stream():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
        stream=True,
    )

    for out in state.text_iter():
        print(out, end="", flush=True)
    print()


def batch():
    states = multi_turn_question.run_batch(
        [
            {
                "question_1": "What is the capital of the United States?",
                "question_2": "List two local attractions.",
            },
            {
                "question_1": "What is the capital of France?",
                "question_2": "What is the population of this city?",
            },
        ]
    )

    for s in states:
        print(s.messages())


if __name__ == "__main__":
    # runtime = sgl.Runtime(model_path="unsloth/Llama-3.2-1B-bnb-4bit", tp_size=1, max_total_tokens=100)
    # runtime = sgl.Runtime(model_path="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", tp_size=1, max_total_tokens=100)
    runtime = sgl.Runtime(
        model_path="meta-llama/Llama-3.2-1B-Instruct",
        modeling="transformers",
        torchao_config="int4wo-128",
        tp_size=2,
        max_total_tokens=100,
    )

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

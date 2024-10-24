"""
Usage:
# Prior to running this script, you need to have a Shortfin server running.
# Build:
#   https://github.com/nod-ai/SHARK-Platform/blob/main/shortfin/README.md
# Run:
#   https://github.com/nod-ai/SHARK-Platform/blob/main/shortfin/python/shortfin_apps/llm/README.md

python3 shortfin_example_chat.py --base_url <url_to_shortfin_server>
"""

import argparse
import os

import sglang as sgl


@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default="http://localhost:8000")
    args = parser.parse_args()
    base_url = args.base_url

    backend = sgl.Shortfin(base_url=base_url)
    sgl.set_default_backend(backend)

    # Run a single request
    print("\n========== single ==========\n")
    single()

    # Stream output
    print("\n========== stream ==========\n")
    stream()

import openai


client = openai.Client(base_url="https://api.openai.com/v1/", api_key="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")


"""
Usage:
export OPENAI_API_KEY=sk-******
python3 openai_example_chat.py
"""

import sglang as sgl


@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_completion_tokens=100))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_completion_tokens=100))


def single():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
    )

    for m in state.messages():
        print(m["role"], ":", m["content"])

    print("\n-- answer_1 --\n", state["answer_1"])





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
    sgl.set_default_backend(sgl.OpenAI("o1"))

    # Run a single request
    print("\n========== single ==========\n")
    single()
    # Run a batch of requests
    print("\n========== batch ==========\n")
    batch()
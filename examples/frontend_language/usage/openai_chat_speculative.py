"""
Usage:
***Note: for speculative execution to work, user must put all "gen" in "assistant".
Show in "assistant" the desired answer format. Each "gen" term should have a stop token.
The stream mode is not supported in speculative execution.

E.g. 
correct: 
    sgl.assistant("\nName:" + sgl.gen("name", stop="\n") + "\nBirthday:" + sgl.gen("birthday", stop="\n") + "\nJob:" + sgl.gen("job", stop="\n"))
incorrect:
    s += sgl.assistant("\nName:" + sgl.gen("name", stop="\n"))
    s += sgl.assistant("\nBirthday:" + sgl.gen("birthday", stop="\n"))
    s += sgl.assistant("\nJob:" + sgl.gen("job", stop="\n"))

export OPENAI_API_KEY=sk-******
python3 openai_chat_speculative.py
"""

import sglang as sgl
from sglang import OpenAI, function, set_default_backend


@function(num_api_spec_tokens=256)
def gen_character_spec(s):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user("Construct a character within the following format:")
    s += sgl.assistant(
        "Name: Steve Jobs.\nBirthday: February 24, 1955.\nJob: Apple CEO.\n"
    )
    s += sgl.user("Please generate new Name, Birthday and Job.\n")
    s += sgl.assistant(
        "Name:"
        + sgl.gen("name", stop="\n")
        + "\nBirthday:"
        + sgl.gen("birthday", stop="\n")
        + "\nJob:"
        + sgl.gen("job", stop="\n")
    )


@function(num_api_spec_tokens=256)
def gen_character_spec_no_few_shot(s):
    s += sgl.user("Construct a character. For each field stop with a newline\n")
    s += sgl.assistant(
        "Name:"
        + sgl.gen("name", stop="\n")
        + "\nAge:"
        + sgl.gen("age", stop="\n")
        + "\nJob:"
        + sgl.gen("job", stop="\n")
    )


@function
def gen_character_normal(s):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user("What's the answer of 23 + 8?")
    s += sgl.assistant(sgl.gen("answer", max_tokens=64))


@function(num_api_spec_tokens=1024)
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user("Answer questions in the following format:")
    s += sgl.user(
        "Question 1: What is the capital of France?\nQuestion 2: What is the population of this city?\n"
    )
    s += sgl.assistant(
        "Answer 1: The capital of France is Paris.\nAnswer 2: The population of Paris in 2024 is estimated to be around 2.1 million for the city proper.\n"
    )
    s += sgl.user("Question 1: " + question_1 + "\nQuestion 2: " + question_2)
    s += sgl.assistant(
        "Answer 1: "
        + sgl.gen("answer_1", stop="\n")
        + "\nAnswer 2: "
        + sgl.gen("answer_2", stop="\n")
    )


def test_spec_single_turn():
    backend.token_usage.reset()

    state = gen_character_spec.run()
    for m in state.messages():
        print(m["role"], ":", m["content"])

    print("\n-- name:", state["name"])
    print("-- birthday:", state["birthday"])
    print("-- job:", state["job"])
    print(backend.token_usage)


def test_inaccurate_spec_single_turn():
    state = gen_character_spec_no_few_shot.run()
    for m in state.messages():
        print(m["role"], ":", m["content"])

    print("\n-- name:", state["name"])
    print("\n-- age:", state["age"])
    print("\n-- job:", state["job"])


def test_normal_single_turn():
    state = gen_character_normal.run()
    for m in state.messages():
        print(m["role"], ":", m["content"])


def test_spec_multi_turn():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions in the capital of the United States.",
    )

    for m in state.messages():
        print(m["role"], ":", m["content"])

    print("\n-- answer_1 --\n", state["answer_1"])
    print("\n-- answer_2 --\n", state["answer_2"])


def test_spec_multi_turn_stream():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
        stream=True,
    )

    for out in state.text_iter():
        print(out, end="", flush=True)


if __name__ == "__main__":
    backend = OpenAI("gpt-4-turbo")
    set_default_backend(backend)

    print("\n========== test spec single turn ==========\n")
    # expect reasonable answer for each field
    test_spec_single_turn()

    print("\n========== test inaccurate spec single turn ==========\n")
    # expect incomplete or unreasonable answers
    test_inaccurate_spec_single_turn()

    print("\n========== test normal single turn ==========\n")
    # expect reasonable answer
    test_normal_single_turn()

    print("\n========== test spec multi turn ==========\n")
    # expect answer with same format as in the few shot
    test_spec_multi_turn()

    print("\n========== test spec multi turn stream ==========\n")
    # expect error in stream_executor: stream is not supported...
    test_spec_multi_turn_stream()

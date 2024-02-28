"""
Usage:
export OPENAI_API_KEY=sk-******
python3 openai_example_chat.py
"""
import sglang as sgl
from sglang import function, gen, set_default_backend, OpenAI


@function(api_num_spec_tokens=512)
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user("Question 1: "+question_1+"\nQuestion 2: "+question_2)
    s += sgl.assistant("Answer 1: "+sgl.gen("answer_1"+"Answer 2: "+ sgl.gen("answer_2"), max_tokens=512))


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
        stream=True
    )

    for out in state.text_iter():
        print(out, end="", flush=True)
    print()




if __name__ == "__main__":
    sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo"))

    # Run a single request
    print("\n========== single ==========\n")
    single()

    # # Stream output
    # print("\n========== stream ==========\n")
    # stream()

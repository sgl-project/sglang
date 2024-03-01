"""
Usage:
***Note: for speculative execution to work, user must put all "gen" in "assistant". 
E.g. 
correct: 
    sgl.assistant("\nName:" + sgl.gen("name", stop="\n") + "\nBirthday:" + sgl.gen("birthday", stop="\n") + "\nJob:" + sgl.gen("job", stop="\n"))
incorrect:     
    s += sgl.assistant("\nName:" + sgl.gen("name", stop="\n"))
    s += sgl.assistant("\nBirthday:" + sgl.gen("birthday", stop="\n"))
    s += sgl.assistant("\nJob:" + sgl.gen("job", stop="\n"))

export OPENAI_API_KEY=sk-******
python3 openaichat_example_chat.py
"""
import sglang as sgl
from sglang import function

@function(api_num_spec_tokens=512)
def gen_character_spec(s):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user("Construct a character within the following format:")
    s += sgl.user("\nName: Steve Jobs.\nBirthday: February 24, 1955.\nJob: Apple CEO.\n")
    s += sgl.user("Please generate new Name, Birthday and Job.\n")
    s += sgl.assistant("\nName:" + sgl.gen("name", stop="\n") + "\nBirthday:" + sgl.gen("birthday", stop="\n") + "\nJob:" + sgl.gen("job", stop="\n"))


@function(api_num_spec_tokens=512)
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(" Question 1: "+question_1+"\nQuestion 2: "+question_2)
    s += sgl.assistant(" Answer 1: "+sgl.gen("answer_1")+"\nAnswer 2: "+ sgl.gen("answer_2"))


def single():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
    )

    for m in state.messages():
        print(m["role"], ":", m["content"])

    print("\n-- answer_1 --\n", state["answer_1"])
    print("\n-- answer_2 --\n", state["answer_2"])


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
    sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo", is_speculative=True))
    state = gen_character_spec.run()

    print("name:", state["name"])
    print("birthday:", state["birthday"])
    print("job:", state["job"])

    # Run a single request
    print("\n========== single ==========\n")
    single()

    # Stream output
    print("\n========== stream ==========\n")
    stream()

"""
Usage:
***Note: for speculative execution to work, user must put all "gen" in "assistant". Show in "assistant" the desired answer format.
E.g. 
correct: 
    sgl.assistant("\nName:" + sgl.gen("name", stop="\n") + "\nBirthday:" + sgl.gen("birthday", stop="\n") + "\nJob:" + sgl.gen("job", stop="\n"))
incorrect:     
    s += sgl.assistant("\nName:" + sgl.gen("name", stop="\n"))
    s += sgl.assistant("\nBirthday:" + sgl.gen("birthday", stop="\n"))
    s += sgl.assistant("\nJob:" + sgl.gen("job", stop="\n"))

export OPENAI_API_KEY=sk-******
python3 openaichat_speculative.py
"""
import sglang as sgl
from sglang import function, gen, set_default_backend, OpenAI

@function(api_num_spec_tokens=512)
def gen_character_spec(s):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user("Construct a character within the following format:")
    s += sgl.assistant("Name: Steve Jobs.\nBirthday: February 24, 1955.\nJob: Apple CEO.\n")
    s += sgl.user("Please generate new Name, Birthday and Job.\n")
    s += sgl.assistant("Name:" + sgl.gen("name", stop="\n") + "\nBirthday:" + sgl.gen("birthday", stop="\n") + "\nJob:" + sgl.gen("job", stop="\n"))


@function(api_num_spec_tokens=512)
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user("Answer questions in the following format:")
    s += sgl.user("Question 1: What is the capital of France?\nQuestion 2: What is the population of this city?\n")
    s += sgl.assistant("Answer 1: The capital of France is Paris.\nAnswer 2: The population of Paris in 2024 is estimated to be around 2.1 million for the city proper.\n")
    s += sgl.user("Question 1: "+question_1+"\nQuestion 2: "+question_2)
    s += sgl.assistant("Answer 1: "+sgl.gen("answer_1", stop="\n") + "\nAnswer 2: "+ sgl.gen("answer_2", stop="\n"))

def single1():

    state = gen_character_spec.run(max_new_tokens=64)
    for m in state.messages():
        print(m["role"], ":", m["content"])


    print("\n-- name:", state["name"])
    print("\n-- birthday:", state["birthday"])
    print("\n-- job:", state["job"])



def single2():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions in the capital of the United States.", max_new_tokens=128
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
    set_default_backend(OpenAI("gpt-3.5-turbo-instruct"))

    # Run a single request
    print("\n========== single 1 ==========\n")
    single1()

    print("\n========== single 2 ==========\n")
    single2()

    # Stream output
    print("\n========== stream ==========\n")
    stream()

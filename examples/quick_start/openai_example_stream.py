from sglang import function, system, user, assistant, gen, set_default_backend, OpenAI


@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))

set_default_backend(OpenAI("gpt-3.5-turbo"))

state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
    stream=True
)

for out in state.text_iter():
    print(out, end="", flush=True)

from sglang import function, user, model, gen, set_default_backend, Gemini


@function
def multi_turn_question(s, question_1, question_2):
    # s += system("You are a helpful assistant.")
    s += user(question_1)
    s += model(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += model(gen("answer_2", max_tokens=256))

set_default_backend(Gemini("gemini-pro"))

state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
    stream=True
)

for out in state.text_iter():
    print(out, end="", flush=True)

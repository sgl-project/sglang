from sglang import function, system, user, assistant, gen, set_default_backend, Runtime


@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))

runtime = Runtime("meta-llama/Llama-2-7b-chat-hf")
set_default_backend(runtime)

state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
    temperature=0,
    stream=True,
)

for out in state.text_iter():
    print(out, end="", flush=True)
print()

runtime.shutdown()

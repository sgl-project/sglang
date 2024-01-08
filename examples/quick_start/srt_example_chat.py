from sglang import function, system, user, assistant, gen, set_default_backend, Runtime


@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))


runtime = Runtime(model_path="meta-llama/Llama-2-7b-chat-hf")
#runtime = Runtime(model_path="mistralai/Mixtral-8x7B-Instruct-v0.1")
set_default_backend(runtime)

state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
)

for m in state.messages():
    print(m["role"], ":", m["content"])


runtime.shutdown()

from sglang import function, gen, set_default_backend, Runtime


@function
def few_shot_qa(s, question):
    s += (
"""The following are questions with answers.
Q: What is the capital of France?
A: Paris
Q: What is the capital of Germany?
A: Berlin
Q: What is the capital of Italy?
A: Rome
""")
    s += "Q: " + question + "\n"
    s += "A:" + gen("answer", stop="\n", temperature=0)


runtime = Runtime(model_path="meta-llama/Llama-2-7b-chat-hf")
set_default_backend(runtime)

state = few_shot_qa.run(question="What is the capital of the United States?")

answer = state["answer"].strip().lower()
assert "washington" in answer, f"answer: {state['answer']}"
print(state.text())

runtime.shutdown()

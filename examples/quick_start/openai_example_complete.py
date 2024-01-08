from sglang import function, gen, set_default_backend, OpenAI


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


set_default_backend(OpenAI("gpt-3.5-turbo-instruct"))

state = few_shot_qa.run(question="What is the capital of the United States?")
answer = state["answer"].strip().lower()

assert "washington" in answer, f"answer: {state['answer']}"

print(state.text())

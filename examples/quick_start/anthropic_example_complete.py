from sglang import function, gen, set_default_backend, Anthropic


@function
def few_shot_qa(s, question):
    s += (
"""
\n\nHuman: What is the capital of France?
\n\nAssistant: Paris
\n\nHuman: What is the capital of Germany?
\n\nAssistant: Berlin
\n\nHuman: What is the capital of Italy?
\n\nAssistant: Rome
""")
    s += "\n\nHuman: " + question + "\n"
    s += "\n\nAssistant:" + gen("answer", stop="\n", temperature=0)


set_default_backend(Anthropic("claude-2"))

state = few_shot_qa.run(question="What is the capital of the United States?")
answer = state["answer"].strip().lower()

assert "washington" in answer, f"answer: {state['answer']}"

print(state.text())

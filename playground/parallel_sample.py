import sglang as sgl

N = 5

@sgl.function
def parallel_sample(s, question):
    s += (
        "Question: Compute 1 + 2 + 3\n"
        "Reasoning: I need to use a calculator.\n"
        "Tool: calculator\n"
        "Answer: 6\n"

        "Question: Compute 3 + 2 + 2\n"
        "Reasoning: I will try a calculator.\n"
        "Tool: calculator\n"
        "Answer: 7\n"
    )
    s += "Question: " + question + "\n"
    forks = s.fork(N)
    forks += "Reasoning:" + sgl.gen("reasoning", stop="\n") + "\n"
    forks += "Tool:" + sgl.gen("tool", choices=["calculator", "browser"]) + "\n"
    forks += "Answer:" + sgl.gen("answer", stop="\n") + "\n"
    forks.join()


sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo-instruct"))
#sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

state = parallel_sample.run(
    question="Compute 5 + 2 + 4.",
    temperature=1.0
)

for i in range(N):
    obj = {
        "reasoning": state["reasoning"][i],
        "tool": state["tool"][i],
        "answer": state["answer"][i],
    }
    print(f"[{i}], {obj}")

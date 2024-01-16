import sglang as sgl


@sgl.function
def tool_use(s, question):
    s += "To answer this question: " + question + ", "
    s += "I need to use a " + sgl.gen("tool", choices=["calculator", "web browser"]) + ". "
    if s["tool"] == "calculator":
        s += "The math expression is" + sgl.gen("expression")
    elif s["tool"] == "web browser":
        s += "The website url is" + sgl.gen("url")


@sgl.function
def tip_suggestion(s):
    s += (
        "Here are two tips for staying healthy: "
        "1. Balanced Diet. 2. Regular Exercise.\n\n"
    )

    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += f"Now, expand tip {i+1} into a paragraph:\n"
        f += sgl.gen(f"detailed_tip", max_tokens=256, stop="\n\n")

    s += "Tip 1:" + forks[0]["detailed_tip"] + "\n"
    s += "Tip 2:" + forks[1]["detailed_tip"] + "\n"
    s += "In summary" + sgl.gen("summary")


@sgl.function
def text_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n")


def driver_tool_use():
    state = tool_use.run(question="What is the capital of the United States?")
    print(state.text())
    print("\n")


def driver_tip_suggestion():
    state = tip_suggestion.run()
    print(state.text())
    print("\n")


def driver_batching():
    states = text_qa.run_batch(
        [
            {"question": "What is the capital of the United Kingdom?"},
            {"question": "What is the capital of France?"},
            {"question": "What is the capital of Japan?"},
        ],
    )

    for s in states:
        print(s.text())
    print("\n")


def driver_stream():
    state = text_qa.run(
        question="What is the capital of France?",
        temperature=0.1)

    for out in state.text_iter():
        print(out, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo-instruct"))

    driver_tool_use()
    driver_tip_suggestion()
    driver_batching()
    driver_stream()

"""
Usage:
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
python choices_logprob.py
"""
import sglang as sgl


@sgl.function
def tool_use(s, question):
    s += "To answer this question: " + question + ", "
    s += "I need to use a " + sgl.gen("tool", choices=["calculator", "search engine"])


def main():
    # Run one case
    question = "What is 5 + 5?"
    state = tool_use.run(question)
    print("questions:", question)
    print("choice:", state["tool"])
    meta_info = state.get_meta_info("tool")
    print("logprobs of choice 1", meta_info["prompt_logprob"][0])
    print("logprobs of choice 2", meta_info["prompt_logprob"][1])
    print('-' * 50)

    # Run a batch
    questions = [
        "What is 5 + 6?",
        "Who is Michael Jordan?",
    ]
    states = tool_use.run_batch([{"question": q} for q in questions])
    for question, state in zip(questions, states):
        print("questions:", question)
        print("choice:", state["tool"])
        meta_info = state.get_meta_info("tool")
        print("logprobs of choice 1", meta_info["prompt_logprob"][0])
        print("logprobs of choice 2", meta_info["prompt_logprob"][1])
        print('-' * 50)


if __name__ == "__main__":
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
    main()

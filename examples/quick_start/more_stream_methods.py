import asyncio
import sglang as sgl


@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))


sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo"))
#sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))


def stream_a_variable():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
        stream=True
    )

    for out in state.text_iter(var_name="answer_2"):
        print(out, end="", flush=True)
    print()


async def async_stream():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
        stream=True
    )

    async for out in state.text_async_iter(var_name="answer_2"):
        print(out, end="", flush=True)
    print()


if __name__ == "__main__":
    #stream_a_variable()
    asyncio.run(async_stream())

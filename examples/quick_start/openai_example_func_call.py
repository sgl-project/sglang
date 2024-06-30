"""
Usage:
export OPENAI_API_KEY=sk-******
python3 openai_example_func_call.py
"""

import sglang as sgl
import json


def get_current_weather(location: str, unit: str = "fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


@sgl.function
def question(s, question, tools=[]):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(
        sgl.gen("answer_1", max_tokens=256, tools=tools, tool_choice="auto")
    )


def single():
    state = question.run(
        question="What's the weather like in San Francisco, Tokyo, Paris, and Beijing?",
        tools=[get_current_weather],
    )

    for m in state.messages():
        print(m["role"], ":", m["content"])

    print("\n-- answer_1 --\n", state["answer_1"])
    # TODO: do we need to add another check for function call results


if __name__ == "__main__":
    sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo"))

    # Run a single request
    print("\n========== single ==========\n")
    single()

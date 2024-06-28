"""
Usage:
export OPENAI_API_KEY=sk-******
python3 openai_example_chat.py
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
def multi_turn_question(s, question_1, functions=[]):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.func_call("func_call_1", tools=functions, tool_choice="auto")
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))


def single():
    state = multi_turn_question.run(
        question_1="What's the weather like in San Francisco, Tokyo, Paris, and Beijing?",
        functions=[get_current_weather],
    )

    for m in state.messages():
        print(m["role"], ":", m["content"])

    print("\n-- answer_1 --\n", state["answer_1"])


if __name__ == "__main__":
    sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo"))

    # Run a single request
    print("\n========== single ==========\n")
    single()

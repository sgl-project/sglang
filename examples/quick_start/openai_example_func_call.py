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


def get_n_day_weather_forecast(location: str, num_days: int, unit: str = "fahrenheit"):
    """Get an N-day weather forecast in a given location"""
    if "tokyo" in location.lower():
        return json.dumps(
            {
                "location": "Tokyo",
                "num_days": num_days,
                "forecast": "all sunny",
                "unit": unit,
            }
        )
    elif "san francisco" in location.lower():
        return json.dumps(
            {
                "location": "San Francisco",
                "num_days": num_days,
                "forecast": "all foggy",
                "unit": unit,
            }
        )
    elif "paris" in location.lower():
        return json.dumps(
            {
                "location": "Paris",
                "num_days": num_days,
                "forecast": "all rainy",
                "unit": unit,
            }
        )
    else:
        return json.dumps({"location": location, "forecast": "unknown"})


@sgl.function
def question(s, question1, question2, tools=[]):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question1)
    s += sgl.assistant(
        sgl.gen(
            "answer_1",
            max_tokens=256,
            tools=tools,
            tool_choice="auto",
        )
    )
    s += sgl.user(question2)
    s += sgl.assistant(
        sgl.gen(
            "answer_2",
            max_tokens=256,
            tools=tools,
            tool_choice="auto",
        )
    )


def single():
    state = question.run(
        question1="What's the weather like in San Francisco, Tokyo, Paris, and Beijing?",
        question2="What's the weather like in San Francisco, Tokyo, Paris, and Beijing in the next 5 days?",
        tools=[get_current_weather, get_n_day_weather_forecast],
    )

    for m in state.messages():
        print(m["role"], ":", m["content"])

    print("\n-- answer_1 --\n", state["answer_1"])
    print("\n-- answer_2 --\n", state["answer_2"])


if __name__ == "__main__":
    sgl.set_default_backend(sgl.OpenAI("gpt-4o"))

    # Run a single request
    print("\n========== single ==========\n")
    single()

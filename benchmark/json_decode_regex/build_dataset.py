import json

import transformers
import wikipedia

model_path = "meta-llama/Llama-2-7b-chat-hf"
t = transformers.AutoTokenizer.from_pretrained(model_path)
city_names = [
    "los angles",
    "london",
    "tokyo",
    "beijing",
    "singapore",
    "paris",
    "dubai",
    "sydney",
    "moscow",
    "rome",
    "toronto",
    "rio de janeiro",
    "istanbul",
    "berlin",
    "auckland",
    "buenos aires",
    "mexico city",
    "mumbai",
    "seoul",
    "bangkok",
    "cairo",
    "athens",
    "jerusalem",
]


def get_content(city_name):
    content = str(wikipedia.page(city_name).content)
    content = content.replace("\n\n", "\n")

    tokens = t.encode(content)

    expected_tokens = 3000
    truncate_len = int((expected_tokens / len(tokens)) * len(content))
    truncate_content = content[:truncate_len]
    truncate_tokens = t.encode(truncate_content)

    # Count token
    print(
        f"city_name: {city_name}, #tokens: {len(tokens)}, #truncate tokens: {len(truncate_tokens)}"
    )

    return truncate_content


if __name__ == "__main__":
    with open("questions.jsonl", "w") as fout:
        for city_name in city_names:
            truncate_content = get_content(city_name)
            fout.write(json.dumps({"document": truncate_content}) + "\n")

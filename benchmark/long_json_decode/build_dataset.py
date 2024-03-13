import json

import transformers
import wikipedia

name = "meta-llama/Llama-2-7b-chat-hf"
t = transformers.AutoTokenizer.from_pretrained(name)
city_names = ["los angles", "london", "tokyo", "beijing", "singapore"]


for city_name in city_names:
    content = str(wikipedia.page(city_name).content)
    content = content.replace("\n\n", "\n")

    tokens = t.encode(content)

    truncate_len = int((10000 / len(tokens)) * len(content))
    truncate_content = content[:truncate_len]
    truncate_tokens = t.encode(truncate_content)

    # Count token
    print(
        f"city_name: {city_name}, #tokens: {len(tokens)}, #truncate tokens: {len(truncate_tokens)}"
    )

    with open("questions.jsonl", "a") as fout:
        fout.write(json.dumps({"document": truncate_content}) + "\n")

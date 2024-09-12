import json

import openai
import requests

import sglang as sgl

lora_path = "/home/ying/test_lora"
prompt_file = "/home/ying/test_prompt/dialogue_choice_prompts.json"
server_url = "http://127.0.0.1:30000"

client = openai.Client(base_url=server_url + "/v1", api_key="EMPTY")


# @sgl.function
# def generate(s, prompt):
#     s += prompt
#     s += sgl.gen("ans")

# sgl.set_default_backend(sgl.RuntimeEndpoint(server_url))


def generate(prompt, lora_path):
    json_data = {
        "text": prompt,
        "sampling_params": {},
        "return_logprob": False,
        "logprob_start_len": None,
        "top_logprobs_num": None,
        "lora_path": lora_path,
    }
    response = requests.post(
        server_url + "/generate",
        json=json_data,
    )
    return json.dumps(response.json())


with open(prompt_file, "r") as f:
    samples = json.load(f)


for sample in samples[:1]:
    assert sample[0]["role"] == "user"
    prompt = sample[0]["content"]
    assert sample[1]["role"] == "assistant"
    ref = sample[1]["content"]

    state = generate(prompt, lora_path)
    print("================================")
    print(ref)
    print("--------------------------------")
    # print(state["ans"])
    print(state)
    print()

# launch server
# python -m sglang.launch_server --model LxzGordon/URM-LLaMa-3.1-8B --is-embedding

import requests

url = "http://127.0.0.1:30000"

PROMPT = (
    "What is the range of the numeric output of a sigmoid node in a neural network?"
)
RESPONSE1 = "The output of a sigmoid node is bounded between -1 and 1."
RESPONSE2 = "The output of a sigmoid node is bounded between 0 and 1."

json_data = {
    "conv": [
        [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": RESPONSE1},
        ],
        [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": RESPONSE2},
        ],
    ],
}
response = requests.post(
    url + "/judge",
    json=json_data,
).json()

print(response)
print("scores:", [x["embedding"] for x in response])

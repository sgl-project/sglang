prompt = "Hello " * 16000

import json

import requests

response = requests.post(
    "http://0.0.0.0:8000/generate",
    json={"text": prompt, "sampling_params": {"temperature": 0}},
)

print(response)
# print("Response status code:", response.status_code)
# print("Response headers:", response.headers)
print("Response content (raw):", response.content)


# prev = 0
# for chunk in response.iter_lines(decode_unicode=False):
#     chunk = chunk.decode("utf-8")
#     if chunk and chunk.startswith("data:"):
#         if chunk == "data: [DONE]":
#             break
#         data = json.loads(chunk[5:].strip("\n"))
#         output = data["text"]
#         print(output[prev:], end="", flush=True)
#         prev = len(output)

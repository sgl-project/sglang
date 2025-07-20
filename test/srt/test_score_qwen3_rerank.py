import requests

# 1.starting a server: python3 -m sglang.launch_server --model-path /root/paddlejob/workspace/liuli44/Qwen/Qwen3-Reranker-0___6B/ --port 8191 --host 0.0.0.0
# 2.then: python test_score_qwen3_rerank.py

# url = "http://127.0.0.1:30000/v1/score"
url = "http://0.0.0.0:8191/v1/score"

payload = {
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": "what is panda?",
    "items": [
        "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
    ],
    "instruction": "Given a web search query, retrieve relevant passages that answer the query",
    "rerank_type": "qwen3-rerank",
    "label_token_ids": [9693, 2152],
}

response = requests.post(url, json=payload)
response_json = response.json()

for item in response_json:
    print(item, response_json[item])

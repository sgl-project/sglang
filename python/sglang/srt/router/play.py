import httpx

client = httpx.Client(base_url="http://127.0.0.1:3000")

data = {'text': 'Once upon a time,', 'sampling_params': {'max_new_tokens': 16, 'temperature': 0}}
headers = httpx.Headers({"Content-Type": "application/json"})
res = client.post(url = "/generate", json = data) # , headers = headers)
print(res)
import requests
port = 10010
url = f"http://10.140.37.157:{port}/v1/chat/completions"

data = {
    "model": "ckpt/Qwen2.5-VL-72B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "/mnt/petrelfs/shaojie/code/speed/example_image.png"
                    },
                },
                {"type": "text", "text": "What’s in this image?"},
            ],
        }
    ],
}

response = requests.post(url, json=data)
# import pdb; pdb.set_trace()
print(response.json())
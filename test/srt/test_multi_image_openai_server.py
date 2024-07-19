import openai
client = openai.Client(api_key="EMPTY", base_url="http://127.0.0.1:12000/v1")
request_1 = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is image 1:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b/resolve/main/TinyLlama_logo.png"
                    },
                },
                {"type": "text", "text": "This is image 2:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/mixtral_8x7b.jpg"
                    },
                },
                {"type": "text", "text": "Now , describe image 1 and image 2."},
            ],
        },
    ],
    temperature=0.7,
    max_tokens=512,
    stream=True,
)
request_2 = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/mixtral_8x7b.jpg"
                    },
                },
            ],
        },
    ],
    temperature=0.7,
    max_tokens=512,
    stream=True,
)
response_1 = ""
response_2 = ""
for chunk in request_1:
    if chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        response_1 += content

for chunk in request_2:
    if chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        response_2 += content
        
print(response_1)
print("----")
print(response_2)

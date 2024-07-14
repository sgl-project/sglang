import openai
client = openai.Client(api_key="EMPTY", base_url="http://127.0.0.1:10028/v1")
response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/mixtral_8x7b.jpg"
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/mixtral_8x7b.jpg"
                    },
                },
                {"type": "text", "text": "Describe this image"},
            ],
        },
        {
            "role": "assistant",
            "content": "A bar chart.",
        },
        {
            "role": "user",
            "content": "More details?"
        }
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)
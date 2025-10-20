from openai import OpenAI

openai_api_key = "Empty"
openai_api_base = "http://127.0.0.1:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
response = client.chat.completions.create(
    model="glm-4.6",
    messages=[
        {"role": "user", "content": "你好，你是谁开发的"},
    ],
    max_tokens=1024,
    temperature=0.0,
    extra_body={
        "do_sample": False,
    },
)
print(response.choices[0].message.content.strip())
print("="*50)
print(response.choices[0].message.reasoning_content.strip())

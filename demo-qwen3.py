from openai import OpenAI

# Modify OpenAI's API key and API base to use SGLang's API server.
openai_api_key = "EMPTY"
openai_api_base = f"http://127.0.0.1:30000/v1" # Use the correct port

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "Qwen/Qwen3-8B" # Use the model loaded by the server
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
        "separate_reasoning": True
    }
)

print("response.choices[0].message.reasoning_content: \n", response.choices[0].message.reasoning_content)
print("response.choices[0].message.content: \n", response.choices[0].message.content)
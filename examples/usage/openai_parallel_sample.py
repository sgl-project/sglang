import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

# Text completion
response = client.completions.create(
    model="default",
    prompt="I am a robot and I want to study like humans. Now let's tell a story. Once upon a time, there was a little",
    n=1,
    temperature=0.8,
    max_tokens=32,
)
print(response)


# Text completion
response = client.completions.create(
    model="default",
    prompt="I am a robot and I want to study like humans. Now let's tell a story. Once upon a time, there was a little",
    n=3,
    temperature=0.8,
    max_tokens=32,
)
print(response)


# Text completion
response = client.completions.create(
    model="default",
    prompt=["The name of the famous soccer player is ", "The capital of US is"],
    n=1,
    temperature=0.8,
    max_tokens=32,
)
print(response)


# Text completion
response = client.completions.create(
    model="default",
    prompt=["The name of the famous soccer player is ", "The capital of US is"],
    n=3,
    temperature=0.8,
    max_tokens=32,
)
print(response)


# Text completion
response = client.completions.create(
    model="default",
    prompt=[
        "The capital of France is",
        "The capital of Germany is",
        "The capital of US is",
    ],
    n=3,
    temperature=0.8,
    max_tokens=32,
)
print(response)

# Chat completion
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0.8,
    max_tokens=64,
    logprobs=True,
    n=4,
)
print(response)

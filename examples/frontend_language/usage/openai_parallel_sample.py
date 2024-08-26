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
    n=5,
    temperature=0.8,
    max_tokens=320,
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
    prompt=["The name of the famous soccer player is"],
    n=1,
    temperature=0.8,
    max_tokens=128,
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


response = client.completions.create(
    model="default",
    prompt=[
        "prompt1: I am a robot and I want to learn like humans. Now let's begin a tale. Once upon a time, there was a small",
        "prompt2: As a robot, my goal is to understand human learning. Let's start a story. In a faraway land, there lived a tiny",
        "prompt3: Being a robot, I aspire to study like people. Let's share a story. Long ago, there was a little",
        "prompt4: I am a robot aiming to learn like humans. Let's narrate a story. Once, in a distant kingdom, there was a young",
        "prompt5: As a robot, I seek to learn in human ways. Let's tell a story. Once upon a time, in a small village, there was a young",
    ],
    n=1,
    temperature=0.8,
    max_tokens=320,
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
    max_tokens=1,
    logprobs=True,
    top_logprobs=3,
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
    max_tokens=1,
    n=1,
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
    max_tokens=1,
    logprobs=True,
    top_logprobs=3,
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
    max_tokens=1,
    n=4,
)
print(response)

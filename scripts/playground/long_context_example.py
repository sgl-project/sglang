from urllib.request import urlopen

from openai import OpenAI

test_cases = {
    "64k": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/64k.txt",
    "200k": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/200k.txt",
    "600k": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/600k.txt",
    "1m": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/1m.txt",
}

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:30000/v1")

for name, url in test_cases.items():
    print(f"\n==== Running test case: {name} ====")
    try:
        with urlopen(url, timeout=10) as response:
            prompt = response.read().decode("utf-8")
    except Exception as e:
        print(f"Failed to load prompt for {name}: {e}")
        continue

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=128,
            temperature=0,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
    except Exception as e:
        print(f"\nError during completion for {name}: {e}")

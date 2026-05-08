"""
OpenAI-compatible LoRA adapter usage with SGLang.

Server Setup:
    python -m sglang.launch_server \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --enable-lora \\
        --lora-paths sql=/path/to/sql python=/path/to/python
"""

import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")


def main():
    print("SGLang OpenAI-Compatible LoRA Examples\n")

    # Example 1: NEW - Adapter in model parameter (OpenAI-compatible)
    print("1. Chat with LoRA adapter in model parameter:")
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct:sql",  # ← adapter:name syntax
        messages=[{"role": "user", "content": "Convert to SQL: show all users"}],
        max_tokens=50,
    )
    print(f"   Response: {response.choices[0].message.content}\n")

    # Example 2: Completions API with adapter
    print("2. Completion with LoRA adapter:")
    response = client.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct:python",
        prompt="def fibonacci(n):",
        max_tokens=50,
    )
    print(f"   Response: {response.choices[0].text}\n")

    # Example 3: OLD - Backward compatible with explicit lora_path
    print("3. Backward compatible (explicit lora_path):")
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Convert to SQL: show all users"}],
        extra_body={"lora_path": "sql"},
        max_tokens=50,
    )
    print(f"   Response: {response.choices[0].message.content}\n")

    # Example 4: Base model (no adapter)
    print("4. Base model without adapter:")
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=30,
    )
    print(f"   Response: {response.choices[0].message.content}\n")

    print("All examples completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print(
            "\nEnsure server is running:\n"
            "  python -m sglang.launch_server --model ... --enable-lora --lora-paths ..."
        )

"""
Usage:

python -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-1.5B-instruct --port 30000
python hidden_states_server.py

Note that each time you change the `return_hidden_states` parameter,
the cuda graph will be recaptured, which might lead to a performance hit.
So avoid getting hidden states and completions alternately.
"""

import requests


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 10,
    }

    json_data = {
        "text": prompts,
        "sampling_params": sampling_params,
        "return_hidden_states": True,
    }

    response = requests.post(
        "http://127.0.0.1:30000/generate",
        json=json_data,
    )

    outputs = response.json()
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(
            f"Prompt: {prompt}\n"
            f"Generated text: {output['text']}\n"
            f"Prompt_Tokens: {output['meta_info']['prompt_tokens']}\t"
            f"Completion_tokens: {output['meta_info']['completion_tokens']}\n"
            f"Hidden states: {output['meta_info']['hidden_states']}"
        )
        print()


if __name__ == "__main__":
    main()

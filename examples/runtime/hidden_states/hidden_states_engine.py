"""
Usage:
python hidden_states_engine.py

CUDA graphs use the configured maximum hidden-state mode. Requests may select
that mode or a weaker one without triggering mode-dependent recapture.
"""

import torch

import sglang as sgl


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create an LLM.
    llm = sgl.Engine(
        model_path="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        return_hidden_states_mode="last",
    )

    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 10,
    }

    outputs = llm.generate(
        prompts, sampling_params=sampling_params, return_hidden_states="last"
    )

    llm.shutdown()

    for prompt, output in zip(prompts, outputs):
        hidden_state = torch.tensor(
            output["meta_info"]["hidden_states"], dtype=torch.bfloat16
        )
        print("===============================")
        print(
            f"Prompt: {prompt}\n"
            f"Generated text: {output['text']}\n"
            f"Prompt_Tokens: {output['meta_info']['prompt_tokens']}\t"
            f"Completion_tokens: {output['meta_info']['completion_tokens']}"
        )
        print("Last hidden state: ")
        print(hidden_state)
        print()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()

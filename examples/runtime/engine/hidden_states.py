"""
Usage:
python hidden_states.py

Note that each time you change the `return_hidden_states` parameter,
the cuda graph will be recaptured, which might lead to a performance hit.
"""

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
    )

    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 10,
        "return_hidden_states": True,
    }

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(
            f"Prompt: {prompt}\nGenerated text: {output['text']}\nPrompt_Tokens: {output['meta_info']['prompt_tokens']}\tCompletion_tokens: {output['meta_info']['completion_tokens']}\nHidden states: {[i.shape for i in output['meta_info']['hidden_states']]}"
        )
        print()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()

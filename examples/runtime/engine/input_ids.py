"""
This example demonstrates how to provide tokenized ids as input instead of text prompt
"""

import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"

def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    # Tokenize inputs
    tokenizer = get_tokenizer(MODEL_PATH)
    token_ids_list = [tokenizer.encode(prompt) for prompt in prompts]
    
    # Create an LLM.
    # You can also specify `skip_tokenizer_init=True`, but it requires explicit detokenization at the end
    llm = sgl.Engine(model_path=MODEL_PATH)

    outputs = llm.generate(input_ids=token_ids_list, sampling_params=sampling_params)
    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated Text: {output['text']}")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()
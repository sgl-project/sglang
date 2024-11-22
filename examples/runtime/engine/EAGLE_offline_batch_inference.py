import sglang as sgl


def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = {"temperature": 0, "max_new_tokens": 256}

    # Create an LLM.
    llm = sgl.Engine(
        model_path="meta-llama/Llama-2-7b-chat-hf",
        draft_model_path="kavio/Sglang-EAGLE-llama2-chat-7B",
        num_speculative_steps=3,
        eagle_topk=4,
        num_draft_tokens=16,
        speculative_algorithm="EAGLE",
        mem_fraction_static=0.70,
    )

    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()

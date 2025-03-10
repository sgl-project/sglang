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
    sampling_params = {"temperature": 0, "max_new_tokens": 30}

    # Create an LLM.
    llm = sgl.Engine(
        model_path="meta-llama/Llama-2-7b-chat-hf",
        speculative_algorithm="EAGLE",
        speculative_draft_model_path="lmsys/sglang-EAGLE-llama2-chat-7B",
        speculative_num_steps=3,
        speculative_eagle_topk=4,
        speculative_num_draft_tokens=16,
        cuda_graph_max_bs=8,
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

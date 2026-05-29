"""Example: offline batch inference with EAGLE3 speculative decoding on Qwen3.

Companion to ``offline_batch_inference_eagle.py`` (Llama-2 + EAGLE V1). This
script targets Qwen3-1.7B paired with the community-maintained
AngelSlim/Qwen3-1.7B_eagle3 draft. The same pattern works for the rest of
the AngelSlim Qwen3 EAGLE3 family by swapping ``model_path`` and
``speculative_draft_model_path`` (4B/8B/14B/32B/30B-A3B).

Reference benchmark (AngelSlim, vLLM v0.11.2, single H20, tp=1, output_len=1024,
num_speculative_tokens=2): Qwen3-1.7B vanilla 381 tok/s vs EAGLE3 643 tok/s
(~1.69x, average accept length 2.17). See the model card for full numbers:
https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3

The defaults below follow AngelSlim's recommended SGLang configuration
(speculative_num_steps=3, eagle_topk=1, num_draft_tokens=4). For short,
template-heavy outputs (tool calling, structured JSON, code completion)
these settings are typically a good starting point; for longer reasoning
outputs you may want to raise ``speculative_num_draft_tokens`` after
profiling acceptance length.
"""

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

    # Create an LLM with EAGLE3 speculative decoding.
    llm = sgl.Engine(
        model_path="Qwen/Qwen3-1.7B",
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="AngelSlim/Qwen3-1.7B_eagle3",
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
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

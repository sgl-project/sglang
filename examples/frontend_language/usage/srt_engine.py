import sglang as sgl

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = {"temperature": 0.8, "top_p": 0.95}

# Create an LLM.
llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for prompt, output in zip(prompts, outputs):
    print("===============================")
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

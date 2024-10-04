import sglang as sgl

# Sample prompts.
prompts = [
    "The capital of China is",
    "The square root of 144 is",
]
# Create a sampling params object.
sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 128}

# Create an LLM.
llm = sgl.Engine(model_path="/shared/public/models/Meta-Llama-3.1-8B-Instruct")

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for prompt, output in zip(prompts, outputs):
    print("===============================")
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

assert "Beijing" in outputs[0]["text"]
assert "12" in outputs[1]["text"]

from sglang import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The capital of China is",
    "What is the meaning of life?",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="deepseek-ai/deepseek-llm-7b-chat", tensor_parallel_size=1)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    index = output["index"]
    prompt = prompts[index]
    answer = output["text"]
    print("===============================")
    print(f"Prompt: {prompt}")
    print(f"Generated text: {output['text']}")

import sglang as sgl

def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = {"temperature": 0.8, "top_p": 0.95}
    llm = sgl.Engine(model_path="nvidia/Llama-3.1-8B-Instruct-FP8", quantization="modelopt")

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

if __name__ == "__main__":
    main()
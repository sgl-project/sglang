import sglang as sgl


def main():
    # Sample prompts.
    prompts = "Elon Musk is"
    
    # Create a sampling params object.
    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    # Create an LLM.
    llm = sgl.Engine(model_path="/shared/public/models/Qwen/Qwen2.5-1.5B-Instruct/", log_level="warning")

    output_iterator = llm.generate(prompts, sampling_params, stream=True)

    for o in output_iterator:
        print(o["text"], end="", flush=True)

    print()
    
    # # Print the outputs.
    # for prompt, output in zip(prompts, outputs):
    #     print("===============================")
    #     print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()

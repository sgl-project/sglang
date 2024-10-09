import sglang as sgl
import asyncio

async def async_streaming(engine):

    generator = await engine.async_generate("Elon Musk is", {"temperature": 0.8, "top_p": 0.95}, stream=True)

    async for output in generator:
        print(output["text"], end="", flush=True)
    print()


def main():
    # Sample prompts.
    prompts = "Elon Musk is"
    
    # Create a sampling params object.
    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    # Create an LLM.
    llm = sgl.Engine(model_path="/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/", log_level="warning")

    # 1. sync + non streaming
    print("\n\n==== 1. sync + non streaming ====")
    output = llm.generate(prompts, sampling_params)

    print(output["text"])

    # 2. sync + streaming
    print("\n\n==== 2. sync + streaming ====")
    output_generator = llm.generate(prompts, sampling_params, stream=True)
    for output in output_generator:
        print(output["text"], end="", flush=True)
    print()


    loop = asyncio.get_event_loop()
    # 3. async + non_streaming
    print("\n\n==== 3. async + non streaming ====")
    output = loop.run_until_complete(llm.async_generate(prompts, sampling_params))
    print(output["text"])

    # 4. async + streaming
    print("\n\n==== 4. async + streaming ====")
    loop.run_until_complete(async_streaming(llm))



# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()

import sglang as sgl
import time

def main():
    # Sample prompts.
    prompts = [
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Where is the capital city of France? ASSISTANT:",
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: 北京今天天气怎么样？ ASSISTANT:"
    ]
    # Create a sampling params object.
    sampling_params = {"temperature": 0, "max_new_tokens": 30}


    # Create an LLM.
    llm = sgl.Engine(model_path="Llama-2-7b-chat-hf", draft_model_path='EAGLE-llama2-chat-7B', disable_cuda_graph=True, num_speculative_steps=5, eagle_topk=8, num_draft_tokens=64, speculative_algorithm='EAGLE', mem_fraction_static=0.60)
    #llm = sgl.Engine(model_path="Llama-2-7b-chat-hf", disable_cuda_graph=False)
    #outputs = llm.generate(prompts, sampling_params)
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print(time.time()-start)
    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()

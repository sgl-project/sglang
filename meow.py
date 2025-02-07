# launch the offline engine
from transformers import AutoTokenizer
import sglang as sgl
import asyncio

def main():
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm = sgl.Engine(model_path=MODEL_NAME, skip_tokenizer_init=True, disable_cuda_graph=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        #"The capital of France is",
        #"The future of AI is",
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 10}

    input_ids = tokenizer(prompts).input_ids
    outputs = llm.generate(input_ids=input_ids, sampling_params=sampling_params)
    for input_id, output in zip(input_ids, outputs):
        print("===============================")
        print(input_id)
        print(output)
        print()
        print(input_id, output['token_ids'], len(input_id), len(output['token_ids']))
        print([i.shape for i in output['meta_info']['hidden_states']], len(output['meta_info']['hidden_states']))
        #print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

if __name__ == "__main__":
    main()

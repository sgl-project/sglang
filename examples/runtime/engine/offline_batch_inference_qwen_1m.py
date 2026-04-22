"""
Usage:
python3 offline_batch_inference.py
"""

from urllib.request import urlopen

import sglang as sgl


def load_prompt() -> str:
    # Test cases with various lengths can be found at:
    #
    # https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/64k.txt
    # https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/200k.txt
    # https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/600k.txt
    # https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/1m.txt

    with urlopen(
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com"
        "/Qwen2.5-1M/test-data/64k.txt",
        timeout=5,
    ) as response:
        prompt = response.read().decode("utf-8")
    return prompt


# Processing the prompt.
def process_requests(llm: sgl.Engine, prompts: list[str]) -> None:
    # Create a sampling params object.
    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "max_new_tokens": 256,
    }
    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt_token_ids = output["meta_info"]["prompt_tokens"]
        generated_text = output["text"]
        print(
            f"Prompt length: {prompt_token_ids}, " f"Generated text: {generated_text!r}"
        )


# Create an LLM.
def initialize_engine() -> sgl.Engine:
    llm = sgl.Engine(
        model_path="Qwen/Qwen2.5-7B-Instruct-1M",
        context_length=1048576,
        page_size=256,
        attention_backend="dual_chunk_flash_attn",
        tp_size=4,
        disable_radix_cache=True,
        enable_mixed_chunk=False,
        enable_torch_compile=False,
        chunked_prefill_size=131072,
        mem_fraction_static=0.6,
        log_level="DEBUG",
    )
    return llm


def main():
    llm = initialize_engine()
    prompt = load_prompt()
    process_requests(llm, [prompt])


if __name__ == "__main__":
    main()

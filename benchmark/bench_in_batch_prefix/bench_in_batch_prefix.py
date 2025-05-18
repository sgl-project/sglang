# Benchmark with lots of common prefixes. Used to benchmark prefix caching performance.
#
# Launch a server:
# python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --log-level-http warning

import random
import string
import time

from tqdm import tqdm
from transformers import AutoTokenizer

import sglang as sgl
from sglang import set_default_backend
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint


def generate_random_string(token_length: int) -> str:
    random_string = "".join(
        random.choices(string.ascii_letters + string.digits, k=token_length * 100)
    )
    tokenized_output = tokenizer.encode(random_string, add_special_tokens=False)[
        :token_length
    ]

    if len(tokenized_output) < token_length:
        tokenized_output = tokenized_output + [tokenizer.pad_token_id] * (
            token_length - len(tokenized_output)
        )

    decoded_string = tokenizer.decode(tokenized_output, skip_special_tokens=False)
    return decoded_string


def generate_unique_prefix(base_text, index):
    return str(index) + base_text[len(str(index)) :]


@sgl.function
def text_qa(s, question, gen_len):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n", temperature=0, max_tokens=gen_len)


def prepare_prompts(num_prefix, num_samples_per_prefix, prefix_length, suffix_length):
    base_prefix = generate_random_string(prefix_length)

    tot_input_len = 0
    all_prompts = []
    for i in tqdm(range(num_prefix), desc="prepare prompts"):
        unique_prefix = generate_unique_prefix(base_prefix, i)
        prompt_list = []
        for j in range(num_samples_per_prefix):
            suffix = generate_random_string(suffix_length)
            prompt = unique_prefix + suffix
            prompt_list.append(prompt)
            tot_input_len += len(tokenizer.encode(prompt))
        all_prompts.append(prompt_list)
    return all_prompts, tot_input_len


def test_batch_by_batch(all_prompts, gen_len):
    backend.flush_cache()

    tot_time = 0
    for i in range(len(all_prompts)):
        tic = time.perf_counter()
        text_qa.run_batch(
            list(zip(all_prompts[i], [gen_len] * len(all_prompts[i]))),
        )
        tot_time += time.perf_counter() - tic

    return tot_time


def test_batch_by_batch_with_hint(all_prompts, gen_len):
    backend.flush_cache()

    tot_time = 0
    for i in range(len(all_prompts)):
        tic = time.perf_counter()
        # Send a hint to cache the prefix
        text_qa.run_batch(list(zip(all_prompts[i][:1], [gen_len])))
        # Send the batch
        text_qa.run_batch(list(zip(all_prompts[i], [gen_len] * len(all_prompts[i]))))

        tot_time += time.perf_counter() - tic

    return tot_time


def test_send_all(all_prompts, gen_len):
    backend.flush_cache()

    all_prompts = [x for prompt_list in all_prompts for x in prompt_list]

    tic = time.perf_counter()
    text_qa.run_batch(
        list(zip(all_prompts, [gen_len] * len(all_prompts))),
    )
    tot_time = time.perf_counter() - tic

    return tot_time


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    backend = RuntimeEndpoint("http://127.0.0.1:30000")
    set_default_backend(backend)

    random.seed(0)
    num_prefix = 10
    num_samples_per_prefix = 32
    prefix_length = 1024
    suffix_length = 128
    gen_len = 1
    all_prompts, tot_input_len = prepare_prompts(
        num_prefix, num_samples_per_prefix, prefix_length, suffix_length
    )

    print(f"Total input token length: {tot_input_len}\n")

    cost = test_batch_by_batch(all_prompts, gen_len)
    print(f"Latency of test_batch_by_batch          : {cost:.4f} s\n")

    cost = test_batch_by_batch_with_hint(all_prompts, gen_len)
    print(f"Latency of test_batch_by_batch_with_hint: {cost:.4f} s\n")

    cost = test_send_all(all_prompts, gen_len)
    print(f"Latency of test_send_all                : {cost:.4f} s\n")

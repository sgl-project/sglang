import json
import random

from datasets import load_dataset

from sglang.bench_serving import get_tokenizer

# Login using e.g. `huggingface-cli login` to access this dataset
metadata = load_dataset("open-thoughts/OpenThoughts-114k", "metadata")
dataset = list(load_dataset("open-thoughts/OpenThoughts-114k", "default")["train"])
tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B-Instruct")

random.seed(42)
random.shuffle(dataset)

NUM_CLIENTS = 256
NUM_ROUNDS = 4
new_dataset = []


COUNTER = 0

input_lengths = []
output_lengths = []


def sample_filter(num_samples):
    global COUNTER
    res = []
    for i in range(COUNTER, len(dataset)):
        d = dataset[i]
        in_token = len(tokenizer(d["conversations"][0]["value"])["input_ids"])
        out_token = len(tokenizer(d["conversations"][1]["value"])["input_ids"])
        if out_token < 8000:
            input_lengths.append(in_token)
            output_lengths.append(out_token)
            COUNTER += 1
            res.append(d)
            if len(res) == num_samples:
                return res


for i in range(NUM_CLIENTS):
    samples = sample_filter(NUM_ROUNDS)
    entry = {
        "system": samples[0]["system"],
        "inputs": [s["conversations"][0]["value"] for s in samples],
        "reference_outputs": [s["conversations"][1]["value"] for s in samples],
    }
    new_dataset.append(entry)

print(
    "Mean input lengths:",
    sum(input_lengths) / len(input_lengths),
)
print(
    "Mean output lengths:",
    sum(output_lengths) / len(output_lengths),
)
open(f"open_thoughts_{NUM_CLIENTS}_{NUM_ROUNDS}.json", "w").write(
    json.dumps(new_dataset, indent=2)
)

import json
import os
import pickle
import random
from typing import Dict, List, Optional, Tuple
import requests
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

# type of content fields, can be only prompts or with images/videos
MsgContent = str

# A list of all the conversations. Each conversation is a list of
# tuples. If multiturn is not enabled, the length of list is 1,
# containing only the first Q&A pair.
# For the shared prefix workload (synthetic, loogle, nextqa), it
# is a list of conversations sharing the same prefix (synthetic,
# doc, video)
SampleOutput = List[List[Tuple[MsgContent, int, int]]]

LoadedDataSet = List[List[Tuple[str, str]]]

def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename

def common_filter_chat(
    num_requests: int,
    new_dataset: LoadedDataSet,
    tokenizer: PreTrainedTokenizerBase,
    min_prompt_len: Optional[int],
    min_output_len: Optional[int],
    max_prompt_len: Optional[int],
    max_output_len: Optional[int],
    fixed_output_len: Optional[int],
    max_sum_len: Optional[int],
    is_loogle: bool = False,
) -> SampleOutput:
    print("[[INFO]] Start filtering......")

    # Filter out sequences that are too long or too short
    filtered_dataset: SampleOutput = []
    l = 0
    input_tokens = 0
    output_tokens = 0
    pbar = tqdm(total=num_requests, desc="Filtering dataset")
    last_prefix = ""
    prefix_tokens = 0
    while l < num_requests:
        for i in range(len(new_dataset)):
            if l == num_requests:
                break
            processed = []
            sum_input_len = 0
            sum_output_len = 0

            for j in new_dataset[i]:
                # Tokenize the prompts and completions.
                prompt = j[0]
                prompt_token_ids = tokenizer.encode(prompt)
                prompt_len = len(prompt_token_ids)
                completion = j[1]
                completion_token_ids = tokenizer.encode(completion)
                output_len = (
                    len(completion_token_ids)
                    if fixed_output_len is None
                    else fixed_output_len
                )
                if (
                    min_prompt_len is not None
                    and prompt_len < min_prompt_len
                    or min_output_len is not None
                    and output_len < min_output_len
                    or max_prompt_len is not None
                    and prompt_len > max_prompt_len
                    or max_output_len is not None
                    and output_len > max_output_len
                ):
                    # Prune too short sequences.
                    continue

                sum_input_len += prompt_len
                sum_output_len += output_len
                processed.append((prompt, prompt_len, output_len))

            sum_len = sum_input_len + sum_output_len
            if max_sum_len is not None and sum_len > max_sum_len or len(processed) == 0:
                continue
            if not is_loogle and len(processed) <= 2:
                continue
            # only commit the change if the sequence is valid
            input_tokens += sum_input_len
            output_tokens += sum_output_len
            filtered_dataset.append(processed)
            if not is_loogle:
                l += 1
                pbar.update(1)
            else:
                assert len(processed) == 1
                if processed[0][0][:100] != last_prefix:
                    l += 1
                    pbar.update(1)
                    last_prefix = processed[0][0][:100]
                    prefix_tokens += processed[0][1]

    pbar.close()
    print(f"# Input tokens:  {input_tokens}")
    print(f"# Output tokens: {output_tokens}")
    if not is_loogle: # multiturn sharegpt
        print(f"# Num samples:   {len(filtered_dataset)}")
        print(f"# Num turns:     {sum(len(x) for x in filtered_dataset)}")
    else:
        print(f"# Prefix tokens: {prefix_tokens}")
        print(f"# Num prefixes:  {l}")
        print(f"# Num turns:     {len(filtered_dataset)}")

    return filtered_dataset

def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    max_sum_len: Optional[int] = None,
) -> SampleOutput:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Download sharegpt if necessary
    if not os.path.isfile(dataset_path):
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Keep one conversation in one list
    new_dataset: LoadedDataSet = []
    for data in dataset:
        total_len = len(data["conversations"])
        if total_len == 0 or total_len % 2 != 0:
            continue
        if data["conversations"][0]["from"] != "human":
            continue

        chat: List[Tuple[str, str]] = []
        for i in range(0, total_len, 2):
            chat.append(
                (
                    data["conversations"][i + 0]["value"],
                    data["conversations"][i + 1]["value"],
                )
            )
        new_dataset.append(chat)

    # Filter out sequences that are too long or too short
    filtered_dataset: SampleOutput = common_filter_chat(
        num_requests, new_dataset, tokenizer, 4, 4, None, None, fixed_output_len, max_sum_len
    )
    return filtered_dataset

def sample_leval_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    max_sum_len: Optional[int] = None
) -> SampleOutput:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    files = []
    for root, dirs, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.endswith(".jsonl"):
                files.append(os.path.join(root, filename))


    new_dataset: LoadedDataSet = []
    for file in files:
        print(f"Processing {file}")
        with open(file, "r") as f:
            for line in f.readlines():
                if line.strip() == "": continue
                data = json.loads(line)
                input = data["input"]
                for (instruction, output) in zip(data["instructions"], data["outputs"]):
                    new_dataset+=[[[input + instruction, output]]]
    filtered_dataset: SampleOutput = common_filter_chat(
        num_requests, new_dataset, tokenizer, 4, None, None, None,
        fixed_output_len, max_sum_len, is_loogle=True
    )
    return filtered_dataset

        

def sample_loogle_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    max_sum_len: Optional[int] = None,
) -> SampleOutput:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset
    dataset = []
    with open(dataset_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            dataset.append(json.loads(line))

    # Keep one conversation in one list
    new_dataset: LoadedDataSet = []
    for data in dataset:
        if (
            "qa_pairs" not in data
            or data["qa_pairs"] == "none"
            or len(data["qa_pairs"]) == 0
        ):
            continue
        prefix = f"Input: {data['input']} Question: "
        qa_pairs = eval(data["qa_pairs"])
        # Flatten the list of QA pairs
        new_dataset += [[(prefix + qa["Q"], qa["A"])] for qa in qa_pairs]

    # Filter out sequences that are too long or too short
    filtered_dataset: SampleOutput = common_filter_chat(
        num_requests, new_dataset, tokenizer, 4, None, None, None,
        fixed_output_len, max_sum_len, is_loogle=True
    )
    return filtered_dataset

def get_dataset_from_openai(
    *_args,
    dataset_name: str,
    dataset_path: str,
    num_prompts: int = int(1e9), # big number
    fixed_output_len: Optional[int] = None,
    max_sum_len: Optional[int] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> SampleOutput:
    FUNC_MAP = {
        "sharegpt": sample_sharegpt_requests,
        "loogle": sample_loogle_requests,
        "leval": sample_leval_requests,
    }

    if f := FUNC_MAP.get(dataset_name):
        return f(
            dataset_path=dataset_path,
            num_requests=num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=fixed_output_len,
            max_sum_len=max_sum_len,
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")

def get_mooncake_trace(
    model_name: str,
    *_args,
    dataset_path: str,
    max_sum_len: int,
) -> Tuple[SampleOutput, List[float]]:
    assert len(_args) == 0, f"Unexpected args: {_args}"
    random.seed(42)

    formal_name = dataset_path.replace(".json", "").replace("/", "_").replace(".", "_")

    home = os.environ["HOME"]

    formal_name += '.' + model_name.replace("/", "_")

    cache_path = f"{home}/.cache/__{formal_name}.pkl"
    if os.path.exists(cache_path):
        print(f"Cache found at {cache_path}, loading dataset")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Cache not found at {cache_path}, generating new dataset")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(dataset_path) as f:
        lines = f.readlines()

    max_vocab_size = tokenizer.vocab_size
    eos_id = tokenizer.eos_token_id
    print(f"{eos_id=}, {max_vocab_size=}")
    assert isinstance(eos_id, int)

    visited: Dict[Tuple[int, ...], int] = {}
    mapping: Dict[int, Tuple[int, ...]] = {}

    def decode(id_list: List[int]) -> str:
        return tokenizer.decode(id_list)

    def encode(raw_str: str) -> List[int]:
        result = tokenizer.encode(raw_str)
        if result[0] == max_vocab_size:
            result = result[1:]
        return result

    def is_stable_list(id_list: List[int]) -> bool:
        return encode(decode(id_list)) == id_list

    def gen_id_list() -> List[int]:
        id_list = [random.randint(0, max_vocab_size - 1) for _ in range(511)]
        for _ in range(256):
            # print(id_list)
            new_id_list = encode(decode(id_list))
            while new_id_list != id_list:
                # print(new_id_list)
                id_list = new_id_list
                new_id_list = encode(decode(id_list))
            if len(id_list) == 511:
                return [eos_id] + id_list
            if len(id_list) > 511:
                id_list = id_list[:511]
            else:
                needed = 511 - len(id_list)
                for _ in range(needed):
                    id_list.append(random.randint(0, max_vocab_size - 1))

        raise ValueError(f"Failed to generate id_list")

    def shrink_list(id_list: List[int], input_length: int) -> Tuple[List[int], str]:
        for _ in range(16):
            raw_str = decode(id_list)
            id_list = encode(raw_str)
            if len(id_list) == input_length:
                return id_list, raw_str
            if len(id_list) < input_length:
                id_list.append(eos_id)
            else:
                id_list = id_list[:input_length]
        raise ValueError(f"Failed to shrink id_list to {input_length}")

    def get_id(id: int) -> Tuple[int, ...]:
        while id not in mapping:
            result = gen_id_list()
            assert is_stable_list(result)
            result = tuple(result)
            if result in visited:
                print("Warning: collision")
                continue
            visited[result] = id
            mapping[id] = result
        return mapping[id]

    def mooncake_process(line: str) -> Tuple[List[Tuple[str, int, int]], float] | None:
        data = json.loads(line)
        input_length = int(data["input_length"])
        output_length = int(data["output_length"])
        if input_length + output_length > max_sum_len:
            return None

        raw_lists = [get_id(hash_id) for hash_id in data["hash_ids"]]
        flat_ids = [x for y in raw_lists for x in y]
        assert input_length <= len(flat_ids)

        final_ids, raw_str = shrink_list(flat_ids, input_length)

        assert raw_str == decode(final_ids)
        assert final_ids == encode(raw_str)
        assert len(final_ids) == input_length
        chunk_size = (input_length // 512) * 512
        assert flat_ids[:chunk_size] == final_ids[:chunk_size]

        return [(raw_str, input_length, output_length)], int(data["timestamp"]) / 1000

    raw_dataset: List[List[Tuple[str, int, int]]] = []
    timestamps: List[float] = []
    for line in tqdm(lines, desc="Processing mooncake trace"):
        if result := mooncake_process(line):
            raw_dataset.append(result[0])
            timestamps.append(result[1])

    with open(cache_path, "wb") as f:
        pickle.dump((raw_dataset, timestamps), f)

    # always set the seed to 42 after returning the dataset
    random.seed(42)

    return raw_dataset, timestamps

if __name__ == "__main__":
    def main():
        dataset_name = "loogle"
        dataset_path = "longdep_qa.json"
        num_prompts  = 100
        _ = get_dataset_from_openai(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            num_prompts=num_prompts
        )
    
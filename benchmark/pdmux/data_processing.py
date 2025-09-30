import json
from typing import List, Optional, Tuple

from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

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
    if not is_loogle:  # multiturn sharegpt
        print(f"# Num samples:   {len(filtered_dataset)}")
        print(f"# Num turns:     {sum(len(x) for x in filtered_dataset)}")
    else:
        print(f"# Prefix tokens: {prefix_tokens}")
        print(f"# Num prefixes:  {l}")
        print(f"# Num turns:     {len(filtered_dataset)}")

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
        num_requests,
        new_dataset,
        tokenizer,
        4,
        None,
        None,
        None,
        fixed_output_len,
        max_sum_len,
        is_loogle=True,
    )
    return filtered_dataset


def get_dataset_from_openai(
    *_args,
    dataset_name: str,
    dataset_path: str,
    num_prompts: int = int(1e9),  # big number
    fixed_output_len: Optional[int] = None,
    max_sum_len: Optional[int] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> SampleOutput:
    FUNC_MAP = {
        "loogle": sample_loogle_requests,
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

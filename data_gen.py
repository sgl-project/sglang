import argparse
import os
import re
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="sglang data gen")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--gpu_index", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--outdir", type=str, default="/root/.cache/hidden_states_dump")
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["sharegpt", "ultrachat", "mixture_of_thoughts"],
        default="sharegpt",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_index))
    MAX_TOKEN_LENGTH = 2048

    # ------------------------ 1. Dataset ------------------------
    # This step converts the dataset into a standard messages format
    if args.dataset == "sharegpt":
        dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    elif args.dataset == "ultrachat":
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    elif args.dataset == "mixture_of_thoughts":
        dataset = load_dataset("open-r1/Mixture-of-Thoughts", "all", split="all")

    dataset = dataset.select(range(args.start, args.end))
    dataset = dataset.shuffle(seed=42)

    # System message that will be prepended to all conversations
    system_message = {
        "role": "system",
        "content": "You are a helpful, respectful and honest assistant.",
    }

    def format_conversation_sharegpt(row, dataset_column="conversations"):
        messages = [system_message]
        current_role = None
        for message in row[dataset_column]:
            if message["from"] == "human":
                messages.append({"role": "user", "content": message["value"]})
            elif message["from"] == "gpt":
                messages.append({"role": "assistant", "content": message["value"]})
            else:
                raise ValueError(f"Unknown role: {message['from']}")

            if current_role is None:
                current_role = messages[-1]["role"]
            else:
                assert (
                    current_role != messages[-1]["role"]
                ), f"Conversation has incorrect role order"
                current_role = messages[-1]["role"]
        return {"messages": messages}

    def format_conversation_ultrachat(row, dataset_column="messages"):
        messages = [system_message]
        for message in row[dataset_column]:
            messages.append(message)
        return {"messages": messages}

    if args.dataset == "sharegpt":
        dataset = dataset.map(format_conversation_sharegpt)
    elif args.dataset == "ultrachat":
        dataset = dataset.map(format_conversation_ultrachat)
    elif args.dataset == "mixture_of_thoughts":
        pass  # no need to format

    # ------------------------ 2. Tokenizer ------------------------
    # This step tokenizes the conversation and creates the loss mask
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Special token sequences used to identify different parts of the conversation
    # For Llama models
    # assistant_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    # user_header = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
    # For Qwen models
    # assistant_header = "<|im_start|>assistant\n"
    # user_header = "<|im_start|>user\n"
    # for Llama-4-Scout-17B-16E-Instruct
    assistant_header = "<|header_start|>assistant<|header_end|>\n\n"
    user_header = "<|header_start|>user<|header_end|>"

    def tokenize_conversation(row, tokenizer, col="messages"):
        formatted_conversation = tokenizer.apply_chat_template(
            row[col], tokenize=False, add_generation_prompt=False
        )
        encoding = tokenizer(
            formatted_conversation, return_offsets_mapping=True, max_length=MAX_TOKEN_LENGTH
        )
        input_ids = encoding.input_ids
        offsets = encoding.offset_mapping
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        # Find spans of assistant responses using regex
        assistant_pattern = (
            re.escape(assistant_header) + r"(.*?)(?=" + re.escape(user_header) + "|$)"
        )
        for match in re.finditer(assistant_pattern, formatted_conversation, re.DOTALL):
            # Assistant response text span (excluding assistant_header itself)
            assistant_start_char = match.start(1)
            assistant_end_char = match.end(1)

            # Mark tokens overlapping with assistant response
            for idx, (token_start, token_end) in enumerate(offsets):
                # Token is part of the assistant response span
                if token_end <= assistant_start_char:
                    continue  # token before assistant text
                if token_start > assistant_end_char:
                    continue  # token after assistant text
                loss_mask[idx] = 1
        return {
            "conversation_str": formatted_conversation,
            "input_ids": input_ids,
            "loss_mask": loss_mask,
        }

    dataset = dataset.map(tokenize_conversation, fn_kwargs={"tokenizer": tokenizer})
    dataset = dataset.remove_columns(
        [
            col
            for col in dataset.column_names
            if col not in ["input_ids", "loss_mask", "conversation_str"]
        ]
    )
    dataset.set_format(type="torch")

    # ------------------------ 3. Compute hidden states ------------------------
    import sglang as sgl

    llm = sgl.Engine(
        model_path="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        skip_tokenizer_init=True,
        enable_return_hidden_states=True,
        tp_size=8,
        context_length=65536,
        disable_cuda_graph=True
    )
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 0,
    }

    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np

    outdir = f"{args.outdir}/{args.index}"
    os.makedirs(outdir, exist_ok=True)
    parquet_file = f"{outdir}/hidden_states.parquet"

    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32())),
        ("loss_mask", pa.list_(pa.int8())),
        ("hidden_state", pa.list_(pa.list_(pa.list_(pa.float16())))),
        ("target_hidden_states", pa.list_(pa.list_(pa.float16()))),
    ])

    writer = pq.ParquetWriter(parquet_file, schema)

    buffer = []
    chunk_size = 5_000 

    for idx, row in tqdm(enumerate(dataset), total=len(dataset)):
        # 推理得到 hidden_states / target_hidden_states
        outputs = llm.generate(
            input_ids=[row["input_ids"].tolist()],
            sampling_params=sampling_params,
            return_hidden_states=True,
        )
        hs_all = outputs[0]["meta_info"]["hidden_states"]   # (N+1, B, S, D)

        tgt_hs = torch.tensor(hs_all[0], dtype=torch.float16).cpu().numpy()  # (S, D)
        hs = torch.tensor(hs_all[1:], dtype=torch.float16).cpu().numpy()     # (N, S, D)

        buffer.append({
            "input_ids": row["input_ids"].tolist(),
            "loss_mask": row["loss_mask"].tolist(),
            "hidden_state": hs.tolist(),
            "target_hidden_states": tgt_hs.tolist(),
        })

        if len(buffer) >= chunk_size:
            table = pa.Table.from_pylist(buffer, schema=schema)
            writer.write_table(table)
            buffer.clear()

    if buffer:
        table = pa.Table.from_pylist(buffer, schema=schema)
        writer.write_table(table)

    writer.close()
    llm.shutdown()
    print(f"✅ Done! 数据已写入 {parquet_file}")



if __name__ == "__main__":
    main()
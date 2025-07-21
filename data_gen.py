import argparse
import os
import re

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="sglang data gen")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
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
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_index))
    MAX_TOKEN_LENGTH = 4096

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
            formatted_conversation,
            return_offsets_mapping=True,
            max_length=MAX_TOKEN_LENGTH,
            truncation=True,
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
    # 上面都是 BaldEagle 里面的。 下面是自己写的 sglang engine。
    # python3 data_gen.py --start 0 --end 2 --index 1 --gpu_index 0 --outdir /root/.cache/hidden_states_dump --model_name meta-llama/Llama-4-Scout-17B-16E-Instruct --dataset sharegpt
    # ------------------------ 3. Compute hidden states ------------------------
    import sglang as sgl

    llm = sgl.Engine(
        model_path=args.model_name,
        skip_tokenizer_init=True,
        enable_return_hidden_states=True,
        return_hidden_state_layers="1,23,44",
        tp_size=8,
        context_length=65536,
        disable_cuda_graph=True,
    )
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 0,
    }

    outdir = f"{args.outdir}/{args.index}"
    os.makedirs(outdir, exist_ok=True)

    buffer = []
    chunk_size = 128
    chunk_idx = 0

    batch_rows = []
    batch_input_ids = []
    for idx, row in tqdm(enumerate(dataset), total=len(dataset)):
        batch_rows.append(row)
        batch_input_ids.append(row["input_ids"].tolist())

        if len(batch_rows) < args.batch_size and idx + 1 < len(dataset):
            continue
        # 推理得到 hidden_states / target_hidden_states
        outputs = llm.generate(
            input_ids=batch_input_ids,
            sampling_params=sampling_params,
            return_hidden_states=True,
        )
        for row, output in zip(batch_rows, outputs):
            hs_all = output["meta_info"]["hidden_states"][0]
            # 每个元素是 4*5120 维度的向量
            # 前 5120 维度是 tgt_hs，后面 3*5120 维度分成 3 份是 hs
            hidden_dim = 5120
            tgt_hs_list = []
            hs_list = [[], [], []]  # 3 layers

            for token_hiddens in hs_all:
                token_hiddens = torch.tensor(token_hiddens, dtype=torch.bfloat16)

                tgt_hs_list.append(token_hiddens[:hidden_dim])

                remaining = token_hiddens[hidden_dim:]
                for i in range(3):
                    start_idx = i * hidden_dim
                    end_idx = (i + 1) * hidden_dim
                    hs_list[i].append(remaining[start_idx:end_idx])
            if len(tgt_hs) == 0:
                continue
            tgt_hs = torch.stack(tgt_hs_list).cpu()  # (S, D)
            hs = torch.stack(
                [torch.stack(layer) for layer in hs_list]
            ).cpu()  # (3, S, D)
            buffer.append(
                {
                    "input_ids": row["input_ids"],
                    "loss_mask": row["loss_mask"],
                    "hidden_state": hs,
                    "target_hidden_states": tgt_hs,
                }
            )

        batch_rows = []
        batch_input_ids = []
        if len(buffer) >= chunk_size:
            # 转为HF datasets支持的格式
            records = []
            for item in buffer:
                records.append(
                    {
                        "input_ids": item["input_ids"].tolist(),
                        "loss_mask": item["loss_mask"].tolist(),
                        "hidden_state": item["hidden_state"].float().numpy().tolist(),
                        "target_hidden_states": item["target_hidden_states"]
                        .float()
                        .numpy()
                        .tolist(),
                    }
                )
            ds = Dataset.from_dict({k: [rec[k] for rec in records] for k in records[0]})
            ds.save_to_disk(f"{outdir}/chunk_{chunk_idx}")
            buffer.clear()
            chunk_idx += 1

    if buffer:
        records = []
        for item in buffer:
            records.append(
                {
                    "input_ids": item["input_ids"].tolist(),
                    "loss_mask": item["loss_mask"].tolist(),
                    "hidden_state": item["hidden_state"].float().numpy().tolist(),
                    "target_hidden_states": item["target_hidden_states"]
                    .float()
                    .numpy()
                    .tolist(),
                }
            )
        ds = Dataset.from_dict({k: [rec[k] for rec in records] for k in records[0]})
        ds.save_to_disk(f"{outdir}/chunk_{chunk_idx}")

    llm.shutdown()
    print(f"✅ Done! 数据已写入 {outdir}")


if __name__ == "__main__":
    main()

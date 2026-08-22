import io
import json
import mimetypes
import os
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pybase64
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    ASSISTANT_SUFFIX,
    BaseDataset,
    DatasetRow,
)
from sglang.benchmark.datasets.image import create_mm_data_row
from sglang.benchmark.utils import encode_image_to_base64, remove_suffix


@dataclass
class CustomDataset(BaseDataset):
    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]
    context_len: Optional[int]
    prompt_suffix: str
    apply_chat_template: bool
    model_path: Optional[str] = None
    backend: str = "vllm-chat"

    @classmethod
    def from_args(cls, args: Namespace) -> "CustomDataset":
        assert not getattr(args, "tokenize_prompt", False)
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
            prompt_suffix=args.prompt_suffix,
            apply_chat_template=args.apply_chat_template,
            model_path=args.model,
            backend=args.backend,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_custom_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
            fixed_output_len=self.fixed_output_len,
            context_len=self.context_len,
            prompt_suffix=self.prompt_suffix,
            apply_chat_template=self.apply_chat_template,
            model_path=self.model_path,
            backend=self.backend,
        )


def sample_custom_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    prompt_suffix: Optional[str] = "",
    apply_chat_template=False,
    model_path: str = None,
    backend: str = "vllm-chat",
) -> List[DatasetRow]:
    """
    Sample requests from a custom JSONL dataset.
    Supports 'content'/'value' as conversation keys and optional 'image' for multimodal data.
    """
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        processor = tokenizer

    # Load the dataset
    dataset = []
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip lines with JSON errors

    # Filter out the conversations with less than 2 turns.
    processed_dataset = []
    for data in dataset:
        convs = data.get("conversations", data.get("conversation", []))
        if len(convs) >= 2:
            user_turn = convs[0].get("content", convs[0].get("value", ""))
            assist_turn = convs[1].get("content", convs[1].get("value", ""))
            processed_dataset.append((user_turn, assist_turn, data))

    dataset = processed_dataset
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[DatasetRow] = []

    dataset_abs_path = os.path.abspath(dataset_path)
    dataset_dir = os.path.dirname(dataset_abs_path)

    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt, completion, original_data = dataset[i]
        image_path = original_data.get("image", None)
        if image_path:
            if not os.path.isabs(image_path):
                image_path = os.path.join(dataset_dir, image_path)

            images_base64_list = []
            images_pil_list = []
            try:
                b64_str = encode_image_to_base64(image_path)
                mime_type, _ = mimetypes.guess_type(image_path)
                mime_type = mime_type or "image/jpeg"
                images_base64_list.append(f"data:{mime_type};base64,{b64_str}")

                raw_bytes = pybase64.b64decode(b64_str)
                img = Image.open(io.BytesIO(raw_bytes))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                images_pil_list.append(img)

                prompt = prompt.replace("<image 1>", "").strip()
                if prompt_suffix:
                    prompt = (
                        remove_suffix(prompt, ASSISTANT_SUFFIX)
                        + prompt_suffix
                        + ASSISTANT_SUFFIX
                    )
                cur_output_len = (
                    len(tokenizer.encode(completion))
                    if fixed_output_len is None
                    else fixed_output_len
                )

                data_row = create_mm_data_row(
                    prompt,
                    images_pil_list,
                    images_base64_list,
                    cur_output_len,
                    processor,
                    backend,
                )

                if data_row.prompt_len < 2 or data_row.output_len < 2:
                    continue

                if (
                    context_len
                    and data_row.prompt_len + data_row.output_len > context_len
                ):
                    continue

                filtered_dataset.append(data_row)
                continue

            except Exception as e:
                print(f"Error handling multimodal row: {e}")
                continue

        if prompt_suffix:
            prompt = (
                remove_suffix(prompt, ASSISTANT_SUFFIX)
                + prompt_suffix
                + ASSISTANT_SUFFIX
            )

        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                return_dict=False,
            )
            if tokenizer.bos_token:
                prompt = prompt.replace(tokenizer.bos_token, "")

        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )

        if prompt_len < 2 or output_len < 2:
            # Prune too short sequences.
            continue

        if context_len and prompt_len + output_len > context_len:
            # Prune too long sequences.
            continue

        filtered_dataset.append(
            DatasetRow(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
            )
        )

    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
    return filtered_dataset

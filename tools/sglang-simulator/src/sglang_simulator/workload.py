import random
from pathlib import Path

import numpy as np
from sglang_simulator.dataset import GenericRequest, SimpleDataset
from sglang_simulator.dataset.autobench import sample_autobench_requests
from transformers import AutoTokenizer

from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.datasets.sharegpt import sample_sharegpt_requests


def _to_simulator_dataset(
    rows: list[DatasetRow],
    *,
    use_timestamps: bool,
    timestamp_scale: float = 1000.0,
) -> SimpleDataset:
    return SimpleDataset(
        reqs=[
            GenericRequest(
                prompt=row.prompt if isinstance(row.prompt, str) else None,
                token_ids=row.prompt if isinstance(row.prompt, list) else None,
                input_length=row.prompt_len,
                output_length=row.output_len,
                custom_params=(
                    {"created_time": row.timestamp / timestamp_scale}
                    if use_timestamps and row.timestamp is not None
                    else {}
                ),
            )
            for row in rows
        ]
    )


def load_inprocess_workload(
    *,
    name: str,
    model_path: str,
    dataset_path: str | None,
    num_prompts: int,
    input_len: int,
    output_len: int,
    timestamp_scale: float,
    seed: int = 42,
) -> SimpleDataset:
    """Use SGLang's benchmark samplers and adapt their rows for SGLang Simulator."""
    if timestamp_scale <= 0:
        raise ValueError("timestamp_scale must be greater than zero")
    random.seed(seed)
    np.random.seed(seed)

    if name == "trace":
        if not dataset_path:
            raise ValueError("--dataset is required for trace")
        rows = sample_autobench_requests(
            dataset_path=str(Path(dataset_path)),
            num_requests=num_prompts,
            tokenizer=None,
        )
        if not rows:
            raise ValueError(f"empty trace: {dataset_path}")
        if any(row.timestamp is None for row in rows):
            raise ValueError("trace rows must contain timestamp")
        rows.sort(key=lambda row: row.timestamp)
        trace_start = rows[0].timestamp
        for row in rows:
            row.timestamp -= trace_start
        return _to_simulator_dataset(
            rows,
            use_timestamps=True,
            timestamp_scale=timestamp_scale,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if name == "sharegpt":
        if not dataset_path:
            raise ValueError("--dataset is required for sharegpt")
        rows = sample_sharegpt_requests(
            dataset_path=str(Path(dataset_path)),
            num_requests=num_prompts,
            tokenizer=tokenizer,
        )
    elif name == "random":
        rows = sample_random_requests(
            input_len=input_len,
            output_len=output_len,
            num_prompts=num_prompts,
            range_ratio=1.0,
            tokenizer=tokenizer,
            dataset_path="",
            random_sample=False,
            return_text=False,
        )
    else:
        raise ValueError(f"unknown workload: {name}")

    return _to_simulator_dataset(rows, use_timestamps=False)

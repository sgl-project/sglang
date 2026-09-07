from sglang_simulator.dataset.base_dataset import (
    BaseDataset,
    GenericRequest,
    SimpleDataset,
)
from sglang_simulator.dataset.dataset_args import DatasetArgs
from sglang_simulator.dataset.random import RandomDataset, RandomIDsDataset
from transformers import PreTrainedTokenizer

dataset_registry: dict[str, BaseDataset] = {
    "random": RandomDataset,
    "random_ids": RandomIDsDataset,
}


def get_dataset(
    dataset_args: DatasetArgs, tokenizer: PreTrainedTokenizer | None = None
) -> BaseDataset:
    if dataset_args.name not in dataset_registry:
        raise ValueError(f"unknown dataset name: {dataset_args.name}")

    dataset: BaseDataset = dataset_registry[dataset_args.name](
        args=dataset_args, tokenizer=tokenizer
    )

    return dataset


__all__ = (
    "DatasetArgs",
    "BaseDataset",
    "SimpleDataset",
    "GenericRequest",
    "get_dataset",
)

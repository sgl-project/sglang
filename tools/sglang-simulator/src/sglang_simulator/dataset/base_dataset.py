from dataclasses import dataclass, field
from typing import Optional, overload

from sglang_simulator.dataset.dataset_args import DatasetArgs
from transformers import PreTrainedTokenizerBase


@dataclass
class GenericRequest:
    prompt: Optional[str] = None
    token_ids: Optional[list[int]] = None
    input_length: int = -1
    output_length: int = -1
    custom_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.prompt is None and self.token_ids is None:
            raise ValueError("Invalid Request")


class BaseDataset:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, args: DatasetArgs):
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.args = args
        self._name = ""

    @overload
    def __getitem__(self, index: int) -> GenericRequest: ...

    @overload
    def __getitem__(self, index: slice) -> list[GenericRequest]: ...

    def __getitem__(self, index):
        """Get item(s) by index or slice. Delegates to _get_single_item for single items."""
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        if index >= len(self):
            raise IndexError
        return self._get_single_item(index)

    def _get_single_item(self, index: int) -> GenericRequest:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return self.__class__.__name__


class SimpleDataset(BaseDataset):
    def __init__(
        self, tokenizer=None, args=None, reqs: list[GenericRequest] | None = None
    ):
        super().__init__(tokenizer, args)
        self.data: list[GenericRequest] = []
        if reqs is not None:
            self.data.extend(reqs)

    def add_request(self, req: GenericRequest):
        self.data.append(req)

    def _get_single_item(self, index: int) -> GenericRequest:
        return self.data[index]

    def __len__(self):
        return len(self.data)

from typing import MutableMapping, TypeVar, Union

from sglang.srt.model_executor.forward_batch_info import ForwardMode

DSACudaGraphMetadataKey = Union[int, tuple[int, ForwardMode]]
MetadataT = TypeVar("MetadataT")


def get_dsa_cuda_graph_metadata_key(
    batch_size: int, forward_mode: ForwardMode
) -> DSACudaGraphMetadataKey:
    """Keep draft-extend graph metadata separate from decode metadata."""
    if forward_mode.is_draft_extend_v2():
        return (batch_size, ForwardMode.DRAFT_EXTEND_V2)
    return batch_size


def store_dsa_cuda_graph_metadata(
    cache: MutableMapping[DSACudaGraphMetadataKey, MetadataT],
    batch_size: int,
    forward_mode: ForwardMode,
    metadata: MetadataT,
) -> DSACudaGraphMetadataKey:
    key = get_dsa_cuda_graph_metadata_key(batch_size, forward_mode)
    cache[key] = metadata
    return key


def load_dsa_cuda_graph_metadata(
    cache: MutableMapping[DSACudaGraphMetadataKey, MetadataT],
    batch_size: int,
    forward_mode: ForwardMode,
) -> MetadataT:
    return cache[get_dsa_cuda_graph_metadata_key(batch_size, forward_mode)]

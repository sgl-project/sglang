"""
Multi-modality utils
"""

import copy
import hashlib
import pickle
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.multimodal import gpu_tensor_hash
from sglang.srt.managers.schedule_batch import (
    CudaIpcTensorTransportProxy,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import flatten_nested_list, is_npu, print_warning_once
from sglang.utils import logger

_is_npu = is_npu()

# NOTE: Using the shared logger from sglang.utils instead of creating a module-specific logger
# to ensure consistent logging behavior across the codebase. This prevents issues with log
# propagation that can cause some log messages (like 'server is fired up') to not appear
# in the console when multimodal support is enabled.

# TODO(mick): nccl
# cuda_ipc: for intranode tensor sharing
TensorTransportMode = Literal["cuda_ipc", "auto", "default"]


_GPU_FEATURE_BUFFER: Optional[torch.Tensor] = None
_BUFFER_OFFSET = 0

_EXTRA_PRE_TOKENS = 0  # pre chunk extra token (0 for the moment)
_EXTRA_POST_TOKENS = 0  # post chunk extra token (0 for the moment)


def init_feature_buffer(device):
    global _GPU_FEATURE_BUFFER, _BUFFER_OFFSET
    if (
        device == "cpu"
        or envs.SGLANG_MM_BUFFER_SIZE_MB.get() == 0
        or _GPU_FEATURE_BUFFER is not None
    ):
        return
    try:
        size_mb = envs.SGLANG_MM_BUFFER_SIZE_MB.get()
        num_elements = int(size_mb * 1024 * 1024 / 4)
        _GPU_FEATURE_BUFFER = torch.empty(
            num_elements, dtype=torch.float32, device=device
        )
        logger.info(f"Preallocated {size_mb}MB GPU buffer")
    except RuntimeError as e:
        _GPU_FEATURE_BUFFER = None


def reset_buffer_offset():
    global _BUFFER_OFFSET
    _BUFFER_OFFSET = 0


def is_feature_buffer_initialized():
    global _GPU_FEATURE_BUFFER
    if _GPU_FEATURE_BUFFER is None:
        return False
    return True


def try_add_to_buffer(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    global _BUFFER_OFFSET

    if _GPU_FEATURE_BUFFER is None:
        return tensor

    tensor_size = tensor.numel()

    if _BUFFER_OFFSET + tensor_size <= _GPU_FEATURE_BUFFER.numel():
        buffer_view = _GPU_FEATURE_BUFFER[_BUFFER_OFFSET : _BUFFER_OFFSET + tensor_size]
        buffer_view.copy_(tensor.flatten(), non_blocking=True)
        result = buffer_view.view(tensor.shape)
        _BUFFER_OFFSET += tensor_size
        return result
    else:
        return tensor


class TransportProxyTensor(torch.Tensor):
    """
    A convenient torch.Tensor subclass that carries extra metadata and supports
    efficient inter-process communications
    """

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        name: Optional[str] = None,
        fields: Optional[Dict[str, Any]] = None,
        transport_mode: TensorTransportMode = "default",
        *args,
        **kwargs,
    ):

        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"Input 'data' must be a torch.Tensor, but got {type(data)}"
            )

        instance = data.as_subclass(cls)

        instance._metadata = {
            "name": name,
            "fields": fields if fields is not None else {},
            "transport_mode": transport_mode,
        }

        return instance

    def __getstate__(self):
        """
        Called during pickling. Implements the serialization logic.
        """
        # acquire all serialize metadata from _metadata
        state = {
            "metadata": self._metadata,
            "tensor_data": None,
            "ipc_extra": None,
        }
        transport_mode = self._metadata.get("transport_mode", "default")

        if transport_mode == "cuda_ipc" and self.is_cuda:
            try:
                storage = self.untyped_storage()
                handle = storage._share_cuda_()

                state["ipc_extra"] = {
                    "handle": handle,
                    "shape": self.shape,
                    "dtype": self.dtype,
                    "stride": self.stride(),
                    "device_index": self.device.index,
                    "storage_offset": self.storage_offset(),
                }
                state["tensor_data"] = None
            except Exception as e:
                # Failed to get CUDA IPC handle (possibly tp). Falling back to default transport.
                state["metadata"]["transport_mode"] = "default"
                state["tensor_data"] = self.as_subclass(torch.Tensor)
        else:
            state["metadata"]["transport_mode"] = "default"
            state["tensor_data"] = self.as_subclass(torch.Tensor)

        return state

    def __setstate__(self, state: Dict[str, Any]):
        """
        Called during unpickling. Implements the deserialization logic.
        """
        self._metadata = state["metadata"]

        transport_mode = self._metadata.get("transport_mode", "default")

        if transport_mode == "cuda_ipc" and state["ipc_extra"] is not None:
            ipc_extra = state["ipc_extra"]
            handle, shape, dtype, stride, source_device_index, s_offset = (
                ipc_extra["handle"],
                ipc_extra["shape"],
                ipc_extra["dtype"],
                ipc_extra["stride"],
                ipc_extra["device_index"],
                ipc_extra["storage_offset"],
            )

            try:
                target_device = torch.device(f"cuda:{source_device_index}")
                with torch.cuda.device(target_device):
                    storage = torch.UntypedStorage._new_shared_cuda(*handle)
                    reconstructed_tensor = torch.empty(
                        0, dtype=dtype, device=target_device
                    ).set_(storage, storage_offset=s_offset, size=shape, stride=stride)
                    self.set_(reconstructed_tensor)
            except Exception as e:
                print(f"Error: Failed to deserialize from CUDA IPC handle ({e}).")
                raise e

        elif state["tensor_data"] is not None:
            self.set_(state["tensor_data"])
        else:
            raise pickle.UnpicklingError(
                "Invalid state for TransportProxyTensor: no tensor data found."
            )

    @property
    def name(self) -> Optional[str]:
        return self._metadata.get("name")

    @property
    def fields(self) -> Dict[str, Any]:
        return self._metadata.get("fields", {})

    @property
    def transport_mode(self) -> TensorTransportMode:
        return self._metadata.get("transport_mode", "default")


class MultiModalityDataPaddingPattern:
    """
    Data tokens (like image tokens) often need special handling during padding
    to maintain model compatibility. This class provides the interface for
    implementing different padding strategies for data tokens
    """

    @abstractmethod
    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """
        Pad the input ids sequence containing data tokens, and replace them with pad_values
        """
        pass


class MultiModalityDataPaddingPatternTokenPairs(MultiModalityDataPaddingPattern):
    """In this pattern, data tokens should be enclosed by special token pairs (e.g. <image>...</image>, data_token_pairs)

    The padded value in a region enclosed by a token pair with be the same one, as the MultimodalDataItem's pad value

    This strategy should be applied when data content is marked by start/end token pairs in the input sequence.
    """

    def __init__(
        self,
        data_token_pairs: Optional[List[Tuple[int, int]]],
        data_start_token_ids: Optional[List[int]] = None,
    ) -> None:
        """

        Args:
            data_start_token_ids marks the start of a single multimodal data
            See Minicpmo's slice_start_id for example
        """
        self.data_token_id_pairs = data_token_pairs
        self.data_start_token_ids = data_start_token_ids or [
            s for s, _e in data_token_pairs
        ]

    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """
        This function will replace the data-tokens in between with pad_values accordingly
        """
        pad_values = [item.pad_value for item in mm_inputs.mm_items]
        data_token_pairs = self.data_token_id_pairs
        mm_inputs.data_offsets = []
        if data_token_pairs is None:
            data_token_pairs = [mm_inputs.im_start_id, mm_inputs.im_end_id]
        if data_token_pairs is None:
            print_warning_once(
                "No data_token_pairs provided, RadixAttention might be influenced."
            )
            return input_ids
        start_token_ids = {s for s, _e in data_token_pairs}
        end_tokens_ids = {e for _s, e in data_token_pairs}

        padded_ids = []
        last_idx = 0
        data_idx = -1

        start_indices = [i for i, x in enumerate(input_ids) if x in start_token_ids]
        end_indices = [i for i, x in enumerate(input_ids) if x in end_tokens_ids]

        if len(start_indices) != len(end_indices):
            return input_ids

        for start_idx, end_idx in zip(start_indices, end_indices):
            padded_ids.extend(input_ids[last_idx : start_idx + 1])

            if input_ids[start_idx] in self.data_start_token_ids:
                data_idx += 1
                mm_inputs.data_offsets += [start_idx]

            if data_idx >= len(pad_values):
                data_idx = len(pad_values) - 1

            num_tokens = end_idx - start_idx - 1
            pad_value = pad_values[data_idx]
            padded_ids.extend([pad_value] * num_tokens)

            last_idx = end_idx

        padded_ids.extend(input_ids[last_idx:])

        assert len(input_ids) == len(padded_ids), "Length validation fails"
        return padded_ids


class MultiModalityDataPaddingPatternMultimodalTokens(MultiModalityDataPaddingPattern):
    """In this pattern, data tokens should be represented as repetitions of a single token
    e.g. <image><image>....<image>, or <audio><audio>...<audio>
    """

    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """
        Replaces multimodal tokens in input_ids with corresponding pad_values from mm_items.
        Each modality (image, audio, video) is handled separately based on its token_id.
        """
        if not input_ids or not mm_inputs.mm_items:
            return input_ids

        input_ids_tensor = torch.as_tensor(input_ids)

        # Check if MM splitting is enabled
        if envs.SGLANG_ENABLE_MM_SPLITTING.get():
            items_by_modality = defaultdict(list)
            for item in mm_inputs.mm_items:
                items_by_modality[item.modality].append(item)

            token_id_map = {
                Modality.IMAGE: mm_inputs.im_token_id,
                Modality.MULTI_IMAGES: mm_inputs.im_token_id,
                Modality.AUDIO: mm_inputs.audio_token_id,
                Modality.VIDEO: mm_inputs.video_token_id,
            }

            for modality, items in items_by_modality.items():
                token_id = token_id_map.get(modality)

                if not items or token_id is None:
                    continue

                for i, item in enumerate(items):
                    for offset in items[i].offsets:
                        input_ids_tensor[offset[0] : offset[1] + 1] = item.pad_value
        else:
            # Create mapping of token_ids to pad_values for each modality
            token_to_pad_mapping = {}
            for item in mm_inputs.mm_items:
                if item.is_image() and mm_inputs.im_token_id is not None:
                    token_to_pad_mapping[mm_inputs.im_token_id] = item.pad_value
                elif item.is_audio() and mm_inputs.audio_token_id is not None:
                    token_to_pad_mapping[mm_inputs.audio_token_id] = item.pad_value
                elif item.is_video() and mm_inputs.video_token_id is not None:
                    token_to_pad_mapping[mm_inputs.video_token_id] = item.pad_value
                else:
                    raise ValueError(
                        f"No multimodal token id provided for {item.modality}"
                    )

            # Apply replacements for all tokens at once
            for token_id, pad_value in token_to_pad_mapping.items():
                input_ids_tensor[input_ids_tensor == token_id] = pad_value

        ret_input_ids = input_ids_tensor.tolist()
        return ret_input_ids


embedding_cache: Optional[MultiModalStaticCache] = None


def init_mm_embedding_cache(max_size: int = 0):
    global embedding_cache
    embedding_cache = MultiModalStaticCache(max_size)


def get_embedding_chunk(
    embedding: torch.Tensor,
    extend_prefix_len: int,
    extend_seq_len: int,
    items_offset: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, int, int]:
    """
    Extract a chunk of embeddings based on the specified prefix length, sequence length, and offset ranges.

    Args:
        embedding: The full embedding tensor to extract a chunk from
        extend_prefix_len: The starting position (prefix length) for extraction
        extend_seq_len: The number of tokens to extract
        items_offset: List of [start, end] offset ranges for multimodal items in the input sequence

    Returns:
        A tuple containing:
        - The extracted embedding chunk as a tensor
        - The start index used for extraction
        - The end index used for extraction

    Note:
        If there's no overlap between the requested range and the offset ranges,
        an empty tensor is returned with zeros for start and end indices.
    """
    start_index, end_index = 0, 0
    extend_start_index = extend_prefix_len
    extend_end_index = extend_prefix_len + extend_seq_len - 1

    for start, end in items_offset:
        if extend_start_index >= start and extend_start_index <= end:
            start_index += extend_start_index - start
        elif extend_start_index > end:
            start_index += end - start + 1

        if extend_end_index >= start and extend_end_index <= end:
            end_index += extend_end_index - start + 1
        elif extend_end_index > end:
            end_index += end - start + 1
    # some models' embedding is 3-dim, reshape it to 2-dim
    embedding = embedding.reshape(-1, embedding.shape[-1])
    embedding_chunk = embedding[start_index:end_index]
    return embedding_chunk, start_index, end_index


def _get_precomputed_embedding(
    items: List[MultimodalDataItem],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Optional[torch.Tensor]:
    """
    If all items have precomputed_embeddings, return their concatenation.
    If some but not all have precomputed_embeddings, raise NotImplementedError.
    If none have precomputed_embeddings, return None.
    """
    precomputed_embeddings = []
    for idx, item in enumerate(items):
        if item.precomputed_embeddings is None:
            precomputed_embeddings.append(None)
            continue
        seq_start_idx = prefix_length[idx]
        seq_end_idx = seq_start_idx + extend_length[idx] - 1
        prefix_embedding_length = []
        extend_embedding_length = []
        for mm_start_idx, mm_end_idx in items_offset_list[idx]:
            if mm_start_idx > seq_end_idx:
                break
            if seq_start_idx > mm_start_idx:
                prefix_embedding_length.append(
                    min(seq_start_idx - mm_start_idx, mm_end_idx - mm_start_idx + 1)
                )
            if mm_end_idx >= seq_start_idx:
                extend_embedding_length.append(
                    min(
                        mm_end_idx - seq_start_idx + 1,
                        seq_end_idx - mm_start_idx + 1,
                        mm_end_idx - mm_start_idx + 1,
                        seq_end_idx - seq_start_idx + 1,
                    )
                )
        prefix_embedding_length = int(np.sum(prefix_embedding_length))
        extend_embedding_length = int(np.sum(extend_embedding_length))
        precomputed_embeddings.append(
            item.precomputed_embeddings[
                prefix_embedding_length : prefix_embedding_length
                + extend_embedding_length
            ]
        )

    if any(feature is not None for feature in precomputed_embeddings):
        if not all(feature is not None for feature in precomputed_embeddings):
            raise NotImplementedError(
                "MM inputs where only some items are precomputed."
            )
        result = torch.concat(precomputed_embeddings)
        # some models embedding is 3-dim, reshape it to 2-dim (similar to get_embedding_chunk)
        result = result.reshape(-1, result.shape[-1])
        return result
    return None


def get_embedding_items_per_chunk_with_extra_padding(
    embedding_items_per_req: List["MultimodalDataItem"],
    extend_prefix_len: int,
    extend_seq_len: int,
    items_offset: List[Tuple[int, int]],
) -> List["MultimodalDataItem"]:
    """
    From all multimodal items of a request, select the subset that is "relevant to
    this prefill chunk", and allow a small amount of extra padding on both sides
    of the chunk boundary (for easier caching or cross-chunk reuse).

    Assumptions:
        - len(embedding_items_per_req) == len(items_offset)
        - items_offset[j] = (start, end), meaning the multimodal tokens of the j-th
        item correspond to [start, end) (left-closed, right-open) in the entire
        token sequence
        - The item order in embedding_items_per_req is one-to-one aligned with
        items_offset

    Args:
        embedding_items_per_req: all items of this modality under the current
            request (e.g. each frame in a 500-frame video)
        extend_prefix_len: number of tokens already prefilled before the current
            chunk
        extend_seq_len: number of tokens in the current chunk
        items_offset: (start, end) position of each item in the whole sentence

    Returns:
        The subset of items to feed into ViT for this chunk (preserving the
        original order)
    """
    assert len(embedding_items_per_req) == len(
        items_offset
    ), f"items_per_req({len(embedding_items_per_req)}) vs items_offset({len(items_offset)}) mismatch"

    if extend_seq_len <= 0:
        return []

    # Current chunk's token range
    chunk_start = extend_prefix_len
    chunk_end = extend_prefix_len + extend_seq_len

    # Current chunk's token range with extra padding
    window_start = max(0, chunk_start - _EXTRA_PRE_TOKENS)
    window_end = chunk_end + _EXTRA_POST_TOKENS

    selected_items: List["MultimodalDataItem"] = []

    for item, (start, end) in zip(embedding_items_per_req, items_offset):
        if start >= end:
            continue

        # Check whether this item has overlap with [window_start, window_end)
        # If has overlap, add the item into selected_item.
        if end > window_start and start < window_end:
            selected_items.append(item)

    return selected_items


# TODO: To be obsoleted.
def _get_chunked_prefill_embedding(
    data_embedding_func: Callable[[List[MultimodalDataItem]], torch.Tensor],
    embedding_items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Optional[torch.Tensor]:
    # Calculate embedding for each request, try to get it from cache to avoid repeated calculation
    embedding_list = []
    # FIXME(Xinyuan): temporary workaround for eagle3, which may have len(items_size) > len(prefix_length)
    max_iterations = min(len(items_size) - 1, len(prefix_length))
    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None, items_offset
        # if all items has been prefixed, we do not need to calculate embedding
        if all([offset_end < prefix_length[i] for _, offset_end in items_offset]):
            continue
        item_hashes = [item.hash for item in embedding_items_per_req]
        embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)
        embedding_per_req = embedding_cache.get(item_hashes)
        if embedding_per_req is None:
            embedding_per_req = data_embedding_func(embedding_items_per_req)
            if not embedding_cache.set(embedding_items_hash, embedding_per_req):
                print_warning_once(
                    "Multimodal embedding cache is full. This typically occurs when a single "
                    "embedding exceeds the cache size limit. Consider increasing the "
                    "`SGLANG_VLM_CACHE_SIZE_MB` environment variable or reducing the input "
                    "embedding size."
                )

        embedding_per_req_chunk, _, _ = get_embedding_chunk(
            embedding=embedding_per_req,
            extend_prefix_len=prefix_length[i],
            extend_seq_len=extend_length[i] if i < len(extend_length) else 0,
            items_offset=items_offset,
        )
        embedding_list.append(embedding_per_req_chunk)
    if len(embedding_list) == 0:
        return None
    return torch.concat(embedding_list, dim=0)


def get_embedding_chunk_remove_extra_padding(
    embedding: torch.Tensor,
    extend_prefix_len: int,
    extend_seq_len: int,
    items_offset: List[Tuple[int, int]],
) -> Tuple[Optional[torch.Tensor], int, int]:
    """
    From the embedding computed on "items related to this chunk + extra padding",
    trim out the token embeddings that are not needed for the current chunk, and
    keep only those mm tokens covered by
    [extend_prefix_len, extend_prefix_len + extend_seq_len).

    Assumptions:
        - Each (start, end) in items_offset represents an item's multimodal token
        interval [start, end) in the whole token sequence, and their order is
        consistent with the order of items in `embedding`.
        - The layout of `embedding`: each selected item is concatenated in order,
        and item j occupies seg_len_j = end_j - start_j rows.

    Args:
        embedding: output of data_embedding_func(embedding_items_per_chunk),
                shape = (T_total, D)
        extend_prefix_len: number of tokens before the chunk (prefix_len)
        extend_seq_len: number of tokens in this chunk (chunk_len)
        items_offset: list of (start, end) for all items of the current request

    Returns:
        - trimmed_embedding: embedding that contains only the mm tokens needed
        by this chunk, concatenated in token order
        - num_tokens_before: number of mm tokens "before the chunk" that are
        trimmed off (optional info, not used by the current caller)
        - num_tokens_after: number of mm tokens "after the chunk" that are
        trimmed off (optional info, not used by the current caller)
    """
    if embedding is None or embedding.numel() == 0:
        return None, 0, 0

    chunk_start = extend_prefix_len
    chunk_end = extend_prefix_len + extend_seq_len

    if extend_seq_len <= 0 or chunk_start >= chunk_end:
        return None, 0, 0

    # The window with extra padding
    window_start = max(0, chunk_start - _EXTRA_PRE_TOKENS)
    window_end = chunk_end + _EXTRA_POST_TOKENS

    # Iterate item_offset to choose item.
    # We need to forward an embedding_idx to locate the item start-end position in embedding.
    embedding_idx = 0
    kept_slices: List[torch.Tensor] = []

    num_tokens_before = 0
    num_tokens_after = 0

    for start, end in items_offset:
        if start >= end:
            continue

        seg_len = end - start

        # Check whether this item has been chosen into embedding_items_per_chunk or not.
        selected = end > window_start and start < window_end

        if not selected:
            # Not in embedding_items_per_chunk, not forward embedding_idx.
            continue

        # embedding has the whole item
        # embedding[embedding_idx : embedding_idx + seg_len]

        # Calculate the overlap range between item and the current chunk
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end)

        if overlap_start < overlap_end:
            # The item has a portion mm tokens in the current chunk
            # The offset inside item
            local_start = overlap_start - start
            local_end = overlap_end - start

            # The embedding index
            slice_start = embedding_idx + local_start
            slice_end = embedding_idx + local_end

            kept_slices.append(embedding[slice_start:slice_end])

            # Stats the token number before and after this chunk
            num_tokens_before += max(0, local_start)
            num_tokens_after += max(0, seg_len - local_end)
        else:
            # Although item is chosen into embedding_items_per_chunk as extra padding,
            # Its mm tokens has no overlap with chunk, so don't count into the current
            # chunk's embedding.
            if end <= chunk_start:
                num_tokens_before += seg_len
            elif start >= chunk_end:
                num_tokens_after += seg_len

        # No matter whether this item has overlap with chunk, once it's selected, it
        # counts seg_len in embedding, so embedding_idx has to forward.
        embedding_idx += seg_len

    if not kept_slices:
        # No mm tokens in this chunk
        return None, num_tokens_before, num_tokens_after

    trimmed_embedding = torch.cat(kept_slices, dim=0)
    return trimmed_embedding, num_tokens_before, num_tokens_after


# This function is for chunked prefill vit for multiple items in the next feature.
def _get_chunked_prefill_embedding_for_chunked_items(
    data_embedding_func: Callable[[List["MultimodalDataItem"]], torch.Tensor],
    embedding_items: List["MultimodalDataItem"],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Optional[torch.Tensor]:
    """
    Multi-modal embedding computation for chunked prefill.

    For each request:
    1. Use items_size to split embedding_items into per-request sublists embedding_items_per_req;
    2. Use get_embedding_items_per_chunk_with_extra_padding to select the subset of items related to this chunk;
    3. Call data_embedding_func (ViT) on this subset to obtain embedding_per_chunk;
    4. Concatenate embedding_per_req_chunk for all requests in order.

    In this way, the ViT for each request only processes the frames / images related to the current chunk,
    avoiding OOM caused by processing all the frames at once.
    """
    # Calculate embedding for each request, try to get it from cache to avoid repeated calculation
    embedding_list = []
    # FIXME(Xinyuan): temporary workaround for eagle3, which may have len(items_size) > len(prefix_length)
    max_iterations = min(len(items_size) - 1, len(prefix_length))

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None, items_offset

        # if all items has been prefixed, we do not need to calculate embedding
        if all([offset_end < prefix_length[i] for _, offset_end in items_offset]):
            continue

        # 1) Pick up items related with this chunk
        embedding_items_per_chunk = get_embedding_items_per_chunk_with_extra_padding(
            embedding_items_per_req,
            extend_prefix_len=prefix_length[i],
            extend_seq_len=extend_length[i] if i < len(extend_length) else 0,
            items_offset=items_offset,
        )

        if not embedding_items_per_chunk:
            continue

        # 2) construct cache key
        # embedding_items_hash = MultiModalStaticCache.combine_hashes(
        #     embedding_items_per_chunk
        # )
        item_hashes = [item.hash for item in embedding_items_per_chunk]
        embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)

        embedding_per_chunk = embedding_cache.get(embedding_items_hash)
        if embedding_per_chunk is None:
            # ViT forward for items related with per chunk
            embedding_per_chunk = data_embedding_func(embedding_items_per_chunk)

            embedding_for_cache = embedding_per_chunk.detach().cpu()
            if not embedding_cache.set(embedding_items_hash, embedding_for_cache):
                print(
                    "[WARN] Multimodal embedding cache is full. "
                    "Consider increasing `SGLANG_VLM_CACHE_SIZE_MB` or reducing "
                    "video frame count / resolution for a single request."
                )
        else:
            target_device = embedding_items_per_req[0].feature.device
            if embedding_per_chunk.device != target_device:
                embedding_per_chunk = embedding_per_chunk.to(target_device)

        # 3) remove extra padding from embedding_per_chunk, only keep current chunk part
        #    We probably don't need this part.
        # embedding_per_req_chunk, _, _ = get_embedding_chunk_remove_extra_padding(
        #     embedding=embedding_per_chunk,
        #     extend_prefix_len=prefix_len,
        #     extend_seq_len=chunk_len,
        #     items_offset=items_offset,
        # )

        if embedding_per_chunk is not None and embedding_per_chunk.numel() > 0:
            embedding_list.append(embedding_per_chunk)

    if not embedding_list:
        return None

    # concat all the request's chunk embedding in token
    return torch.cat(embedding_list, dim=0)


def _get_multimodal_mask(
    input_ids: torch.Tensor, placeholder_tensor: torch.Tensor
) -> torch.Tensor:
    return torch.isin(input_ids, placeholder_tensor).unsqueeze(-1)


def _adjust_embedding_length(
    embedding: torch.Tensor,
    mask: torch.Tensor,
    logger,
) -> torch.Tensor:
    num_mm_tokens_in_embedding = embedding.shape[0]
    num_mm_tokens_in_input_ids = mask.sum().item()
    if num_mm_tokens_in_input_ids != num_mm_tokens_in_embedding:
        logger.warning(
            f"Number of tokens in multimodal embedding does not match those in the input text. "
            f"Got {num_mm_tokens_in_input_ids} tokens in the text but {num_mm_tokens_in_embedding} "
            f"tokens from multimodal embeddings."
        )
        if num_mm_tokens_in_input_ids < num_mm_tokens_in_embedding:
            chunked_prefill_size = get_global_server_args().chunked_prefill_size
            if chunked_prefill_size != -1:
                logger.warning(
                    "You may want to avoid this issue by raising `chunked_prefill_size`, or disabling chunked prefill"
                )
            # extract from the end: this is a compromise
            if embedding.dim() == 2:
                embedding = embedding[-num_mm_tokens_in_input_ids:, :]
            else:
                num_multimodal = num_mm_tokens_in_input_ids // embedding.shape[0]
                embedding = embedding[-num_multimodal:, :]
        else:
            raise RuntimeError(
                f"Insufficient multimodal embedding length: {num_mm_tokens_in_input_ids=} vs {num_mm_tokens_in_embedding=}. This is an internal error"
            )
    return embedding


def get_embedding_and_mask(
    data_embedding_func: Callable[[List[MultimodalDataItem]], torch.Tensor],
    embedding_items: List[MultimodalDataItem],
    placeholder_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate multimodal embeddings and create a mask for identifying their positions in the input sequence.

    Args:
        data_embedding_func: Function that generates embeddings for multimodal items
        embedding_items: List of multimodal items to embed
        placeholder_tensor: Tensor containing token IDs that serve as placeholders for multimodal content
        input_ids: The input token IDs tensor
        items_size: Cumulative sizes of multimodal items per request
        prefix_length: Prefix lengths for each request
        extend_length: Sequence lengths for each request
        items_offset_list: List of offset ranges for multimodal items in each request

    Returns:
        A tuple containing:
        - The generated embeddings tensor
        - A boolean mask tensor indicating where these embeddings should be placed
    """
    # 1. Get embedding
    embedding = _get_precomputed_embedding(
        embedding_items, prefix_length, extend_length, items_offset_list
    )
    if embedding is None:
        embedding = _get_chunked_prefill_embedding(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
        )
        if embedding is None:
            return None, None
    # 2. Get mask
    if _is_npu:
        torch.npu.current_stream().synchronize()
    special_multimodal_mask = _get_multimodal_mask(input_ids, placeholder_tensor)
    # 3. Adjust embedding length if needed
    embedding = _adjust_embedding_length(embedding, special_multimodal_mask, logger)
    return embedding, special_multimodal_mask


def embed_mm_inputs(
    mm_inputs_list: List[MultimodalInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    input_embedding: nn.Embedding,
    multimodal_model: nn.Module = None,
    data_embedding_func_mapping: Dict[
        Modality, Callable[[List[MultimodalDataItem]], torch.Tensor]
    ] = None,
    placeholder_tokens: dict[Modality, List[int]] = None,
    use_deepstack: Dict[Modality, bool] = {},
) -> Optional[torch.Tensor]:
    """
    Embed multimodal inputs and integrate them with text token embeddings.

    Args:
        mm_inputs_list: List of multimodal inputs to process
        extend_prefix_lens: Prefix lengths for each request
        extend_seq_lens: Sequence lengths for each request
        input_ids: Input token IDs tensor
        input_embedding: Embedding layer for text tokens
        placeholder_tokens: Token IDs for multimodal placeholders (uses pad_values if None)

    Returns:
        Combined embedding tensor with multimodal content integrated
    """
    other_info = {}
    if mm_inputs_list is None:
        return None

    # 1. Calculate the multimodal data which exists in input_ids, with the help of pad_values
    # we assume that multimodal data are represented with its pad_values in input_ids
    item_flatten_list = []
    for mm_inputs in mm_inputs_list:
        item_flatten_list += [item for item in mm_inputs.mm_items if item is not None]

    # deepstack_embeddings: per-modality
    modalities, embeddings, masks, deepstack_embeddings = [], [], [], []

    # 2. Get multimodal embedding separately
    # Try get mm embedding if any
    for modality in Modality.all():
        items = [
            item for item in item_flatten_list if item.is_modality(modality=modality)
        ]
        embedder = (
            None
            if data_embedding_func_mapping is None
            else data_embedding_func_mapping.get(modality, None)
        )
        if embedder is None:
            # "image", "video", etc
            modality_id = modality.name.lower()
            embedder = getattr(multimodal_model, f"get_{modality_id}_feature", None)
        if len(items) != 0:
            assert embedder is not None, f"no embedding method found for {modality}"
            placeholder_tensor = torch.as_tensor(
                [item.pad_value for item in items],
                device=input_ids.device,
            )
            # calculate per request items length offset
            items_size = torch.zeros(len(mm_inputs_list) + 1, dtype=int)
            items_offsets = []
            for i, mm_inputs in enumerate(mm_inputs_list):
                mm_items = [
                    item
                    for item in mm_inputs.mm_items
                    if item.is_modality(modality=modality)
                ]
                items_size[i + 1] = len(mm_items)
                items_offsets.append(
                    flatten_nested_list([item.offsets for item in mm_items])
                )
            items_size = torch.cumsum(items_size, dim=0).tolist()

            embedding, mask = get_embedding_and_mask(
                data_embedding_func=embedder,
                embedding_items=items,
                placeholder_tensor=placeholder_tensor,
                input_ids=input_ids,
                items_size=items_size,
                prefix_length=extend_prefix_lens,
                extend_length=extend_seq_lens,
                items_offset_list=items_offsets,
            )

            if use_deepstack.get(modality, None) and embedding is not None:
                embedding, deepstack_embedding = (
                    multimodal_model.separate_deepstack_embeds(embedding)
                )
                deepstack_embeddings += [deepstack_embedding]
            modalities += [modality]
            embeddings += [embedding]
            masks += [mask]

    # 3. Get input embeddings
    vocab_size = input_embedding.num_embeddings
    # Important: clamp after getting original multimodal regions
    # Clamp input ids. This is because the input_ids for the multimodal tokens are
    # filled with the hash values of the multimodal for the prefix matching in the radix attention.
    # There values are useless because their embeddings will be replaced by vision embeddings anyway.
    input_ids.clamp_(min=0, max=vocab_size - 1)
    input_embeds = input_embedding(input_ids)

    # deepstack embedding
    if use_deepstack:
        num_deepstack_embeddings = len(multimodal_model.deepstack_visual_indexes)

        deepstack_embedding_shape = input_embeds.shape[:-1] + (
            input_embeds.shape[-1] * num_deepstack_embeddings,
        )
        # a zero-filled embedding, with the same length of input_embeds, but different hidden_size
        input_deepstack_embeds = torch.zeros(
            deepstack_embedding_shape,
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )

        other_info["input_deepstack_embeds"] = input_deepstack_embeds

    # 4. scatter embeddings into input embedding
    for i, modality, embedding, mask in zip(
        range(len(embeddings)), modalities, embeddings, masks
    ):
        if embedding is None or mask is None:
            continue
        # in-place update
        indices = torch.where(mask.squeeze(dim=-1))[0]
        input_embeds[indices] = embedding.to(input_embeds.device, input_embeds.dtype)
        if use_deepstack.get(modality, None):
            input_deepstack_embeds[indices] = deepstack_embeddings[i].to(
                input_embeds.device, input_embeds.dtype
            )

    return input_embeds, other_info


def general_mm_embed_routine(
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    language_model: nn.Module,
    multimodal_model: Optional[nn.Module] = None,
    data_embedding_funcs: Dict[
        Modality, Callable[[List[MultimodalDataItem]], torch.Tensor]
    ] = None,
    placeholder_tokens: Optional[dict[Modality, List[int]]] = None,
    use_deepstack: Dict[Modality, bool] = {},
    **kwargs,
) -> torch.Tensor:
    """
    Process multimodal inputs and forward through language model.

    Args:
        input_ids: Input token IDs tensor
        forward_batch: Batch information for model forward pass
        language_model: Base language model to use
        data_embedding_funcs: A dictionary mapping from modality type to the corresponding embedding function.
        placeholder_tokens: Token IDs for multimodal placeholders
        use_deepstack: Whether to use deepstack embeddings for each modality, default False
        **kwargs: Additional arguments passed to language model

    Returns:
        Hidden states from language model forward pass
    """
    assert hasattr(language_model, "get_input_embeddings")
    embed_tokens = language_model.get_input_embeddings()
    if not hasattr(language_model, "pp_group") or language_model.pp_group.is_first_rank:
        if (
            not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.contains_mm_inputs()
        ):
            mm_inputs_list = [
                mm_input for mm_input in forward_batch.mm_inputs if mm_input is not None
            ]
            extend_prefix_lens = [
                prefix_len
                for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            extend_seq_lens = [
                seq_len
                for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            input_embeds, other_info = embed_mm_inputs(
                mm_inputs_list=mm_inputs_list,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens=extend_seq_lens,
                input_ids=input_ids,
                multimodal_model=multimodal_model,
                input_embedding=embed_tokens,
                data_embedding_func_mapping=data_embedding_funcs,
                placeholder_tokens=placeholder_tokens,
                use_deepstack=use_deepstack,
            )
            # add for qwen3_vl deepstack
            if use_deepstack:
                kwargs["input_deepstack_embeds"] = other_info["input_deepstack_embeds"]
            # once used, mm_inputs is useless, considering chunked-prefill is disabled for multimodal models
            # just being defensive here
            forward_batch.mm_inputs = None
        else:
            input_embeds = embed_tokens(input_ids)
        # Copy to pre-allocated buffer if available (for CUDA graph address stability)
        if forward_batch.input_embeds is not None:
            forward_batch.input_embeds.copy_(input_embeds)
            input_embeds = forward_batch.input_embeds
    else:
        input_embeds = None

    hidden_states = language_model(
        input_ids=None,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        **kwargs,
    )
    return hidden_states


def get_multimodal_data_bounds(
    input_ids: torch.Tensor, pad_values: List[int], token_pairs: List[Tuple[int, int]]
) -> torch.Tensor:
    """
    Returns a tensor indicating the bounds of multimodal data (images, video, audio, etc.)

    Returns:
        [bounds_count, 2]
    """
    # All the multimodal data in the batch should share the same special bound token ids.
    start_tokens = {s for s, _e in token_pairs}
    end_tokens = {e for _s, e in token_pairs}

    assert all(isinstance(t, int) for t in start_tokens)
    assert all(isinstance(t, int) for t in end_tokens)

    start_cond = torch.isin(
        input_ids, torch.as_tensor(start_tokens, device=input_ids.device)
    )
    end_cond = torch.isin(
        input_ids, torch.as_tensor(end_tokens, device=input_ids.device)
    )

    (data_start_tokens,) = torch.where(start_cond)
    (data_end_tokens,) = torch.where(end_cond)

    data_start_tokens_cpu = data_start_tokens.cpu().tolist()
    data_end_tokens_cpu = data_end_tokens.cpu().tolist()

    # the im_start_id sometimes can be cached as prefix, but it is needed for the embedding of the multimodal data
    if len(data_start_tokens_cpu) != len(data_end_tokens_cpu):
        if (
            len(data_start_tokens_cpu) + 1 == len(data_end_tokens_cpu)
            and input_ids[0].item() in pad_values
            and data_end_tokens_cpu
            and data_start_tokens_cpu
            and data_end_tokens_cpu[0] < data_start_tokens_cpu[0]
        ):
            data_start_tokens_cpu.insert(0, 0)
    valid_mm_data_nums = min(len(data_start_tokens_cpu), len(data_end_tokens_cpu))

    if valid_mm_data_nums == 0:
        return torch.zeros((0, 2), device=input_ids.device)

    # Filter out pairs where start_token >= end_token
    valid_pairs = []
    for i in range(valid_mm_data_nums):
        start_token = data_start_tokens_cpu[i]
        end_token = data_end_tokens_cpu[i]
        if start_token < end_token:
            valid_pairs.append((start_token + 1, end_token - 1))

    if not valid_pairs:
        return torch.zeros((0, 2), device=input_ids.device)

    # Convert valid pairs to tensor
    valid_pairs_tensor = torch.as_tensor(valid_pairs, device=input_ids.device)
    return valid_pairs_tensor


def data_hash(data) -> int:
    hash_bytes = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def tensor_hash(tensor_list) -> int:
    """
    hash a tensor or a tensor list
    """
    tensor = tensor_list
    if isinstance(tensor_list, list):
        tensor_list = flatten_nested_list(tensor_list)
        tensor_list = [
            x.flatten() if isinstance(x, torch.Tensor) else x for x in tensor_list
        ]
        tensor = torch.concat(tensor_list)
    if tensor.is_cuda:
        return gpu_tensor_hash(tensor.cuda())
    tensor = tensor.detach().contiguous()

    if tensor.dtype == torch.bfloat16:
        # memoryview() doesn't support PyTorch's BFloat16 dtype
        tensor = tensor.float()

    assert isinstance(tensor, torch.Tensor)
    tensor_cpu = tensor.cpu()

    mv = memoryview(tensor_cpu.numpy())
    return data_hash(mv.tobytes())


def hash_feature(f):
    if isinstance(f, list):
        if isinstance(f[0], torch.Tensor):
            return tensor_hash(f)
        return data_hash(tuple(flatten_nested_list(f)))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        arr_bytes = arr.tobytes()
        return data_hash(arr_bytes)
    elif isinstance(f, torch.Tensor):
        return tensor_hash([f])
    elif isinstance(f, CudaIpcTensorTransportProxy):
        reconstruct_t = f.reconstruct_on_target_device(torch.cuda.current_device())
        return tensor_hash([reconstruct_t])
    return data_hash(f)


def extend_mrope_positions_for_retracted_request(
    mrope_positions: torch.Tensor, output_ids_len: int
) -> torch.Tensor:
    """
    Extend mrope_positions for retracted requests by appending positions for output_ids.

    When a request is retracted and has multimodal inputs with mrope_positions,
    we need to extend the positions to cover the output_ids that were already generated.
    For pure text tokens, all three dimensions use the same incremental sequence.

    Args:
        mrope_positions: The original mrope positions tensor, shape (3, origin_input_ids_len)
        output_ids_len: The number of output tokens to generate positions for

    Returns:
        Extended mrope_positions tensor with shape (3, origin_input_ids_len + output_ids_len)
    """
    if output_ids_len <= 0:
        return mrope_positions

    # Get the last position value corresponding to origin_input_ids
    # mrope_positions shape: (3, origin_input_ids_len)
    last_position = mrope_positions[:, -1]  # shape: (3,)

    # Generate pure text mrope positions for output_ids
    # All three dimensions for pure text are the same incremental sequence
    start_pos = last_position[0] + 1  # Start from last position + 1
    output_positions = (
        torch.arange(
            start_pos,
            start_pos + output_ids_len,
            dtype=torch.int64,
            device=mrope_positions.device,
        )
        .unsqueeze(0)
        .expand(3, -1)
    )  # shape: (3, output_ids_len)

    # Concatenate to the original mrope_positions
    return torch.cat([mrope_positions, output_positions], dim=1)


def _get_length(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.shape[0] if value.ndim > 0 else None
    if isinstance(value, np.ndarray):
        return value.shape[0] if value.ndim > 0 else None
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _slice_value(value, start, end):
    if isinstance(value, torch.Tensor):
        return value[start:end]
    if isinstance(value, np.ndarray):
        return value[start:end]
    if isinstance(value, list):
        return value[start:end]
    if isinstance(value, tuple):
        return value[start:end]
    try:
        return value[start:end]
    except Exception:
        return value


def _slice_model_data(
    data: dict,
    index: int,
    start: int,
    end: int,
    num_items: int,
    total_feature_len: Optional[int],
):
    sliced = {}
    for key, value in data.items():
        length = _get_length(value)
        if length == num_items:
            sliced[key] = _slice_value(value, index, index + 1)
        elif total_feature_len is not None and length == total_feature_len:
            sliced[key] = _slice_value(value, start, end)
        else:
            sliced[key] = value
    return sliced


def get_new_expanded_mm_items(original_mm_items):
    expanded_mm_items = []
    for item in original_mm_items:
        is_bundled = item.offsets is not None and len(item.offsets) > 1

        if is_bundled:
            num_items = len(item.offsets)

            if item.is_image():
                image_grid_thw = item.model_specific_data.get("image_grid_thw")
                grid_len = _get_length(image_grid_thw)
                if image_grid_thw is None or grid_len != num_items:
                    expanded_mm_items.append(item)
                    continue

                patches_per_item = []
                for grid in image_grid_thw:
                    grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                    patches_per_item.append(int(torch.prod(grid_tensor).item()))

                cumulative = torch.cumsum(
                    torch.tensor(patches_per_item, dtype=torch.long), dim=0
                )
                slice_indices = [0] + cumulative.tolist()

                feature_len = _get_length(item.feature)
                if feature_len is None:
                    feature_len = _get_length(item.precomputed_embeddings)
                if feature_len is None or slice_indices[-1] != feature_len:
                    expanded_mm_items.append(item)
                    continue

                total_feature_len = feature_len
                for i in range(num_items):
                    start, end = slice_indices[i], slice_indices[i + 1]
                    new_item = copy.deepcopy(item)
                    if item.feature is not None:
                        new_item.feature = _slice_value(item.feature, start, end)
                    if item.precomputed_embeddings is not None:
                        new_item.precomputed_embeddings = _slice_value(
                            item.precomputed_embeddings, start, end
                        )
                    new_item.offsets = [item.offsets[i]]
                    new_item.model_specific_data = _slice_model_data(
                        item.model_specific_data,
                        index=i,
                        start=start,
                        end=end,
                        num_items=num_items,
                        total_feature_len=total_feature_len,
                    )
                    new_item.hash = None
                    expanded_mm_items.append(new_item)

            elif item.is_video():
                video_grid_thw = item.model_specific_data.get("video_grid_thw")
                if video_grid_thw is None:
                    expanded_mm_items.append(item)
                    continue

                # video_grid_thw shape: [num_videos, 3] where each row is [T, H, W]
                # When T > 1, item.offsets contains frames (num_items = total frames)
                # grid_len = num_videos, num_items = sum(T for each video) = total frames
                grid_len = _get_length(video_grid_thw)
                num_videos = grid_len

                # Calculate total frames and frames per video
                frames_per_video = []
                total_frames = 0
                for i in range(num_videos):
                    grid = video_grid_thw[i]
                    if isinstance(grid, torch.Tensor):
                        T = int(grid[0].item())  # T is the first element [T, H, W]
                    else:
                        grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                        T = int(grid_tensor[0].item())
                    frames_per_video.append(T)
                    total_frames += T

                # num_items should equal total_frames when T > 1
                if num_items != total_frames:
                    expanded_mm_items.append(item)
                    continue

                # Calculate patches per video: T * H * W for each video
                patches_per_video = []
                for i in range(num_videos):
                    grid = video_grid_thw[i]
                    if isinstance(grid, torch.Tensor):
                        patches_per_video.append(int(torch.prod(grid).item()))
                    else:
                        grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                        patches_per_video.append(int(torch.prod(grid_tensor).item()))

                # Calculate cumulative patches to get slice indices for each video
                cumulative = torch.cumsum(
                    torch.tensor(patches_per_video, dtype=torch.long), dim=0
                )
                slice_indices = [0] + cumulative.tolist()

                feature_len = _get_length(item.feature)
                if feature_len is None:
                    feature_len = _get_length(item.precomputed_embeddings)
                if feature_len is None or slice_indices[-1] != feature_len:
                    expanded_mm_items.append(item)
                    continue

                total_feature_len = feature_len
                # Group frames by video: calculate frame indices for each video
                frame_start_indices = [0]
                for i in range(num_videos):
                    frame_start_indices.append(
                        frame_start_indices[-1] + frames_per_video[i]
                    )

                # Expand each video into a separate item
                for video_idx in range(num_videos):
                    start, end = (
                        slice_indices[video_idx],
                        slice_indices[video_idx + 1],
                    )
                    frame_start, frame_end = (
                        frame_start_indices[video_idx],
                        frame_start_indices[video_idx + 1],
                    )

                    new_item = copy.deepcopy(item)
                    if item.feature is not None:
                        new_item.feature = _slice_value(item.feature, start, end)
                    if item.precomputed_embeddings is not None:
                        new_item.precomputed_embeddings = _slice_value(
                            item.precomputed_embeddings, start, end
                        )
                    # Group offsets for this video (all frames of this video)
                    new_item.offsets = item.offsets[frame_start:frame_end]
                    # For video_grid_thw, slice the corresponding row [T, H, W] for this video
                    new_item.model_specific_data = _slice_model_data(
                        item.model_specific_data,
                        index=video_idx,
                        start=start,
                        end=end,
                        num_items=num_videos,
                        total_feature_len=total_feature_len,
                    )
                    new_item.hash = None
                    expanded_mm_items.append(new_item)
            else:
                expanded_mm_items.append(item)

        else:
            expanded_mm_items.append(item)
    return expanded_mm_items

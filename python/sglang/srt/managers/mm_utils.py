"""
Multi-modality utils
"""

import asyncio
import copy
import hashlib
import pickle
from abc import abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.environ import envs
from sglang.srt.layers.multimodal import gpu_tensor_hash
from sglang.srt.managers.schedule_batch import (
    CudaIpcTensorTransportProxy,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.multimodal.evs import EVSEmbeddingResult
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

_is_default_tensor_transport = None


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

        # Replace multimodal tokens using per-item offsets
        items_by_modality = defaultdict(list)
        for item in mm_inputs.mm_items:
            items_by_modality[item.modality].append(item)

        token_id_map = {
            Modality.IMAGE: mm_inputs.im_token_id,
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

        ret_input_ids = input_ids_tensor.tolist()
        return ret_input_ids


embedding_cache: Optional[MultiModalStaticCache] = None
mm_global_cache_controller: Optional[Any] = None
_mm_global_cache_executor: Optional[ThreadPoolExecutor] = None


def init_mm_embedding_cache(max_size: int = 0):
    global embedding_cache
    embedding_cache = MultiModalStaticCache(max_size)


def init_mm_global_cache(controller: Optional[Any] = None):
    """Initialize the global cache controller for multi-node embedding sharing."""
    global mm_global_cache_controller, _mm_global_cache_executor
    mm_global_cache_controller = controller

    # Create a dedicated thread pool for global cache background writes
    if controller is not None and _mm_global_cache_executor is None:
        _mm_global_cache_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mm_global_cache_writer"
        )


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
    items_size: List[int],
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
    max_iterations = min(len(items_size) - 1, len(prefix_length))

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue

        items_per_req = items[items_size[i] : items_size[i + 1]]
        extend_len = extend_length[i] if i < len(extend_length) else 0
        items_offset = items_offset_list[i]

        if any(item.precomputed_embeddings is None for item in items_per_req):
            chunk = None
        else:
            req_embeddings = torch.concat(
                [item.precomputed_embeddings for item in items_per_req]
            )
            chunk, _, _ = get_embedding_chunk(
                embedding=req_embeddings,
                extend_prefix_len=prefix_length[i],
                extend_seq_len=extend_len,
                items_offset=items_offset,
            )

        if chunk is None and len(items_per_req) > 1:
            return None
        precomputed_embeddings.append(chunk)

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


DataEmbeddingFunc = Callable[
    [List[MultimodalDataItem]], torch.Tensor | EVSEmbeddingResult
]


def _can_skip_pre_embed_feature_move(data_embedding_func: DataEmbeddingFunc) -> bool:
    """qwen-vl visual forward already moves batched features to the target device.

    instead of performing multiple H2D for each mm feature from all mm_items (followed by concatenation on device),
    for some models which internally performs H2D on concated mm feature, these small H2D calls could be replaced with a single big H2D
    """
    owner = getattr(data_embedding_func, "__self__", None)
    if owner is None:
        return False
    if getattr(data_embedding_func, "__name__", None) not in (
        "get_image_feature",
        "get_video_feature",
    ):
        return False
    return owner.__class__.__name__ in {
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }


def _move_items_to_device(
    items: List[MultimodalDataItem], device: torch.device
) -> None:
    """Move item features to the target device (in-place, non-blocking).
    Saves a CPU reference so the offload path can restore without GPU->CPU copy."""
    for item in items:
        if isinstance(item.feature, torch.Tensor) and item.feature.device != device:
            item._cpu_feature = item.feature
            item.feature = item.feature.to(device, non_blocking=True)


def get_chunked_embedding_legacy(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items_per_req: List[MultimodalDataItem],
    items_offset: List[Tuple[int, int]],
    extend_prefix_len: int,
    extend_seq_len: int,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Fallback: encode all items at once, cache combined result, extract chunk.
    Used for non-bundled items or EVS results.
    """
    item_hashes = [item.hash for item in embedding_items_per_req]
    embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)
    embedding_per_req = embedding_cache.get(item_hashes)

    if embedding_per_req is None:
        if not _can_skip_pre_embed_feature_move(data_embedding_func):
            _move_items_to_device(embedding_items_per_req, device)
        embedding = data_embedding_func(embedding_items_per_req)
        embedding_per_req = (
            EmbeddingResult(embedding=embedding)
            if isinstance(embedding, torch.Tensor)
            else embedding
        )
        embedding_cache.set(embedding_items_hash, embedding_per_req)

    if isinstance(embedding_per_req, EVSEmbeddingResult):
        item = embedding_items_per_req[0]
        input_ids, items_offset = (
            embedding_per_req.redistribute_pruned_frames_placeholders(
                input_ids,
                items_offset,
                item=item,
                extend_prefix_len=extend_prefix_len,
                extend_seq_len=extend_seq_len,
            )
        )

    embedding_per_req_chunk, _, _ = get_embedding_chunk(
        embedding=embedding_per_req.embedding,
        extend_prefix_len=extend_prefix_len,
        extend_seq_len=extend_seq_len,
        items_offset=items_offset,
    )
    return embedding_per_req_chunk, input_ids


def find_chunk_items_and_check_cache(
    embedding_items_per_req: List[MultimodalDataItem],
    items_offset: List[Tuple[int, int]],
    chunk_start: int,
    chunk_end: int,
) -> List[Tuple[MultimodalDataItem, Optional[torch.Tensor], int, int]]:
    """Return (item, cached_embedding_or_None, start, end) for items in [chunk_start, chunk_end)."""
    chunk_entries = []
    for item, (start, end) in zip(embedding_items_per_req, items_offset):
        if end >= chunk_start and start < chunk_end:
            cached = embedding_cache.get_single(item.hash)
            emb = cached.embedding if cached is not None else None
            chunk_entries.append((item, emb, start, end))
    return chunk_entries


def assemble_chunk_embedding(
    chunk_entries: List[Tuple[Any, torch.Tensor, int, int]],
    chunk_start: int,
    chunk_end: int,
) -> Optional[torch.Tensor]:
    """
    Assemble a chunk of embeddings by slicing each item's embedding
    to the portion that falls within [chunk_start, chunk_end).
    """
    chunk_slices = []
    for _, emb, start, end in chunk_entries:
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end - 1)  # inclusive
        local_start = overlap_start - start
        local_end = overlap_end - start + 1  # exclusive for slicing
        chunk_slices.append(emb[local_start:local_end])

    if not chunk_slices:
        return None
    return torch.cat(chunk_slices, dim=0)


def get_chunked_prefill_embedding_legacy(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
    input_ids: torch.Tensor,
    max_iterations: int,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Non-per-image path: encode each request independently."""
    embedding_list = []
    device = input_ids.device

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None, items_offset

        extend_prefix_len = prefix_length[i]
        extend_seq_len = extend_length[i] if i < len(extend_length) else 0

        chunk_embedding, input_ids = get_chunked_embedding_legacy(
            data_embedding_func,
            embedding_items_per_req,
            items_offset,
            extend_prefix_len,
            extend_seq_len,
            input_ids,
            device,
        )
        if chunk_embedding is not None:
            embedding_list.append(chunk_embedding)

    if len(embedding_list) == 0:
        return None, input_ids
    return torch.concat(embedding_list, dim=0), input_ids


def _get_chunked_prefill_embedding(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """
    Chunked prefill embedding: collect cache misses across all per-image
    requests, batch them into a single ViT call, then assemble per-request
    chunk embeddings from the results.

    Implements three-tier cache: L1 (local) -> L2 (Global/Mooncake) -> Compute (ViT)
    """
    embedding_list = []
    device = input_ids.device
    # FIXME(Xinyuan): temporary workaround for eagle3
    # FIXME(yhyang201): check this
    max_iterations = min(len(items_size) - 1, len(prefix_length))

    per_image_process = (
        len(embedding_items) > 0 and len(embedding_items[0].offsets) == 1
    )

    if not per_image_process:
        return get_chunked_prefill_embedding_legacy(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
            input_ids,
            max_iterations,
        )

    # 1. Collect chunk entries per request, accumulate all L1 misses
    pending_requests = []
    all_l1_miss_items = []
    all_l1_miss_token_counts = []

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue
        extend_seq_len = extend_length[i] if i < len(extend_length) else 0
        if extend_seq_len <= 0:
            continue

        extend_prefix_len = prefix_length[i]
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None, items_offset

        chunk_start = extend_prefix_len
        chunk_end = extend_prefix_len + extend_seq_len
        chunk_entries = find_chunk_items_and_check_cache(
            embedding_items_per_req,
            items_offset,
            chunk_start,
            chunk_end,
        )
        if not chunk_entries:
            continue

        for item, emb, start, end in chunk_entries:
            if emb is None:
                all_l1_miss_items.append(item)
                all_l1_miss_token_counts.append(end - start + 1)

        pending_requests.append((chunk_entries, chunk_start, chunk_end))

    if not pending_requests:
        return None, input_ids

    # 2. Check L2 (Global Cache) for L1 misses if available
    hash_to_l2_emb = {}  # hash -> embedding (for L2 hits and TP sync)
    l2_miss_items = []  # Items that miss both L1 and L2
    l2_miss_token_counts = []

    if mm_global_cache_controller is not None and all_l1_miss_items:
        max_batch = envs.SGLANG_MM_GLOBAL_CACHE_MAX_BATCH.get()
        exist_timeout = envs.SGLANG_MM_GLOBAL_CACHE_EXIST_TIMEOUT.get()
        prefetch_timeout_per_item = envs.SGLANG_MM_GLOBAL_CACHE_PREFETCH_TIMEOUT.get()
        prefetch_max_timeout = envs.SGLANG_MM_GLOBAL_CACHE_PREFETCH_MAX_TIMEOUT.get()

        # Limit batch size to avoid long blocking
        if len(all_l1_miss_items) > max_batch:
            logger.debug(
                f"L1 miss items ({len(all_l1_miss_items)}) exceeds max batch size "
                f"({max_batch}), processing first batch only"
            )
            items_to_check = all_l1_miss_items[:max_batch]
            overflow_items = all_l1_miss_items[max_batch:]
            overflow_counts = all_l1_miss_token_counts[max_batch:]
        else:
            items_to_check = all_l1_miss_items
            overflow_items = []
            overflow_counts = []

        check_hashes = [item.hash for item in items_to_check]
        try:

            def check_global_cache():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        mm_global_cache_controller.batch_is_exist(check_hashes)
                    )
                finally:
                    loop.close()

            # Quick check with short timeout to avoid blocking encoder
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(check_global_cache)
                try:
                    exist_mask = future.result(timeout=exist_timeout)
                except TimeoutError:
                    logger.debug(
                        "Global cache existence check timeout, fallback to ViT"
                    )
                    exist_mask = [False] * len(check_hashes)

            # Categorize items by L2 existence
            global_hit_items = []  # (item, token_count) for L2 hits
            global_hit_hashes = []
            l2_check_miss = []  # (item, token_count) for L2 misses

            for i, item in enumerate(items_to_check):
                if exist_mask[i]:
                    global_hit_hashes.append(item.hash)
                    global_hit_items.append((item, all_l1_miss_token_counts[i]))
                else:
                    l2_check_miss.append((item, all_l1_miss_token_counts[i]))

            # Add overflow items directly to ViT
            for item, tc in zip(overflow_items, overflow_counts):
                l2_check_miss.append((item, tc))

            # Fetch embeddings from global cache for hits
            if global_hit_hashes:
                expected_tokens = [tc for _, tc in global_hit_items]
                req_id = (
                    f"prefetch_{str(global_hit_hashes[0])[:16]}"
                    f"_{id(data_embedding_func)}"
                )

                modality = global_hit_items[0][0].modality if global_hit_items else None
                mm_global_cache_controller.prefetch(
                    req_id, global_hit_hashes, expected_tokens, modality
                )

                # Wait for prefetch completion with adaptive timeout
                import time

                start_wait = time.time()
                prefetch_timeout = min(
                    prefetch_max_timeout, len(global_hit_hashes) * prefetch_timeout_per_item
                )
                while not mm_global_cache_controller.check_prefetch_progress(req_id):
                    if time.time() - start_wait > prefetch_timeout:
                        logger.debug(
                            f"Global cache prefetch timeout ({prefetch_timeout:.2f}s), "
                            f"falling back {len(global_hit_hashes)} items to ViT"
                        )
                        # Move all global hits back to L2 misses
                        for item, tc in global_hit_items:
                            l2_check_miss.append((item, tc))
                        global_hit_items = []
                        global_hit_hashes = []
                        break
                    time.sleep(0.002)  # 2ms polling interval

                # Get embeddings from global cache
                if global_hit_hashes:
                    global_embeddings = mm_global_cache_controller.get_embeddings(
                        global_hit_hashes
                    )
                    successful_items = []
                    for (item, token_count), emb in zip(
                        global_hit_items, global_embeddings
                    ):
                        if emb is None:
                            logger.warning(
                                f"Global cache returned None for hash {item.hash}, "
                                f"fallback to ViT"
                            )
                            l2_check_miss.append((item, token_count))
                            continue
                        hash_to_l2_emb[item.hash] = emb
                        # Also write to L1 cache for future local hits
                        emb_result = EmbeddingResult(embedding=emb)
                        embedding_cache.set(item.hash, emb_result)
                        successful_items.append((item, token_count))

                    # Update global_hit_items to only include successful hits
                    global_hit_items = successful_items

                    logger.debug(
                        f"Global cache hit: {len(global_hit_items)} items, "
                        f"L2 miss/ViT: {len(l2_check_miss)} items"
                    )

            l2_miss_items = [item for item, _ in l2_check_miss]
            l2_miss_token_counts = [tc for _, tc in l2_check_miss]

        except Exception as e:
            logger.warning(f"Global cache check failed: {e}, falling back to ViT")
            l2_miss_items = list(all_l1_miss_items)
            l2_miss_token_counts = list(all_l1_miss_token_counts)
    else:
        l2_miss_items = list(all_l1_miss_items)
        l2_miss_token_counts = list(all_l1_miss_token_counts)

    # 3. TP-Group synchronization: ensure all ranks have consistent cache hit results
    # Synchronize L1 + L2 hit status across all ranks using MIN operation (intersection)
    # This prevents hang when different ranks have inconsistent cache hit states
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > 1 and mm_global_cache_controller is not None:
        # Collect all items across all pending requests for TP sync
        all_items_for_sync = []
        for chunk_entries, _, _ in pending_requests:
            for item, emb, start, end in chunk_entries:
                all_items_for_sync.append((item, emb))

        # Sort by hash to ensure consistent order across all ranks
        sorted_items = sorted(all_items_for_sync, key=lambda x: x[0].hash)

        # Build hit mask: 1 for hit (L1 or L2), 0 for miss
        # All ranks have same order after sorting, use MIN to find intersection
        hit_mask = []
        for item, emb in sorted_items:
            is_hit = emb is not None or item.hash in hash_to_l2_emb
            hit_mask.append(1 if is_hit else 0)

        # Convert to tensor for all_reduce
        hit_tensor = torch.tensor(hit_mask, dtype=torch.int32, device="cpu")

        # MIN operation: if any rank has 0 (miss), result is 0
        # If all ranks have 1 (hit), result is 1
        # Use cache controller's prefetch_tp_group (gloo backend for CPU tensors)
        torch.distributed.all_reduce(
            hit_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=mm_global_cache_controller.prefetch_tp_group,
        )

        # synced_mask[i] == 1 means ALL ranks have this item cached
        # synced_mask[i] == 0 means at least one rank missed
        synced_mask = hit_tensor.tolist()

        # Build set of hashes that all ranks have (intersection)
        synced_hash_set = set()
        for i, (item, emb) in enumerate(sorted_items):
            if synced_mask[i] == 1:
                synced_hash_set.add(item.hash)

        # Items that this rank hit but other ranks missed need to go to ViT
        # Update chunk_entries and collect items for recompute
        items_to_recompute = []
        for req_idx, (chunk_entries, chunk_start, chunk_end) in enumerate(
            pending_requests
        ):
            updated_entries = []
            for item, emb, start, end in chunk_entries:
                if item.hash not in synced_hash_set and (
                    emb is not None or item.hash in hash_to_l2_emb
                ):
                    # Local hit but remote miss - need to recompute
                    items_to_recompute.append((item, end - start + 1))
                    # Set emb to None to mark as miss
                    updated_entries.append((item, None, start, end))
                    # Remove from L2 cache results
                    if item.hash in hash_to_l2_emb:
                        del hash_to_l2_emb[item.hash]
                else:
                    updated_entries.append((item, emb, start, end))
            pending_requests[req_idx] = (updated_entries, chunk_start, chunk_end)

        if items_to_recompute:
            logger.info(
                f"TP sync: {len(items_to_recompute)} items moved to ViT due to "
                f"inconsistent cache hit across ranks (local hit but remote miss)"
            )
            l2_miss_items.extend(item for item, _ in items_to_recompute)
            l2_miss_token_counts.extend(tc for _, tc in items_to_recompute)

        # Log sync results
        total_items = len(all_items_for_sync)
        synced_hits = sum(synced_mask)
        synced_misses = total_items - synced_hits
        logger.info(
            f"=== TP Cache Sync === Total: {total_items} | "
            f"Synced Hits (all ranks): {synced_hits} | "
            f"Synced Misses (ViT): {synced_misses}"
        )

    # 4. Batch encode all L2 cache-miss items in one ViT call
    locally_encoded = []  # (item, embedding) for write-back to L2
    if l2_miss_items:
        if not _can_skip_pre_embed_feature_move(data_embedding_func):
            _move_items_to_device(l2_miss_items, device)
        all_miss_embedding = data_embedding_func(l2_miss_items)
        all_miss_embedding = all_miss_embedding.reshape(
            -1, all_miss_embedding.shape[-1]
        )
        miss_embeddings = list(
            torch.split(all_miss_embedding, l2_miss_token_counts, dim=0)
        )
        for item, emb in zip(l2_miss_items, miss_embeddings):
            embedding_cache.set(item.hash, EmbeddingResult(embedding=emb))
            locally_encoded.append((item, emb))

    # 5. Write locally encoded embeddings back to L2 (Global Cache) for multi-node sharing
    # Use background thread to avoid blocking the encoder
    # Every rank writes so all local caches benefit
    if mm_global_cache_controller is not None and locally_encoded:
        global _mm_global_cache_executor
        if _mm_global_cache_executor is not None:
            hashes_to_insert = [item.hash for item, _ in locally_encoded]
            embeddings_to_insert = [emb for _, emb in locally_encoded]

            def _background_insert():
                try:
                    mm_global_cache_controller.insert_batch(
                        hashes_to_insert, embeddings_to_insert
                    )
                except Exception as e:
                    # Log but don't fail - global cache is best-effort
                    logger.warning(f"Background global cache write failed: {e}")

            # Submit to background thread pool (non-blocking)
            _mm_global_cache_executor.submit(_background_insert)
            logger.debug(
                f"Submitted {len(locally_encoded)} embeddings to global cache "
                f"(background)"
            )

    # 6. Fill in resolved embeddings and assemble per-request chunks
    # Build hash -> embedding map for all non-L1-hit items (L2 hits + ViT results)
    hash_to_resolved_emb = {}
    hash_to_resolved_emb.update(hash_to_l2_emb)
    for item, emb in locally_encoded:
        hash_to_resolved_emb[item.hash] = emb

    for chunk_entries, chunk_start, chunk_end in pending_requests:
        updated_entries = []
        for item, emb, start, end in chunk_entries:
            if emb is not None:
                # L1 cache hit
                updated_entries.append((item, emb, start, end))
            elif item.hash in hash_to_resolved_emb:
                # L2 hit or ViT computed
                updated_entries.append(
                    (item, hash_to_resolved_emb[item.hash], start, end)
                )
            else:
                # Should not happen if logic is correct
                logger.warning(
                    f"Item {item.hash} has no embedding after all cache tiers"
                )

        chunk_embedding = assemble_chunk_embedding(
            updated_entries, chunk_start, chunk_end
        )
        if chunk_embedding is not None:
            embedding_list.append(chunk_embedding)

    if len(embedding_list) == 0:
        return None, input_ids
    return torch.concat(embedding_list, dim=0), input_ids


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
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    placeholder_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
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
        - If EVS is used, the pruned input ids tensor; otherwise, the original input ids tensor
    """
    # 1. Get embedding
    embedding = _get_precomputed_embedding(
        embedding_items, items_size, prefix_length, extend_length, items_offset_list
    )
    if embedding is None:
        embedding, input_ids = _get_chunked_prefill_embedding(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
            input_ids,
        )
        if embedding is None:
            return None, None, input_ids
    # 2. Get mask
    if _is_npu:
        torch.npu.current_stream().synchronize()
    special_multimodal_mask = _get_multimodal_mask(input_ids, placeholder_tensor)
    # 3. Adjust embedding length if needed
    embedding = _adjust_embedding_length(embedding, special_multimodal_mask, logger)
    return embedding, special_multimodal_mask, input_ids


def embed_mm_inputs(
    mm_inputs_list: List[MultimodalInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    input_embedding: nn.Embedding,
    multimodal_model: nn.Module = None,
    data_embedding_func_mapping: Dict[Modality, DataEmbeddingFunc] = None,
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
            items_size = [0]
            items_offsets = []
            for mm_inputs in mm_inputs_list:
                mm_items = [
                    item
                    for item in mm_inputs.mm_items
                    if item.is_modality(modality=modality)
                ]
                items_size.append(items_size[-1] + len(mm_items))
                items_offsets.append(
                    flatten_nested_list([item.offsets for item in mm_items])
                )

            embedding, mask, input_ids = get_embedding_and_mask(
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
            else:
                deepstack_embeddings += [None]
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


def _embed_mm_inputs_with_split(
    mm_inputs_list: List[MultimodalInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embedding: nn.Embedding,
    multimodal_model: nn.Module = None,
    data_embedding_func_mapping: Dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: dict[Modality, List[int]] = None,
    use_deepstack: Dict[Modality, bool] = {},
):
    """Split batch into precomputed vs non-precomputed, embed each group, merge back."""
    precomputed_req_indices = []
    non_precomputed_req_indices = []
    for idx, mm_input in enumerate(mm_inputs_list):
        items = [item for item in mm_input.mm_items if item is not None]
        if items and all(
            getattr(item, "precomputed_embeddings", None) is not None for item in items
        ):
            precomputed_req_indices.append(idx)
        else:
            non_precomputed_req_indices.append(idx)

    embed_kwargs = dict(
        multimodal_model=multimodal_model,
        input_embedding=input_embedding,
        data_embedding_func_mapping=data_embedding_func_mapping,
        placeholder_tokens=placeholder_tokens,
        use_deepstack=use_deepstack,
    )

    if not precomputed_req_indices or not non_precomputed_req_indices:
        return embed_mm_inputs(
            mm_inputs_list=mm_inputs_list,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            input_ids=input_ids,
            **embed_kwargs,
        )

    all_seq_lens = forward_batch.extend_seq_lens_cpu
    mm_batch_indices = [
        i for i, mm in enumerate(forward_batch.mm_inputs) if mm is not None
    ]
    token_starts = []
    cumulative = 0
    for sl in all_seq_lens:
        token_starts.append(cumulative)
        cumulative += sl

    vocab_size = input_embedding.num_embeddings
    input_embeds = input_embedding(input_ids.clamp(min=0, max=vocab_size - 1))
    other_info = {}

    input_deepstack_embeds = None
    if use_deepstack and multimodal_model is not None:
        num_deepstack_embeddings = len(multimodal_model.deepstack_visual_indexes)
        input_deepstack_embeds = torch.zeros(
            input_ids.shape[0],
            input_embedding.embedding_dim * num_deepstack_embeddings,
            device=input_ids.device,
            dtype=input_embedding.weight.dtype,
        )
        other_info["input_deepstack_embeds"] = input_deepstack_embeds

    for group_req_indices in [precomputed_req_indices, non_precomputed_req_indices]:
        sub_mm_inputs = [mm_inputs_list[i] for i in group_req_indices]
        sub_prefix_lens = [extend_prefix_lens[i] for i in group_req_indices]
        sub_seq_lens = [extend_seq_lens[i] for i in group_req_indices]
        group_batch_indices = [mm_batch_indices[i] for i in group_req_indices]
        sub_slices = [
            input_ids[token_starts[bi] : token_starts[bi] + all_seq_lens[bi]]
            for bi in group_batch_indices
        ]
        sub_input_ids = torch.cat(sub_slices)

        sub_embeds, sub_info = embed_mm_inputs(
            mm_inputs_list=sub_mm_inputs,
            extend_prefix_lens=sub_prefix_lens,
            extend_seq_lens=sub_seq_lens,
            input_ids=sub_input_ids,
            **embed_kwargs,
        )

        offset = 0
        for bi in group_batch_indices:
            req_len = all_seq_lens[bi]
            start = token_starts[bi]
            input_embeds[start : start + req_len] = sub_embeds[
                offset : offset + req_len
            ]
            if (
                input_deepstack_embeds is not None
                and "input_deepstack_embeds" in sub_info
            ):
                input_deepstack_embeds[start : start + req_len] = sub_info[
                    "input_deepstack_embeds"
                ][offset : offset + req_len]
            offset += req_len

    return input_embeds, other_info


def offload_mm_features_to_cpu(mm_inputs_list: List[MultimodalInputs]):
    """Free GPU features after embedding. CPU copies are kept for later use
    (e.g. chunked prefill or recovery after retraction)."""
    language_only = get_global_server_args().language_only
    for mm_input in mm_inputs_list or []:
        if not mm_input or not hasattr(mm_input, "mm_items"):
            continue
        for item in mm_input.mm_items:
            if isinstance(item.feature, torch.Tensor) and item.feature.is_cuda:
                if item._cpu_feature is not None:
                    item.feature = item._cpu_feature
                else:
                    item.feature = item.feature.to("cpu", non_blocking=True)
            if language_only:
                pe = item.precomputed_embeddings
                if isinstance(pe, torch.Tensor) and pe.is_cuda:
                    item.precomputed_embeddings = pe.to("cpu", non_blocking=True)


def general_mm_embed_routine(
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    language_model: nn.Module,
    multimodal_model: Optional[nn.Module] = None,
    data_embedding_funcs: Dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: Optional[dict[Modality, List[int]]] = None,
    use_deepstack: Dict[Modality, bool] = {},
    **kwargs,
) -> torch.Tensor:
    """
    Process multimodal inputs and forward through language model.
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
            server_args = get_global_server_args()
            if server_args and server_args.enable_adaptive_dispatch_to_encoder:
                # Split by precomputed vs non-precomputed so get_embedding_and_mask only sees uniform batches
                input_embeds, other_info = _embed_mm_inputs_with_split(
                    mm_inputs_list=mm_inputs_list,
                    extend_prefix_lens=extend_prefix_lens,
                    extend_seq_lens=extend_seq_lens,
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    input_embedding=embed_tokens,
                    multimodal_model=multimodal_model,
                    data_embedding_func_mapping=data_embedding_funcs,
                    placeholder_tokens=placeholder_tokens,
                    use_deepstack=use_deepstack,
                )
            else:
                input_embeds, other_info = embed_mm_inputs(
                    mm_inputs_list=mm_inputs_list,
                    extend_prefix_lens=extend_prefix_lens,
                    extend_seq_lens=extend_seq_lens,
                    input_ids=input_ids,
                    input_embedding=embed_tokens,
                    multimodal_model=multimodal_model,
                    data_embedding_func_mapping=data_embedding_funcs,
                    placeholder_tokens=placeholder_tokens,
                    use_deepstack=use_deepstack,
                )

            # add for qwen3_vl deepstack
            if use_deepstack:
                kwargs["input_deepstack_embeds"] = other_info["input_deepstack_embeds"]
            # Free GPU features after embedding. CPU copies are kept for
            # later use (e.g. chunked prefill or recovery after retraction).
            offload_mm_features_to_cpu(mm_inputs_list)
            forward_batch.mm_inputs = None
            forward_batch.mm_input_embeds = input_embeds
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
        tensors = [
            x.flatten() if isinstance(x, torch.Tensor) else x for x in tensor_list
        ]
        # GPU path: concat + triton hash (unchanged)
        if any(isinstance(t, torch.Tensor) and t.is_cuda for t in tensors):
            tensor = torch.concat(tensors)
            return gpu_tensor_hash(tensor.cuda())
        # CPU path: hash each tensor incrementally without concat
        hasher = hashlib.sha256()
        for t in tensors:
            t = t.detach().contiguous()
            hasher.update(memoryview(t.view(torch.uint8).numpy()))
        hash_bytes = hasher.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big", signed=False)

    # Single tensor
    if tensor.is_cuda:
        return gpu_tensor_hash(tensor.cuda())
    tensor = tensor.detach().contiguous()
    hasher = hashlib.sha256()
    hasher.update(memoryview(tensor.view(torch.uint8).numpy()))
    hash_bytes = hasher.digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def hash_feature(f):
    if isinstance(f, list):
        if isinstance(f[0], torch.Tensor):
            return tensor_hash(f)
        return data_hash(tuple(flatten_nested_list(f)))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        hasher = hashlib.sha256()
        hasher.update(memoryview(arr))
        hash_bytes = hasher.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big", signed=False)
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


def _try_simple_split(item, num_items, expanded_mm_items):
    """Try to split a bundled item by matching feature dim-0 to offset count.
    Returns True if split succeeded, False otherwise."""
    feature = item.feature if item.feature is not None else item.precomputed_embeddings
    if feature is None:
        return False

    if isinstance(feature, (torch.Tensor, np.ndarray)):
        feature_count = feature.shape[0]
    elif isinstance(feature, (list, tuple)):
        feature_count = len(feature)
    else:
        return False

    if feature_count != num_items:
        return False

    for i in range(num_items):
        new_item = copy.copy(item)
        if item.feature is not None:
            if isinstance(item.feature, (list, tuple)):
                new_item.feature = [item.feature[i]]
            else:
                new_item.feature = item.feature[i : i + 1]
        if item.precomputed_embeddings is not None:
            if isinstance(item.precomputed_embeddings, (list, tuple)):
                new_item.precomputed_embeddings = [item.precomputed_embeddings[i]]
            else:
                new_item.precomputed_embeddings = item.precomputed_embeddings[i : i + 1]
        new_item.offsets = [item.offsets[i]]
        new_data = {}
        for k, v in item.model_specific_data.items():
            if isinstance(v, (list, tuple)) and len(v) == num_items:
                new_data[k] = [v[i]]
            elif (
                isinstance(v, (torch.Tensor, np.ndarray))
                and len(v.shape) > 0
                and v.shape[0] == num_items
            ):
                new_data[k] = v[i : i + 1]
            else:
                new_data[k] = v
        new_item.model_specific_data = new_data
        new_item.hash = None
        expanded_mm_items.append(new_item)
    return True


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
                    # No grid info — fall back to simple split by feature dim-0
                    if not _try_simple_split(item, num_items, expanded_mm_items):
                        expanded_mm_items.append(item)
                    continue

                if isinstance(image_grid_thw, torch.Tensor):
                    patches_per_item = (
                        torch.prod(image_grid_thw, dim=-1).long().tolist()
                    )
                else:
                    patches_per_item = [int(np.prod(grid)) for grid in image_grid_thw]

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
                    new_item = copy.copy(item)
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
                    if not _try_simple_split(item, num_items, expanded_mm_items):
                        expanded_mm_items.append(item)
                    continue

                # video_grid_thw shape: [num_videos, 3] where each row is [T, H, W]
                # When T > 1, item.offsets contains frames (num_items = total frames)
                # grid_len = num_videos, num_items = sum(T for each video) = total frames
                grid_len = _get_length(video_grid_thw)
                num_videos = grid_len

                # Calculate total frames and frames per video
                if isinstance(video_grid_thw, torch.Tensor):
                    frames_per_video = video_grid_thw[:, 0].long().tolist()
                else:
                    frames_per_video = [int(grid[0]) for grid in video_grid_thw]
                total_frames = sum(frames_per_video)

                # num_items should equal total_frames when T > 1
                if num_items != total_frames:
                    expanded_mm_items.append(item)
                    continue

                # Calculate patches per video: T * H * W for each video
                if isinstance(video_grid_thw, torch.Tensor):
                    patches_per_video = (
                        torch.prod(video_grid_thw, dim=-1).long().tolist()
                    )
                else:
                    patches_per_video = [int(np.prod(grid)) for grid in video_grid_thw]

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

                    new_item = copy.copy(item)
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
                if not _try_simple_split(item, num_items, expanded_mm_items):
                    expanded_mm_items.append(item)

        else:
            expanded_mm_items.append(item)
    return expanded_mm_items


class ShmPointerMMData:
    """
    Wraps a tensor to be sent via a shared memory handle.
    This acts as a "pointer" to the tensor data across process boundaries.
    """

    def __init__(self, tensor: torch.Tensor):
        if not tensor.is_cpu:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        nbytes = tensor.numel() * tensor.element_size()
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        try:
            dst = torch.frombuffer(shm.buf, dtype=torch.uint8)
            dst.copy_(tensor.view(torch.uint8).reshape(-1))
        except BaseException:
            shm.close()
            shm.unlink()
            raise
        self.shm_name = shm.name
        shm.close()
        self._shm_handle = None

    def __getstate__(self):
        return {
            "shm_name": self.shm_name,
            "shape": self.shape,
            "dtype": self.dtype,
        }

    def __setstate__(self, state):
        self.shm_name = state["shm_name"]
        self.shape = state["shape"]
        self.dtype = state["dtype"]
        self._shm_handle = shared_memory.SharedMemory(name=self.shm_name)
        # Zero-copy view into shared memory (no clone, no unlink)
        self.tensor = torch.frombuffer(self._shm_handle.buf, dtype=self.dtype).reshape(
            self.shape
        )

    def materialize(self) -> torch.Tensor:
        """Clone tensor from shm to owned memory, then release shm handle."""
        tensor = self.tensor.clone()
        if self._shm_handle is not None:
            self._shm_handle.close()
            try:
                self._shm_handle.unlink()
            except FileNotFoundError:
                pass  # Another rank already unlinked
            self._shm_handle = None
        return tensor

    def __del__(self):
        # Only close; never unlink. Unlinking is materialize()'s job.
        if getattr(self, "_shm_handle", None) is not None:
            self._shm_handle.close()
            self._shm_handle = None


def _get_is_default_transport():
    global _is_default_tensor_transport
    if _is_default_tensor_transport is None:
        from sglang.srt.managers.tokenizer_manager import (
            _determine_tensor_transport_mode,
        )

        _is_default_tensor_transport = (
            _determine_tensor_transport_mode(get_global_server_args()) == "default"
        )
    return _is_default_tensor_transport


def _wrap_tensor_or_list(value):
    """Wrap a CPU tensor (or list of CPU tensors) in ShmPointerMMData."""
    if isinstance(value, torch.Tensor) and value.is_cpu:
        return ShmPointerMMData(value)
    elif isinstance(value, (list, tuple)):
        wrapped = [
            (ShmPointerMMData(t) if isinstance(t, torch.Tensor) and t.is_cpu else t)
            for t in value
        ]
        return type(value)(wrapped) if isinstance(value, tuple) else wrapped
    return value


def wrap_shm_features(obj):
    """
    Scan the object for multimodal tensors and wrap them in SHM pointers.
    """
    if _get_is_default_transport() or get_global_server_args().skip_tokenizer_init:
        return obj

    if hasattr(obj, "mm_inputs") and obj.mm_inputs:
        for item in obj.mm_inputs.mm_items:
            if hasattr(item, "feature") and item.feature is not None:
                item.feature = _wrap_tensor_or_list(item.feature)
            if (
                hasattr(item, "precomputed_embeddings")
                and item.precomputed_embeddings is not None
            ):
                item.precomputed_embeddings = _wrap_tensor_or_list(
                    item.precomputed_embeddings
                )
    return obj


def _feature_has_shm(feat) -> bool:
    """Check whether a single feature (tensor, ShmPointer, or list) contains ShmPointerMMData."""
    if isinstance(feat, ShmPointerMMData):
        return True
    if isinstance(feat, (list, tuple)):
        return any(isinstance(t, ShmPointerMMData) for t in feat)
    return False


def has_shm_features(recv_reqs):
    """Return True if any request in the list contains ShmPointerMMData."""
    for req in recv_reqs:
        if hasattr(req, "batch"):
            if has_shm_features(req.batch):
                return True
        elif hasattr(req, "mm_inputs") and req.mm_inputs:
            for item in req.mm_inputs.mm_items:
                if _feature_has_shm(item.feature):
                    return True
                if _feature_has_shm(getattr(item, "precomputed_embeddings", None)):
                    return True
    return False


def _unwrap_tensor_or_list(value):
    """Restore ShmPointerMMData wrappers back into standard torch.Tensors."""
    if isinstance(value, ShmPointerMMData):
        return value.materialize()
    elif isinstance(value, (list, tuple)):
        unwrapped = [
            t.materialize() if isinstance(t, ShmPointerMMData) else t for t in value
        ]
        return type(value)(unwrapped) if isinstance(value, tuple) else unwrapped
    return value


def unwrap_shm_features(obj):
    """
    Restore ShmPointerMMData wrappers back into standard torch.Tensors.
    Handles both single requests and batch requests.
    """
    if _get_is_default_transport() or get_global_server_args().skip_tokenizer_init:
        return obj
    # Handle batch requests
    if hasattr(obj, "batch"):
        for sub_obj in obj.batch:
            unwrap_shm_features(sub_obj)
        return obj
    # Handle single requests
    if hasattr(obj, "mm_inputs") and obj.mm_inputs:
        for item in obj.mm_inputs.mm_items:
            if hasattr(item, "feature") and item.feature is not None:
                item.feature = _unwrap_tensor_or_list(item.feature)
            if (
                hasattr(item, "precomputed_embeddings")
                and item.precomputed_embeddings is not None
            ):
                item.precomputed_embeddings = _unwrap_tensor_or_list(
                    item.precomputed_embeddings
                )
    return obj

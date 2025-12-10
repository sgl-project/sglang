"""
Multi-modality utils
"""

import hashlib
import pickle
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn

from sglang.srt.distributed.parallel_state import get_tp_group
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
                raise ValueError(f"No multimodal token id provided for {item.modality}")

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
) -> Optional[torch.Tensor]:
    """
    If all items have precomputed_embeddings, return their concatenation.
    If some but not all have precomputed_embeddings, raise NotImplementedError.
    If none have precomputed_embeddings, return None.
    """
    precomputed_embeddings = [item.precomputed_embeddings for item in items]
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
    print(f"{max_iterations=}")
    if max_iterations > 1:
        batch_compute_embedding = _get_chunked_prefill_embedding2(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
        )
        return batch_compute_embedding
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
            # torch.cuda.synchronize()
            # s_time = time.time()
            # if max_iterations > 1:
            #     print(f"{embedding_items_per_req=}")
            embedding_per_req = data_embedding_func(embedding_items_per_req)
            # torch.cuda.synchronize()
            # e_time = time.time()
            # print(f"data_embedding_func cost {(e_time - s_time) * 1000} ms")
            if not embedding_cache.set(embedding_items_hash, embedding_per_req):
                print_warning_once(
                    "Multimodal embedding cache is full. This typically occurs when a single "
                    "embedding exceeds the cache size limit. Consider increasing the "
                    "`SGLANG_VLM_CACHE_SIZE_MB` environment variable or reducing the input "
                    "embedding size."
                )
        # if max_iterations > 1:
        #     print(f"i, embeddings_per_req {embedding_per_req.shape}, {embedding_per_req[0]}")

        embedding_per_req_chunk, _, _ = get_embedding_chunk(
            embedding=embedding_per_req,
            extend_prefix_len=prefix_length[i],
            extend_seq_len=extend_length[i] if i < len(extend_length) else 0,
            items_offset=items_offset,
        )
        # if max_iterations > 1:
        #     print(f"i, embedding_per_req_chunk {embedding_per_req_chunk.shape}, {embedding_per_req_chunk[0]}")

        embedding_list.append(embedding_per_req_chunk)
    # if max_iterations > 1:
    #     single_compute_embedding = torch.concat(embedding_list, dim=0)
    #     assert torch.equal(batch_compute_embedding, single_compute_embedding)
    #     print("equal!!!")
    if len(embedding_list) == 0:
        return None
    return torch.concat(embedding_list, dim=0)


def _get_chunked_prefill_embedding2(
    data_embedding_func: Callable[[List[MultimodalDataItem]], torch.Tensor],
    embedding_items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Optional[torch.Tensor]:
    # Calculate embedding for each request, try to get it from cache to avoid repeated calculation
    # FIXME(Xinyuan): temporary workaround for eagle3, which may have len(items_size) > len(prefix_length)
    max_iterations = min(len(items_size) - 1, len(prefix_length))
    print(f"{max_iterations=}")

    # step1: check cache and collect cache-miss items
    all_req_embedding_map = []  # List[Tuple[req_idx,embeddings]]
    new_compute_reqs = {}  # key: req_idx, value: list of MultimodalDataItem
    token_num_list = []
    modality = None
    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None, items_offset
        # if all items has been prefixed, we do not need to calculate embedding
        if all([offset_end < prefix_length[i] for _, offset_end in items_offset]):
            continue
        if modality is None:
            modality = embedding_items_per_req[0].modality
        token_num = 0
        for s, e in items_offset:
            token_num += e - s + 1
        token_num_list.append(token_num)

        item_hashes = [item.hash for item in embedding_items_per_req]
        embedding_per_req = embedding_cache.get(item_hashes)
        if embedding_per_req is None:
            new_compute_reqs[i] = embedding_items_per_req
            all_req_embedding_map.append((i, None))
        else:
            all_req_embedding_map.append((i, embedding_per_req))
        # if max_iterations > 1:
        #     print(f"{embedding_items_per_req=}")
    # print(f"{all_req_embedding_map=}, len {len(all_req_embedding_map)}")
    token_num_list = [0] + list(np.cumsum(token_num_list))
    # if max_iterations > 1:
    #     print(f"{items_offset_list=}")
    #     print(f"{token_num_list=}")

    pixel_values_list = []
    grid_thw_list = []
    for items in new_compute_reqs.values():
        # print(f"feature shape {items[0].feature.shape}")
        pixel_values_list.extend([item.feature for item in items])
        # print(f"grid_thw shape {items[0].image_grid_thw.shape}")
        grid_thw_list.extend([item.image_grid_thw for item in items])
    # if max_iterations > 1:
    #     print(f"pixel_values_list len {len(pixel_values_list)}")
    #     print(f"grid_thw_list len {len(grid_thw_list)}")

    # Check if enable encoder DP
    use_encoder_dp = get_global_server_args().mm_enable_dp_encoder
    if use_encoder_dp:
        print("use_encoder_dp")
        # split images/videos to different dp rank
        embedding_items_local_rank, items_size_local_rank = None, None
    else:
        # feature = torch.cat(pixel_values_list)
        # print(f"feature shape {feature.shape}")
        # grid_thw = torch.cat(grid_thw_list)
        # print(f"grid_thw shape {grid_thw.shape}")
        assert modality is not None
        embedding_items_local_rank = MultimodalDataItem(
            modality=modality,
            feature=torch.cat(pixel_values_list),
            model_specific_data={"image_grid_thw": torch.cat(grid_thw_list)},
        )
    # enable_vlm_batch_compute = get_global_server_args().enable_vlm_batch_compute
    # batch compute embeddings
    embeddings_local_rank = data_embedding_func([embedding_items_local_rank])
    assert (
        embeddings_local_rank is not None
    ), f"failed to calculate mm embeddings, use_encoder_dp is {use_encoder_dp}"
    # print(f"embeddings_local_rank shape {embeddings_local_rank.shape}")

    if use_encoder_dp:
        # gather embeddings from different dp rank
        print("embeddings_local_rank")

    # split embeddings to different requests and chunk
    embedding_list = []
    for i, embeddings in all_req_embedding_map:
        # print(f"{i}, embeddings {embeddings}")
        embeddings_per_req = None
        if embeddings is not None:
            embeddings_per_req = embeddings
        else:
            embeddings_per_req = embeddings_local_rank[
                token_num_list[i] : token_num_list[i + 1]
            ]
        # print(f"{i}, embeddings_per_req {embeddings_per_req.shape}, {embeddings_per_req[0]}")

        embedding_per_req_chunk, _, _ = get_embedding_chunk(
            embedding=embeddings_per_req,
            extend_prefix_len=prefix_length[i],
            extend_seq_len=extend_length[i] if i < len(extend_length) else 0,
            items_offset=items_offset_list[i],
        )
        # print(f"{i}, embedding_per_req_chunk {embedding_per_req_chunk.shape}, {embedding_per_req_chunk[0]}")
        embedding_list.append(embedding_per_req_chunk)
    if len(embedding_list) == 0:
        return None
    return torch.concat(embedding_list, dim=0)


# def _get_embeddings_per_req(
#     data_embedding_func: Callable[[List[MultimodalDataItem]], torch.Tensor],
#     embedding_items: List[MultimodalDataItem],
#     items_size: List[int],
#     prefix_length: List[int],
#     extend_length: List[int],
#     items_offset_list: List[List[Tuple[int, int]]],
# ) -> List[torch.Tensor]:
#     # Calculate embedding for each request, try to get it from cache to avoid repeated calculation
#     embedding_list = []
#     # FIXME(Xinyuan): temporary workaround for eagle3, which may have len(items_size) > len(prefix_length)
#     max_iterations = min(len(items_size) - 1, len(prefix_length))
#     for i in range(max_iterations):
#         if items_size[i] == items_size[i + 1]:
#             continue
#         embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
#         items_offset = items_offset_list[i]
#         assert items_offset is not None, items_offset
#         # if all items has been prefixed, we do not need to calculate embedding
#         if all([offset_end < prefix_length[i] for _, offset_end in items_offset]):
#             continue
#         item_hashes = [item.hash for item in embedding_items_per_req]
#         embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)
#         embedding_per_req = embedding_cache.get(item_hashes)
#         if embedding_per_req is None:
#             embedding_per_req = data_embedding_func(embedding_items_per_req)
#             if not embedding_cache.set(embedding_items_hash, embedding_per_req):
#                 print_warning_once(
#                     "Multimodal embedding cache is full. This typically occurs when a single "
#                     "embedding exceeds the cache size limit. Consider increasing the "
#                     "`SGLANG_VLM_CACHE_SIZE_MB` environment variable or reducing the input "
#                     "embedding size."
#                 )

#         # embedding_per_req_chunk, _, _ = get_embedding_chunk(
#         #     embedding=embedding_per_req,
#         #     extend_prefix_len=prefix_length[i],
#         #     extend_seq_len=extend_length[i] if i < len(extend_length) else 0,
#         #     items_offset=items_offset,
#         # )
#         embedding_list.append(embedding_per_req)
#     if len(embedding_list) == 0:
#         return None
#     return embedding_list

# def _get_embeddings_batch(
#     data_embedding_func: Callable[[List[MultimodalDataItem]], torch.Tensor],
#     embedding_items: List[MultimodalDataItem],
#     items_size: List[int],
#     prefix_length: List[int],
#     extend_length: List[int],
#     items_offset_list: List[List[Tuple[int, int]]],
# ) -> Optional[torch.Tensor]:
#     """
#     Calculate embeddings for all requests with cache support, batch computation and encode dp.

#     Workflow:
#     1. Check cache for each request and collect cache-miss items
#     2. Batch compute all cache-miss items together in one call
#     3. Split batch results back to individual requests using embedding sizes
#     4. Apply get_embedding_chunk for each request
#     5. Return concatenated results
#     """
#     # FIXME(Xinyuan): temporary workaround for eagle3, which may have len(items_size) > len(prefix_length)
#     max_iterations = min(len(items_size) - 1, len(prefix_length))

#     # Phase 1: Separate cache hits from misses, and collect cache-miss items metadata
#     req_info_list = []  # List of (req_idx, items, offset, cached_embedding_or_none)
#     all_items_to_compute = []  # All items that need computation
#     item_to_req_mapping = []  # Maps item index in all_items_to_compute to (req_idx, item_idx_in_req)
#     req_to_items_mapping = {} # Maps req_idx to items in all_items_to_compute, {req_idx: List[MultimodalDataItem]}, here item refers to a single image/video

#     for i in range(max_iterations):
#         if items_size[i] == items_size[i + 1]:
#             continue

#         embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
#         items_offset = items_offset_list[i]
#         assert items_offset is not None, items_offset

#         # Skip if all items are prefixed
#         if all([offset_end < prefix_length[i] for _, offset_end in items_offset]):
#             continue

#         # Check cache
#         item_hashes = [item.hash for item in embedding_items_per_req]
#         embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)
#         cached_embedding = embedding_cache.get(item_hashes)

#         if cached_embedding is not None:
#             # Cache hit - store for later use
#             req_info_list.append((i, embedding_items_per_req, items_offset, cached_embedding))
#         else:
#             # Cache miss - add to batch computation
#             req_info_list.append((i, embedding_items_per_req, items_offset, None))
#             for item_idx_in_req, item in enumerate(embedding_items_per_req):
#                 # split item into single image/video
#                 if 'image_grid_thw' not in item.model_specific_data:
#                     return None
#                 img_count = len(items_offset)
#                 grid_thw_tensor = item.model_specific_data["image_grid_thw"]
#                 grid_thw_list = grid_thw_tensor.tolist()
#                 patches_per_image = [math.prod(grid_thw) for grid_thw in grid_thw_list]
#                 cum_patches_per_image = [0, *itertools.accumulate(patches_per_image)]

#                 all_computed_patches = []
#                 all_grid_thws = []
#                 for j in range(img_count):
#                     feature = item.feature[cum_patches_per_image[j] : cum_patches_per_image[j + 1]]
#                     single_item = MultimodalDataItem(
#                         modality=item.modality,
#                         offsets=items_offset[j],
#                         feature=feature,
#                         model_specific_data={
#                             "image_grid_thw": grid_thw_tensor[j]
#                         },
#                     )
#                     all_items_to_compute.append(item)
#                     item_to_req_mapping.append((i, item_idx_in_req))
#                 if i not in req_to_items_mapping:
#                     req_to_items_mapping[i] = []
#                 req_to_items_mapping[i].append(item)
#             assert len(all_items_to_compute) == len(item_to_req_mapping)

#     # Phase 2: Batch compute all cache-miss items
#     if len(all_items_to_compute) > 0:
#         # Try batch computation with size tracking
#         # Fall back to per-request if not supported
#         batch_success = False

#         # Group items by request for potential fallback
#         # req_to_items = {}
#         # for item_idx, (req_idx, _) in enumerate(item_to_req_mapping):
#         #     if req_idx not in req_to_items:
#         #         req_to_items[req_idx] = []
#         #     req_to_items[req_idx].append(all_items_to_compute[item_idx])

#         # Attempt optimized batch computation with DP support
#         try:
#             # Check if enable encoder DP
#             use_encoder_dp = get_global_server_args().mm_enable_dp_encoder

#             if use_encoder_dp:
#                 # Import DP utilities
#                 from sglang.srt.multimodal.mm_utils import get_dp_encoder_lb_assignment
#                 from sglang.srt.distributed.parallel_state import (
#                     get_tensor_model_parallel_world_size,
#                     get_tensor_model_parallel_rank,
#                     tensor_model_parallel_all_gather,
#                 )
#                 import itertools
#                 import math

#                 # Extract metadata
#                 first_item = all_items_to_compute[0]

#                 if hasattr(first_item, 'image_grid_thw') or hasattr(first_item, 'video_grid_thw'):
#                     # Extract grid_thw for all items
#                     grid_thw_list = []
#                     for item in all_items_to_compute:
#                         if hasattr(item, 'image_grid_thw') and item.image_grid_thw is not None:
#                             grid_thw = item.image_grid_thw
#                         elif hasattr(item, 'video_grid_thw') and item.video_grid_thw is not None:
#                             grid_thw = item.video_grid_thw
#                         else:
#                             continue

#                         if isinstance(grid_thw, torch.Tensor):
#                             grid_thw = grid_thw.tolist()
#                         if isinstance(grid_thw, list) and len(grid_thw) > 0 and isinstance(grid_thw[0], list):
#                             grid_thw = grid_thw[0]
#                         grid_thw_list.append(grid_thw)

#                     # DP load balancing
#                     tp_size = get_tensor_model_parallel_world_size()
#                     tp_rank_local = get_tensor_model_parallel_rank()

#                     patches_per_image = [math.prod(grid_thw) for grid_thw in grid_thw_list]
#                     (image_to_tp_rank, gpu_sample_counts, grouped_pixel_values_len) = (
#                         get_dp_encoder_lb_assignment(patches_per_image, tp_size)
#                     )

#                     cum_gpu_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]
#                     image_idxs_local = image_to_tp_rank[
#                         cum_gpu_sample_counts[tp_rank_local] : cum_gpu_sample_counts[tp_rank_local + 1]
#                     ]

#                     # Get local items for this rank
#                     local_items = [all_items_to_compute[i] for i in image_idxs_local]

#                     # Compute embeddings for local items
#                     if len(local_items) > 0:
#                         image_embeds_local = data_embedding_func(local_items)
#                     else:
#                         # Empty tensor for this rank
#                         image_embeds_local = torch.empty((0, 0), device='cuda')

#                     # Calculate embedding dimension reduction factor (spatial_merge_size^2)
#                     # Assume 2 for Qwen-VL
#                     spatial_merge_size = 2
#                     embed_dim_reduction_factor = spatial_merge_size ** 2

#                     # Pad for allgather
#                     max_len_per_rank = max(grouped_pixel_values_len) // embed_dim_reduction_factor
#                     current_len = image_embeds_local.shape[0]

#                     if current_len < max_len_per_rank:
#                         padding_size = max_len_per_rank - current_len
#                         if image_embeds_local.numel() > 0:
#                             padding = torch.empty(
#                                 (padding_size, image_embeds_local.shape[1]),
#                                 dtype=image_embeds_local.dtype,
#                                 device=image_embeds_local.device,
#                             )
#                             image_embeds_local_padded = torch.cat([image_embeds_local, padding], dim=0)
#                         else:
#                             # Handle empty case
#                             image_embeds_local_padded = torch.empty(
#                                 (max_len_per_rank, 0), device='cuda'
#                             )
#                     else:
#                         image_embeds_local_padded = image_embeds_local

#                     # AllGather
#                     gathered_embeds = tensor_model_parallel_all_gather(image_embeds_local_padded, dim=0)

#                     # Remove padding and reconstruct per-rank embeddings
#                     rank_embeddings = []
#                     for rank in range(tp_size):
#                         start_idx = rank * max_len_per_rank
#                         end_idx = start_idx + (grouped_pixel_values_len[rank] // embed_dim_reduction_factor)
#                         rank_embeddings.append(gathered_embeds[start_idx:end_idx])

#                     # Calculate embedding sizes per item
#                     patches_per_output_image = [
#                         (patch_size // embed_dim_reduction_factor) for patch_size in patches_per_image
#                     ]

#                     # Reconstruct embeddings in original order
#                     original_order_embeddings = [None] * len(all_items_to_compute)
#                     current_idx = 0
#                     for rank in range(tp_size):
#                         count = gpu_sample_counts[rank]
#                         if count > 0:
#                             rank_images = image_to_tp_rank[current_idx : current_idx + count]
#                             rank_embed = rank_embeddings[rank]

#                             embed_start = 0
#                             for img_idx in rank_images:
#                                 img_patches = patches_per_output_image[img_idx]
#                                 original_order_embeddings[img_idx] = rank_embed[
#                                     embed_start : embed_start + img_patches
#                                 ]
#                                 embed_start += img_patches
#                             current_idx += count

#                     batch_embeddings = torch.cat(original_order_embeddings, dim=0)
#                     item_embedding_sizes = patches_per_output_image

#                     # Phase 3: Split batch results using size information
#                     cumulative_sizes = [0]
#                     for size in item_embedding_sizes:
#                         cumulative_sizes.append(cumulative_sizes[-1] + size)

#                     # Group items by request and extract embeddings
#                     req_to_item_indices = {}
#                     for item_idx, (req_idx, _) in enumerate(item_to_req_mapping):
#                         if req_idx not in req_to_item_indices:
#                             req_to_item_indices[req_idx] = []
#                         req_to_item_indices[req_idx].append(item_idx)

#                     batch_success = True
#             else:
#                 all_embeddings = data_embedding_func(all_items_to_compute)
#         except Exception as e:
#             # Fall back to per-request computation on any error
#             import traceback
#             print_warning_once(f"Encoder DP batch computation failed: {e}\n{traceback.format_exc()}")
#             pass

#         # Fall back to per-request computation if batch failed
#         if not batch_success:
#             return _get_embeddings_per_req(
#                 data_embedding_func,
#                 embedding_items,
#                 items_size,
#                 prefix_length,
#                 extend_length,
#                 items_offset_list,
#             )

#     # Phase 4: Extract and concatenate embeddings for each request
#     computed_req_embeddings = {}  # req_idx -> computed embedding
#     embedding_list = []
#     for req_idx, embedding_items_per_req, items_offset, cached_embedding in req_info_list:
#         if cached_embedding is not None:
#             embedding_list.append(cached_embedding)
#         else:

#             embedding_list.append(None)

#     # Extract and concatenate embeddings for each request
#     for req_idx, indices in req_to_item_indices.items():
#         req_embedding_parts = []
#         for item_idx in indices:
#             start = cumulative_sizes[item_idx]
#             end = cumulative_sizes[item_idx + 1]
#             req_embedding_parts.append(batch_embeddings[start:end])

#         req_embedding = torch.cat(req_embedding_parts, dim=0)
#         computed_req_embeddings[req_idx] = req_embedding

#         # Cache the result
#         req_items = [all_items_to_compute[idx] for idx in indices]
#         item_hashes = [item.hash for item in req_items]
#         embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)
#         if not embedding_cache.set(embedding_items_hash, req_embedding):
#             print_warning_once(
#                 "Multimodal embedding cache is full. Consider increasing "
#                 "`SGLANG_VLM_CACHE_SIZE_MB`."
#             )

#     for req_idx, embedding_items_per_req, items_offset, cached_embedding in req_info_list:
#         # Get embedding (cached or computed)
#         if cached_embedding is not None:
#             embedding_per_req = cached_embedding
#         elif req_idx in computed_req_embeddings:
#             embedding_per_req = computed_req_embeddings[req_idx]
#         else:
#             assert embedding_per_req is not None, f"Failed to get embedding for request {req_idx}"
#             continue

#         # Apply chunking
#         # embedding_per_req_chunk, _, _ = get_embedding_chunk(
#         #     embedding=embedding_per_req,
#         #     extend_prefix_len=prefix_length[req_idx],
#         #     extend_seq_len=extend_length[req_idx] if req_idx < len(extend_length) else 0,
#         #     items_offset=items_offset,
#         # )
#         # embedding_list.append(embedding_per_req_chunk)

#     if len(embedding_list) == 0:
#         return None
#     return torch.concat(embedding_list, dim=0)


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
    embedding = _get_precomputed_embedding(embedding_items)
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
    # Lazy import to allow some monkey patch of piecewise_cuda_graph_runner
    from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
        use_original_ca_comm,
    )

    tp_group = get_tp_group()

    with use_original_ca_comm(tp_group):
        # We disable custom allreduce in piecewise cuda graph.
        # However, because we only capture the language model part, the multimodal can still use custom allreduce.
        assert hasattr(language_model, "get_input_embeddings")
        embed_tokens = language_model.get_input_embeddings()
        if (
            not hasattr(language_model, "pp_group")
            or language_model.pp_group.is_first_rank
        ):
            if (
                not forward_batch.forward_mode.is_decode()
                and not forward_batch.forward_mode.is_target_verify()
                and forward_batch.contains_mm_inputs()
            ):
                mm_inputs_list = [
                    mm_input
                    for mm_input in forward_batch.mm_inputs
                    if mm_input is not None
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
                    kwargs["input_deepstack_embeds"] = other_info[
                        "input_deepstack_embeds"
                    ]
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

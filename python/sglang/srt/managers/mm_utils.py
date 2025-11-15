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

from sglang.srt.layers.multimodal import gpu_tensor_hash
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
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


def _get_multimodal_mask(
    input_ids: torch.Tensor, placeholder_tensor: torch.Tensor
) -> torch.Tensor:
    return torch.isin(input_ids, placeholder_tensor).unsqueeze(-1)


def _adjust_embedding_length(
    embedding: torch.Tensor, special_multimodal_mask: torch.Tensor, logger
) -> torch.Tensor:
    """Make sure the multimodal embedding length matches the number of MM tokens.

    - If we have MORE embeddings than MM tokens: truncate
    - If we have FEWER embeddings than MM tokens: pad zeros instead of raising.
    """
    num_mm_tokens_in_input_ids = int(special_multimodal_mask.sum().item())
    num_mm_tokens_in_embedding = int(embedding.size(0))

    if num_mm_tokens_in_embedding == num_mm_tokens_in_input_ids:
        return embedding

    if num_mm_tokens_in_embedding < num_mm_tokens_in_input_ids:
        logger.warning(
            "Multimodal embedding shorter than expected: "
            "num_mm_tokens_in_input_ids=%d vs num_mm_tokens_in_embedding=%d. "
            "Padding %d zero embeddings.",
            num_mm_tokens_in_input_ids,
            num_mm_tokens_in_embedding,
            num_mm_tokens_in_input_ids - num_mm_tokens_in_embedding,
        )

        hidden_size = embedding.size(-1)

        if num_mm_tokens_in_embedding == 0:
            return embedding.new_zeros(num_mm_tokens_in_input_ids, hidden_size)

        pad = embedding.new_zeros(
            num_mm_tokens_in_input_ids - num_mm_tokens_in_embedding,
            hidden_size,
        )
        return torch.cat([embedding, pad], dim=0)

    logger.warning(
        "Multimodal embedding longer than expected: "
        "num_mm_tokens_in_input_ids=%d vs num_mm_tokens_in_embedding=%d. "
        "Truncating extra %d embeddings.",
        num_mm_tokens_in_input_ids,
        num_mm_tokens_in_embedding,
        num_mm_tokens_in_embedding - num_mm_tokens_in_input_ids,
    )
    return embedding[:num_mm_tokens_in_input_ids]


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

    Returns:
        (inputs_embeds, other_info)
    """
    other_info = {}
    if mm_inputs_list is None:
        return None

    # 1) 扁平化所有 mm items
    item_flatten_list = []
    for mm_inputs in mm_inputs_list:
        item_flatten_list += [item for item in mm_inputs.mm_items if item is not None]

    modalities: List[Modality] = []
    embeddings: List[Optional[torch.Tensor]] = []
    masks: List[Optional[torch.Tensor]] = []
    deepstack_embeddings: List[torch.Tensor] = []

    target_hidden = input_embedding.embedding_dim

    # —— helper: 仅在必要时做 projector（先 deepstack，再 projector）
    def _maybe_project_feature(modality_str: str, feat: torch.Tensor) -> torch.Tensor:
        if feat is None:
            return None
        if feat.shape[-1] == target_hidden:
            return feat  # 已经对齐

        # 候选 projector 名称（尽量覆盖常见实现与别名）
        candidates = [
            f"project_{modality_str}_feature",
            f"{modality_str}_proj",
            f"{modality_str}_projector",
            "project_mm_feature",
            "project_feature",
            "mm_projector",
            "mm_proj",
            "vision_proj" if modality_str == "image" else None,
            "visual_proj" if modality_str == "image" else None,
            "visual_projector" if modality_str == "image" else None,
            "visual_mlp" if modality_str == "image" else None,
            "audio_proj" if modality_str == "audio" else None,
            "audio_projector" if modality_str == "audio" else None,
        ]
        candidates = [c for c in candidates if c]

        projector = None

        # 直接在 multimodal_model 上找
        for name in candidates:
            if hasattr(multimodal_model, name):
                projector = getattr(multimodal_model, name)
                break

        # 常见的嵌套路径再探测一轮
        if projector is None:
            nest_paths = [
                "model",
                "language_model",
                "vision_model",
                "vision_tower",
                "mm",
                "modules",
            ]
            for p in nest_paths:
                cur = getattr(multimodal_model, p, None)
                if cur is None:
                    continue
                for name in candidates:
                    if hasattr(cur, name):
                        projector = getattr(cur, name)
                        break
                if projector is not None:
                    break

        if projector is not None:
            out = projector(feat)
            if out.shape[-1] != target_hidden:
                raise RuntimeError(
                    f"[embed_mm_inputs] projector `{projector.__class__.__name__}` "
                    f"outputs dim={out.shape[-1]} but hidden={target_hidden}."
                )
            return out

        # 没 projector 就显式报错（避免静默降级为随机线性层）
        raise RuntimeError(
            "[embed_mm_inputs] Found multimodal feature dim "
            f"{feat.shape[-1]} but hidden size is {target_hidden}; no projector found. "
            "Expose a projector on the model or make `get_*_feature` return aligned features."
        )

    # 2) 逐模态取特征与 mask
    for modality in Modality.all():
        items = [it for it in item_flatten_list if it.is_modality(modality=modality)]

        if len(items) == 0:
            continue

        # 选择 embedder
        embedder = None
        if data_embedding_func_mapping is not None:
            embedder = data_embedding_func_mapping.get(modality, None)
        if embedder is None and multimodal_model is not None:
            modality_id = modality.name.lower()
            embedder = getattr(multimodal_model, f"get_{modality_id}_feature", None)

        assert embedder is not None, f"no embedding method found for {modality}"

        placeholder_tensor = torch.as_tensor(
            [item.pad_value for item in items],
            device=input_ids.device,
        )

        # per-request offsets/lengths
        items_size = torch.zeros(len(mm_inputs_list) + 1, dtype=int)
        items_offsets = []
        for i, mm_inputs in enumerate(mm_inputs_list):
            mm_items = [
                it for it in mm_inputs.mm_items if it.is_modality(modality=modality)
            ]
            items_size[i + 1] = len(mm_items)
            items_offsets.append(flatten_nested_list([it.offsets for it in mm_items]))
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

        if embedding is not None and mask is not None:
            modality_str = modality.name.lower()

            # ✅ 先做 deepstack 拆分（Qwen3-Omni: 8192=4×2048 等典型情况）
            if use_deepstack.get(modality, False) and hasattr(
                multimodal_model, "separate_deepstack_embeds"
            ):
                # 如果最后一维是 hidden 的整数倍且不等于 hidden，优先拆分
                if (
                    embedding.shape[-1] % target_hidden == 0
                    and embedding.shape[-1] != target_hidden
                ):
                    embedding, deep = multimodal_model.separate_deepstack_embeds(
                        embedding
                    )
                    if deep is not None:
                        # 校验 deep 的维度符合 k×hidden
                        if deep.shape[-1] % target_hidden != 0:
                            raise RuntimeError(
                                f"[embed_mm_inputs] deepstack last dim {deep.shape[-1]} "
                                f"is not multiple of hidden {target_hidden}."
                            )
                        deepstack_embeddings.append(deep)

            # deepstack 后如仍未对齐，再尝试 projector
            if embedding.shape[-1] != target_hidden:
                embedding = _maybe_project_feature(modality_str, embedding)

            modalities.append(modality)
            embeddings.append(embedding)
            masks.append(mask)
        else:
            modalities.append(modality)
            embeddings.append(None)
            masks.append(None)

    # 3) 计算 text embedding
    vocab_size = input_embedding.num_embeddings
    input_ids.clamp_(min=0, max=vocab_size - 1)
    inputs_embeds = input_embedding(input_ids)

    # 为 deepstack 预分配容器（仅在任一模态启用时）
    input_deepstack_embeds = None
    if any(use_deepstack.values()):
        # 兼容：没有暴露 deepstack_visual_indexes 时按 1 处理
        idxs = getattr(multimodal_model, "deepstack_visual_indexes", None)
        if idxs is None:
            num_deep = 1
        else:
            num_deep = idxs if isinstance(idxs, int) else len(idxs)
        deepstack_shape = inputs_embeds.shape[:-1] + (
            inputs_embeds.shape[-1] * num_deep,
        )
        input_deepstack_embeds = torch.zeros(
            deepstack_shape, device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        other_info["input_deepstack_embeds"] = input_deepstack_embeds

    # 4) scatter 到 text embedding（以及 deepstack 同步）
    for i in range(len(embeddings)):
        emb = embeddings[i]
        msk = masks[i]
        modality = modalities[i] if i < len(modalities) else None

        if emb is None or msk is None:
            continue

        # mask 形状健壮化
        if msk.dim() == 2 and msk.shape[-1] == 1:
            mflat = msk.squeeze(-1)
        else:
            mflat = msk
        if mflat.dim() != 1:
            mflat = mflat.view(-1)

        indices = torch.where(mflat.to(dtype=torch.bool))[0]

        if emb.shape[-1] != target_hidden:
            raise RuntimeError(
                f"[embed_mm_inputs] Embedding dim mismatch after alignment: "
                f"{emb.shape[-1]} vs hidden {target_hidden} (modality={modality})."
            )
        if emb.shape[0] != indices.shape[0]:
            # 与上游 zero padding 行为对齐，裁到最短
            min_len = min(emb.shape[0], indices.shape[0])
            emb = emb[:min_len]
            indices = indices[:min_len]

        inputs_embeds[indices] = emb.to(inputs_embeds.device, inputs_embeds.dtype)

        # deepstack 同步（仅当该模态启用）
        if input_deepstack_embeds is not None and use_deepstack.get(modality, False):
            deep = (
                deepstack_embeddings.pop(0) if len(deepstack_embeddings) > 0 else None
            )
            if deep is not None:
                if deep.shape[0] != len(indices):
                    min_len = min(deep.shape[0], len(indices))
                    deep = deep[:min_len]
                    indices = indices[:min_len]
                if deep.shape[-1] % target_hidden != 0:
                    raise RuntimeError(
                        f"[embed_mm_inputs] deepstack last dim {deep.shape[-1]} "
                        f"is not multiple of hidden {target_hidden}."
                    )
                input_deepstack_embeds[indices] = deep.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )

    return inputs_embeds, other_info


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
            inputs_embeds, other_info = embed_mm_inputs(
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
            inputs_embeds = embed_tokens(input_ids)
    else:
        inputs_embeds = None

    hidden_states = language_model(
        input_ids=None,
        forward_batch=forward_batch,
        input_embeds=inputs_embeds,
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


def data_hash(data: Any) -> int:
    """
    Robust hashing that returns an int (as previously expected by callers).
    - bytes/bytearray/memoryview: use directly
    - torch.Tensor: detach+cpu+contiguous, then view as uint8 to read raw bytes
    - np.ndarray: ascontiguousarray + view uint8 to read raw bytes
    - others: pickle.dumps fallback
    Finally: sha256(...).digest()[:8] interpreted as little-endian integer.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        buf = bytes(data)
    elif torch.is_tensor(data):
        t = data.detach().cpu().contiguous()
        # reinterpret as raw bytes regardless of dtype (supports bfloat16, float16, etc.)
        buf = t.view(torch.uint8).numpy().tobytes()
    elif isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data)
        buf = arr.view(np.uint8).tobytes()
    else:
        buf = pickle.dumps(data, protocol=4)

    digest8 = hashlib.sha256(buf).digest()[:8]
    return int.from_bytes(digest8, byteorder="little", signed=False)


def hash_feature(feature: Any) -> int:
    # keep API identical to before
    return data_hash(feature)


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

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Source: https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/llava/mm_utils.py
"""
Utilities for multi-modal models.

This python file mainly contains utilities that were used in the
image processing logic of llava-next including operations such as
anyres and anyres_max

Currently supports the anyres and anyres_max operation for CLIP and
SigLip. For more information, you may refer to the paper or the blog

LLaVA-NeXT : https://llava-vl.github.io/blog/2024-01-30-llava-next/
LLaVA-Onevision : https://arxiv.org/pdf/2408.03326

"""

import ast
import itertools
import math
import re
from io import BytesIO
from typing import Callable, List, Literal, Optional, Sequence

import numpy as np
import pybase64
import torch
from PIL import Image

from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import flatten_nested_list


def ensure_numpy(x):
    """Convert torch.Tensor to numpy array if needed (v5 compat)."""
    return x.numpy() if isinstance(x, torch.Tensor) else x


def has_valid_data(data) -> bool:
    if data is None:
        return False
    if isinstance(data, list):
        return any(has_valid_data(item) for item in flatten_nested_list(data))
    return True


def materialize_multimodal_features(
    features: Sequence[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Concatenate variable-length feature tensors into one destination buffer.

    A multimodal item can arrive as a CPU tensor, a CUDA-IPC reconstruction,
    or an already resident tensor with a different dtype.  Calling ``to`` on
    every item and then ``torch.cat`` creates one temporary tensor per item
    before allocating the final packed input.  Allocate the final buffer once
    and copy each item directly into its slice instead; ``copy_`` performs the
    required device and dtype conversion in the destination copy.

    All tensors must agree on dimensions after the leading token dimension.
    The leading dimension may differ because images commonly have different
    numbers of vision patches.
    """

    if not features:
        raise ValueError("features must contain at least one tensor")

    first = features[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(first)}")
    if first.ndim == 0:
        raise ValueError("multimodal feature tensors must have a leading dimension")
    trailing_shape = first.shape[1:]
    total_tokens = 0
    for feature in features:
        if not isinstance(feature, torch.Tensor):
            raise TypeError(f"expected torch.Tensor, got {type(feature)}")
        if feature.ndim == 0 or feature.shape[1:] != trailing_shape:
            raise ValueError(
                "multimodal feature tensors must have matching trailing shapes: "
                f"expected {trailing_shape}, got {feature.shape}"
            )
        total_tokens += feature.shape[0]

    output = torch.empty(
        (total_tokens, *trailing_shape),
        device=device,
        dtype=dtype,
    )
    offset = 0
    for feature in features:
        length = feature.shape[0]
        output[offset : offset + length].copy_(feature, non_blocking=True)
        offset += length
    return output


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )

        # Calculate effective and wasted resolutions
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [
            224,
            336,
            384,
            448,
            512,
        ], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [
            (i, j)
            for i in range(range_start[0], range_end[0] + 1)
            for j in range(range_start[1], range_end[1] + 1)
        ]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        np.array: An np array containing the processed image patches.
    """
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [
            224,
            336,
            384,
            448,
            512,
        ], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [
            (i, j)
            for i in range(range_start[0], range_end[0] + 1)
            for j in range(range_start[1], range_end[1] + 1)
        ]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    # For Siglip processor, only have size but no crop size.
    # In transformers v5, crop_size may exist but be None.
    crop_size = (
        processor.crop_size["height"]
        if getattr(processor, "crop_size", None) is not None
        else processor.size["height"]
    )
    shortest_edge = (
        processor.size["shortest_edge"]
        if "shortest_edge" in processor.size
        else processor.size["height"]
    )
    patches = divide_to_patches(image_padded, crop_size)

    image_original_resize = image.resize((shortest_edge, shortest_edge))

    image_patches = [image_original_resize] + patches
    image_patches = [
        processor.preprocess(image_patch.convert("RGB"))["pixel_values"][0]
        for image_patch in image_patches
    ]
    # In transformers v5, image processors may return torch.Tensor instead of numpy arrays
    image_patches = [ensure_numpy(p) for p in image_patches]
    return np.stack(image_patches, axis=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(pybase64.b64decode(image, validate=True)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    if pil_img.mode == "L":
        pil_img = pil_img.convert("RGB")
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


def unpad_image_shape(current_height, current_width, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image
    and returns the new shape.
    """
    original_width, original_height = original_size

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        new_shape = (current_height - 2 * padding, current_width)
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        new_shape = (current_height, current_width - 2 * padding)

    return new_shape


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
            image = image_processor.preprocess(image)["pixel_values"][0]
            new_images.append(image)
    elif "anyres" in image_aspect_ratio:
        for image in images:
            image = process_anyres_image(
                image, image_processor, model_cfg.image_grid_pinpoints
            )
            new_images.append(image)
    else:
        return image_processor(images)["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = np.stack(new_images, axis=0)
    return new_images


def dp_encoder_num_patches(grid_thw) -> int:
    """Canonical per-image load metric for DP-encoder load balancing.

    Both the scheduler-side owner assignment and the vision-model runner derive
    per-image load from this single definition so ownership can never diverge.
    """
    return int(math.prod(grid_thw))


def assign_dp_encoder_owner_ranks(sizes: List[int], tp_size: int) -> List[int]:
    """Return the owning attention-TP rank for each item.

    The single, authoritative DP-encoder ownership decision, computed once per
    request in the scheduler and stored as a persistent per-item tag. ``sizes``
    is the per-item load metric (:func:`dp_encoder_num_patches`); the result is
    deterministic given ``(sizes, tp_size)`` so every rank agrees without
    communication.
    """
    shuffle_indices, gpu_sample_counts, _ = get_dp_encoder_lb_assignment(sizes, tp_size)
    owner_ranks = [0] * len(sizes)
    pos = 0
    for rank in range(tp_size):
        for _ in range(gpu_sample_counts[rank]):
            owner_ranks[shuffle_indices[pos]] = rank
            pos += 1
    return owner_ranks


def _dp_encoder_owner_groups(
    owner_ranks: List[int], patches_per_image: List[int], tp_size: int
):
    """Group encoded items by their persistent owner tag (no load balancing).

    Returns the same ``(rank_image_order, gpu_sample_counts,
    grouped_pixel_values_len)`` triple as the legacy path, but derived purely
    from the fixed owner tags of the current encoded subset. Item indices within
    each rank stay ascending, matching the order features are concatenated into
    ``pixel_values``.
    """
    rank_image_order: List[List[int]] = [[] for _ in range(tp_size)]
    for i, rank in enumerate(owner_ranks):
        rank_image_order[rank].append(i)
    gpu_sample_counts = [len(idxs) for idxs in rank_image_order]
    grouped_pixel_values_len = [
        sum(patches_per_image[i] for i in idxs) for idxs in rank_image_order
    ]
    return rank_image_order, gpu_sample_counts, grouped_pixel_values_len


class DpEncoderDispatch:
    """Process-wide selector for the DP-encoder sharding strategy.

    A single global instance (:func:`get_dp_encoder_dispatch`) is the one place
    that decides which DP-encoder path is active, so scheduler tagging, the
    encode-path drop, and the runner consult the same switch. Lookups are
    read-through (``server_args`` and TP getters resolved lazily each call) and
    degrade to safe legacy behavior (disabled / tp_size 1) when runtime state is
    not ready.
    """

    @staticmethod
    def _server_args():
        try:
            from sglang.srt.runtime_context import get_server_args

            return get_server_args()
        except Exception:
            return None

    def tp_size(self) -> int:
        try:
            return get_parallel().attn_tp_size
        except Exception:
            return 1

    def tp_rank(self) -> int:
        try:
            return get_parallel().attn_tp_rank
        except Exception:
            return 0

    def shard_by_owner_enabled(self) -> bool:
        """True when the experimental owner-tag sharding path should be used.

        Requires ``mm_enable_dp_encoder`` AND ``mm_dp_encoder_shard_by_owner``
        AND an attention-TP world size > 1. When False, the DP encoder (if
        enabled at all) falls back to the legacy runner-side load balancing.
        """
        sa = self._server_args()
        if sa is None:
            return False
        if not getattr(sa, "mm_enable_dp_encoder", False):
            return False
        if not getattr(sa, "mm_dp_encoder_shard_by_owner", False):
            return False
        return self.tp_size() > 1


_dp_encoder_dispatch = DpEncoderDispatch()


def get_dp_encoder_dispatch() -> DpEncoderDispatch:
    """Return the process-wide :class:`DpEncoderDispatch` singleton."""
    return _dp_encoder_dispatch


def get_dp_encoder_lb_assignment(
    sizes: list[int],
    num_gpus: int = 2,
) -> tuple[list[int], list[int], list[int]]:
    """
    Generate load balancing assignment and metadata
    for distributing data across GPUs.
    The load is determined by the total image sizes,
    not the number of images.

    Args:
        sizes: The size of each image
        num_gpus: Number of GPUs to balance across

    Returns:
        shuffle_indices:
            Indices to reorder data for balanced loading
        gpu_sample_counts:
            Number of samples assigned to each GPU
        grouped_sizes_per_gpu:
            Total size assigned to each GPU

    Example:
        ```
        sizes = [1000, 100, 200, 50]
        num_gpus = 2
        ```

    """

    n_samples = len(sizes)

    # Handle edge cases
    if n_samples == 0:
        return [], [0] * num_gpus, [0] * num_gpus

    # Use greedy algorithm - balance by total size, not sample count
    gpu_assignments = [list[int]() for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus  # This tracks total SIZE, not sample count

    # Sort indices by size (largest first for better load balancing)
    # sizes = [1000, 100, 200, 50]
    # large_to_small_indices = [0, 2, 1, 3]
    large_to_small_indices = sorted(
        range(n_samples), key=lambda i: sizes[i], reverse=True
    )

    for idx in large_to_small_indices:
        # Find GPU with minimum current load (by total size)
        min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
        gpu_assignments[min_gpu].append(idx)
        gpu_loads[min_gpu] += sizes[idx]

    # Create shuffle indices and counts
    shuffle_indices = list[int]()
    gpu_sample_counts = list[int]()
    for gpu_id in range(num_gpus):
        # GPU_0 = [1000] = [0]
        # GPU_1 = [200, 100, 50] = [2, 1, 3]
        # shuffle_indices = [0, 2, 1, 3]
        shuffle_indices.extend(gpu_assignments[gpu_id])
        # GPU_0 = [1]
        # GPU_1 = [3]
        # gpu_sample_counts = [1, 3]
        gpu_sample_counts.append(len(gpu_assignments[gpu_id]))

    return (shuffle_indices, gpu_sample_counts, gpu_loads)


def _pad_mrope_vision_embeddings_for_tp_gather(
    image_embeds_local: torch.Tensor, max_len_per_rank: int
) -> torch.Tensor:
    """Pad the DP encoder output for a fixed-shape TP all-gather.

    Allocating the padding fragment and then concatenating it creates two
    temporary buffers on every underfilled rank. Allocate the final
    fixed-shape input directly and copy just the valid embeddings instead.
    """

    current_len = image_embeds_local.shape[0]
    if current_len >= max_len_per_rank:
        return image_embeds_local

    padded = torch.empty(
        (max_len_per_rank, *image_embeds_local.shape[1:]),
        dtype=image_embeds_local.dtype,
        device=image_embeds_local.device,
    )
    if current_len > 0:
        padded[:current_len].copy_(image_embeds_local)
    return padded


# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/vision.py
def run_dp_sharded_vision_model(
    image_input: torch.Tensor, vision_model: torch.nn.Module
) -> torch.Tensor:
    """Run a vision model with data parallelism (DP) sharding. The function
    will shard the input image tensor on the first dimension and run the vision
    model

    Args:
        image_input (torch.Tensor): Image input tensor.
        vision_model (torch.nn.Module): Vision model.
    Returns:
        torch.Tensor: Output image embeddings
    """

    num_chunks = image_input.shape[0]
    mp_world_size = get_parallel().tp_size
    num_chunks_per_rank = (num_chunks + mp_world_size - 1) // mp_world_size
    num_padded_chunks = num_chunks_per_rank * mp_world_size - num_chunks
    pad = (0,) * (2 * (image_input.dim() - 1)) + (0, num_padded_chunks)
    image_input_padded = torch.nn.functional.pad(image_input, pad)
    rank = get_parallel().tp_rank
    image_input_per_rank = image_input_padded[
        rank * num_chunks_per_rank : (rank + 1) * num_chunks_per_rank, ...
    ]

    vision_embeddings = vision_model(image_input_per_rank)
    # Ensure tensor is contiguous before all_gather
    vision_embeddings = vision_embeddings.last_hidden_state.contiguous()
    vision_embeddings = tensor_model_parallel_all_gather(vision_embeddings, dim=0)
    vision_embeddings = vision_embeddings[:num_chunks, ...]
    return vision_embeddings


# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/vision.py
def run_dp_sharded_mrope_vision_model(
    vision_model: torch.nn.Module,
    pixel_values: Optional[torch.Tensor],
    grid_thw_list: list,
    *,
    rope_type: Literal["rope_3d", "rope_2d", "rope_2d_packed"],
    owner_ranks: Optional[List[int]] = None,
    load_local_pixel_values: Optional[Callable[[list[int]], torch.Tensor]] = None,
    pixel_values_device: Optional[torch.device] = None,
    pixel_values_dtype: Optional[torch.dtype] = None,
):
    """Run a vision model with data parallelism (DP) sharding.
    The function will shard the input image tensor on the
    first dimension and run the vision model.
    This function is used to run the vision model with mrope.

    Args:
        vision_model (torch.nn.Module): Vision model.
        pixel_values (torch.Tensor): Image/Video input tensor. When
            ``owner_ranks`` is ``None`` this is the full concat of all items'
            patches (runner slices out this rank's shard); when provided it is
            already this rank's shard, in ascending item-index order.
        grid_thw_list: Grid dimensions for all images in the encoded subset (one
            row per item), used for output reassembly. When ``owner_ranks`` is
            provided, ``len(owner_ranks) == len(grid_thw_list)``.
        rope_type: Type of rope used in the vision model.
                   Different rope types have different dimension to do ViT.
                   "rope_3d" for 3D rope (e.g., Qwen2.5-VL)
                   "rope_2d" for 2D rope (e.g., Kimi-VL)
                   "rope_2d_packed" for packed 2D rope outputs that accept
                   ``grid_thws`` positionally (e.g., Kimi-K2.5/K2.7)
        owner_ranks: Optional per-item owner attention-TP rank (the persistent
            ``dp_encoder_owner_rank`` tag). When provided the runner groups the
            encoded subset by these fixed tags instead of recomputing a
            load-balancing assignment, treating ``pixel_values`` as this rank's
            shard.
    Returns:
        torch.Tensor: Output image embeddings

    Example:
        ```
        vision_model.out_hidden_size = 64
        vision_model.spatial_merge_size = 2
        pixel_values.shape = (1350, channel), or a local loader supplies
        per-image features after the data-parallel assignment is known.
        grid_thw_list = [[1, 10, 100], [1, 10, 10], [1, 10, 20], [1, 50]]
        tp_size = 2
        ```

    """
    if pixel_values is None and load_local_pixel_values is None:
        raise ValueError("pixel_values or load_local_pixel_values must be provided")

    input_device = (
        pixel_values.device if pixel_values is not None else pixel_values_device
    )
    input_dtype = pixel_values.dtype if pixel_values is not None else pixel_values_dtype
    if input_device is None or input_dtype is None:
        raise ValueError(
            "pixel_values_device and pixel_values_dtype are required with a local loader"
        )

    tp_size = get_parallel().attn_tp_size
    if tp_size == 1:
        if pixel_values is None:
            pixel_values = load_local_pixel_values(list(range(len(grid_thw_list))))
        grid_thw = torch.tensor(
            grid_thw_list,
            # MoonViT's 2D RoPE implementation combines the grid metadata
            # with CUDA activations. Keep the metadata colocated in that
            # path; other encoders retain their existing CPU contract.
            device=pixel_values.device if rope_type == "rope_2d" else None,
        )
        if rope_type == "rope_2d":
            image_embeds = vision_model(
                pixel_values,
                grid_hw=grid_thw,
                max_seqlen=max(math.prod(grid) for grid in grid_thw_list),
            )
            # MoonViT returns one tensor per image. The multi-GPU path below
            # already concatenates these tensors before returning, so keep the
            # TP=1 DP-encoder path on the same projector-facing contract.
            if isinstance(image_embeds, list):
                return torch.cat(image_embeds, dim=0)
            return image_embeds
        if rope_type == "rope_2d_packed":
            image_embeds = vision_model(pixel_values, grid_thw)
            if isinstance(image_embeds, list):
                return torch.cat(image_embeds, dim=0)
            return image_embeds
        return vision_model(pixel_values, grid_thw=grid_thw)

    # GPU_0 tp_rank_local = 0
    # GPU_1 tp_rank_local = 1
    tp_rank_local = get_parallel().attn_tp_rank

    patches_per_image = [dp_encoder_num_patches(g) for g in grid_thw_list]
    # patches_per_image = [0, 1000, 1100, 1300, 1350]
    cum_patches_per_image = [0, *itertools.accumulate(patches_per_image)]

    # Ownership is decided once in the scheduler and stored as a persistent
    # per-item tag. When the encode path forwards those tags (``owner_ranks``)
    # we group the encoded subset by them rather than recomputing a load-
    # balancing assignment, so ownership never diverges from the pre-H2D drop.
    # Otherwise (legacy / video path with a full ``pixel_values``) we compute
    # the assignment here.
    pixel_values_is_sharded = owner_ranks is not None
    if owner_ranks is not None:
        assert len(owner_ranks) == len(grid_thw_list), (
            f"owner_ranks has {len(owner_ranks)} entries but grid_thw_list has "
            f"{len(grid_thw_list)}; the tags and the encoded item list disagree."
        )
        rank_image_order, gpu_sample_counts, grouped_pixel_values_len = (
            _dp_encoder_owner_groups(owner_ranks, patches_per_image, tp_size)
        )
    else:
        image_to_tp_rank, gpu_sample_counts, grouped_pixel_values_len = (
            get_dp_encoder_lb_assignment(patches_per_image, tp_size)
        )
        cum_gpu_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]
        rank_image_order = [
            sorted(
                image_to_tp_rank[
                    cum_gpu_sample_counts[rank] : cum_gpu_sample_counts[rank + 1]
                ]
            )
            for rank in range(tp_size)
        ]

    image_idxs_local = rank_image_order[tp_rank_local]

    if pixel_values_is_sharded:
        # pixel_values is exactly this rank's shard (features for non-owned
        # items were dropped to None before H2D). Assert it lines up with the
        # owner-tag grouping.
        expected_local_patches = grouped_pixel_values_len[tp_rank_local]
        assert pixel_values.shape[0] == expected_local_patches, (
            f"pre-sharded pixel_values has {pixel_values.shape[0]} patches but the "
            f"DP-encoder owner tags expect {expected_local_patches} for this rank."
        )
        pixel_values_local = pixel_values
    elif len(image_idxs_local) > 0:
        if load_local_pixel_values is not None:
            pixel_values_local = load_local_pixel_values(image_idxs_local)
        else:
            assert pixel_values is not None
            pixel_values_local = torch.cat(
                [
                    pixel_values[
                        cum_patches_per_image[i] : cum_patches_per_image[i + 1]
                    ]
                    for i in image_idxs_local
                ]
            )
    else:
        pixel_values_local = None
    # embed_dim_reduction_factor = 2 * 2
    packed_2d_rope = rope_type in ("rope_2d", "rope_2d_packed")
    if packed_2d_rope:
        embed_dim_reduction_factor = (
            vision_model.merge_kernel_size[0] * vision_model.merge_kernel_size[1]
        )
    else:
        embed_dim_reduction_factor = (
            vision_model.spatial_merge_size * vision_model.spatial_merge_size
        )

    # Find the max length across all ranks
    # The output embedding of every DP rank has to be
    # padded to this length for tensor_model_parallel_all_gather
    # to work
    max_len_per_rank = max(grouped_pixel_values_len) // embed_dim_reduction_factor
    local_grid_thw_list = [grid_thw_list[i] for i in image_idxs_local]

    # Run the vision model on the local pixel_values_local
    if packed_2d_rope:
        if pixel_values_local is not None and pixel_values_local.shape[0] > 0:
            local_grid_thw = torch.tensor(
                local_grid_thw_list, device=pixel_values_local.device
            )
            if rope_type == "rope_2d":
                image_embeds_local = vision_model(
                    pixel_values_local,
                    grid_hw=local_grid_thw,
                    max_seqlen=max(math.prod(grid) for grid in local_grid_thw_list),
                )
            else:
                image_embeds_local = vision_model(pixel_values_local, local_grid_thw)
            if isinstance(image_embeds_local, list):
                image_embeds_local = torch.cat(image_embeds_local, dim=0)
        else:
            out_dim = getattr(vision_model.config, "hidden_size", None)
            image_embeds_local = torch.empty(
                (0, embed_dim_reduction_factor, out_dim),
                device=input_device,
                dtype=input_dtype,
            )
    else:
        if pixel_values_local is not None and pixel_values_local.shape[0] > 0:
            # print(f"{local_grid_thw_list = }", flush=True)
            image_embeds_local = vision_model(
                pixel_values_local, torch.tensor(local_grid_thw_list)
            )
            if isinstance(image_embeds_local, list):
                image_embeds_local = torch.cat(image_embeds_local, dim=0)
        else:
            # Handle empty case
            out_dim = getattr(vision_model, "out_hidden_size", None)
            if out_dim is None:
                out_dim = vision_model.config.hidden_size
            image_embeds_local = torch.empty(
                (0, out_dim),
                device=input_device,
                dtype=input_dtype,
            )

    # The TP all-gather needs a common first dimension. Allocate that final
    # shape directly instead of materializing a padding fragment and catting it.
    image_embeds_local_padded = _pad_mrope_vision_embeddings_for_tp_gather(
        image_embeds_local, max_len_per_rank
    )

    # Do all_gather to collect embeddings from all ranks
    gathered_embeds = get_parallel().attn_tp_group.all_gather(
        image_embeds_local_padded, dim=0
    )

    # Remove padding and reconstruct per-rank embeddings
    rank_embeddings = list[torch.Tensor]()
    for rank in range(tp_size):
        start_idx = rank * max_len_per_rank
        end_idx = start_idx + (
            grouped_pixel_values_len[rank] // embed_dim_reduction_factor
        )
        rank_embeddings.append(gathered_embeds[start_idx:end_idx])

    patches_per_output_image = [
        (patch_size // embed_dim_reduction_factor) for patch_size in patches_per_image
    ]

    # Reconstruct embeddings in the original order
    original_order_embeddings = [None] * len(grid_thw_list)
    for rank in range(tp_size):
        count = gpu_sample_counts[rank]
        if count > 0:
            # Get images assigned to this rank in canonical (sorted) order,
            # matching the within-rank order used to build pixel_values_local.
            rank_images = rank_image_order[rank]

            rank_embed = rank_embeddings[rank]
            # Split rank embeddings back to individual images
            embed_start = 0
            for img_idx in rank_images:
                img_patches = patches_per_output_image[img_idx]
                original_order_embeddings[img_idx] = rank_embed[
                    embed_start : embed_start + img_patches
                ]
                embed_start += img_patches
    out_embeddings = torch.cat(original_order_embeddings, dim=0)
    return out_embeddings

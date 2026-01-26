"""Multimodal scheduler policies for handling image/video requests across encoder instances."""

from __future__ import annotations

import itertools
import logging
import math
from enum import Enum
from typing import Callable

import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather

logger = logging.getLogger(__name__)


class MMPackPolicy(Enum):
    """Strategies for packing multiple images/videos in a single encoder instance."""

    ALL_PACK = "all_pack"  # Pack all images/videos together
    PROFILE_PACK = "profile_pack"  # Use offline profile data for optimal packing


class MMDPSchedulePolicy(Enum):
    """Strategies for distributing multiple images/videos to different encoder instances to run vit."""

    LOAD_BALANCE = "load_balance"


class MMScheduler:
    """
    Multi-modal Scheduler, use MMDPSchedulePolicy to distribute multiple images/videos to different encoder instances and
    MMPackPolicy to pack multiple images/videos in a single encoder instance.
    """

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

    @classmethod
    def get_dp_encoder_assignment(
        self,
        grid_thw_list: list,
        schedule_policy: MMDPSchedulePolicy = MMDPSchedulePolicy.LOAD_BALANCE,
    ):
        """
        Assign images/videos to different dp rank according to MMDPSchedulePolicy, and calculate some metadata info for later use

        Args:
        grid_thw_list: List of grid dimensions for each image
        schedule_policy: MMDPSchedulePolicy
        Returns:
            patches_per_image:
                Patch num for each image
            image_to_tp_rank:
                Image indices reordered according to its assigned DP rank, needs to be used in conjunction with gpu_sample_counts
            gpu_sample_counts:
                Number of samples assigned to each GPU
            grouped_sizes_per_gpu:
                Total size assigned to each GPU

        Example:
            ```
            grid_thw_list = [[1, 10, 100], [1, 10, 10], [1, 10, 20], [1, 50]]
            ```
        """
        tp_size = get_tensor_model_parallel_world_size()

        # patches_per_image = [1000, 100, 200, 50]
        patches_per_image = [math.prod(grid_thw) for grid_thw in grid_thw_list]
        # print(f"{patches_per_image = }")

        # Get load balancing assignment with all metadata
        # image_to_tp_rank = [0, 2, 1, 3]
        # gpu_sample_counts = [1, 3]
        # grouped_pixel_values_len = [1000, 350]
        if schedule_policy == MMDPSchedulePolicy.LOAD_BALANCE:
            (image_to_tp_rank, gpu_sample_counts, grouped_pixel_values_len) = (
                self.get_dp_encoder_lb_assignment(patches_per_image, tp_size)
            )

            return (
                patches_per_image,
                image_to_tp_rank,
                gpu_sample_counts,
                grouped_pixel_values_len,
            )

        else:
            logger.warning(
                "MMDPScheduler only supports MMDPSchedulePolicy.LOAD_BALANCE now"
            )
            return [], [], [], []

    @classmethod
    def split_mm_to_dp_encoder(
        self,
        pixel_values: torch.Tensor,
        grid_thw_list: list,
        patches_per_image: list[int],
        image_to_tp_rank: list[int],
        gpu_sample_counts: list[int],
    ):
        """
        Split mm data to different DP rank according to assignment result

        Args:
            pixel_values: Image/Video input tensor.
            grid_thw_list: List of grid dimensions for each image
            patches_per_image: Patch num for each image
            image_to_tp_rank: Image indices reordered according to its assigned DP rank, needs to be used in conjunction with gpu_sample_counts
            gpu_sample_counts: Number of samples assigned to each GPU
        Returns:
            pixel_values_local: Image/Video input tensor for local DP rank
            local_grid_thw_list: List of grid dimensions for each image on local DP rank

        Example:
            ```
            pixel_values.shape = (1350, channel)
            image_to_tp_rank = [0, 2, 1, 3]
            gpu_sample_counts = [1, 3]
            ```
        """
        # GPU_0 tp_rank_local = 0
        # GPU_1 tp_rank_local = 1
        tp_rank_local = get_tensor_model_parallel_rank()

        # cu_gpu_sample_counts = [0, 1, 4]
        cum_gpu_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]

        # GPU_0 image_idxs_local = [0]
        # GPU_1 image_idxs_local = [2, 1, 3]
        image_idxs_local = image_to_tp_rank[
            cum_gpu_sample_counts[tp_rank_local] : cum_gpu_sample_counts[
                tp_rank_local + 1
            ]
        ]

        # cum_patches_per_image = [0, 1000, 1100, 1300, 1350]
        cum_patches_per_image = [0, *itertools.accumulate(patches_per_image)]
        # Get the pixel values for the local images based on the image_idxs_local
        if len(image_idxs_local) > 0:
            pixel_values_local = torch.cat(
                [
                    pixel_values[
                        cum_patches_per_image[i] : cum_patches_per_image[i + 1]
                    ]
                    for i in image_idxs_local
                ]
            )
        else:
            # Handle case where this rank has no images
            pixel_values_local = torch.empty(
                (0, pixel_values.shape[1]),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )

        local_grid_thw_list = [grid_thw_list[i] for i in image_idxs_local]

        return pixel_values_local, local_grid_thw_list

    @classmethod
    def get_mm_pack_result(
        self,
        pixel_values: torch.Tensor,
        grid_thw_list: list,
        modality,
        schedule_policy: MMPackPolicy = MMPackPolicy.ALL_PACK,
    ):
        """
        Pack images/videos in a single encoder instance to multiple computing groups according to MMPackPolicy

        Args:
            pixel_values: Image/Video input tensor for local DP rank
            grid_thw_list: List of grid dimensions for each image on local DP rank
            modality: Modality of the input data
            schedule_policy: MMPackPolicy
        Returns:
            embedding_items_list_local_rank:
                List of MultimodalDataItem to be computed for local DP rank, each item represents a computing group

        Example:
            ```
            pixel_values.shape = (350, channel)
            grid_thw_list = [[1, 10, 10], [1, 10, 20], [1, 50]]
            ```
        """
        from sglang.srt.managers.mm_utils import MultimodalDataItem

        if schedule_policy == MMPackPolicy.ALL_PACK:
            return [
                [
                    MultimodalDataItem(
                        modality=modality,
                        feature=pixel_values,
                        model_specific_data={
                            "image_grid_thw": torch.tensor(grid_thw_list)
                        },
                    )
                ]
            ]
        elif schedule_policy == MMPackPolicy.PROFILE_PACK:
            # TODO
            logger.warning("MMDPScheduler only supports MMPackPolicy.ALL_PACK now")
        return [
            [
                MultimodalDataItem(
                    modality=modality,
                    feature=pixel_values,
                    model_specific_data={"image_grid_thw": torch.tensor(grid_thw_list)},
                )
            ]
        ]

    @classmethod
    def gather_dp_result(
        self,
        patches_per_image: list[int],
        image_to_tp_rank: list[int],
        gpu_sample_counts: list[int],
        grouped_pixel_values_len: list[int],
        grid_thw_list: list[int],
        image_embeds_local: torch.Tensor,
        get_mm_dp_metadata_func: Callable,
    ):
        """
        Gather mm embeddings from different dp rank to local dp rank

        Args:
            patches_per_image: Patch num for each image
            image_to_tp_rank: Image indices reordered according to its assigned DP rank, needs to be used in conjunction with gpu_sample_counts
            gpu_sample_counts: Number of samples assigned to each GPU
            grouped_pixel_values_len: Total size assigned to each GPU
            grid_thw_list: List of grid dimensions for all image
            image_embeds_local: Image/Video embeddings for local DP rank
            get_mm_dp_metadata_func: Function to get mm dp metadata, can be used to get rope_type and embed_dim_reduction_factor of different model
        Returns:
            embeddings_all_rank: Embeddings for all input images/videos in original order

        Example:
            ```
            patches_per_image = [1000, 100, 200, 50]
            image_to_tp_rank = [0, 2, 1, 3]
            gpu_sample_counts = [1, 3]
            grouped_pixel_values_len = [1000, 350]
            grid_thw_list = [[1, 10, 100], [1, 10, 10], [1, 10, 20], [1, 50]]
            ```
        """
        embed_dim_reduction_factor, rope_type = get_mm_dp_metadata_func()

        # Pad the output based on max_len_per_rank
        # for tensor_model_parallel_all_gather to work
        max_len_per_rank = max(grouped_pixel_values_len) // embed_dim_reduction_factor
        current_len = image_embeds_local.shape[0]
        if current_len < max_len_per_rank:
            padding_size = max_len_per_rank - current_len
            if rope_type == "rope_2d":
                padding = torch.empty(
                    (
                        padding_size,
                        image_embeds_local.shape[1],
                        image_embeds_local.shape[2],
                    ),
                    dtype=image_embeds_local.dtype,
                    device=image_embeds_local.device,
                )
            else:
                padding = torch.empty(
                    (padding_size, image_embeds_local.shape[1]),
                    dtype=image_embeds_local.dtype,
                    device=image_embeds_local.device,
                )
            image_embeds_local_padded = torch.cat([image_embeds_local, padding], dim=0)
        else:
            image_embeds_local_padded = image_embeds_local

        # Do all_gather to collect embeddings from all ranks
        gathered_embeds = tensor_model_parallel_all_gather(
            image_embeds_local_padded, dim=0
        )

        # Remove padding and reconstruct per-rank embeddings
        tp_size = get_tensor_model_parallel_world_size()

        rank_embeddings = list[torch.Tensor]()
        for rank in range(tp_size):
            start_idx = rank * max_len_per_rank
            end_idx = start_idx + (
                grouped_pixel_values_len[rank] // embed_dim_reduction_factor
            )
            rank_embeddings.append(gathered_embeds[start_idx:end_idx])

        embedding_len_per_output_image = [
            (patch_size // embed_dim_reduction_factor)
            for patch_size in patches_per_image
        ]

        # Reconstruct embeddings in the original order
        original_order_embeddings = [None] * len(grid_thw_list)
        current_idx = 0
        for rank in range(tp_size):
            count = gpu_sample_counts[rank]
            if count > 0:
                # Get images assigned to this rank in shuffled order
                # GPU_0 = image_idxs_local  [0]
                # GPU_1 = image_idxs_local  [2, 1, 3]
                rank_images = image_to_tp_rank[current_idx : current_idx + count]

                rank_embed = rank_embeddings[rank]
                # Split rank embeddings back to individual images
                embed_start = 0
                for img_idx in rank_images:
                    img_embedding_len = embedding_len_per_output_image[img_idx]
                    original_order_embeddings[img_idx] = rank_embed[
                        embed_start : embed_start + img_embedding_len
                    ]
                    embed_start += img_embedding_len
                current_idx += count
        out_embeddings = torch.cat(original_order_embeddings, dim=0)
        return out_embeddings

# Copyright 2025 SGLang Team
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
"""Search-space construction for the block-wise W8A8 GEMM tuner.

This module deliberately has no torch/triton dependency so the tile selection
can be unit-tested on a CPU-only machine.
"""

from typing import Dict, List

# K-dimension tile candidates for the CUDA kernels. A tile has to divide the
# quantization block's K dimension so that every stored block scale covers a
# whole number of K tiles.
_DEFAULT_CUDA_BLOCK_K_TILES = (64, 128)
_SMALL_CUDA_BLOCK_K_TILES = (32, 16)


def get_cuda_block_k_tiles(block_k: int) -> List[int]:
    """Return the K tile sizes to try for a quantization block of ``block_k``.

    The historical search space of 64/128 is kept exactly whenever it is
    valid. Smaller quantization blocks (for example the ``[32, 32]`` recipe)
    fall back to the largest supported tile that divides ``block_k``, which
    keeps the search space non-empty instead of filtering every candidate out.
    """
    if block_k <= 0:
        raise ValueError(f"block_k must be positive, got {block_k}")

    default_tiles = [
        tile for tile in _DEFAULT_CUDA_BLOCK_K_TILES if block_k % tile == 0
    ]
    if default_tiles:
        return default_tiles

    for tile in _SMALL_CUDA_BLOCK_K_TILES:
        if block_k % tile == 0:
            return [tile]

    raise ValueError(
        f"block_k={block_k} has no supported CUDA K tile. Expected a multiple "
        "of 16 (for example 16, 32, 64 or 128)."
    )


def get_rocm_configs_compute_bound() -> List[Dict[str, int]]:
    configs = []
    waves_per_eu_range = 0
    for num_stages in [2]:
        for block_m in [32, 64, 128, 256]:
            for block_k in [32, 64, 128, 256]:
                for block_n in [16, 32, 64, 128, 256]:
                    for num_warps in [4, 8]:
                        for group_size in [1, 4, 8, 16, 32]:
                            configs.append(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": group_size,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                    "waves_per_eu": waves_per_eu_range,
                                }
                            )
    return configs


def get_configs_compute_bound(is_hip: bool, block_k: int) -> List[Dict[str, int]]:
    """Build the candidate kernel configurations for the given block size."""
    if is_hip:
        return get_rocm_configs_compute_bound()

    # Only try K tiles that can actually divide the quantization block. This
    # keeps the list non-empty for small blocks such as [32, 32].
    k_tiles = get_cuda_block_k_tiles(block_k)
    configs = []
    for num_stages in [2, 3, 4, 5]:
        for block_m in [16, 32, 64, 128, 256]:
            for block_size_k in k_tiles:
                for block_n in [32, 64, 128, 256]:
                    for num_warps in [4, 8]:
                        for group_size in [1, 16, 32, 64]:
                            configs.append(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_size_k,
                                    "GROUP_SIZE_M": group_size,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                }
                            )
    return configs

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from sglang.srt.models.constant_size_cache import PAD_SLOT_ID, ConstantSizeCache


@dataclass
class MinimaxCacheParams:
    minimax_cache: torch.Tensor = torch.Tensor()
    state_indices_tensor: torch.Tensor = torch.Tensor()

    def at_layer_idx(self, layer_idx):
        return MinimaxCacheParams(
            self.minimax_cache[layer_idx, ...], self.state_indices_tensor
        )


class MinimaxCacheManager(ConstantSizeCache):

    def __init__(self, dtype, cache_shape):
        super().__init__(cache_shape[1])  # max_batch_size is cache_shape[1]
        self._minimax_cache = torch.empty(size=cache_shape, dtype=dtype, device="cuda")

        # Pre-allocate state_indices_tensor buffers for different batch sizes
        # to avoid creating tensors during CUDA graph capture
        self._max_batch_size = cache_shape[1]
        self._state_indices_buffers = {}

        # Pre-create buffers for all possible batch sizes up to max_batch_size
        # This ensures we never need to create tensors during CUDA graph capture
        for batch_size in range(1, self._max_batch_size + 1):
            self._state_indices_buffers[batch_size] = torch.full(
                (batch_size,), PAD_SLOT_ID, dtype=torch.int32, device="cuda"
            )

    @property
    def cache(self):
        return self._minimax_cache

    def _copy_cache(self, from_index: int, to_index: int):
        assert len(self.cache) > 0
        self.cache[:, to_index].copy_(self.cache[:, from_index], non_blocking=True)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        """
        Provide the CUDA graph capture runs with a buffer in adjusted size.
        The buffer is used to maintain the Cache during the CUDA graph replay
        runs.
        """
        # Use pre-allocated buffer - this should always exist since we pre-allocate
        # all possible batch sizes during initialization
        if batch_size in self._state_indices_buffers:
            state_indices_tensor = self._state_indices_buffers[batch_size]
        else:
            # This should never happen if batch_size <= _max_batch_size
            raise RuntimeError(
                f"Requested batch_size {batch_size} is larger than max_batch_size "
                f"{self._max_batch_size}. Cannot create tensor during CUDA graph capture."
            )

        return (self.cache, state_indices_tensor)

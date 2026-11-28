# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

import torch

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.device import is_host_cpu_arm64


class ArmAttentionBackend(TorchNativeAttnBackend):
    """Arm attention backend using PyTorch native attention."""

    def __init__(self, model_runner):
        if model_runner.device != "cpu":
            raise ValueError("ArmAttentionBackend requires --device cpu.")
        if not is_host_cpu_arm64():
            raise ValueError("ArmAttentionBackend requires an Arm64 CPU.")
        super().__init__(model_runner)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize metadata for eager forward and CPU graph replay."""
        self._init_swa_forward_metadata(forward_batch.out_cache_loc)

    def _init_swa_forward_metadata(self, out_cache_loc: torch.Tensor | None) -> None:
        """Initialize sliding-window KV write locations for a forward pass."""
        if self.use_sliding_window_kv_pool and out_cache_loc is not None:
            self.swa_out_cache_loc = self.runner_ids.ids_in(
                IdSpaceKind.SLIDING_WINDOW, out_cache_loc
            )
        else:
            self.swa_out_cache_loc = None

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        # Native attention needs no additional CPU graph workspace.
        pass

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        self._init_swa_forward_metadata(out_cache_loc)

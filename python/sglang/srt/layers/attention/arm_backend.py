# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

import torch

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend


class ArmAttentionBackend(TorchNativeAttnBackend):
    """Arm attention backend using PyTorch native attention."""

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
        pass

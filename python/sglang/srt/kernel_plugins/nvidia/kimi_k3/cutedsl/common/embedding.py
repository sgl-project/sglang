# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Local CuTe vocabulary gather plus the ordinary framework TP all-reduce."""

import torch

from sglang.kernels.jit.kimi_k3.host import architecture
from sglang.kernels.ops.embeddings.kimi_k3_specialized.embedding import compile_gather
from sglang.srt.distributed.parallel_state import get_tp_group

SGLANG_READS = {
    "srt/distributed/parallel_state.py": ("get_tp_group",),
}


class PreparedEmbedding:
    def __init__(self, embedding, tp, rank):
        weight = embedding.weight
        # The gather moves 7 x 1024 BF16 columns of contiguous rows.
        if weight.dim() != 2 or weight.shape[1] != 7168 or not weight.is_contiguous():
            raise ValueError(
                f"embedding shard must be a contiguous [V, 7168] tensor, got "
                f"{tuple(weight.shape)}"
            )
        if weight.dtype != torch.bfloat16:
            raise ValueError(f"embedding shard must be BF16, got {weight.dtype}")
        self.group, self.weight = get_tp_group(), weight.detach()
        self.start, self.vocab = rank * weight.shape[0], embedding.org_vocab_size
        with torch.accelerator.device_index(weight.device.index):
            arch, ids = architecture(weight.device), (torch.int32, torch.int64)
            self.kernels = {t: compile_gather(t, weight.shape[0], arch) for t in ids}

    def run(self, ids, out):
        if ids.dim() != 1 or ids.dtype not in self.kernels:
            raise ValueError(
                f"token ids must be 1-D int32/int64, got {ids.dtype} {tuple(ids.shape)}"
            )
        if out.shape != (ids.shape[0], 7168) or not out.is_contiguous():
            raise ValueError(
                f"embedding output must be contiguous [{ids.shape[0]}, 7168]"
            )
        self.kernels[ids.dtype](ids, self.weight, out, self.start, self.vocab)
        reduced = self.group.all_reduce(out)
        if reduced is not out:
            out.copy_(reduced)
        return out

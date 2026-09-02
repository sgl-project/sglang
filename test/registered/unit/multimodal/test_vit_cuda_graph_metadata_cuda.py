import sys

import pytest
import torch
from torch import nn

from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


class _BoundaryBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = type(
            "AttentionConfig",
            (),
            {
                "num_attention_heads_per_partition": 1,
                "head_size": 1,
                "qkv_backend_name": "triton_attn",
            },
        )()

    def forward(
        self,
        x,
        *,
        cu_seqlens,
        position_embeddings,
        output_ws=None,
    ):
        boundary = cu_seqlens[0][1].to(x.dtype)
        position = position_embeddings[0][: x.shape[0], :1].unsqueeze(1)
        return x + boundary + position


class _Merger(nn.Module):
    def forward(self, x):
        return x.squeeze(1)


class _VisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_BoundaryBlock()])
        self.merger = _Merger()
        self.use_data_parallel = True
        self.deepstack_visual_indexes = []
        self.deepstack_merger_list = None
        self.max_context_len = None
        self.register_buffer("anchor", torch.empty(0, device="cuda"))

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return torch.float32


def test_vit_graph_replays_current_attention_and_position_metadata():
    runner = ViTCudaGraphRunner(_VisionTower())

    def run(seq_len, boundaries, position):
        x = torch.zeros(seq_len, 1, device="cuda")
        cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
        positions = torch.full((seq_len, 1), position, device="cuda")
        output = runner.run(x, cu_seqlens, None, (positions, positions))
        torch.cuda.synchronize()
        return output.cpu()

    first = run(4, [0, 2, 4], 1)
    different_layout = run(4, [0, 1, 4], 1)
    run(8, [0, 8], 7)
    small_after_growth = run(4, [0, 2, 4], 5)

    torch.testing.assert_close(first, torch.full_like(first, 3))
    torch.testing.assert_close(different_layout, torch.full_like(different_layout, 2))
    torch.testing.assert_close(
        small_after_growth, torch.full_like(small_after_growth, 7)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

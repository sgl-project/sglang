# SPDX-License-Identifier: Apache-2.0


from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    scatter_mamba_states_after_mtp_verify,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.mamba_checkpoint_pool import MambaCheckpointPool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def enable_fused_codec():
    with envs.SGLANG_EXPERIMENTAL_FUSED_INT8_CHECKPOINT.override(True):
        yield


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tracking", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_verify_commit_checkpoint_restore(dtype, tracking, graph_mode):
    torch.manual_seed(901)
    num_layers, n, h, v, k, c, d = 3, 4, 3, 17, 33, 12, 8
    device = "cuda"
    temporal = torch.full((num_layers, c, h, v, k), -5.0, dtype=dtype, device=device)
    intermediate = (
        torch.randn((num_layers, n, d, h, v, k), dtype=dtype, device=device) * 0.1
    )
    conv = torch.full((num_layers, c, 10, 3), -5.0, dtype=dtype, device=device)
    interm_conv = torch.randn((num_layers, n, d, 10, 3), dtype=dtype, device=device)
    state = SimpleNamespace(
        temporal=temporal,
        intermediate_ssm=intermediate,
        conv=[conv],
        intermediate_conv_window=[interm_conv],
    )
    active_idx = torch.tensor([1, 4, 6, 8], device=device)
    steps = torch.tensor([0, 3, 7, 1], device=device)
    track_idx = torch.tensor([2, 5, 7, 9], device=device) if tracking else None
    track_steps = torch.tensor([-1, 2, -1, 0], device=device) if tracking else None
    selected = (
        torch.tensor([1, 4, 6, 8, 5, 9], device=device) if tracking else active_idx
    )
    ckpt_idx = (
        torch.tensor([11, 2, 7, 5, 1, 9], device=device)
        if tracking
        else torch.tensor([11, 2, 7, 5], device=device)
    )
    restored_idx = (
        torch.tensor([3, 8, 1, 7, 5, 10], device=device)
        if tracking
        else torch.tensor([3, 8, 1, 7], device=device)
    )
    checkpoint = MambaCheckpointPool(
        num_layers=num_layers,
        num_slots=12,
        num_heads=h,
        head_v_dim=v,
        head_k_dim=k,
        conv_shapes=[(10, 3)],
        conv_dtype=dtype,
        temporal_dtype=dtype,
        device=device,
    )
    active = SimpleNamespace(mamba_cache=state)
    restored = SimpleNamespace(
        mamba_cache=SimpleNamespace(
            temporal=torch.zeros_like(temporal), conv=[torch.zeros_like(conv)]
        )
    )

    def pipeline():
        scatter_mamba_states_after_mtp_verify(
            state, active_idx, steps, track_idx, track_steps
        )
        checkpoint.store_from_active(active, selected, ckpt_idx)
        checkpoint.load_to_active(restored, ckpt_idx, restored_idx)

    pipeline()
    torch.cuda.synchronize()
    if graph_mode:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            pipeline()
        intermediate.mul_(0.75)
        interm_conv.mul_(0.75)
        g.replay()
        torch.cuda.synchronize()
    expected = torch.full_like(temporal, -5.0)
    expected_conv = torch.full_like(conv, -5.0)
    for j in range(n):
        expected[:, active_idx[j]] = intermediate[:, j, steps[j]]
        expected_conv[:, active_idx[j]] = interm_conv[:, j, steps[j]]
        if tracking and track_steps[j] >= 0:
            expected[:, track_idx[j]] = intermediate[:, j, track_steps[j]]
            expected_conv[:, track_idx[j]] = interm_conv[:, j, track_steps[j]]
    torch.testing.assert_close(temporal, expected, rtol=0, atol=0)
    torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
    q, s = checkpoint.temporal.quantize(expected[:, selected])
    torch.testing.assert_close(
        checkpoint.temporal.qdata[:, ckpt_idx], q, rtol=0, atol=0
    )
    torch.testing.assert_close(
        checkpoint.temporal.scale[:, ckpt_idx], s, rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored.mamba_cache.temporal[:, restored_idx],
        checkpoint.temporal.dequantize(q, s, dtype),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        restored.mamba_cache.conv[0][:, restored_idx],
        expected_conv[:, selected],
        rtol=0,
        atol=0,
    )


if __name__ == "__main__":
    import sys

    args = [arg for arg in sys.argv[1:] if arg != "-f"]
    raise SystemExit(pytest.main([__file__, "-v", *args]))

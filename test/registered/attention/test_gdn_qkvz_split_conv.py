import sys

import pytest
import torch

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvz_split_conv1d_update_contiguous,
    fused_qkvzba_split_reshape_cat_contiguous,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    PAD_SLOT_ID,
    causal_conv1d_update,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")

QK_HEADS = 8
V_HEADS = 16
HEAD_DIM = 128
QKV_DIM = 4096
Z_DIM = 2048
BA_DIM = 32


def _make_inputs(batch: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    projected_qkvz = torch.randn(
        batch,
        QKV_DIM + Z_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    projected_ba = torch.randn(
        batch,
        BA_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    weight = torch.randn(
        QKV_DIM,
        4,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    storage = torch.randn(
        batch + 9,
        3,
        QKV_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    state = storage.transpose(1, 2)
    cache_indices = torch.randperm(
        batch + 9,
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )[:batch]
    if batch > 1:
        cache_indices[-1] = PAD_SLOT_ID
    return projected_qkvz, projected_ba, weight, state, cache_indices


def _reference(projected_qkvz, projected_ba, weight, state, cache_indices):
    mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
        projected_qkvz,
        projected_ba,
        QK_HEADS,
        V_HEADS,
        HEAD_DIM,
        HEAD_DIM,
    )
    mixed_qkv = causal_conv1d_update(
        mixed_qkv,
        state,
        weight,
        activation="silu",
        conv_state_indices=cache_indices,
    )
    return mixed_qkv, z, b, a


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch", [1, 32])
def test_matches_split_then_conv(batch):
    projected_qkvz, projected_ba, weight, state, cache_indices = _make_inputs(
        batch, seed=batch
    )
    reference_state = state.clone(memory_format=torch.preserve_format)
    mixed_ref, z_ref, b_ref, a_ref = _reference(
        projected_qkvz,
        projected_ba,
        weight,
        reference_state,
        cache_indices,
    )

    mixed = torch.empty_like(mixed_ref)
    z = torch.empty_like(z_ref)
    b = torch.empty_like(b_ref)
    a = torch.empty_like(a_ref)
    fused_qkvz_split_conv1d_update_contiguous(
        projected_qkvz,
        projected_ba,
        mixed,
        z,
        b,
        a,
        state,
        weight,
        cache_indices,
        PAD_SLOT_ID,
    )

    valid = cache_indices != PAD_SLOT_ID
    torch.testing.assert_close(mixed[valid], mixed_ref[valid], atol=0, rtol=0)
    torch.testing.assert_close(z, z_ref, atol=0, rtol=0)
    torch.testing.assert_close(b, b_ref, atol=0, rtol=0)
    torch.testing.assert_close(a, a_ref, atol=0, rtol=0)
    torch.testing.assert_close(state, reference_state, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_replay():
    batch = 32
    projected_qkvz, projected_ba, weight, state, cache_indices = _make_inputs(
        batch, seed=101
    )
    initial_state = state.clone(memory_format=torch.preserve_format)
    mixed = torch.empty(batch, QKV_DIM, dtype=torch.bfloat16, device="cuda")
    z = torch.empty(batch, V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    b = torch.empty(batch, V_HEADS, dtype=torch.bfloat16, device="cuda")
    a = torch.empty_like(b)

    fused_qkvz_split_conv1d_update_contiguous(
        projected_qkvz,
        projected_ba,
        mixed,
        z,
        b,
        a,
        state,
        weight,
        cache_indices,
        PAD_SLOT_ID,
    )
    state.copy_(initial_state)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_qkvz_split_conv1d_update_contiguous(
            projected_qkvz,
            projected_ba,
            mixed,
            z,
            b,
            a,
            state,
            weight,
            cache_indices,
            PAD_SLOT_ID,
        )

    new_qkvz, new_ba, new_weight, new_state, _ = _make_inputs(batch, seed=202)
    projected_qkvz.copy_(new_qkvz)
    projected_ba.copy_(new_ba)
    weight.copy_(new_weight)
    state.copy_(new_state)
    reference_state = new_state.clone(memory_format=torch.preserve_format)
    mixed_ref, z_ref, b_ref, a_ref = _reference(
        projected_qkvz,
        projected_ba,
        weight,
        reference_state,
        cache_indices,
    )
    graph.replay()
    torch.cuda.synchronize()

    valid = cache_indices != PAD_SLOT_ID
    torch.testing.assert_close(mixed[valid], mixed_ref[valid], atol=0, rtol=0)
    torch.testing.assert_close(z, z_ref, atol=0, rtol=0)
    torch.testing.assert_close(b, b_ref, atol=0, rtol=0)
    torch.testing.assert_close(a, a_ref, atol=0, rtol=0)
    torch.testing.assert_close(state, reference_state, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

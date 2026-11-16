"""``page_unified`` Mamba state transfers through the piece-table kernel."""

import sys

import pytest
import torch

from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.page_unified import (
    build_mamba_page_unified_layout,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="nightly-amd-kernel-1-gpu", nightly=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Mamba transfer kernel tests require CUDA."
)

LAYERS, SLOTS = 3, 8
# (temporal shape, conv shape, conv widths, conv axis, groups). GDN conv is
# cat(q, k, v) along axis 0; KDA conv is [K-1, cat(v, q, k)] along axis 1.
MODELS = {
    "gdn": ((4, 8, 16), (64, 4), (16, 16, 32), 0, 2),
    "kda": ((4, 8, 8), (3, 96), (32, 32, 32), 1, 4),
}


def _state(shape, dtype, strided):
    # Arena slots stride over every layer's state.
    dims = (SLOTS, LAYERS) if strided else (LAYERS, SLOTS)
    state = torch.randn(dims + shape, device="cuda").to(dtype)
    return state.transpose(0, 1) if strided else state


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("model", list(MODELS))
def test_page_unified_backup_load_round_trip(model, strided):
    temporal_shape, conv_shape, widths, axis, groups = MODELS[model]
    states = (
        _state(temporal_shape, torch.float32, strided),
        _state(conv_shape, torch.bfloat16, strided),
    )
    info = MambaStateBufferInfo(
        temporal=states[0],
        conv=states[1:],
        conv_shard_groups=widths,
        conv_slice_axis=axis,
    )
    layout = build_mamba_page_unified_layout(
        info, layer_num=LAYERS, attn_tp_size=1, group_num=groups
    )
    host = MambaPoolHost(
        info,
        host_to_device_ratio=2,
        host_size=0,
        device_capacity=SLOTS,
        layout="page_unified",
        page_unified_layout=layout,
    )
    try:
        device_indices = torch.tensor([1, 6, 3], device="cuda")
        host_indices = torch.tensor([4, 0, 9], device="cuda")
        host.backup_from_device_all_layer(None, host_indices, device_indices)
        torch.cuda.synchronize()
        for host_index, slot in zip(host_indices.tolist(), device_indices.tolist()):
            page = host.unified_buffer[host_index]
            for part, layer, offset, page_offset, length in layout.pieces():
                source = states[part][layer, slot].contiguous().view(torch.uint8)
                expected = source.flatten()[offset : offset + length].cpu()
                assert torch.equal(page[page_offset : page_offset + length], expected)

        expected = [state[:, device_indices].clone() for state in states]
        for state in states:
            state.zero_()
        load_indices = torch.tensor([2, 7, 5], device="cuda")
        for layer in range(LAYERS):
            host.load_to_device_per_layer(None, host_indices, load_indices, layer)
            torch.cuda.synchronize()
            # A per-layer load touches only its own layer.
            for state, original in zip(states, expected):
                assert torch.equal(state[layer, load_indices], original[layer])
                assert not torch.count_nonzero(state[layer + 1 :])
        for state in states:
            assert not torch.count_nonzero(state[:, [0, 1, 3, 4, 6]])
    finally:
        host.destroy()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

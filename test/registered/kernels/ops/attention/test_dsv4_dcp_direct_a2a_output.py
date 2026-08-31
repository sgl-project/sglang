from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("aiter")

from sglang.kernels.ops.attention.dcp_a2a import (
    DCPA2AOutputWorkspace,
    DCPA2APackedOutput,
)
from sglang.kernels.ops.attention.dcp_kernels import (
    dcp_lse_combine_triton,
    dcp_pack_a2a_send,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
)
from sglang.srt.layers.dcp import dcp_a2a_lse_reduce
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_hip() or not is_gfx95_supported(),
    reason="Direct DSV4 DCP A2A output requires ROCm gfx950",
)

HEADS = 128
HEAD_DIM = 512
GUARD = 64
GUARD_WORD = 0x5A5A


def _guarded_workspace(
    world_size: int,
) -> tuple[DCPA2AOutputWorkspace, torch.Tensor, torch.Tensor]:
    shape = (world_size, 1, HEADS // world_size, HEAD_DIM + 2)
    payload_numel = world_size * shape[1] * shape[2] * shape[3]
    send_storage = torch.full(
        (payload_numel + 2 * GUARD,),
        GUARD_WORD,
        dtype=torch.int16,
        device="cuda",
    )
    recv_storage = torch.full_like(send_storage, GUARD_WORD)
    send = send_storage[GUARD : GUARD + payload_numel].view(torch.bfloat16).view(shape)
    recv = recv_storage[GUARD : GUARD + payload_numel].view(torch.bfloat16).view(shape)
    return DCPA2AOutputWorkspace(send, recv), send_storage, recv_storage


def _assert_guards(storage: torch.Tensor) -> None:
    assert torch.equal(storage[:GUARD], torch.full_like(storage[:GUARD], GUARD_WORD))
    assert torch.equal(storage[-GUARD:], torch.full_like(storage[-GUARD:], GUARD_WORD))


def _decode_inputs(
    world_size: int,
    *,
    rank_major: bool,
    kv_len: int = 73,
    seed: int = 20260829,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q_full = torch.randn(
        (1, HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    if rank_major:
        local_heads = HEADS // world_size
        # Combined Q/top-k has padding after each source-rank Q row.
        backing = torch.empty(
            (1, world_size, local_heads + 2, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        q = backing[:, :, :local_heads, :]
        q.copy_(q_full.view(1, world_size, local_heads, HEAD_DIM))
    else:
        q = q_full
    unified_kv = torch.randn(
        (kv_len, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    kv_indices = torch.arange(kv_len, dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda")
    sink = torch.randn(
        (HEADS,), dtype=torch.float32, device="cuda", generator=generator
    )
    return q, unified_kv, kv_indices, kv_indptr, sink


def _decode(
    inputs: tuple[torch.Tensor, ...],
    *,
    output_workspace: DCPA2AOutputWorkspace | None = None,
    kv_splits: int = 64,
):
    q, unified_kv, kv_indices, kv_indptr, sink = inputs
    return _sparse_attn_v4_paged_decode_triton(
        q,
        unified_kv,
        kv_indices,
        kv_indptr,
        sink,
        HEAD_DIM**-0.5,
        block_h=16,
        kv_splits=kv_splits,
        block_k=16,
        return_lse=True,
        output_workspace=output_workspace,
    )


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("rank_major", [False, True])
def test_direct_producer_is_bitwise_old_pack_layout(
    world_size: int, rank_major: bool
) -> None:
    inputs = _decode_inputs(world_size, rank_major=rank_major)
    old_out, old_lse = _decode(inputs)
    expected = torch.empty(
        (world_size, 1, HEADS // world_size, HEAD_DIM + 2),
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected.view(torch.int16).fill_(GUARD_WORD)
    dcp_pack_a2a_send(
        old_out,
        old_lse,
        expected[..., :HEAD_DIM],
        expected.view(torch.float32)[..., HEAD_DIM // 2],
    )

    workspace, send_storage, recv_storage = _guarded_workspace(world_size)
    packed = _decode(inputs, output_workspace=workspace)
    assert isinstance(packed, DCPA2APackedOutput)
    assert torch.equal(
        workspace.send_combined.view(torch.int16), expected.view(torch.int16)
    )
    _assert_guards(send_storage)
    _assert_guards(recv_storage)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_direct_producer_mutated_graph_replay_matches_old_pack(
    world_size: int,
) -> None:
    inputs = _decode_inputs(world_size, rank_major=False)
    warmup_workspace, _, _ = _guarded_workspace(world_size)
    _decode(inputs, output_workspace=warmup_workspace)
    torch.cuda.synchronize()

    graph_workspace, send_storage, recv_storage = _guarded_workspace(world_size)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed = _decode(inputs, output_workspace=graph_workspace)
    assert isinstance(packed, DCPA2APackedOutput)

    q, unified_kv, _, _, sink = inputs
    for replay, seed in enumerate((41, 97, 193), start=1):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        q.copy_(
            torch.randn(
                q.shape,
                dtype=q.dtype,
                device=q.device,
                generator=generator,
            )
        )
        unified_kv.copy_(
            torch.randn(
                unified_kv.shape,
                dtype=unified_kv.dtype,
                device=unified_kv.device,
                generator=generator,
            )
        )
        sink.copy_(
            torch.randn(
                sink.shape,
                dtype=sink.dtype,
                device=sink.device,
                generator=generator,
            )
            * replay
        )

        old_out, old_lse = _decode(inputs)
        expected = torch.empty_like(graph_workspace.send_combined)
        expected.view(torch.int16).fill_(GUARD_WORD)
        dcp_pack_a2a_send(
            old_out,
            old_lse,
            expected[..., :HEAD_DIM],
            expected.view(torch.float32)[..., HEAD_DIM // 2],
        )
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            graph_workspace.send_combined.view(torch.int16),
            expected.view(torch.int16),
        )
        _assert_guards(send_storage)
        _assert_guards(recv_storage)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_prepacked_a2a_random_and_special_lse_is_exact(world_size: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(1000 + world_size)
    partial_out = torch.randn(
        (1, HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    partial_lse = torch.randn(
        (1, HEADS), dtype=torch.float32, device="cuda", generator=generator
    )
    special = torch.tensor(
        [
            -float("inf"),
            float("inf"),
            float("nan"),
            -0.0,
            0.0,
            -88.0,
            88.0,
            1.0000001192092896,
        ],
        dtype=torch.float32,
        device="cuda",
    )
    partial_lse[:, : special.numel()] = special

    group = MagicMock()
    group.world_size = world_size
    group.all_to_all_single.side_effect = lambda output, input_: output.copy_(input_)
    old = dcp_a2a_lse_reduce(partial_out, partial_lse, group)

    workspace, send_storage, recv_storage = _guarded_workspace(world_size)
    dcp_pack_a2a_send(
        partial_out,
        partial_lse,
        workspace.send_combined[..., :HEAD_DIM],
        workspace.send_combined.view(torch.float32)[..., HEAD_DIM // 2],
    )
    packed = DCPA2APackedOutput(
        workspace=workspace,
        batch_size=1,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        is_lse_base_on_e=True,
    )
    direct = dcp_a2a_lse_reduce(packed, None, group)
    assert torch.equal(old.view(torch.int16), direct.view(torch.int16))
    _assert_guards(send_storage)
    _assert_guards(recv_storage)


def test_fused_or_missing_workspace_uses_original_outputs() -> None:
    inputs = _decode_inputs(8, rank_major=False)
    workspace, send_storage, recv_storage = _guarded_workspace(8)

    fused = _decode(inputs, output_workspace=workspace, kv_splits=1)
    missing = _decode(inputs, output_workspace=None)
    assert isinstance(fused, tuple)
    assert isinstance(missing, tuple)
    assert fused[0].shape == missing[0].shape == (1, HEADS, HEAD_DIM)
    assert fused[1].shape == missing[1].shape == (1, HEADS)
    assert torch.all(send_storage == GUARD_WORD)
    assert torch.all(recv_storage == GUARD_WORD)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_lse_combine_selects_completed_epoch_plane_on_device(
    world_size: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(7000 + world_size)
    local_heads = HEADS // world_size
    outputs = torch.randn(
        (2, world_size, 1, local_heads, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    lses = torch.randn(
        (2, world_size, 1, local_heads),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    epoch = torch.tensor([1], dtype=torch.int32, device="cuda")
    for completed_epoch in (1, 2):
        plane = (completed_epoch - 1) & 1
        expected, _ = dcp_lse_combine_triton(outputs[plane], lses[plane])
        epoch.fill_(completed_epoch)
        actual, _ = dcp_lse_combine_triton(
            outputs,
            lses,
            plane_epoch=epoch,
        )
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

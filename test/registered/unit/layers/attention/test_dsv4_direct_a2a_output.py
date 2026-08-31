from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.kernels.ops.attention.dcp_a2a import (
    DCPA2AOutputWorkspace,
    DCPA2APackedOutput,
    can_use_dsv4_dcp_direct_a2a_output,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEADS = 128
HEAD_DIM = 512


def _route_inputs(
    world_size: int = 8, *, rank_major: bool = False
) -> tuple[torch.Tensor, torch.Tensor, DCPA2AOutputWorkspace]:
    if rank_major:
        local_heads = HEADS // world_size
        # Match combined-Q storage: Q occupies a prefix of each padded rank row.
        backing = torch.empty(
            (1, world_size, local_heads + 2, HEAD_DIM), dtype=torch.bfloat16
        )
        q = backing[:, :, :local_heads, :]
    else:
        q = torch.empty((1, HEADS, HEAD_DIM), dtype=torch.bfloat16)
    unified_kv = torch.empty((17, HEAD_DIM), dtype=torch.bfloat16)
    workspace = DCPA2AOutputWorkspace.allocate(
        world_size=world_size,
        max_batch_size=1,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        device="cpu",
    )
    return q, unified_kv, workspace


def _eligible_kwargs(world_size: int = 8, *, rank_major: bool = False) -> dict:
    q, unified_kv, workspace = _route_inputs(world_size, rank_major=rank_major)
    return {
        "q": q,
        "unified_kv": unified_kv,
        "workspace": workspace,
        "feature_enabled": True,
        "is_gfx950": True,
        "dcp_size": world_size,
        "comm_backend": "a2a",
        "is_plain_decode": True,
        "batch_size": 1,
        "speculative": False,
        "tbo": False,
        "memory_saver": False,
        "hisparse": False,
        "state_capture": False,
        "piecewise_graph": False,
        "breakable_graph": False,
    }


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("rank_major", [False, True])
def test_direct_a2a_route_accepts_only_supported_dsv4_sources(
    world_size: int, rank_major: bool
) -> None:
    assert can_use_dsv4_dcp_direct_a2a_output(
        **_eligible_kwargs(world_size, rank_major=rank_major)
    )


@pytest.mark.parametrize(
    ("case", "overrides"),
    [
        ("flag_off", {"feature_enabled": False}),
        ("not_gfx950", {"is_gfx950": False}),
        ("dcp1", {"dcp_size": 1}),
        ("unsupported_dcp", {"dcp_size": 16}),
        ("ag_rs", {"comm_backend": "ag_rs"}),
        ("fi_a2a", {"comm_backend": "fi_a2a"}),
        ("prefill", {"is_plain_decode": False}),
        ("idle", {"is_plain_decode": False}),
        ("target_verify", {"is_plain_decode": False}),
        ("batch_gt_one", {"batch_size": 2}),
        ("speculative", {"speculative": True}),
        ("tbo", {"tbo": True}),
        ("memory_saver", {"memory_saver": True}),
        ("hisparse", {"hisparse": True}),
        ("state_capture", {"state_capture": True}),
        ("piecewise_graph", {"piecewise_graph": True}),
        ("breakable_graph", {"breakable_graph": True}),
        ("missing_workspace", {"workspace": None}),
    ],
)
def test_direct_a2a_route_falls_back_for_unsupported_modes(
    case: str, overrides: dict
) -> None:
    del case
    kwargs = _eligible_kwargs()
    kwargs.update(overrides)
    assert not can_use_dsv4_dcp_direct_a2a_output(**kwargs)


def test_direct_a2a_route_rejects_unsupported_tensor_layouts() -> None:
    kwargs = _eligible_kwargs()

    bad_qs = [
        torch.empty((2, HEADS, HEAD_DIM), dtype=torch.bfloat16),
        torch.empty((1, HEADS // 2, HEAD_DIM), dtype=torch.bfloat16),
        torch.empty((1, HEADS, HEAD_DIM), dtype=torch.float16),
        torch.empty((1, HEADS, HEAD_DIM * 2), dtype=torch.bfloat16)[..., ::2],
    ]
    for q in bad_qs:
        assert not can_use_dsv4_dcp_direct_a2a_output(**(kwargs | {"q": q}))

    bad_kv = torch.empty((17, HEAD_DIM), dtype=torch.float16)
    assert not can_use_dsv4_dcp_direct_a2a_output(**(kwargs | {"unified_kv": bad_kv}))

    bad_workspace = DCPA2AOutputWorkspace(
        send_combined=torch.empty(
            (8, 1, HEADS // 8, HEAD_DIM + 4), dtype=torch.bfloat16
        ),
        recv_combined=torch.empty(
            (8, 1, HEADS // 8, HEAD_DIM + 4), dtype=torch.bfloat16
        ),
    )
    assert not can_use_dsv4_dcp_direct_a2a_output(
        **(kwargs | {"workspace": bad_workspace})
    )


def test_workspace_views_preserve_existing_transport_strides_and_storage() -> None:
    _, _, workspace = _route_inputs(world_size=8)
    output, lse = workspace.send_output_and_lse_views(
        batch_size=1,
        head_dim=HEAD_DIM,
    )

    assert output.shape == (8, 1, 16, HEAD_DIM)
    assert output.stride() == (16 * 514, 16 * 514, 514, 1)
    assert lse.shape == (8, 1, 16)
    assert lse.stride() == (16 * 257, 16 * 257, 257)
    assert output.untyped_storage().data_ptr() == lse.untyped_storage().data_ptr()
    assert lse.data_ptr() - output.data_ptr() == HEAD_DIM * 2


def test_prepacked_handle_skips_pack_and_keeps_collective_combine() -> None:
    import sglang.srt.layers.dcp.comm as comm

    _, _, workspace = _route_inputs(world_size=8)
    workspace.send_combined.view(torch.int16).copy_(
        torch.arange(workspace.send_combined.numel(), dtype=torch.int16).view_as(
            workspace.send_combined
        )
    )
    packed = DCPA2APackedOutput(
        workspace=workspace,
        batch_size=1,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        is_lse_base_on_e=True,
    )
    group = MagicMock()
    group.world_size = 8
    group.all_to_all_single.side_effect = lambda output, input_: output.copy_(input_)
    combined = torch.empty((1, HEADS // 8, HEAD_DIM), dtype=torch.bfloat16)

    with (
        patch.object(
            comm,
            "dcp_pack_a2a_send",
            side_effect=AssertionError("prepacked path must not launch pack"),
        ) as pack,
        patch.object(
            comm,
            "dcp_lse_combine_triton",
            return_value=(combined, None),
        ) as combine,
    ):
        result = comm.dcp_a2a_lse_reduce(packed, None, group)

    assert result is combined
    pack.assert_not_called()
    group.all_to_all_single.assert_called_once()
    assert torch.equal(
        workspace.recv_combined.view(torch.int16),
        workspace.send_combined.view(torch.int16),
    )
    recv_output, recv_lse = combine.call_args.args
    assert recv_output.shape == (8, 1, 16, HEAD_DIM)
    assert recv_output.stride() == (16 * 514, 16 * 514, 514, 1)
    assert recv_lse.shape == (8, 1, 16)
    assert recv_lse.stride() == (16 * 257, 16 * 257, 257)

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.kernels.ops.attention.dcp_a2a import (
    DCPA2AOutputWorkspace,
    DCPA2APackedOutput,
    DCPDestinationPushOutput,
    DCPDestinationPushWorkspace,
    can_use_dsv4_dcp_destination_push_output,
    dsv4_dcp_decode_reducer_communicator,
    prepare_dsv4_dcp_destination_push_workspace,
    prepare_dsv4_dcp_full_model_destination_push,
    select_dsv4_dcp_full_model_output_workspaces,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEADS = 128
HEAD_DIM = 512
DCP_SIZE = 8


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    backing = torch.empty(
        (1, DCP_SIZE, HEADS // DCP_SIZE + 2, HEAD_DIM), dtype=torch.bfloat16
    )
    return backing[:, :, : HEADS // DCP_SIZE, :], torch.empty(
        (17, HEAD_DIM), dtype=torch.bfloat16
    )


def _push_workspace(source_rank: int = 0) -> DCPDestinationPushWorkspace:
    return DCPDestinationPushWorkspace(
        recv_planes=torch.empty(
            (2, DCP_SIZE, 1, HEADS // DCP_SIZE, HEAD_DIM + 2),
            dtype=torch.bfloat16,
        ),
        peer_recv_ptrs=torch.zeros(DCP_SIZE, dtype=torch.uint64),
        epoch=torch.zeros(1, dtype=torch.int32),
        source_rank=source_rank,
    )


def _direct_workspace() -> DCPA2AOutputWorkspace:
    return DCPA2AOutputWorkspace.allocate(
        world_size=DCP_SIZE,
        max_batch_size=1,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        device="cpu",
    )


def _push_route_kwargs() -> dict:
    q, unified_kv = _inputs()
    communicator = MagicMock()
    communicator.rank_in_group = 0
    communicator.should_dsv4_dcp_destination_push.return_value = True
    return {
        "q": q,
        "unified_kv": unified_kv,
        "workspace": _push_workspace(),
        "communicator": communicator,
        "feature_enabled": True,
        "is_gfx950": True,
        "dcp_size": DCP_SIZE,
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


def _rank_group(*, rank: int = 0, capable: bool = True) -> MagicMock:
    group = MagicMock()
    group.ranks = list(range(DCP_SIZE))
    group.rank_in_group = rank
    group.world_size = DCP_SIZE
    group.all_reduce.side_effect = lambda value: value.fill_(
        DCP_SIZE if bool(value.item()) else 0
    )
    if capable:
        group.ca_comm.disabled = False
        group.ca_comm.prepare_dsv4_destination_push_workspace = MagicMock()
        group.ca_comm.should_dsv4_destination_push = MagicMock()
        group.ca_comm.dsv4_destination_push_ready = MagicMock()
        registered = _push_workspace(rank)
        group.prepare_dsv4_dcp_destination_push_workspace.return_value = (
            registered.recv_planes,
            registered.peer_recv_ptrs,
            registered.epoch,
        )
        group.should_dsv4_dcp_destination_push.return_value = True
    else:
        group.ca_comm = None
    return group


def test_destination_push_gate_and_layout_are_strict() -> None:
    kwargs = _push_route_kwargs()
    assert can_use_dsv4_dcp_destination_push_output(**kwargs)
    for override in (
        {"feature_enabled": False},
        {"is_gfx950": False},
        {"comm_backend": "ag_rs"},
        {"is_plain_decode": False},
        {"batch_size": 2},
        {"speculative": True},
        {"tbo": True},
        {"memory_saver": True},
        {"hisparse": True},
        {"state_capture": True},
        {"piecewise_graph": True},
        {"breakable_graph": True},
        {"workspace": None},
    ):
        assert not can_use_dsv4_dcp_destination_push_output(**(kwargs | override))

    workspace = kwargs["workspace"]
    output, lse = workspace.receive_output_and_lse_views(
        batch_size=1, head_dim=HEAD_DIM
    )
    assert output.shape == (2, 8, 1, 16, HEAD_DIM)
    assert lse.shape == (2, 8, 1, 16)
    assert output.untyped_storage().data_ptr() == lse.untyped_storage().data_ptr()


def test_destination_push_precedes_direct_and_falls_back_cleanly() -> None:
    kwargs = _push_route_kwargs()
    common = {
        key: value
        for key, value in kwargs.items()
        if key not in {"workspace", "feature_enabled"}
    }
    push, direct = select_dsv4_dcp_full_model_output_workspaces(
        destination_push_workspace=kwargs["workspace"],
        direct_output_workspace=_direct_workspace(),
        destination_push_enabled=True,
        direct_output_enabled=True,
        **common,
    )
    assert push is kwargs["workspace"]
    assert direct is None

    push, direct = select_dsv4_dcp_full_model_output_workspaces(
        destination_push_workspace=kwargs["workspace"],
        direct_output_workspace=_direct_workspace(),
        destination_push_enabled=False,
        direct_output_enabled=True,
        **common,
    )
    assert push is None
    assert direct is not None

    push, direct = select_dsv4_dcp_full_model_output_workspaces(
        destination_push_workspace=kwargs["workspace"],
        direct_output_workspace=_direct_workspace(),
        destination_push_enabled=True,
        direct_output_enabled=True,
        **(common | {"is_plain_decode": False}),
    )
    assert push is direct is None


def test_direct_platform_gate_remains_broader_than_exact_push_gate() -> None:
    kwargs = _push_route_kwargs()
    common = {
        key: value
        for key, value in kwargs.items()
        if key not in {"workspace", "feature_enabled", "is_gfx950"}
    }
    push, direct = select_dsv4_dcp_full_model_output_workspaces(
        destination_push_workspace=kwargs["workspace"],
        direct_output_workspace=_direct_workspace(),
        destination_push_enabled=True,
        direct_output_enabled=True,
        is_gfx950=False,
        direct_platform_supported=True,
        **common,
    )
    assert push is None
    assert direct is not None


def test_destination_push_ready_precedes_device_epoch_combine() -> None:
    from sglang.srt.layers.dcp import comm

    workspace = _push_workspace()
    output = DCPDestinationPushOutput(
        workspace=workspace,
        batch_size=1,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        is_lse_base_on_e=True,
    )
    group = MagicMock()
    group.world_size = DCP_SIZE
    group.rank_in_group = 0
    group.should_dsv4_dcp_destination_push.return_value = True
    events = []
    group.dsv4_dcp_destination_push_ready.side_effect = lambda *_: events.append(
        "ready"
    )
    combined = torch.empty((1, HEADS // DCP_SIZE, HEAD_DIM), dtype=torch.bfloat16)

    def _combine(*args, **kwargs):
        events.append("combine")
        assert kwargs["plane_epoch"] is workspace.epoch
        return combined, None

    with patch.object(comm, "dcp_lse_combine_triton", side_effect=_combine):
        result = comm.dcp_registered_destination_push_lse_reduce(output, group)
    assert result is combined
    assert events == ["ready", "combine"]


def test_selected_destination_push_ready_error_propagates() -> None:
    from sglang.srt.layers.dcp import comm

    workspace = _push_workspace()
    output = DCPDestinationPushOutput(
        workspace=workspace,
        batch_size=1,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        is_lse_base_on_e=True,
    )
    group = MagicMock(world_size=DCP_SIZE, rank_in_group=0)
    group.world_size = DCP_SIZE
    group.rank_in_group = 0
    group.should_dsv4_dcp_destination_push.return_value = True
    group.dsv4_dcp_destination_push_ready.side_effect = RuntimeError("ready")
    with pytest.raises(RuntimeError, match="ready"):
        comm.dcp_registered_destination_push_lse_reduce(output, group)


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ("missing", "ca_comm missing"),
        ("disabled", "disabled=True"),
        ("none", "AITER prepare returned None"),
        ("supports", "candidate supports rejected"),
        ("should", "should rejected"),
        ("registration", "registration exception=RuntimeError('register')"),
    ],
)
def test_prepare_reports_prelaunch_failure_reasons(
    setup: str, message: str, caplog
) -> None:
    group = _rank_group()
    if setup == "missing":
        group.ca_comm = None
    elif setup == "disabled":
        group.ca_comm.disabled = True
    elif setup == "none":
        group.prepare_dsv4_dcp_destination_push_workspace.return_value = None
    elif setup == "supports":
        registered = list(
            group.prepare_dsv4_dcp_destination_push_workspace.return_value
        )
        registered[0] = torch.empty((1,), dtype=torch.bfloat16)
        group.prepare_dsv4_dcp_destination_push_workspace.return_value = tuple(
            registered
        )
    elif setup == "should":
        group.should_dsv4_dcp_destination_push.return_value = False
    else:
        group.prepare_dsv4_dcp_destination_push_workspace.side_effect = RuntimeError(
            "register"
        )
    group.all_reduce.side_effect = lambda value: value.fill_(0)

    assert (
        prepare_dsv4_dcp_destination_push_workspace(
            communicator=group,
            agreement_group=group,
            device=torch.device("cpu"),
            candidate_name="dcp",
        )
        is None
    )
    assert message in caplog.text


def test_prepare_reports_ready_agreement_failures(caplog) -> None:
    group = _rank_group()
    group.all_reduce.side_effect = lambda value: value.fill_(7)
    assert (
        prepare_dsv4_dcp_destination_push_workspace(
            communicator=group,
            agreement_group=group,
            device=torch.device("cpu"),
            candidate_name="dcp",
        )
        is None
    )
    assert "ready_count=7/8" in caplog.text

    caplog.clear()
    group.all_reduce.side_effect = RuntimeError("collective")
    assert (
        prepare_dsv4_dcp_destination_push_workspace(
            communicator=group,
            agreement_group=group,
            device=torch.device("cpu"),
            candidate_name="dcp",
        )
        is None
    )
    assert "ready all-reduce exception=RuntimeError('collective')" in caplog.text


def test_full_model_selects_first_equivalent_capable_group() -> None:
    dcp = _rank_group(capable=False)
    tp = _rank_group()
    attn_tp = _rank_group()
    workspace, selected = prepare_dsv4_dcp_full_model_destination_push(
        dcp_group=dcp,
        candidate_groups=[("dcp", dcp), ("tp", tp), ("attn_tp", attn_tp)],
        dcp_rank=0,
        device=torch.device("cpu"),
    )
    assert workspace is not None
    assert selected is tp
    tp.prepare_dsv4_dcp_destination_push_workspace.assert_called_once()
    attn_tp.prepare_dsv4_dcp_destination_push_workspace.assert_not_called()


def test_full_model_rejects_non_equivalent_group_with_diagnostics(caplog) -> None:
    dcp = _rank_group(capable=False)
    candidate = _rank_group()
    candidate.ranks = list(reversed(range(DCP_SIZE)))
    workspace, selected = prepare_dsv4_dcp_full_model_destination_push(
        dcp_group=dcp,
        candidate_groups=[("tp", candidate)],
        dcp_rank=0,
        device=torch.device("cpu"),
    )
    assert workspace is selected is None
    assert "group equivalence mismatch" in caplog.text
    assert "candidate_ranks=" in caplog.text
    assert "dcp_rank=0" in caplog.text


def test_reducer_uses_selected_group_only_for_push_output() -> None:
    push_group = object()
    dcp_group = object()
    workspace = _push_workspace()
    pushed = DCPDestinationPushOutput(
        workspace=workspace,
        batch_size=1,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        is_lse_base_on_e=True,
    )
    packed = DCPA2APackedOutput(
        workspace=_direct_workspace(),
        batch_size=1,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        is_lse_base_on_e=True,
    )
    assert (
        dsv4_dcp_decode_reducer_communicator(
            pushed,
            destination_push_communicator=push_group,
            dcp_group=dcp_group,
        )
        is push_group
    )
    assert (
        dsv4_dcp_decode_reducer_communicator(
            packed,
            destination_push_communicator=push_group,
            dcp_group=dcp_group,
        )
        is dcp_group
    )


def test_destination_push_flag_defaults_off() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SGLANG_DSV4_DCP_REGISTERED_DESTINATION_PUSH", None)
        assert envs.SGLANG_DSV4_DCP_REGISTERED_DESTINATION_PUSH.get() is False

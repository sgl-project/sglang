from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
    IpcA2AState,
    _peer_cuda_device,
    _Unsupported,
    ipc_a2a_ready,
    ipc_shareable_zeros,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a_multi import (
    IpcA2AMultiState,
    _PipelinePool,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

_IPC = "sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a"
_MULTI = f"{_IPC}_multi"

# ipc_a2a_ready() only sets up the transport on CUDA. Elsewhere the platform may
# lack is_full_nvlink and torch may lack the accelerator allocator-settings API.
requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the IPC all-to-all transport is CUDA-only"
)


@requires_cuda
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("nvlink", [False, True])
def test_ipc_requires_nvlink_before_mapping_peer_memory(monkeypatch, rank, nvlink):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    state = IpcA2AState()
    with (
        patch(f"{_IPC}.dist.get_rank", return_value=rank),
        patch(f"{_IPC}.dist.get_process_group_ranks", return_value=[0, 1]),
        patch(f"{_IPC}.torch.cuda.current_device", return_value=rank),
        patch(f"{_IPC}._peer_cuda_device", return_value=1 - rank),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_full_nvlink",
            return_value=nvlink,
        ) as topology,
        patch(f"{_IPC}.torch.cuda.can_device_access_peer", return_value=True) as peer,
        patch("ctypes.CDLL") as cudart,
        patch(f"{_IPC}.load_ipc_a2a_sync") as kernels,
        patch(f"{_IPC}.torch.zeros"),
        patch.object(state, "_share"),
    ):
        if nvlink:
            state.init(object())
            assert state.inited
            peer.assert_called_once_with(rank, 1 - rank)
            kernels.assert_called_once()
        else:
            with pytest.raises(_Unsupported, match="NVLink"):
                state.init(object())
            assert not state.inited
            peer.assert_not_called()
            cudart.assert_not_called()
            kernels.assert_not_called()
        topology.assert_called_once_with([1, 3])


@pytest.mark.parametrize(
    "new_ranks", [[0, 2], [0, 1]], ids=["different_pair", "same_pair"]
)
def test_ipc_transport_reinitializes_only_for_a_different_rank_pair(new_ranks):
    """A second group handle over the same pair (the SP device group that
    AllToAll4D passes vs the Ulysses group USP passes) must reuse the transport."""
    state = IpcA2AState()
    old_group = object()
    new_group = object()
    state.inited = True
    state.group = old_group
    state.ranks = [0, 1]
    state.calls = 7

    def initialize(group):
        state.inited = True
        state.group = group

    with (
        patch(f"{_IPC}.IPC_A2A", state),
        patch(f"{_IPC}.envs.SGLANG_DIFFUSION_IPC_A2A", True),
        patch(f"{_IPC}.dist.get_process_group_ranks", return_value=new_ranks),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_cuda",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.distributed.get_tp_world_size",
            return_value=1,
        ),
        patch.object(state, "init", side_effect=initialize) as init,
        patch(f"{_IPC}.torch.cuda.is_current_stream_capturing", return_value=False),
    ):
        assert ipc_a2a_ready(new_group)

    replaced = new_ranks != [0, 1]
    assert init.called == replaced
    assert state.group is (new_group if replaced else old_group)
    assert state.calls == (0 if replaced else 7)


def test_peer_cuda_device_uses_the_ulysses_group_mapping():
    group = object()
    members = [("node-a", 2), ("node-a", 3)]

    def gather(output, value, *, group):
        assert value == ("node-a", 3)
        output[:] = members

    with (
        patch(f"{_IPC}.socket.gethostname", return_value="node-a"),
        patch(f"{_IPC}.dist.get_world_size", return_value=2),
        patch(f"{_IPC}.dist.all_gather_object", side_effect=gather),
    ):
        assert _peer_cuda_device(group, rank=1, device=3) == 2


def test_drop_staging_clears_cached_buffers_and_nothing_else():
    from collections import OrderedDict

    from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
        IpcA2AState,
    )

    state = IpcA2AState()
    state.inited = True
    state.calls = 7
    state.staging = OrderedDict(
        {(4, 4, "bf16"): ("local", "peer"), (8, 8, "bf16"): ("l", "p")}
    )

    state.drop_staging()

    assert state.staging == OrderedDict()
    assert state.inited is True and state.calls == 7
    state.drop_staging()  # idempotent on an empty cache


def test_drop_a2a_staging_buffers_clears_the_ulysses_cache():
    import torch

    from sglang.multimodal_gen.runtime.layers import usp

    usp._A2A_STAGING_BUFFERS[("qkv", torch.float16, 0)] = torch.empty(
        8, dtype=torch.float16
    )
    with patch.object(torch.cuda, "is_available", return_value=False):
        usp.drop_a2a_staging_buffers()
        usp.drop_a2a_staging_buffers()  # idempotent on an empty cache
    assert usp._A2A_STAGING_BUFFERS == {}


@requires_cuda
@pytest.mark.parametrize("expandable", [False, True])
def test_ipc_buffers_are_allocated_outside_expandable_segments(expandable):
    settings = "expandable_segments:True" if expandable else ""
    calls = []
    with (
        patch(
            f"{_IPC}.torch._C._accelerator_getAllocatorSettings",
            return_value=settings,
        ),
        patch(
            f"{_IPC}.torch._C._accelerator_setAllocatorSettings",
            side_effect=lambda conf: calls.append(conf),
        ),
        patch(f"{_IPC}.torch.zeros", side_effect=lambda *a, **k: calls.append("alloc")),
    ):
        ipc_shareable_zeros(2, 8, dtype=None)
    expected = ["expandable_segments:False", "alloc", "expandable_segments:True"]
    assert calls == (expected if expandable else ["alloc"])


def test_multi_rank_transport_exports_only_ipc_shareable_buffers():
    # under expandable segments a plain allocation exports a handle whose import
    # needs pidfd_getfd, so one such buffer sends the whole group to NCCL
    state = IpcA2AMultiState()
    shareable, exported = [], []

    def shareable_zeros(*shape, dtype):
        shareable.append(MagicMock())
        return shareable[-1]

    def share(t, group):
        exported.append(t)
        return [t, t]

    agreed = MagicMock()
    agreed.item.return_value = 1
    with (
        patch(f"{_MULTI}.ipc_shareable_zeros", side_effect=shareable_zeros),
        patch(f"{_MULTI}.torch.zeros"),
        patch(f"{_MULTI}.torch.tensor", return_value=agreed),
        patch(f"{_MULTI}.torch.cuda.current_device", return_value=0),
        patch(f"{_MULTI}.torch.cuda.can_device_access_peer", return_value=True),
        patch(f"{_MULTI}.torch.cuda.is_current_stream_capturing", return_value=False),
        patch(f"{_MULTI}.dist.get_rank", return_value=0),
        patch(f"{_MULTI}.dist.get_world_size", return_value=2),
        patch(f"{_MULTI}.dist.all_reduce"),
        patch(f"{_MULTI}._member_devices", return_value=[0, 1]),
        patch(f"{_MULTI}._StreamMemOps"),
        patch(f"{_MULTI}.load_ipc_a2a_sync"),
        patch("ctypes.CDLL"),
        patch.object(state, "_share", side_effect=share),
    ):
        state.init(object())
        state.get_staging(8, torch.bfloat16)
        _PipelinePool(state, torch.bfloat16, 8, 8, 2)
    # flags, timeout flag, staging, pipeline receive/output boxes and counters
    assert [any(t is s for s in shareable) for t in exported] == [True] * 7

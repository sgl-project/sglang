from unittest.mock import patch

import pytest

from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
    IpcA2AState,
    _peer_cuda_device,
    _Unsupported,
    ipc_a2a_ready,
)

_IPC = "sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a"


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("nvlink", [False, True])
def test_ipc_requires_nvlink_before_mapping_peer_memory(monkeypatch, rank, nvlink):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    state = IpcA2AState()
    with (
        patch(f"{_IPC}.dist.get_rank", return_value=rank),
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


def test_reinitializes_ipc_transport_for_replaced_process_group():
    state = IpcA2AState()
    old_group = object()
    new_group = object()
    state.inited = True
    state.group = old_group
    state.calls = 7

    def initialize(group):
        state.inited = True
        state.group = group

    with (
        patch(f"{_IPC}.IPC_A2A", state),
        patch(f"{_IPC}.envs.SGLANG_DIFFUSION_IPC_A2A", True),
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

    init.assert_called_once_with(new_group)
    assert state.group is new_group
    assert state.calls == 0


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

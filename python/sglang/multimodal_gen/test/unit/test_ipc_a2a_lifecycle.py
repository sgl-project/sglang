from unittest.mock import patch

from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
    IpcA2AState,
    ipc_a2a_ready,
)

_IPC = "sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a"


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

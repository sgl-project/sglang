from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import call, patch

from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
    IPC_A2A,
)
from sglang.multimodal_gen.runtime.distributed.parallel_groups import PROCESS_GROUP
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    initialize_parallel_runtime,
)

_UTILS = "sglang.multimodal_gen.test.single_test_file.component_accuracy.utils"


def _server_args(*, ulysses_degree: int, ring_degree: int) -> SimpleNamespace:
    return SimpleNamespace(
        tp_size=1,
        sp_degree=2,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        dp_size=1,
        cfg_parallel_degree=1,
    )


def _patch_current_topology(
    stack: ExitStack, *, ulysses_degree: int, ring_degree: int
) -> None:
    for context_manager in (
        patch(f"{_UTILS}.model_parallel_is_initialized", return_value=True),
        patch(f"{_UTILS}.get_tensor_model_parallel_world_size", return_value=1),
        patch(f"{_UTILS}.get_sequence_parallel_world_size", return_value=2),
        patch(
            f"{_UTILS}.get_ulysses_parallel_world_size",
            return_value=ulysses_degree,
        ),
        patch(f"{_UTILS}.get_ring_parallel_world_size", return_value=ring_degree),
        patch(f"{_UTILS}.get_data_parallel_world_size", return_value=1),
        patch(f"{_UTILS}.get_classifier_free_guidance_world_size", return_value=1),
    ):
        stack.enter_context(context_manager)


def test_reinitializes_when_sp_decomposition_changes():
    with ExitStack() as stack:
        _patch_current_topology(stack, ulysses_degree=2, ring_degree=1)
        stack.enter_context(
            patch(f"{_UTILS}.torch.distributed.is_initialized", return_value=True)
        )
        barrier = stack.enter_context(patch(f"{_UTILS}.torch.distributed.barrier"))
        destroy = stack.enter_context(patch(f"{_UTILS}.destroy_model_parallel"))
        initialize = stack.enter_context(
            patch(f"{_UTILS}.maybe_init_distributed_environment_and_model_parallel")
        )
        initialize_parallel_runtime(_server_args(ulysses_degree=1, ring_degree=2))

    destroy.assert_called_once_with()
    initialize.assert_called_once_with(
        tp_size=1,
        sp_size=2,
        cfg_degree=1,
        ulysses_degree=1,
        ring_degree=2,
        dp_size=1,
    )
    assert barrier.call_count == 2


def test_reuses_matching_sp_decomposition():
    with ExitStack() as stack:
        _patch_current_topology(stack, ulysses_degree=1, ring_degree=2)
        stack.enter_context(
            patch(f"{_UTILS}.torch.distributed.is_initialized", return_value=True)
        )
        destroy = stack.enter_context(patch(f"{_UTILS}.destroy_model_parallel"))
        initialize = stack.enter_context(
            patch(f"{_UTILS}.maybe_init_distributed_environment_and_model_parallel")
        )
        initialize_parallel_runtime(_server_args(ulysses_degree=1, ring_degree=2))

    destroy.assert_not_called()
    initialize.assert_not_called()


def test_destroy_releases_sequence_parallel_subgroups_after_partial_init():
    ulysses_group = object()
    ring_group = object()

    with ExitStack() as stack:
        for name in (
            "_TP",
            "_SP",
            "_DP",
            "_CFG",
            "_PP",
            "_VAE_DECODE",
            "_DIT",
            "_VAE",
        ):
            stack.enter_context(patch.object(parallel_state, name, None))
        stack.enter_context(patch.object(PROCESS_GROUP, "ULYSSES_PG", ulysses_group))
        stack.enter_context(patch.object(PROCESS_GROUP, "RING_PG", ring_group))
        reset_ipc = stack.enter_context(patch.object(IPC_A2A, "reset"))
        destroy_group = stack.enter_context(
            patch.object(parallel_state.torch.distributed, "destroy_process_group")
        )

        parallel_state.destroy_model_parallel()

        reset_ipc.assert_called_once_with()
        assert destroy_group.call_args_list == [call(ulysses_group), call(ring_group)]
        assert PROCESS_GROUP.ULYSSES_PG is None
        assert PROCESS_GROUP.RING_PG is None

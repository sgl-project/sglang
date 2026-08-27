from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import call, patch

import torch
from torch import nn

from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
    IPC_A2A,
)
from sglang.multimodal_gen.runtime.distributed.parallel_groups import PROCESS_GROUP
from sglang.multimodal_gen.test.single_test_file.component_accuracy.engine import (
    AccuracyEngine,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    initialize_parallel_runtime,
)
from sglang.srt.distributed import parallel_state as srt_parallel_state

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


def test_srt_attention_tp_group_tracks_diffusion_tp_group():
    tp_group = object()

    with (
        patch.object(parallel_state, "_TP", tp_group),
        patch.object(srt_parallel_state, "_TP", None),
        patch.object(srt_parallel_state, "_ATTN_TP", None),
    ):
        parallel_state._sync_srt_tp_group()

        assert srt_parallel_state._TP is tp_group
        assert srt_parallel_state._ATTN_TP is tp_group

        parallel_state._clear_srt_tp_group()

        assert srt_parallel_state._TP is None
        assert srt_parallel_state._ATTN_TP is None


def test_srt_owned_groups_are_not_overwritten_or_cleared():
    diffusion_tp_group = object()
    srt_tp_group = object()
    srt_attention_tp_group = object()

    with (
        patch.object(parallel_state, "_TP", diffusion_tp_group),
        patch.object(srt_parallel_state, "_TP", srt_tp_group),
        patch.object(srt_parallel_state, "_ATTN_TP", srt_attention_tp_group),
    ):
        parallel_state._sync_srt_tp_group()
        parallel_state._clear_srt_tp_group()

        assert srt_parallel_state._TP is srt_tp_group
        assert srt_parallel_state._ATTN_TP is srt_attention_tp_group


def test_srt_tp_groups_follow_encoder_folding_context():
    original_diffusion_tp_group = object()
    original_srt_tp_group = object()
    original_srt_attention_tp_group = object()
    folding_tp_group = object()

    with (
        patch.object(parallel_state, "_TP", original_diffusion_tp_group),
        patch.object(srt_parallel_state, "_TP", original_srt_tp_group),
        patch.object(
            srt_parallel_state,
            "_ATTN_TP",
            original_srt_attention_tp_group,
        ),
    ):
        with parallel_state.use_tensor_parallel_group(folding_tp_group):
            assert parallel_state._TP is folding_tp_group
            assert srt_parallel_state._TP is folding_tp_group
            assert srt_parallel_state._ATTN_TP is folding_tp_group

        assert parallel_state._TP is original_diffusion_tp_group
        assert srt_parallel_state._TP is original_srt_tp_group
        assert srt_parallel_state._ATTN_TP is original_srt_attention_tp_group


def test_encoder_folding_context_is_nested_and_restores_each_group():
    original_tp_group = object()
    outer_tp_group = object()
    inner_tp_group = object()

    with (
        patch.object(parallel_state, "_TP", original_tp_group),
        patch.object(srt_parallel_state, "_TP", original_tp_group),
        patch.object(srt_parallel_state, "_ATTN_TP", original_tp_group),
    ):
        with parallel_state.use_tensor_parallel_group(outer_tp_group):
            with parallel_state.use_tensor_parallel_group(inner_tp_group):
                assert parallel_state._TP is inner_tp_group
                assert srt_parallel_state._TP is inner_tp_group
                assert srt_parallel_state._ATTN_TP is inner_tp_group

            assert parallel_state._TP is outer_tp_group
            assert srt_parallel_state._TP is outer_tp_group
            assert srt_parallel_state._ATTN_TP is outer_tp_group

        assert parallel_state._TP is original_tp_group
        assert srt_parallel_state._TP is original_tp_group
        assert srt_parallel_state._ATTN_TP is original_tp_group


def test_weight_transfer_uses_loader_for_implicit_srt_shard():
    source = nn.Module()
    source.weight = nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(4, 2))
    target = nn.Module()
    target.weight = nn.Parameter(torch.empty(2, 2))

    def load_first_shard(param, loaded_weight):
        param.data.copy_(loaded_weight[:2])

    target.weight.weight_loader = load_first_shard

    with patch(
        "sglang.multimodal_gen.test.single_test_file.component_accuracy.engine.model_parallel_is_initialized",
        return_value=False,
    ):
        AccuracyEngine.transfer_weights(
            source,
            target,
            min_match_ratio=1.0,
            target_device=torch.device("cpu"),
        )

    torch.testing.assert_close(
        target.weight,
        source.weight[:2].to(dtype=torch.bfloat16),
    )

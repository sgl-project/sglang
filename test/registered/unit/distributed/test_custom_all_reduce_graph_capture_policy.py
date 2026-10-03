import sys
import types
from functools import partial
from unittest import mock

import torch

from sglang.srt.distributed.device_communicators import custom_all_reduce
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_registered_graph_inputs_enabled_by_default():
    with (
        mock.patch.object(
            custom_all_reduce.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH,
            "get",
            return_value=False,
        ),
        mock.patch(
            "sglang.srt.runtime_context.attention_backends",
            return_value=("tilelang", "tilelang"),
        ),
        mock.patch("sglang.srt.runtime_context.get_parallel") as get_parallel,
    ):
        get_parallel.return_value.enable_dp_attention = True
        assert custom_all_reduce._enable_register_for_capturing()


def test_registered_graph_inputs_disabled_for_dsa_dp_attention():
    with (
        mock.patch.object(
            custom_all_reduce.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH,
            "get",
            return_value=False,
        ),
        mock.patch(
            "sglang.srt.runtime_context.attention_backends",
            return_value=("aiter", "dsa"),
        ),
        mock.patch("sglang.srt.runtime_context.get_parallel") as get_parallel,
    ):
        get_parallel.return_value.enable_dp_attention = True
        assert not custom_all_reduce._enable_register_for_capturing()


def test_registered_graph_inputs_remain_enabled_for_dsa_without_dp_attention():
    with (
        mock.patch.object(
            custom_all_reduce.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH,
            "get",
            return_value=False,
        ),
        mock.patch(
            "sglang.srt.runtime_context.attention_backends",
            return_value=("tilelang", "dsa"),
        ),
        mock.patch("sglang.srt.runtime_context.get_parallel") as get_parallel,
    ):
        get_parallel.return_value.enable_dp_attention = False
        assert custom_all_reduce._enable_register_for_capturing()


def test_memory_saver_always_uses_copy_in():
    with mock.patch.object(
        custom_all_reduce.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH,
        "get",
        return_value=True,
    ):
        assert not custom_all_reduce._enable_register_for_capturing()


def test_policy_is_forwarded_to_sglang_custom_allreduce():
    with (
        mock.patch.object(custom_all_reduce, "_is_cuda", False),
        mock.patch.object(custom_all_reduce, "_is_musa", False),
        mock.patch.object(custom_all_reduce, "_is_hip", True),
        mock.patch.object(
            custom_all_reduce, "_use_amd_deterministic_impl", return_value=False
        ),
        mock.patch.object(
            custom_all_reduce, "_enable_register_for_capturing", return_value=False
        ),
        mock.patch.object(custom_all_reduce, "get_bool_env_var", return_value=False),
    ):
        factory = custom_all_reduce.dispatch_custom_allreduce(
            group=mock.sentinel.group,
            device=torch.device("cpu"),
        )

    assert isinstance(factory, partial)
    assert factory.func is custom_all_reduce.CustomAllreduce
    assert factory.keywords["enable_register_for_capturing"] is False


def test_policy_is_forwarded_to_aiter_custom_allreduce():
    fake_aiter_module = types.ModuleType(
        "aiter.dist.device_communicators.custom_all_reduce"
    )

    class FakeAiterCustomAllreduce:
        pass

    fake_aiter_module.CustomAllreduce = FakeAiterCustomAllreduce
    with (
        mock.patch.dict(
            sys.modules,
            {"aiter.dist.device_communicators.custom_all_reduce": (fake_aiter_module)},
        ),
        mock.patch.object(custom_all_reduce, "_is_cuda", False),
        mock.patch.object(custom_all_reduce, "_is_musa", False),
        mock.patch.object(custom_all_reduce, "_is_hip", True),
        mock.patch.object(
            custom_all_reduce, "_use_amd_deterministic_impl", return_value=False
        ),
        mock.patch.object(
            custom_all_reduce, "_enable_register_for_capturing", return_value=False
        ),
        mock.patch.object(custom_all_reduce, "get_bool_env_var", return_value=True),
    ):
        factory = custom_all_reduce.dispatch_custom_allreduce(
            group=mock.sentinel.group,
            device=torch.device("cpu"),
        )

    assert isinstance(factory, partial)
    assert factory.func is FakeAiterCustomAllreduce
    assert factory.keywords["enable_register_for_capturing"] is False

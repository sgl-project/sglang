import sys
import types
from contextlib import contextmanager
from functools import partial
from unittest import mock

import pytest
import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.device_communicators import custom_all_reduce
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def _published_policy_inputs(*, decode_backend, attn_dp_enabled):
    with (
        mock.patch.object(
            custom_all_reduce.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH,
            "get",
            return_value=False,
        ),
        mock.patch(
            "sglang.srt.runtime_context.get_context",
            return_value=types.SimpleNamespace(
                is_config_namespace_published=lambda name: True
            ),
        ),
        mock.patch(
            "sglang.srt.runtime_context.attention_backends",
            return_value=("aiter", decode_backend),
        ),
        mock.patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=types.SimpleNamespace(attn_dp_enabled=attn_dp_enabled),
        ),
    ):
        yield


@pytest.mark.parametrize(
    "decode_backend, attn_dp_enabled, expected",
    [
        ("tilelang", True, True),
        ("dsa", True, False),
        # The deprecated alias builds the same per-lane dense/sparse graphs.
        ("nsa", True, False),
        ("dsa", False, True),
    ],
)
def test_registered_graph_inputs_policy(decode_backend, attn_dp_enabled, expected):
    with _published_policy_inputs(
        decode_backend=decode_backend, attn_dp_enabled=attn_dp_enabled
    ):
        assert custom_all_reduce._enable_register_for_capturing() is expected


def test_unpublished_config_keeps_registered_graph_inputs():
    """Dispatch before publish must not raise: the caller turns any exception
    into "custom all-reduce disabled" for the whole group."""
    with (
        mock.patch.object(
            custom_all_reduce.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH,
            "get",
            return_value=False,
        ),
        mock.patch(
            "sglang.srt.runtime_context.get_context",
            return_value=types.SimpleNamespace(
                is_config_namespace_published=lambda name: False
            ),
        ),
        mock.patch(
            "sglang.srt.runtime_context.attention_backends",
            side_effect=ValueError("config namespace 'exec' not published"),
        ),
    ):
        assert custom_all_reduce._enable_register_for_capturing()


@pytest.mark.parametrize(
    "ca_registers, copy_in_env, expected",
    [(True, False, True), (False, False, False), (True, True, False)],
)
def test_aiter_all_gather_and_reduce_scatter_follow_capture_policy(
    ca_registers, copy_in_env, expected
):
    """A communicator built for copy-in must not register all-gather or
    reduce-scatter inputs either; DP lanes replaying different graphs then read
    another graph's peer buffers."""
    ca_comm = types.SimpleNamespace(enable_register_for_capturing=ca_registers)
    with (
        mock.patch.object(
            parallel_state.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH,
            "get",
            return_value=False,
        ),
        mock.patch.object(
            parallel_state.envs.SGLANG_AITER_CAPTURE_COPY_IN,
            "get",
            return_value=copy_in_env,
        ),
    ):
        assert parallel_state._aiter_capture_registered(ca_comm) is expected


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

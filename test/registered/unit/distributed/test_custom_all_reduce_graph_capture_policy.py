import sys
import types
from contextlib import contextmanager
from functools import partial
from unittest import mock

import pytest
import torch

from sglang.srt.distributed.device_communicators import custom_all_reduce
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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
        ("dsa", False, True),
    ],
)
def test_registered_graph_inputs_policy(decode_backend, attn_dp_enabled, expected):
    with _published_policy_inputs(
        decode_backend=decode_backend, attn_dp_enabled=attn_dp_enabled
    ):
        assert custom_all_reduce._enable_register_for_capturing() is expected


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

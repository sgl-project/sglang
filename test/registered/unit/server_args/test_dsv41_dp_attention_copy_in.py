"""DeepSeek-V4.1 with DP attention on ROCm keeps every aiter custom collective in its
decode graphs off per-graph registered buffers; otherwise DP ranks that replay
different graph variants decode wrong tokens."""

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.distributed.device_communicators import custom_all_reduce
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_COPY_IN = "SGLANG_AITER_CAPTURE_COPY_IN"


class _PastCopyIn(Exception):
    """Validation read a server-args field after the copy-in decision."""


class _View(SimpleNamespace):
    def __getattr__(self, name):
        raise _PastCopyIn(name)


def _environ(env):
    environ = {k: v for k, v in os.environ.items() if k != _COPY_IN}
    if env is not None:
        environ[_COPY_IN] = env
    return environ


class TestDsv41DpAttentionCopyIn(CustomTestCase):
    def _validate(self, *, env):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41")
        )
        with (
            patch.dict(os.environ, _environ(env), clear=True),
            patch.object(deepseek_v4_hook, "resolving_view", return_value=_View()),
            patch.object(
                deepseek_v4_hook, "model_config_of", return_value=model_config
            ),
            patch.object(
                deepseek_v4_hook,
                "get_platform",
                return_value=SimpleNamespace(is_hip=True, is_cuda=False),
            ),
            patch.object(deepseek_v4_hook, "attn_dp_enabled_of", return_value=True),
        ):
            with self.assertRaises(_PastCopyIn):
                deepseek_v4_hook.validate_deepseek_v41_features(object())
            return os.environ.get(_COPY_IN)

    def test_dp_attention_on_rocm_turns_copy_in_on_unless_set(self):
        value = self._validate(env=None)
        self.assertIsNotNone(value)
        self.assertTrue(envs.SGLANG_AITER_CAPTURE_COPY_IN.parse(value))
        self.assertEqual(self._validate(env="0"), "0")

    def test_copy_in_keeps_the_aiter_all_reduce_off_graph_registration(self):
        aiter_ar = types.ModuleType("aiter.dist.device_communicators.custom_all_reduce")
        aiter_ar.CustomAllreduce = object
        modules = {
            "aiter": types.ModuleType("aiter"),
            "aiter.dist": types.ModuleType("aiter.dist"),
            "aiter.dist.device_communicators": types.ModuleType(
                "aiter.dist.device_communicators"
            ),
            "aiter.dist.device_communicators.custom_all_reduce": aiter_ar,
        }
        for env, registered in (("1", False), (None, True)):
            with (
                self.subTest(env=env),
                patch.dict(sys.modules, modules),
                patch.dict(os.environ, _environ(env), clear=True),
                patch.object(custom_all_reduce, "_is_cuda", False),
                patch.object(custom_all_reduce, "_is_musa", False),
                patch.object(custom_all_reduce, "_is_hip", True),
                patch.object(
                    custom_all_reduce, "_use_amd_deterministic_impl", return_value=False
                ),
            ):
                factory = custom_all_reduce.dispatch_custom_allreduce(
                    group=None, device=None
                )
                self.assertEqual(
                    factory.keywords["enable_register_for_capturing"], registered
                )


if __name__ == "__main__":
    unittest.main()

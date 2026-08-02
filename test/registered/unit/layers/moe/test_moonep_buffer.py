import sys
import types
import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.moonep import MoonEPBuffer
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeMoonEPBuffer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.destroy_calls = 0
        self.__class__.instances.append(self)

    def destroy(self):
        self.destroy_calls += 1


def _fake_moonep_module():
    module = types.ModuleType("moonep")
    module.Buffer = _FakeMoonEPBuffer
    return module


class TestMoonEPBuffer(unittest.TestCase):
    def setUp(self):
        reset_context()
        _FakeMoonEPBuffer.instances.clear()

    def tearDown(self):
        try:
            MoonEPBuffer.destroy_all_buffers()
        finally:
            reset_context()
            _FakeMoonEPBuffer.instances.clear()

    def test_lazily_constructs_and_reuses_buffer_for_static_key(self):
        group = object()

        with (
            patch.dict(sys.modules, {"moonep": _fake_moonep_module()}),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_world_size",
                return_value=4,
            ),
        ):
            buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=1024,
                router_topk=8,
                num_experts=64,
                num_max_dispatch_tokens_per_rank=256,
                num_prefetch_slots=16,
                token_padding=64,
                num_sms=20,
            )
            same_buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=1024,
                router_topk=8,
                num_experts=64,
                num_max_dispatch_tokens_per_rank=256,
                num_prefetch_slots=16,
                token_padding=64,
                num_sms=20,
            )
            larger_buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=1024,
                router_topk=8,
                num_experts=64,
                num_max_dispatch_tokens_per_rank=512,
                num_prefetch_slots=16,
                token_padding=64,
                num_sms=20,
            )

        self.assertIs(buffer, same_buffer)
        self.assertIsNot(buffer, larger_buffer)
        self.assertEqual(len(_FakeMoonEPBuffer.instances), 2)
        self.assertEqual(
            buffer.kwargs,
            {
                "S": 256,
                "H": 1024,
                "K": 8,
                "E": 64,
                "num_ep_ranks": 4,
                "num_sms": 20,
                "token_padding": 64,
                "B": 16,
                "group": group,
            },
        )
        self.assertIs(MoonEPBuffer.get_existing_buffer(), larger_buffer)

    def test_resolves_env_defaults_and_training_safe_prefetch_slots(self):
        group = object()

        with (
            envs.SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(384),
            envs.SGLANG_MOONEP_NUM_PREFETCH_SLOTS.override(-1),
            envs.SGLANG_MOONEP_TOKEN_PADDING.override(32),
            envs.SGLANG_MOONEP_NUM_SMS.override(18),
            patch.dict(sys.modules, {"moonep": _fake_moonep_module()}),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_world_size",
                return_value=8,
            ),
        ):
            buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=2048,
                router_topk=6,
                num_experts=128,
            )

        self.assertEqual(buffer.kwargs["S"], 384)
        self.assertEqual(buffer.kwargs["token_padding"], 32)
        self.assertEqual(buffer.kwargs["num_sms"], 18)
        self.assertEqual(buffer.kwargs["B"], 16)

    def test_rejects_non_divisible_experts_before_allocating(self):
        with (
            patch.dict(sys.modules, {"moonep": _fake_moonep_module()}),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_world_size",
                return_value=6,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "divisible"):
                MoonEPBuffer.get_moonep_buffer(
                    group=object(),
                    hidden_size=1024,
                    router_topk=8,
                    num_experts=64,
                )

        self.assertEqual(_FakeMoonEPBuffer.instances, [])

    def test_destroy_all_buffers_releases_cached_buffers(self):
        group = object()

        with (
            patch.dict(sys.modules, {"moonep": _fake_moonep_module()}),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_world_size",
                return_value=4,
            ),
        ):
            buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=1024,
                router_topk=8,
                num_experts=64,
                num_max_dispatch_tokens_per_rank=256,
            )

        MoonEPBuffer.destroy_all_buffers()

        self.assertEqual(buffer.destroy_calls, 1)
        self.assertIsNone(MoonEPBuffer.get_existing_buffer())


if __name__ == "__main__":
    unittest.main()

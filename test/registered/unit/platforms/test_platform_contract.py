"""SRTPlatform rejects a subclass whose public surface drifts from the base.

Both rejected shapes below are copied from shipped out-of-tree plugins: one
defined six hooks core had renamed or never had, the other overrode a base
method without its ``device`` parameter. Each ran for months without a
signal; the check moves the failure to class-definition time.
"""

import unittest

from sglang.srt.platforms.device_mixin import DeviceMixin
from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_DEAD_NAMES = (
    "default_page_size",
    "get_nsa_kv_pool_cls",
    "supports_torch_compile",
    "use_pynccl_by_default",
    "should_use_fallback_rotary_embedding",
    "get_group_coordinator_device",
)


class TestPlatformContract(CustomTestCase):
    def test_dead_names_from_a_shipped_plugin_are_rejected(self):
        with self.assertRaises(TypeError) as ctx:

            class NeuronLike(SRTPlatform):
                default_page_size = 32

                def get_nsa_kv_pool_cls(self) -> type:
                    raise NotImplementedError

                def supports_torch_compile(self) -> bool:
                    return True

                def use_pynccl_by_default(self, backend: str) -> bool:
                    return False

                def should_use_fallback_rotary_embedding(
                    self, *, head_size: int
                ) -> bool:
                    return False

                def get_group_coordinator_device(
                    self, local_rank: int, *, one_visible_device_per_process=False
                ):
                    return None

        message = str(ctx.exception)
        for name in _DEAD_NAMES:
            self.assertIn(name, message)

    def test_override_missing_a_base_parameter_is_rejected(self):
        with self.assertRaises(TypeError) as ctx:

            class SpyreLike(SRTPlatform):
                def is_pin_memory_available(self) -> bool:
                    return False

        self.assertIn("is_pin_memory_available", str(ctx.exception))
        self.assertIn("'device'", str(ctx.exception))

    def test_override_requiring_an_extra_parameter_is_rejected(self):
        with self.assertRaises(TypeError) as ctx:

            class P(SRTPlatform):
                def get_compile_backend(self, mode=None, extra=None, *, must):
                    return "x"

        self.assertIn("'must'", str(ctx.exception))

    def test_mixin_names_are_checked_in_either_mro_order(self):
        class M(DeviceMixin):
            def vendor_only(self):
                return 1

        with self.assertRaises(TypeError):

            class P1(SRTPlatform, M):
                pass

        with self.assertRaises(TypeError):

            class P2(M, SRTPlatform):
                pass

    def test_documented_extension_patterns_pass(self):
        class P(SRTPlatform):
            capabilities = PlatformCapabilities(supports_triton=True)

            def _vendor_helper(self):
                return 1

            def get_compile_backend(self, mode=None, **kwargs):
                return "vendor"

            def get_device(self, device_id=0):
                return None

            @classmethod
            def seed_everything(cls, seed=None):
                pass

        self.assertTrue(P.capabilities.supports_triton)
        self.assertEqual(P().get_compile_backend(mode="x"), "vendor")


if __name__ == "__main__":
    unittest.main()

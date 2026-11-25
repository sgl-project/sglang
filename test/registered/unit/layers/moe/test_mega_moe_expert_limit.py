"""DeepGEMM MegaMoE expert-count limits are checked while the model is built."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.layers.moe import MoeA2ABackend, mega_moe
from sglang.srt.models import kimi_k3
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _ReachedLayerConstruction(Exception):
    pass


class TestMegaMoeNumExperts(CustomTestCase):
    def test_accepts_layouts_within_the_limits(self):
        # Kimi-K3 at EP8 / EP2, and both limits exactly.
        for num_experts, ep_size in [(896, 8), (896, 2), (512, 1), (2048, 4)]:
            with self.subTest(num_experts=num_experts, ep_size=ep_size):
                mega_moe.check_mega_moe_num_experts(num_experts, ep_size)

    def test_rejects_more_than_512_experts_per_rank(self):
        with self.assertRaisesRegex(
            ValueError,
            r"at most 2048 routed experts and 512 routed experts per rank; got "
            r"896 experts over an EP group of 1 rank\(s\) = 896 per rank.*"
            r"--tp-size.*--moe-a2a-backend",
        ):
            mega_moe.check_mega_moe_num_experts(896, 1)
        with self.assertRaisesRegex(ValueError, "= 513 per rank"):
            mega_moe.check_mega_moe_num_experts(1026, 2)

    def test_rejects_more_than_2048_experts(self):
        # 512 per rank, but the EP group holds more experts than the signal block.
        with self.assertRaisesRegex(ValueError, "got 4096 experts"):
            mega_moe.check_mega_moe_num_experts(4096, 8)


class TestKimiK3MegaMoeNumExperts(CustomTestCase):
    def setUp(self):
        super().setUp()
        scope = get_context().override_server_args(
            enable_eplb=False, ep_num_redundant_experts=0
        )
        scope.install()
        self.addCleanup(scope.restore)
        self.config = KimiLinearConfig(num_experts=896)

    @contextmanager
    def _build(self, a2a_backend, ep_size, is_hip=False):
        """Patch a KimiK3MoE build to stop at its first sub-layer; yields the
        (wrapped) expert-count check."""
        with (
            patch.object(kimi_k3, "get_moe_a2a_backend", return_value=a2a_backend),
            patch.object(
                kimi_k3,
                "get_parallel",
                return_value=SimpleNamespace(moe_ep_size=ep_size, tp_size=ep_size),
            ),
            patch.object(kimi_k3, "_is_hip", is_hip),
            patch.object(kimi_k3, "MoEGate", side_effect=_ReachedLayerConstruction),
            patch.object(
                mega_moe,
                "check_mega_moe_num_experts",
                wraps=mega_moe.check_mega_moe_num_experts,
            ) as check,
        ):
            yield check

    def _assert_builds(self, a2a_backend, ep_size, is_hip=False):
        with self._build(a2a_backend, ep_size, is_hip) as check:
            with self.assertRaises(_ReachedLayerConstruction):
                kimi_k3.KimiK3MoE(self.config)
        return check

    def test_megamoe_rejects_all_experts_on_one_rank(self):
        with self._build(MoeA2ABackend.MEGAMOE, 1):
            # Raised before any sub-layer is created.
            with self.assertRaisesRegex(ValueError, "= 896 per rank"):
                kimi_k3.KimiK3MoE(self.config)

    def test_megamoe_accepts_supported_ep_sizes(self):
        for ep_size in (2, 8, 16):
            with self.subTest(ep_size=ep_size):
                check = self._assert_builds(MoeA2ABackend.MEGAMOE, ep_size)
                check.assert_called_once_with(896, ep_size)

    def test_other_backends_are_not_checked(self):
        # The default runner serves --ep-size 1 (896 experts on every rank).
        for backend in (MoeA2ABackend.NONE, MoeA2ABackend.DEEPEP):
            with self.subTest(backend=backend):
                self._assert_builds(backend, 1).assert_not_called()

    def test_rocm_megamoe_is_not_checked(self):
        # megamoe on ROCm is aiter MegaMoEv2, not DeepGEMM.
        self._assert_builds(MoeA2ABackend.MEGAMOE, 1, is_hip=True).assert_not_called()


if __name__ == "__main__":
    unittest.main()

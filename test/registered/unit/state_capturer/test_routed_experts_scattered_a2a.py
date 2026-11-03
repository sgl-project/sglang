"""RoutedExpertsCapturer: DeepEP-family backend recognition and capture width."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.state_capturer import routed_experts as re_mod
from sglang.srt.state_capturer.base import BaseDeviceCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestGetLocalSliceBackendBranch(CustomTestCase):
    T, L, K = 16, 3, 4

    def _capturer(self):
        cap = object.__new__(re_mod.RoutedExpertsCapturer)
        buf = torch.arange(self.T * self.L * self.K, dtype=torch.int32).reshape(
            self.T, self.L, self.K
        )
        cap.device_cache = SimpleNamespace(buffer=buf)
        cap.topk_size = self.K
        return cap, buf

    def _slice(self, cap, n_local):
        fb = SimpleNamespace(out_cache_loc=torch.empty(n_local))
        return cap._get_local_slice(fb, can_run_graph=False, cuda_graph_batch=None)

    def test_deepep_v2_reads_buffer_head(self):
        cap, buf = self._capturer()
        with (
            mock.patch.object(re_mod, "is_dp_attention_enabled", return_value=True),
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("deepep_v2")
            ),
        ):
            out = self._slice(cap, n_local=5)
        self.assertTrue(torch.equal(out, buf[0:5, :, : self.K]))

    def test_deepep_v2_matches_deepep(self):
        cap, _ = self._capturer()
        outs = []
        for backend in ("deepep", "deepep_v2"):
            with (
                mock.patch.object(re_mod, "is_dp_attention_enabled", return_value=True),
                mock.patch.object(
                    re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend(backend)
                ),
            ):
                outs.append(self._slice(cap, n_local=7))
        self.assertTrue(torch.equal(outs[0], outs[1]))

    def test_tp_moe_reads_global_offset(self):
        cap, buf = self._capturer()
        with (
            mock.patch.object(re_mod, "is_dp_attention_enabled", return_value=True),
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("none")
            ),
            mock.patch.object(re_mod, "get_dp_local_slice_cpu", return_value=(6, 4)),
        ):
            out = self._slice(cap, n_local=999)
        self.assertTrue(torch.equal(out, buf[6:10, :, : self.K]))


class TestCaptureRoutedIdsOnly(CustomTestCase):
    TOPK_SIZE = 3

    def _capturer(self):
        cap = object.__new__(re_mod.RoutedExpertsCapturer)
        cap.num_fused_shared_experts = 1
        cap.device_cache = BaseDeviceCache(
            max_batch_size=8,
            num_layers=2,
            topk_size=self.TOPK_SIZE + 1,
            device="cpu",
            name="routed_experts",
        )
        cap.gather_buffer = torch.empty((8, self.TOPK_SIZE + 1), dtype=torch.int32)
        return cap

    def _capture(self, cap, ids, backend):
        with (
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend(backend)
            ),
            mock.patch.object(
                re_mod,
                "attn_tp_all_gather_into_tensor",
                side_effect=lambda out, inp: out.copy_(inp),
            ),
            mock.patch.object(
                re_mod, "get_parallel", return_value=SimpleNamespace(attn_tp_size=1)
            ),
        ):
            cap.capture(layer_id=1, topk_indices=ids)

    def test_routed_only_ids_fill_the_routed_columns(self):
        """Routed-only capture must fill the routed columns, including scattered a2a."""
        ids = torch.arange(4 * self.TOPK_SIZE, dtype=torch.int32).view(4, -1)
        for backend in ("none", "deepep"):
            with self.subTest(backend=backend):
                cap = self._capturer()
                self._capture(cap, ids, backend)
                self.assertTrue(
                    torch.equal(cap.device_cache.buffer[:4, 1, : self.TOPK_SIZE], ids)
                )

    def test_other_widths_are_rejected(self):
        """Only the routed-only width may be padded; any other width is a caller bug."""
        ids = torch.zeros((4, self.TOPK_SIZE - 1), dtype=torch.int32)
        with self.assertRaises(ValueError):
            self._capture(self._capturer(), ids, "none")


class TestCapturerWidthFollowsTheInstalledCount(CustomTestCase):
    def test_a_wrapper_model_gets_room_for_the_fused_shared_column(self):
        """Capture sizing must use the supplied wrapper model's shared-expert layout."""
        seen = {}

        def record(self, model_config, **kwargs):
            seen.update(kwargs)

        features = SimpleNamespace(enable_return_routed_experts=True)
        with (
            mock.patch.object(
                re_mod, "get_exec", return_value=SimpleNamespace(features=features)
            ),
            mock.patch.object(re_mod.RoutedExpertsCapturer, "__init__", record),
        ):
            re_mod.RoutedExpertsCapturer.create(
                model=SimpleNamespace(
                    shared_experts_config=SharedExpertsConfig(
                        num_fused_shared_experts=1
                    )
                ),
                model_config=SimpleNamespace(),
                num_tokens=8,
                max_running_requests=2,
                device="cpu",
            )
        self.assertEqual(seen["num_fused_shared_experts"], 1)


if __name__ == "__main__":
    unittest.main()

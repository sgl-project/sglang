"""Routed-expert capture with global DP rows and local sequence shards."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.state_capturer import routed_experts as re_mod
from sglang.srt.state_capturer.base import BaseDeviceCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestScatteredA2ABackendHelper(CustomTestCase):
    def test_classification(self):
        expected = {
            "deepep": True,
            "deepep_v2": True,
            "none": False,
            "mooncake": False,
        }
        for value, exp in expected.items():
            with mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend(value)
            ):
                self.assertEqual(
                    re_mod._is_scattered_a2a_backend(), exp, f"backend={value}"
                )


class TestGetLocalSliceBackendBranch(CustomTestCase):
    T, L, K = 16, 3, 4

    def _capturer(self):
        cap = object.__new__(re_mod.RoutedExpertsCapturer)
        buf = torch.arange(self.T * self.L * self.K, dtype=torch.int32).reshape(
            self.T, self.L, self.K
        )
        cap.device_cache = BaseDeviceCache(
            self.T, self.L, self.K, "cpu", "routed_experts"
        )
        cap.device_cache.buffer.copy_(buf)
        cap.topk_size = self.K
        cap.capture_local = False
        cap.gather_buffer = torch.empty(self.T * 2, self.K, dtype=torch.int32)
        cap.host_cache = SimpleNamespace(
            buffer=torch.full((32, self.L, self.K), -1, dtype=torch.int32)
        )
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

    def test_local_capture_survives_dp_offsets_sharding_and_overlap(self):
        from sglang.srt.layers.moe import topk

        cap, _ = self._capturer()
        expected = torch.arange(6 * self.L * self.K, dtype=torch.int32).reshape(
            6, self.L, self.K
        )
        locations = torch.tensor([4, 9, 13, 21, 7])
        forward_batch = SimpleNamespace(out_cache_loc=locations)
        request_pool = SimpleNamespace(
            req_to_token=torch.tensor([[4, 9, 13, 21, 7, 0]])
        )
        # Switch layouts on one capturer as real unaligned-prefill fallbacks do.
        for sharded, graph in ((True, False), (False, False), (True, True)):
            with self.subTest(sharded=sharded, graph=graph):
                config = topk.TopKConfig(
                    top_k=self.K,
                    routed_experts_capture_local=True,
                    routed_experts_capture_sequence_sharded=sharded,
                )
                cap.device_cache.buffer.fill_(-900)
                with (
                    mock.patch.object(
                        topk, "get_global_experts_capturer", return_value=cap
                    ),
                    mock.patch.object(
                        re_mod, "is_dp_attention_enabled", return_value=True
                    ),
                    mock.patch.object(
                        re_mod,
                        "get_moe_a2a_backend",
                        return_value=MoeA2ABackend("flashinfer"),
                    ),
                    mock.patch.object(
                        re_mod,
                        "get_parallel",
                        return_value=SimpleNamespace(attn_tp_size=2),
                    ),
                    mock.patch.object(
                        re_mod, "get_dp_local_slice_cpu", return_value=(8, 5)
                    ) as global_slice,
                    mock.patch.object(
                        re_mod, "attn_tp_all_gather_into_tensor"
                    ) as gather,
                ):
                    for layer in range(self.L):
                        rows = expected[:, layer, :]

                        def gather_rows(output, local):
                            self.assertTrue(torch.equal(local, rows[:3]))
                            output.copy_(rows)

                        gather.side_effect = gather_rows
                        topk.capture_routed_experts_if_allowed(
                            config, layer, rows[:3] if sharded else rows[:5]
                        )
                    pending = cap.on_forward_end(
                        forward_batch,
                        can_run_graph=graph,
                        cuda_graph_batch=8 if graph else None,
                        no_copy_to_cpu=True,
                    )
                    global_slice.assert_not_called()
                    self.assertEqual(gather.call_count, self.L if sharded else 0)
                # Another forward may reuse the device buffer before D2H finishes.
                cap.device_cache.buffer.fill_(-800)
                pending.map_device_tensors(lambda tensor: tensor.clone())
                pending.finalize()
                result = cap.get_topk(0, 6, request_pool)
                self.assertTrue(torch.equal(result, expected[:5]))


if __name__ == "__main__":
    unittest.main()

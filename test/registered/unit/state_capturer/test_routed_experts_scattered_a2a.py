"""RoutedExpertsCapturer: DeepEP-family backend recognition, capture width,
and routed-layer host storage."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.configs.nemotron_h import NemotronHConfig
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.state_capturer import base as base_mod
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
        cap._slot_of_layer = {0: 0, 1: 1}
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


class TestRoutedLayerHostStorage(CustomTestCase):
    PATTERN = "ME*EM"
    K = 2
    NUM_TOKENS = 16
    OUT_CACHE_LOC = torch.tensor([3, 7, 1, 12, 5])

    @staticmethod
    def _fake_host_alloc(dims, dtype, **_):
        # Like the real mmap-backed allocator, reject a zero-byte mapping.
        if torch.Size(dims).numel() == 0:
            raise OSError("Invalid argument")
        return torch.empty(dims, dtype=dtype)

    def _capturer(self, hf_text_config):
        parallel = SimpleNamespace(num_dp_ranks=1)
        with (
            mock.patch.object(re_mod, "get_parallel", return_value=parallel),
            mock.patch.object(
                re_mod,
                "get_schedule",
                return_value=SimpleNamespace(chunked_prefill_size=8),
            ),
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("none")
            ),
            mock.patch.dict(
                base_mod.ALLOC_MEMORY_FUNCS, {"cpu": self._fake_host_alloc}
            ),
        ):
            return re_mod.RoutedExpertsCapturer(
                SimpleNamespace(hf_text_config=hf_text_config),
                num_tokens=self.NUM_TOKENS,
                max_running_requests=4,
                num_fused_shared_experts=0,
                device="cpu",
            )

    def _run_forward(self, cap, routed_layers):
        with (
            mock.patch.object(re_mod, "is_dp_attention_enabled", return_value=False),
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("none")
            ),
        ):
            for layer_id in routed_layers:
                topk = torch.randint(
                    0, 64, (len(self.OUT_CACHE_LOC), self.K), dtype=torch.int32
                )
                cap.capture(layer_id=layer_id, topk_indices=topk)
            return cap.on_forward_end(
                SimpleNamespace(
                    out_cache_loc=self.OUT_CACHE_LOC, out_cache_loc_virtual=None
                ),
                can_run_graph=False,
                cuda_graph_batch=None,
            )

    def _read(self, cap):
        pool = SimpleNamespace(req_to_token=self.OUT_CACHE_LOC.unsqueeze(0))
        return cap.get_topk(
            req_pool_idx=0, seqlen=5, req_to_token_pool=pool, start_len=1
        )

    def _nemotron_h(self, pattern):
        return NemotronHConfig(
            hybrid_override_pattern=pattern, num_experts_per_tok=self.K
        )

    def test_routed_layers_read_back_like_all_layer_storage(self):
        config = self._nemotron_h(self.PATTERN)
        all_layers_config = SimpleNamespace(
            num_hidden_layers=len(self.PATTERN), num_experts_per_tok=self.K
        )
        routed = self._capturer(config)
        oracle = self._capturer(all_layers_config)
        self.assertEqual(routed.host_cache.buffer.shape[1], self.PATTERN.count("E"))

        torch.manual_seed(0)
        self._run_forward(routed, config.moe_layer_ids)
        torch.manual_seed(0)
        self._run_forward(oracle, config.moe_layer_ids)
        # A 2-token chunk splits the 3 returned tokens across a chunk boundary.
        with mock.patch.object(re_mod, "_READBACK_CHUNK_TOKENS", 2):
            routed_out = self._read(routed)
        self.assertEqual(routed_out.shape, (3, len(self.PATTERN), self.K))
        self.assertTrue(torch.equal(routed_out, self._read(oracle)))

    def test_model_without_routed_layers_returns_zeros(self):
        """A Nemotron-H pattern with no MoE layer still returns all-zero rows."""
        pattern = "M*-M"
        cap = self._capturer(self._nemotron_h(pattern))
        self._run_forward(cap, routed_layers=[])
        self.assertTrue(
            torch.equal(
                self._read(cap),
                torch.zeros((3, len(pattern), self.K), dtype=torch.int32),
            )
        )


if __name__ == "__main__":
    unittest.main()

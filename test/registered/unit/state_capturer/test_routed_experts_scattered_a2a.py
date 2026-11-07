"""RoutedExpertsCapturer: DeepEP-family backend recognition, capture width,
and the output-rank host cache."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.runtime_context import get_parallel
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
        cap.device_cache = BaseDeviceCache(
            max_batch_size=8,
            num_layers=2,
            topk_size=self.TOPK_SIZE + 1,
            device="cpu",
            name="routed_experts",
        )
        cap.device_topk_size = self.TOPK_SIZE + 1
        cap.local_cache = None
        cap.gather_buffer = torch.empty((8, self.TOPK_SIZE + 1), dtype=torch.int32)
        return cap

    def _capture(self, cap, ids, backend):
        cap._gathers_per_layer = MoeA2ABackend(backend).is_deepep()
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

        features = SimpleNamespace(
            enable_return_routed_experts=True, enable_decoder_swa_bounded_replay=False
        )
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
                    ),
                    modules=lambda: iter(()),
                ),
                model_config=SimpleNamespace(),
                num_tokens=8,
                max_running_requests=2,
                device="cpu",
            )
        self.assertEqual(seen["num_fused_shared_experts"], 1)


class TestOutputRankHostCache(CustomTestCase):
    NUM_LAYERS = 3
    K = 2
    NUM_TOKENS = 16
    OUT_CACHE_LOC = torch.tensor([3, 7, 1, 12, 5])

    def _capturer(self, attn_tp_rank, backend="none"):
        hf_text_config = SimpleNamespace(
            num_hidden_layers=self.NUM_LAYERS, num_experts_per_tok=self.K
        )
        with (
            get_parallel().override(
                num_dp_ranks=1,
                pp_rank=0,
                attn_tp_rank=attn_tp_rank,
                attn_tp_size=2,
                attn_cp_rank=0,
            ),
            mock.patch.object(
                re_mod,
                "get_schedule",
                return_value=SimpleNamespace(chunked_prefill_size=8),
            ),
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend(backend)
            ),
            mock.patch.object(re_mod, "is_dp_attention_enabled", return_value=True),
            mock.patch.dict(
                base_mod.ALLOC_MEMORY_FUNCS,
                {"cpu": lambda dims, dtype, **_: torch.empty(dims, dtype=dtype)},
            ),
        ):
            return re_mod.RoutedExpertsCapturer(
                SimpleNamespace(hf_text_config=hf_text_config),
                num_tokens=self.NUM_TOKENS,
                max_running_requests=4,
                num_fused_shared_experts=0,
                device="cpu",
            )

    def _run_forward(self, cap, topk):
        with (
            mock.patch.object(re_mod, "is_dp_attention_enabled", return_value=False),
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("none")
            ),
        ):
            for layer_id in range(self.NUM_LAYERS):
                cap.capture(layer_id=layer_id, topk_indices=topk[:, layer_id])
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

    def test_only_output_rank_keeps_rows(self):
        """Only the rank that sends outputs to the detokenizer keeps routed
        experts; every other rank kept a device buffer and pinned a host copy
        that no one read."""
        topk = torch.randint(
            0,
            64,
            (len(self.OUT_CACHE_LOC), self.NUM_LAYERS, self.K),
            dtype=torch.int32,
        )

        output_rank = self._capturer(attn_tp_rank=0)
        self.assertTrue(output_rank.has_host_cache)
        self.assertIsNone(self._run_forward(output_rank, topk))
        self.assertTrue(torch.equal(self._read(output_rank), topk[1:4]))

        other_rank = self._capturer(attn_tp_rank=1)
        self.assertFalse(other_rank.has_host_cache)
        self.assertIsNone(other_rank.device_cache)
        self.assertIsNone(self._run_forward(other_rank, topk))
        with self.assertRaises(RuntimeError):
            self._read(other_rank)

    def test_scattered_other_rank_only_joins_the_all_gather(self):
        """Without a known last layer, other ranks still join each layer's all-gather."""
        other_rank = self._capturer(attn_tp_rank=1, backend="deepep")
        self.assertIsNone(other_rank.device_cache)
        gathered_shapes = []
        with (
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("deepep")
            ),
            mock.patch.object(
                re_mod, "get_parallel", return_value=SimpleNamespace(attn_tp_size=2)
            ),
            mock.patch.object(
                re_mod,
                "attn_tp_all_gather_into_tensor",
                side_effect=lambda out, inp: gathered_shapes.append(tuple(out.shape)),
            ),
        ):
            for layer_id in range(self.NUM_LAYERS):
                other_rank.capture(
                    layer_id=layer_id,
                    topk_indices=torch.zeros((3, self.K), dtype=torch.int32),
                )
        self.assertEqual(gathered_shapes, [(6, self.K)] * self.NUM_LAYERS)


class TestLastLayerGather(CustomTestCase):
    """DeepEP: each attention-TP rank keeps its own tokens' ids, and the last
    routed layer all-gathers every layer at once."""

    NUM_LAYERS = 3
    K = 2
    LOCAL_ROWS = 3
    OUT_CACHE_LOC = torch.tensor([3, 7, 1, 12, 5, 9])

    def _parallel(self, attn_tp_rank, pp_rank=0):
        return dict(
            num_dp_ranks=1,
            pp_rank=pp_rank,
            pp_size=2,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=2,
            attn_cp_rank=0,
        )

    def _capturer(
        self,
        parallel,
        routed_layer_ids=tuple(range(NUM_LAYERS)),
        chunked_prefill_size=8,
    ):
        hf_text_config = SimpleNamespace(
            num_hidden_layers=self.NUM_LAYERS, num_experts_per_tok=self.K
        )
        with (
            get_parallel().override(**parallel),
            mock.patch.object(
                re_mod,
                "get_schedule",
                return_value=SimpleNamespace(chunked_prefill_size=chunked_prefill_size),
            ),
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("deepep")
            ),
            mock.patch.object(re_mod, "is_dp_attention_enabled", return_value=True),
            mock.patch.dict(
                base_mod.ALLOC_MEMORY_FUNCS,
                {"cpu": lambda dims, dtype, **_: torch.empty(dims, dtype=dtype)},
            ),
        ):
            return re_mod.RoutedExpertsCapturer(
                SimpleNamespace(hf_text_config=hf_text_config),
                num_tokens=16,
                max_running_requests=4,
                num_fused_shared_experts=0,
                device="cpu",
                routed_layer_ids=None
                if routed_layer_ids is None
                else list(routed_layer_ids),
            )

    def _capture_layers(self, cap, parallel, local_topk, all_gather, layers=None):
        with (
            get_parallel().override(**parallel),
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("deepep")
            ),
            mock.patch.object(
                re_mod, "attn_tp_all_gather_into_tensor", side_effect=all_gather
            ),
        ):
            for layer_id in range(self.NUM_LAYERS) if layers is None else layers:
                cap.capture(layer_id=layer_id, topk_indices=local_topk[:, layer_id])

    def _local_topk(self):
        return torch.randint(
            0, 64, (self.LOCAL_ROWS, self.NUM_LAYERS, self.K), dtype=torch.int32
        )

    def test_last_layer_all_gather_rebuilds_the_per_layer_rows(self):
        shares = [self._local_topk() for _ in range(2)]
        sent, calls = {}, []

        def all_gather(rank):
            # The last rank to join sees every share, like the real collective.
            def _all_gather(output, input_):
                calls.append(rank)
                sent[rank] = input_.clone()
                if len(sent) == 2:
                    output.copy_(torch.cat([sent[0], sent[1]], dim=0))

            return _all_gather

        output_parallel, peer_parallel = self._parallel(0), self._parallel(1)
        output_rank = self._capturer(output_parallel)
        peer = self._capturer(peer_parallel)
        self.assertIsNone(peer.device_cache)
        self.assertIsNotNone(peer.all_gather_scratch)
        self.assertIsNone(output_rank.all_gather_scratch)

        self._capture_layers(peer, peer_parallel, shares[1], all_gather(1))
        self._capture_layers(output_rank, output_parallel, shares[0], all_gather(0))
        # One all-gather per rank per forward, not one per layer.
        self.assertEqual(calls, [1, 0])

        with (
            get_parallel().override(**output_parallel),
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("deepep")
            ),
            mock.patch.object(re_mod, "is_dp_attention_enabled", return_value=True),
        ):
            output_rank.on_forward_end(
                SimpleNamespace(
                    out_cache_loc=self.OUT_CACHE_LOC, out_cache_loc_virtual=None
                ),
                can_run_graph=False,
                cuda_graph_batch=None,
            )
        # What the per-layer all-gather wrote: rank 0's rows, then rank 1's.
        pool = SimpleNamespace(req_to_token=self.OUT_CACHE_LOC.unsqueeze(0))
        self.assertTrue(
            torch.equal(
                output_rank.get_topk(
                    req_pool_idx=0, seqlen=7, req_to_token_pool=pool, start_len=0
                ),
                torch.cat(shares, dim=0),
            )
        )

    def test_whole_shares_fit_when_capacity_is_not_a_multiple(self):
        parallel = self._parallel(0)
        cap = self._capturer(parallel, chunked_prefill_size=9)
        self.assertEqual(cap.device_cache.buffer.shape[0], 10)
        sizes = []
        self._capture_layers(
            cap,
            parallel,
            torch.zeros((5, self.NUM_LAYERS, self.K), dtype=torch.int32),
            lambda output, input_: sizes.append((output.shape[0], input_.shape[0])),
        )
        self.assertEqual(sizes, [(10, 5)])

    def test_capture_after_the_gather_layer_raises(self):
        parallel = self._parallel(0)
        cap = self._capturer(parallel, routed_layer_ids=(0, 1))
        with self.assertRaisesRegex(RuntimeError, "after the all-gather layer 1"):
            self._capture_layers(
                cap, parallel, self._local_topk(), lambda output, input_: None
            )

    def test_unequal_rows_across_layers_raise(self):
        parallel = self._parallel(0)
        cap = self._capturer(parallel)
        local_topk = self._local_topk()
        with self.assertRaisesRegex(RuntimeError, "captured 2 routed-expert rows"):
            with (
                get_parallel().override(**parallel),
                mock.patch.object(
                    re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("deepep")
                ),
            ):
                cap.capture(layer_id=0, topk_indices=local_topk[:, 0])
                cap.capture(layer_id=1, topk_indices=local_topk[:2, 1])

    def test_bounded_replay_keeps_per_layer_gathers(self):
        """Bounded replay runs late layers on fewer rows than early ones."""
        seen = {}
        features = SimpleNamespace(
            enable_return_routed_experts=True, enable_decoder_swa_bounded_replay=True
        )
        with (
            mock.patch.object(
                re_mod, "get_exec", return_value=SimpleNamespace(features=features)
            ),
            mock.patch.object(
                re_mod.RoutedExpertsCapturer,
                "__init__",
                lambda self, model_config, **kwargs: seen.update(kwargs),
            ),
        ):
            re_mod.RoutedExpertsCapturer.create(
                model=SimpleNamespace(
                    shared_experts_config=SharedExpertsConfig(
                        num_fused_shared_experts=0
                    ),
                    modules=lambda: iter(()),
                ),
                model_config=SimpleNamespace(),
                num_tokens=8,
                max_running_requests=2,
                device="cpu",
            )
        self.assertIsNone(seen["routed_layer_ids"])

    def test_unknown_last_layer_gathers_per_layer(self):
        parallel = self._parallel(1)
        cap = self._capturer(parallel, routed_layer_ids=None)
        self.assertIsNone(cap.local_cache)
        calls = []
        self._capture_layers(
            cap,
            parallel,
            self._local_topk(),
            lambda output, input_: calls.append(output.shape[0]),
        )
        self.assertEqual(calls, [2 * self.LOCAL_ROWS] * self.NUM_LAYERS)

    def test_later_pipeline_stages_send_nothing(self):
        parallel = self._parallel(1, pp_rank=1)
        cap = self._capturer(parallel)
        self.assertIsNone(cap.local_cache)
        self.assertIsNone(cap.device_cache)
        self._capture_layers(
            cap,
            parallel,
            self._local_topk(),
            AssertionError("a later pipeline stage joined the all-gather"),
        )


class TestRoutedLayerIds(CustomTestCase):
    def _topk(self, layer_id, captures=True):
        module = object.__new__(TopK)
        torch.nn.Module.__init__(module)
        module.layer_id = layer_id
        module.topk_config = SimpleNamespace(allow_routed_experts_capture=captures)
        return module

    def test_capturing_topk_layers_in_order(self):
        model = torch.nn.ModuleList(
            [
                self._topk(3),
                torch.nn.Linear(2, 2),
                self._topk(7),
                self._topk(9, captures=False),
            ]
        )
        self.assertEqual(re_mod._routed_layer_ids(model), [3, 7])

    def test_no_capturing_topk(self):
        model = torch.nn.ModuleList([self._topk(2, captures=False)])
        self.assertEqual(re_mod._routed_layer_ids(model), [])


if __name__ == "__main__":
    unittest.main()

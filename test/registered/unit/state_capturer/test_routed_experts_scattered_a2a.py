"""RoutedExpertsCapturer: DeepEP-family backend recognition, capture width,
and the output-rank host cache."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.batch_overlap.operations import _StageExecutor
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_parallel
from sglang.srt.state_capturer import base as base_mod
from sglang.srt.state_capturer import routed_experts as re_mod
from sglang.srt.state_capturer.base import BaseDeviceCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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

    def test_tbo_children_write_their_parent_rows(self):
        """Children must not overwrite each other, even when padded to a common length."""
        cap = self._capturer()
        for token_range, value in (((0, 1), 1), ((1, 4), 2)):
            ids = torch.full((3, self.TOPK_SIZE), value, dtype=torch.int32)
            with forward_context(
                ForwardContext(attn_backend=None, tbo_parent_token_range=token_range)
            ):
                self._capture(cap, ids, "deepep")
        self.assertEqual(cap.device_cache.buffer[:4, 1, 0].tolist(), [1, 2, 2, 2])

    def _read_non_ep_dp_tbo(self, rank, parent_sizes, split):
        """Capture both gathered children on ``rank`` and read back its parent rows."""
        cap = self._capturer()
        cap.device_cache.buffer.fill_(-7)
        cap.topk_size = self.TOPK_SIZE
        ranges = [
            [(0, split[r]), (split[r], size)] for r, size in enumerate(parent_sizes)
        ]
        padded = [max(e - s for s, e in rank_ranges) for rank_ranges in ranges]
        offset = sum(parent_sizes[:rank])
        with (
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("none")
            ),
            mock.patch.object(re_mod, "is_dp_attention_enabled", return_value=True),
            mock.patch.object(re_mod, "get_dp_global_num_tokens", return_value=padded),
            mock.patch.object(re_mod, "dp_slot_in", return_value=rank),
            mock.patch("sglang.srt.layers.dp_attention.dp_slot_in", return_value=rank),
        ):
            for child in range(2):
                gathered = []
                for r, rank_ranges in enumerate(ranges):
                    s, e = rank_ranges[child]
                    gathered += [100 * (r + 1) + i for i in range(s, e)]
                    gathered += [-1] * (padded[r] - (e - s))
                ids = torch.tensor(gathered, dtype=torch.int32)
                ids = ids.unsqueeze(1).expand(-1, self.TOPK_SIZE + 1)
                with forward_context(
                    ForwardContext(
                        attn_backend=None,
                        tbo_parent_token_range=ranges[rank][child],
                        tbo_parent_dp_offset=offset,
                    )
                ):
                    cap.capture(layer_id=1, topk_indices=ids)
            batch = SimpleNamespace(
                global_num_tokens_cpu=parent_sizes,
                out_cache_loc=torch.empty(parent_sizes[rank]),
            )
            return cap._get_local_slice(batch, False, None)[:, 1, 0].tolist()

    def test_non_ep_dp_tbo_reads_each_ranks_rows(self):
        """Gathered children must land where each DP rank's readback looks."""
        for parent_sizes, split in (
            ([4, 4], [2, 2]),
            ([2, 6], [1, 3]),
            ([6, 2], [4, 1]),
            ([0, 4], [0, 2]),
            ([3, 1], [3, 0]),
        ):
            for rank, size in enumerate(parent_sizes):
                with self.subTest(parent_sizes=parent_sizes, rank=rank):
                    self.assertEqual(
                        self._read_non_ep_dp_tbo(rank, parent_sizes, split),
                        [100 * (rank + 1) + i for i in range(size)],
                    )

    def test_stage_executor_exposes_parent_rows(self):
        """A child's ForwardContext must carry the rows capture writes."""
        batch = SimpleNamespace(
            moe_num_token_non_padded=None,
            tbo_parent_token_range=(2, 5),
            tbo_parent_dp_offset=6,
            global_dp_buffer_len=None,
            tbo_padded_len=3,
            global_num_tokens_cpu=None,
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: False),
        )
        executor = _StageExecutor(
            "test",
            [],
            {"forward_batch": batch},
            child_ctx=ForwardContext(attn_backend=None),
        )
        self.assertEqual(executor._child_ctx.tbo_parent_token_range, (2, 5))
        self.assertEqual(executor._child_ctx.tbo_parent_dp_offset, 6)

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
        """Under DeepEP every rank must join each layer's all-gather, or the
        output rank hangs; the other ranks still keep no rows."""
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


if __name__ == "__main__":
    unittest.main()

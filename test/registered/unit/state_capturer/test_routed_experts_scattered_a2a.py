"""RoutedExpertsCapturer: DeepEP-family backend recognition, capture width,
the output-rank host cache, and storage dtype."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pybase64
import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
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


def _report_num_experts(num_experts):
    """Make the model report num_experts routed experts (None: no report)."""
    return mock.patch.object(
        re_mod.ModelConfigForExpertLocation,
        "from_model_config",
        return_value=(
            None
            if num_experts is None
            else ModelConfigForExpertLocation(
                num_layers=1, num_logical_experts=num_experts
            )
        ),
    )


def _create_kwargs(
    *, routed_experts_dtype="int32", num_experts=None, num_fused_shared_experts=0
):
    seen = {}

    def record(self, model_config, **kwargs):
        seen.update(kwargs)

    with (
        get_context().override_server_args(
            enable_return_routed_experts=True,
            routed_experts_dtype=routed_experts_dtype,
        ),
        mock.patch.object(re_mod.RoutedExpertsCapturer, "__init__", record),
        _report_num_experts(num_experts),
    ):
        re_mod.RoutedExpertsCapturer.create(
            model=SimpleNamespace(
                shared_experts_config=SharedExpertsConfig(
                    num_fused_shared_experts=num_fused_shared_experts
                )
            ),
            model_config=SimpleNamespace(),
            num_tokens=8,
            max_running_requests=2,
            device="cpu",
        )
    return seen


class TestCapturerWidthFollowsTheInstalledCount(CustomTestCase):
    def test_a_wrapper_model_gets_room_for_the_fused_shared_column(self):
        """Capture sizing must use the supplied wrapper model's shared-expert layout."""
        seen = _create_kwargs(num_fused_shared_experts=1)
        self.assertEqual(seen["num_fused_shared_experts"], 1)
        # int32 never needs the expert count, which this model does not report.
        self.assertEqual(seen["dtype"], torch.int32)


class TestInt16Storage(CustomTestCase):
    NUM_LAYERS = 3
    K = 2
    OUT_CACHE_LOC = torch.tensor([3, 7, 1, 12, 5])

    def setUp(self):
        super().setUp()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(get_context().override_server_args(chunked_prefill_size=8))
        stack.enter_context(get_flags().moe.override(a2a_backend=MoeA2ABackend.NONE))
        stack.enter_context(get_flags().dp.override(enabled=False))
        # The output rank keeps the caches.
        stack.enter_context(
            get_parallel().override(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0)
        )
        # The real allocator registers host memory with CUDA; CPU tensors suffice.
        stack.enter_context(
            mock.patch.dict(
                base_mod.ALLOC_MEMORY_FUNCS,
                {"cpu": lambda dims, dtype, **_: torch.empty(dims, dtype=dtype)},
            )
        )

    def _capturer(self):
        hf_text_config = SimpleNamespace(
            num_hidden_layers=self.NUM_LAYERS, num_experts_per_tok=self.K
        )
        return re_mod.RoutedExpertsCapturer(
            SimpleNamespace(hf_text_config=hf_text_config),
            num_tokens=16,
            max_running_requests=4,
            num_fused_shared_experts=0,
            device="cpu",
            dtype=torch.int16,
        )

    def test_expert_ids_round_trip_through_int16(self):
        """The highest expert id and the -1 padding mark survive int16 storage
        and come back through the response tag."""
        top_id = torch.iinfo(torch.int16).max
        topk = torch.tensor(
            [[0, 1], [top_id, -1], [-1, -1], [17, top_id - 1], [2, 3]],
            dtype=torch.int32,
        )
        forward_batch = SimpleNamespace(
            out_cache_loc=self.OUT_CACHE_LOC, out_cache_loc_virtual=None
        )
        pool = SimpleNamespace(req_to_token=self.OUT_CACHE_LOC.unsqueeze(0))
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                cap = self._capturer()
                # Capture narrows on the device, so the D2H moves int16.
                self.assertEqual(cap.device_cache.buffer.dtype, torch.int16)
                self.assertEqual(cap.host_cache.buffer.dtype, torch.int16)
                for layer_id in range(self.NUM_LAYERS):
                    cap.capture(layer_id=layer_id, topk_indices=topk)
                output = cap.on_forward_end(
                    forward_batch,
                    can_run_graph=False,
                    cuda_graph_batch=None,
                    no_copy_to_cpu=overlap,
                )
                if overlap:
                    output.map_device_tensors(lambda t: t.cpu())
                    output.finalize()
                out = cap.get_topk(
                    req_pool_idx=0, seqlen=5, req_to_token_pool=pool, start_len=1
                )
                self.assertEqual(out.dtype, torch.int16)
                decoded = re_mod.extract_routed_experts_from_meta_info(
                    {"meta_info": _response_meta_info(out, "int16")}
                )
                expected = topk[1:4, None].expand(-1, self.NUM_LAYERS, -1)
                self.assertTrue(np.array_equal(decoded, expected.flatten().numpy()))


def _response_meta_info(ids, dtype=None):
    # Encoded as the detokenizer does; dtype is the tag the response carries.
    meta_info = {"routed_experts": pybase64.b64encode(ids.numpy().tobytes()).decode()}
    if dtype is not None:
        meta_info["routed_experts_dtype"] = dtype
    return meta_info


class TestIdsDtype(CustomTestCase):
    def test_untagged_routed_experts_decode_as_int32(self):
        """Servers that predate --routed-experts-dtype send int32 without a tag."""
        ids = torch.tensor([0, 70000, -1], dtype=torch.int32)
        decoded = re_mod.extract_routed_experts_from_meta_info(
            {"meta_info": _response_meta_info(ids)}
        )
        self.assertEqual(decoded.tolist(), [0, 70000, -1])

    def test_create_uses_the_configured_dtype(self):
        seen = _create_kwargs(routed_experts_dtype="int16", num_experts=896)
        self.assertEqual(seen["dtype"], torch.int16)

    def test_int16_must_hold_the_largest_expert_id(self):
        """Ids run from -1 to num_experts - 1, so int16 holds 32768 experts."""
        with _report_num_experts(32768):
            self.assertEqual(re_mod._ids_dtype("int16", SimpleNamespace()), torch.int16)
        rejected = {"too many experts": 32769, "unknown expert count": None}
        for case, num_experts in rejected.items():
            with (
                self.subTest(case),
                _report_num_experts(num_experts),
                self.assertRaises(ValueError),
            ):
                re_mod._ids_dtype("int16", SimpleNamespace())


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

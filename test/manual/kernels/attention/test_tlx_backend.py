"""GPU tests through SGLang forward_extend/forward_decode, without model weights."""

import inspect
import unittest

import torch

from sglang.kernels.ops.attention.tlx.paged import unified_attention, unsupported_reason
from sglang.srt.layers.attention.aiter_backend import ForwardMetadata
from sglang.srt.layers.attention.tlx_backend import TLX_PHASES, TlxAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.kernels.tlx_attention import make_paged_case, paged_reference
from sglang.test.test_utils import CustomTestCase


class TensorPool:
    def __init__(self, k, v):
        self.k, self.v = k.flatten(0, 1), v.flatten(0, 1)
        self.start_layer, self.layer_num = 0, 1
        self.layer_transfer_counter = None
        self.layer_router = PlainPoolRouter(self, "full")

    def get_kv_buffer(self, layer):
        return self.k, self.v

    def get_key_buffer(self, layer):
        return self.k

    def get_value_buffer(self, layer):
        return self.v


class TestTlxBackend(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if (
            torch.version.hip is None
            or not torch.cuda.is_available()
            or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250")
        ):
            raise unittest.SkipTest("requires gfx1250 and the uTLX TDM extension")

    def case(self, qlens, klens, window=-1):
        args, reference = make_paged_case(
            fp8=True, qlens=qlens, klens=klens, window=window
        )
        backend = object.__new__(TlxAttnBackend)
        backend._tlx_attention = unified_attention
        backend._tlx_parameters = frozenset(
            inspect.signature(unified_attention).parameters
        )
        backend._tlx_unsupported_reason = unsupported_reason
        backend._tlx_fallback_reasons = set()
        backend._tlx_dispatched = set()
        # Every phase on uTLX, so the cases below exercise the kernels.
        backend._tlx_phases = frozenset(TLX_PHASES)
        backend._tlx_routed_phases = set()
        backend.use_mla = False
        backend.use_triton_unified_attention = True
        backend._use_unified_verify = True
        backend.use_sliding_window_kv_pool = False
        backend.kv_cache_is_vectorized_5d = False
        # AiterAttnBackend's Muse fast paths are gfx95-only, so gfx1250 runs them disabled.
        backend.use_muse_attention_paths = False
        backend._muse_ck_block_prefill = False
        backend.muse_reuse_extend_metadata = False
        backend.muse_reuse_verify_lengths = False
        backend.muse_extend_fp8_q = False
        backend.muse_extend_gluon = False
        backend.muse_extend_swa_gluon = False
        backend.muse_extend_fly = False
        backend._muse_extend_fp8_logged = False
        backend.muse_verify_kernels = frozenset()
        backend.dcp_world_size = 1
        backend.kv_cache_dtype = args["k"].dtype
        backend.input_dtype = args["out"].dtype
        backend.page_size = args["k"].shape[1]
        backend.device = args["q"].device
        backend.logits_soft_cap = 0.0
        backend.q_scale = torch.tensor([0.63], device="cuda")
        backend.k_scale, backend.v_scale = args["k_descale"], args["v_descale"]
        backend.token_to_kv_pool = TensorPool(args["k"], args["v"])
        backend.deterministic_prefill_max_q = None
        backend.forward_metadata = ForwardMetadata(
            kv_indptr=torch.tensor(
                [0, *torch.tensor(klens).cumsum(0).tolist()],
                device="cuda",
                dtype=torch.int32,
            ),
            kv_indices=args["block_table"],
            qo_indptr=args["cu_seqlens_q"],
            kv_last_page_len=torch.tensor(
                [n % backend.page_size or backend.page_size for n in klens],
                device="cuda",
                dtype=torch.int32,
            ),
            max_q_len=max(qlens),
            max_kv_len=max(klens),
        )
        # Use real request metadata and RadixAttention objects. Only the pool
        # allocation and ModelRunner startup are replaced with small fixtures.
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=len(qlens),
            input_ids=torch.zeros(sum(qlens), device="cuda", dtype=torch.int64),
            req_pool_indices=torch.arange(len(qlens), device="cuda", dtype=torch.int64),
            seq_lens=args["seqused_k"].clone(),
            out_cache_loc=torch.zeros(sum(qlens), device="cuda", dtype=torch.int64),
            seq_lens_sum=sum(klens),
        )
        batch.seq_lens_cpu = torch.tensor(klens)
        batch.extend_seq_lens = torch.tensor(qlens, device="cuda", dtype=torch.int64)
        batch.extend_prefix_lens_cpu = [k - q for k, q in zip(klens, qlens)]
        layer = RadixAttention(
            args["q"].shape[1],
            args["q"].shape[2],
            args["softmax_scale"],
            args["k"].shape[2],
            0,
            sliding_window_size=window + 1 if window >= 0 else -1,
        )
        layer.k_scale, layer.v_scale = args["k_descale"], args["v_descale"]
        tokens = torch.arange(
            args["block_table"].shape[1] * backend.page_size, device="cuda"
        )
        backend.req_to_token = (
            args["block_table"][:, tokens // backend.page_size].long()
            * backend.page_size
            + tokens % backend.page_size
        ).contiguous()
        return backend, batch, layer, args, reference

    def assert_accuracy(self, actual, reference):
        self.assertTrue(torch.isfinite(actual).all().item())
        error = (
            (actual.reshape(reference.shape).double() - reference)
            .square()
            .mean()
            .sqrt()
        )
        self.assertLess((error / reference.square().mean().sqrt()).item(), 0.02)

    def test_prefill_and_cached_prefix(self):
        for window in (-1, 31, 2047):
            with self.subTest(window=window):
                backend, batch, layer, args, reference = self.case(
                    (17, 129), (17, 2063), window
                )
                output = backend.forward_extend(
                    args["q"], None, None, layer, batch, save_kv_cache=False
                )
                self.assert_accuracy(output, reference)
                self.assertFalse(backend._tlx_fallback_reasons)

    def test_dflash_verify(self):
        for window in (-1, 31, 2047):
            with self.subTest(window=window):
                backend, batch, layer, args, reference = self.case(
                    (8, 8), (71, 2051), window
                )
                batch.forward_mode = ForwardMode.TARGET_VERIFY
                batch.seq_lens -= 8
                output = backend.forward_extend(
                    args["q"], None, None, layer, batch, save_kv_cache=False
                )
                self.assert_accuracy(output, reference)
                self.assertFalse(backend._tlx_fallback_reasons)

    def test_decode_fp8_query(self):
        backend, batch, layer, args, _ = self.case((1, 1), (71, 2051), 2047)
        batch.forward_mode = ForwardMode.DECODE
        output = backend.forward_decode(
            args["q"], None, None, layer, batch, save_kv_cache=False
        )
        args["q"] = (args["q"].float() / backend.q_scale).to(torch.float8_e4m3fn)
        args["q_descale"] = backend.q_scale
        reference = paged_reference(args, (1, 1), (71, 2051))
        self.assert_accuracy(output, reference)
        self.assertFalse(backend._tlx_fallback_reasons)

    def test_verify_graph_replay_updates_lengths_and_pages(self):
        backend, batch, layer, args, _ = self.case((8, 8), (71, 2051), 31)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.seq_lens -= 8
        backend.forward_extend(args["q"], None, None, layer, batch, save_kv_cache=False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = backend.forward_extend(
                args["q"], None, None, layer, batch, save_kv_cache=False
            )
        batch.seq_lens.copy_(torch.tensor((31 - 8, 1025 - 8), device="cuda"))
        args["block_table"][0, :2] = args["block_table"][0, :2].flip(0)
        expected = paged_reference(args, (8, 8), (31, 1025))
        for _ in range(30):
            graph.replay()
            self.assert_accuracy(output, expected)

    def test_prefill_writes_calibrated_cache(self):
        qlens, klens = (17, 9), (31, 79)
        backend, batch, layer, args, _ = self.case(qlens, klens)
        locations = torch.cat(
            [
                backend.req_to_token[i, k - q : k]
                for i, (q, k) in enumerate(zip(qlens, klens))
            ]
        )
        batch.out_cache_loc = locations
        incoming_k = torch.randn(
            (sum(qlens), 2, 64), device="cuda", dtype=torch.bfloat16
        )
        incoming_v = torch.randn_like(incoming_k)
        expected_k, expected_v = args["k"].float(), args["v"].float()
        expected_k.flatten(0, 1)[locations] = (
            (incoming_k.float() / args["k_descale"]).to(torch.float8_e4m3fn).float()
        )
        expected_v.flatten(0, 1)[locations] = (
            (incoming_v.float() / args["v_descale"]).to(torch.float8_e4m3fn).float()
        )
        reference = paged_reference(
            args | {"k": expected_k, "v": expected_v}, qlens, klens
        )
        output = backend.forward_extend(
            args["q"], incoming_k, incoming_v, layer, batch, save_kv_cache=True
        )
        self.assert_accuracy(output, reference)
        torch.testing.assert_close(args["k"].float(), expected_k, rtol=0, atol=0)
        torch.testing.assert_close(args["v"].float(), expected_v, rtol=0, atol=0)

    def test_softcap_falls_back_with_correct_output(self):
        from aiter.ops.triton.attention.unified_attention import (
            unified_attention as aiter_attention,
        )

        backend, _, _, args, _ = self.case((3, 3), (31, 79))
        args["softcap"] = 10.0
        args["q_descale"] = None
        expected = torch.empty_like(args["out"])
        aiter_attention(**(args | {"out": expected, "q_descale": None}))
        backend._run_unified_attention(**args)
        torch.testing.assert_close(args["out"], expected)
        self.assertTrue(backend._tlx_fallback_reasons)

    def test_phase_classification(self):
        for max_seqlen_q, window, phase in (
            (1, (-1, -1), "decode"),
            (8, (2046, 0), "decode"),
            (17, (-1, -1), "prefill_short"),
            (1023, (-1, -1), "prefill_short"),
            (1024, (-1, -1), "prefill_long"),
            (4096, (2046, 0), "prefill_window"),
        ):
            with self.subTest(max_seqlen_q=max_seqlen_q, window=window):
                self.assertEqual(
                    TlxAttnBackend._phase(
                        {"max_seqlen_q": max_seqlen_q, "window_size": window}
                    ),
                    phase,
                )

    def test_routed_phase_runs_aiter(self):
        from aiter.ops.triton.attention.unified_attention import (
            unified_attention as aiter_attention,
        )

        backend, _, _, args, _ = self.case((17, 129), (17, 2063))
        backend._tlx_phases = frozenset({"decode"})
        args["softcap"] = 0.0
        args["q_descale"] = None
        expected = torch.empty_like(args["out"])
        aiter_attention(**(args | {"out": expected}))
        backend._run_unified_attention(**args)
        torch.testing.assert_close(args["out"], expected)
        self.assertEqual(backend._tlx_routed_phases, {"prefill_short"})
        self.assertFalse(backend._tlx_dispatched)


if __name__ == "__main__":
    unittest.main()

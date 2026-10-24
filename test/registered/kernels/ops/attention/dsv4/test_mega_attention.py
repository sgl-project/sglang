import copy
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.flash_mla_interface import (
    flash_mla_mega_decode,
    flash_mla_mega_prefill,
)
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    DSV4Metadata,
)
from sglang.srt.layers.attention.deepseek_v4_mega_backend import (
    interleave_output_columns,
    interleave_query_rows,
    pack_query,
    quantize_wo_a,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v4 import (
    MqaAttentionBase,
    MQALayer,
    _fp8_wo_a_einsum,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.dsv41_kv_quant_reference import (
    interleave_pages,
    quantize_k_cache_v41,
    quantize_k_cache_v41_fp4,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def unpack_output(data, scale):
    tokens, groups, hidden = data.shape
    sf = scale.contiguous().view(torch.uint8).float()
    values = data.float().view(tokens, groups, hidden // 32, 32)
    values *= torch.exp2(sf - 127).unsqueeze(-1)
    return (
        values.view(tokens, groups, 16, 8, 32)
        .transpose(2, 3)
        .reshape(tokens, groups, hidden)
    )


def paged_view(raw, page_size, bytes_per_token):
    return raw[:, : page_size * bytes_per_token].view(
        raw.shape[0], page_size, 1, bytes_per_token
    )


def pack_query_reference(q, padded_heads):
    # pack_query as it was before the persistent buffer: fresh zeros every call.
    tokens, heads, _ = q.shape
    out = q.new_zeros(tokens, 32, padded_heads, 16)
    out[:, :, :heads] = q.view(tokens, heads, 32, 16).transpose(1, 2)
    return out.view(tokens, padded_heads, 512)


class _MegaQueryLayer:
    """The state MQALayer._forward_mega reads before the attention call, which
    records the packed query it is handed."""

    head_dim = 512
    mega_wq_b_permuted = False
    _zero_padded_q = MqaAttentionBase._zero_padded_q
    _forward_mega = MQALayer._forward_mega

    def __init__(self):
        self.packed = []

    def _mega_core(
        self, q, attn_sink, forward_batch, attn_backend, allow_mxfp8_output=False
    ):
        self.packed.append(q.clone())
        return q


def new_metadata():
    return DSV4Metadata(core_attn_metadata=None, indexer_metadata=None)


def new_tail_metadata():
    return DSV4Metadata(
        core_attn_metadata=SimpleNamespace(request_window_layout=None),
        indexer_metadata=None,
        late_layer_tail=SimpleNamespace(real_rows_count=None, cp_metadata=None),
    )


class _KeptMetadataBackend:
    """The state DeepseekV4AttnBackend reads to share the packed query buffer
    between metadata that captured graphs keep, and to switch the late layers
    onto their tail metadata."""

    _share_q_mega_pad_buffer = DeepseekV4AttnBackend._share_q_mega_pad_buffer
    enter_late_layer_tail = DeepseekV4AttnBackend.enter_late_layer_tail
    exit_late_layer_tail = DeepseekV4AttnBackend.exit_late_layer_tail

    def __init__(self):
        self.forward_metadata = None
        self.tail_forward_metadata = None
        self._last_kept_metadata = None
        self.token_to_kv_pool = SimpleNamespace(request_window=None)


class TestMegaPackedQueryBuffer(CustomTestCase):
    """The packed query lives in one zeroed buffer per metadata that pack_query
    writes only the real heads of, so its padding must stay zero across layers,
    shrinking and growing token counts, and CUDA graph capture and replay."""

    def _forward(self, layer, backend, q, padded_heads, mode=ForwardMode.DECODE):
        forward_batch = SimpleNamespace(forward_mode=mode)
        return layer._forward_mega(q, padded_heads, None, forward_batch, backend)

    def test_matches_fresh_zeros_across_calls(self):
        torch.manual_seed(0)
        decode_tokens = (8, 48, 3, 48, 64, 1)
        cases = [
            (heads, padded, ForwardMode.DECODE, decode_tokens)
            for heads, padded in ((8, 64), (16, 64), (32, 64), (64, 64), (128, 128))
        ]
        cases.append((16, 64, ForwardMode.EXTEND, (8192, 1000, 8192, 4096)))
        for heads, padded, mode, token_counts in cases:
            with self.subTest(heads=heads, padded=padded, mode=mode):
                layer = _MegaQueryLayer()
                backend = SimpleNamespace(forward_metadata=new_metadata())
                for tokens in token_counts:
                    for _ in range(3):  # layers of one forward
                        q = torch.randn(
                            tokens, heads, 512, device="cuda", dtype=torch.bfloat16
                        )
                        out = self._forward(layer, backend, q, padded, mode)
                        layer.packed.clear()
                        buf = backend.forward_metadata.q_mega_pad_buffer
                        self.assertEqual(out.data_ptr(), buf.data_ptr())
                        self.assertTrue(
                            torch.equal(out, pack_query_reference(q, padded))
                        )
                        self.assertTrue(torch.equal(pack_query(q, padded), out))
                self.assertEqual(buf.shape[0], max(token_counts))
                # q_pad_buffer's layout pads where the packed layout holds real heads.
                self.assertIsNone(backend.forward_metadata.q_pad_buffer)

    def test_cuda_graph_reuses_buffer_allocated_before_capture(self):
        # Metadata kept across warmup and capture (prefill graphs, and decode
        # graphs that keep the full metadata): warmup allocates and zeroes the
        # buffer eagerly, so the graph holds no memset and replays reuse it.
        torch.manual_seed(1)
        layer = _MegaQueryLayer()
        backend = SimpleNamespace(forward_metadata=new_metadata())
        static_q = [
            torch.randn(48, 16, 512, device="cuda", dtype=torch.bfloat16)
            for _ in range(4)
        ]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for q in static_q:
                self._forward(layer, backend, q, 64)
        torch.cuda.current_stream().wait_stream(stream)
        meta = backend.forward_metadata
        warm_ptr = meta.q_mega_pad_buffer.data_ptr()
        self.assertFalse(meta.q_mega_pad_captured)

        layer.packed.clear()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for q in static_q:
                self._forward(layer, backend, q, 64)
        self.assertTrue(meta.q_mega_pad_captured)
        self.assertEqual(meta.q_mega_pad_buffer.data_ptr(), warm_ptr)
        for _ in range(3):
            for q in static_q:
                q.copy_(torch.randn_like(q))
            graph.replay()
            torch.cuda.synchronize()
            for q, packed in zip(static_q, layer.packed, strict=True):
                self.assertTrue(torch.equal(packed, pack_query_reference(q, 64)))

        # A captured buffer may not be replaced by a larger one.
        with self.assertRaisesRegex(AssertionError, "grew after a CUDA graph"):
            self._forward(
                layer,
                backend,
                torch.randn(64, 16, 512, device="cuda", dtype=torch.bfloat16),
                64,
            )

    def test_cuda_graph_rezeroes_buffer_allocated_in_capture(self):
        # Metadata built inside the captured forward (raw decode metadata upgraded
        # in the graph): the buffer comes from the graph pool, so each replay must
        # zero it again whatever that memory held in between.
        torch.manual_seed(2)
        layer = _MegaQueryLayer()
        backend = SimpleNamespace(forward_metadata=None)
        static_q = [
            torch.randn(40, 16, 512, device="cuda", dtype=torch.bfloat16)
            for _ in range(4)
        ]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            backend.forward_metadata = new_metadata()
            for q in static_q:
                self._forward(layer, backend, q, 64)
        meta = backend.forward_metadata
        self.assertTrue(meta.q_mega_pad_captured)
        for _ in range(3):
            meta.q_mega_pad_buffer.fill_(float("nan"))
            for q in static_q:
                q.copy_(torch.randn_like(q))
            graph.replay()
            torch.cuda.synchronize()
            for q, packed in zip(static_q, layer.packed, strict=True):
                self.assertTrue(torch.equal(packed, pack_query_reference(q, 64)))

    def _prefill_forward(self, layer, backend, forward_batch, queries):
        # Two early layers over every row, then two late layers over the tail.
        for q in queries[:2]:
            self._forward(layer, backend, q, 64, ForwardMode.EXTEND)
        saved = backend.enter_late_layer_tail(forward_batch)
        for q in queries[2:]:
            self._forward(layer, backend, q, 64, ForwardMode.EXTEND)
        backend.exit_late_layer_tail(saved, forward_batch)

    def test_kept_metadata_share_one_buffer(self):
        # Breakable prefill graphs keep one metadata per captured shape, built
        # largest first and warmed up eagerly before capture, and switch the late
        # layers onto a tail metadata. They share one buffer, and each graph still
        # replays exactly after the others ran. A shape that needs more rows than
        # the shared buffer gets its own; the graphs before it keep theirs.
        torch.manual_seed(3)
        layer = _MegaQueryLayer()
        backend = _KeptMetadataBackend()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            attn_cp_metadata=None,
            global_num_token_non_padded_cpu=None,
            global_num_token_non_padded=None,
            num_token_non_padded=None,
        )
        captured = []
        for tokens in (512, 256, 48, 1024):
            meta, tail = new_metadata(), new_tail_metadata()
            backend._share_q_mega_pad_buffer(meta)
            backend.forward_metadata, backend.tail_forward_metadata = meta, tail
            queries = [
                torch.randn(rows, 16, 512, device="cuda", dtype=torch.bfloat16)
                for rows in (tokens, tokens, tokens // 4, tokens // 4)
            ]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._prefill_forward(layer, backend, forward_batch, queries)
            torch.cuda.current_stream().wait_stream(stream)
            layer.packed.clear()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._prefill_forward(layer, backend, forward_batch, queries)
            self.assertTrue(meta.q_mega_pad_captured and tail.q_mega_pad_captured)
            captured.append((meta, tail, graph, queries, list(layer.packed)))
            layer.packed.clear()

        shared = captured[0][0].q_mega_pad_buffer
        self.assertEqual(shared.shape[0], 512)
        for meta, tail, *_ in captured[:3]:
            self.assertIs(meta.q_mega_pad_buffer, shared)
            self.assertIs(tail.q_mega_pad_buffer, shared)
        meta, tail, *_ = captured[3]
        self.assertEqual(meta.q_mega_pad_buffer.shape[0], 1024)
        self.assertIs(tail.q_mega_pad_buffer, meta.q_mega_pad_buffer)

        for i in (2, 0, 3, 1, 2):
            _, _, graph, queries, packed = captured[i]
            for q in queries:
                q.copy_(torch.randn_like(q))
            graph.replay()
            torch.cuda.synchronize()
            for q, p in zip(queries, packed, strict=True):
                self.assertTrue(torch.equal(p, pack_query_reference(q, 64)))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "MegaAttention requires SM100/SM103",
)
class TestMegaAttention(CustomTestCase):
    def test_wq_b_permutation_matches_gemm_then_pack(self):
        from sglang.srt.layers.quantization import fp8_utils
        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
        from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

        for backend in (
            fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTLASS,
            fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL,
        ):
            with patch.object(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", backend):
                method = Fp8LinearMethod(
                    Fp8Config(
                        is_checkpoint_fp8_serialized=True,
                        use_mxfp8=True,
                        weight_block_size=[1, 32],
                        scale_fmt="ue8m0",
                    )
                )
            with self.subTest(backend=backend):
                linear = torch.nn.Module()
                linear.quant_method = method
                linear.weight = torch.randn(64 * 512, 1280, device="cuda").to(
                    torch.float8_e4m3fn
                )
                linear.weight_scale_inv = torch.randint(
                    119, 129, (64 * 512, 40), device="cuda", dtype=torch.uint8
                )
                method._process_mxfp8_linear_weight_scale(linear)
                reference = copy.deepcopy(linear)
                for param in (linear.weight, linear.weight_scale_inv):
                    param.copy_(interleave_query_rows(param, 64))
                method._process_mxfp8_linear_weight_scale(linear)
                x = torch.randn(1536, 1280, device="cuda", dtype=torch.bfloat16)
                quantized = Mxfp8SwizzledInput(
                    *fp8_utils.flashinfer_mxfp8_quantize(
                        x[:3], is_sf_swizzled_layout=True, alignment=32
                    )
                )
                for inputs in (x, quantized):
                    torch.testing.assert_close(
                        method.apply(linear, inputs),
                        pack_query(
                            method.apply(reference, inputs).view(-1, 64, 512), 64
                        ).view(-1, 64 * 512),
                        rtol=0,
                        atol=0,
                    )

    def test_wo_a_projection_matches_reference(self):
        torch.manual_seed(456)
        positions = torch.tensor([0, 7, 127], device="cuda", dtype=torch.int32)
        angle = (
            torch.arange(256, device="cuda")[:, None]
            / torch.arange(1, 33, device="cuda")[None]
        )
        table = torch.cat((angle.cos(), angle.sin()), dim=-1).float()
        main = paged_view(
            quantize_k_cache_v41(
                torch.randn(1, 128, 512, device="cuda", dtype=torch.bfloat16)
            ),
            128,
            528,
        )
        extra = paged_view(
            quantize_k_cache_v41_fp4(
                torch.randn(1, 64, 512, device="cuda", dtype=torch.bfloat16)
            ),
            64,
            288,
        )
        indices = torch.arange(128, device="cuda", dtype=torch.int32).repeat(3, 1)
        extra_indices = torch.arange(64, device="cuda", dtype=torch.int32).repeat(3, 1)
        for heads in (16, 64):
            with self.subTest(heads=heads):
                q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
                data, scale, _ = flash_mla_mega_decode(
                    pack_query(q, 64),
                    main,
                    indices,
                    positions,
                    table,
                    512**-0.5,
                    attn_sink=torch.randn(64, device="cuda", dtype=torch.float32),
                    topk_length=torch.tensor(
                        [128, 96, 1], device="cuda", dtype=torch.int32
                    ),
                    extra_kv=extra,
                    extra_indices=extra_indices,
                    extra_topk_length=torch.tensor(
                        [64, 32, 0], device="cuda", dtype=torch.int32
                    ),
                )
                groups = heads // 8
                weight = torch.randn(
                    groups, 64, 4096, device="cuda", dtype=torch.bfloat16
                )
                actual = _fp8_wo_a_einsum(
                    data[:, :groups],
                    scale[:, :groups],
                    *quantize_wo_a(interleave_output_columns(weight)),
                    recipe=(1, 1, 32),
                )
                expected = torch.einsum(
                    "tgd,grd->tgr",
                    unpack_output(data, scale)[:, :groups],
                    weight.float(),
                )
                error = (actual.float() - expected).square().mean().sqrt()
                self.assertLess(
                    error.item() / expected.square().mean().sqrt().item(), 0.04
                )

    def test_mega_kernels_leave_q_unchanged(self):
        # The model zeroes the packed query's padding once per forward and every
        # layer reuses the buffer, so no MegaAttention kernel may write its query.
        torch.manual_seed(789)
        tokens, topk, extra_topk, page = 48, 128, 512, 64
        positions = torch.randint(0, 4096, (tokens,), device="cuda", dtype=torch.int32)
        angle = (
            torch.arange(4096, device="cuda")[:, None]
            / torch.arange(1, 33, device="cuda")[None]
        )
        table = torch.cat((angle.cos(), angle.sin()), dim=-1).float()
        main = quantize_k_cache_v41(
            torch.randn(6, page, 512, device="cuda", dtype=torch.bfloat16) / 4
        )
        extra = quantize_k_cache_v41_fp4(
            torch.randn(9, page, 512, device="cuda", dtype=torch.bfloat16) / 4
        )
        decode_caches = {
            True: (paged_view(main, page, 528), paged_view(extra, page, 288)),
            False: (
                paged_view(interleave_pages(main, page, 512, 16), page, 528),
                paged_view(interleave_pages(extra, page, 256, 32), page, 288),
            ),
        }
        indices = torch.randint(
            0, 6 * page, (tokens, topk), device="cuda", dtype=torch.int32
        )
        extra_indices = torch.randint(
            0, 9 * page, (tokens, extra_topk), device="cuda", dtype=torch.int32
        )
        lengths = torch.randint(1, topk + 1, (tokens,), device="cuda").int()
        extra_lengths = torch.randint(0, extra_topk + 1, (tokens,), device="cuda").int()
        prefill_kv = torch.randn(512, 1, 512, device="cuda", dtype=torch.bfloat16) / 4
        prefill_indices = torch.randint(
            0, 512, (tokens, 1, topk), device="cuda", dtype=torch.int32
        )
        for (heads, padded), use_legacy, op in itertools.product(
            ((16, 64), (64, 64), (128, 128)), (True, False), ("decode", "prefill")
        ):
            with self.subTest(heads=heads, use_legacy=use_legacy, op=op):
                q_real = torch.randn(
                    tokens, heads, 512, device="cuda", dtype=torch.bfloat16
                )
                q = pack_query(q_real, padded)
                common = dict(
                    attn_sink=torch.randn(padded, device="cuda"),
                    topk_length=lengths,
                    use_legacy=use_legacy,
                )
                if op == "decode":
                    main_kv, extra_kv = decode_caches[use_legacy]
                    flash_mla_mega_decode(
                        q,
                        main_kv,
                        indices,
                        positions,
                        table,
                        512**-0.5,
                        extra_kv=extra_kv,
                        extra_indices=extra_indices,
                        extra_topk_length=extra_lengths,
                        **common,
                    )
                else:
                    flash_mla_mega_prefill(
                        q,
                        prefill_kv,
                        prefill_indices,
                        positions,
                        table,
                        512**-0.5,
                        **common,
                    )
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(q, pack_query_reference(q_real, padded)))


if __name__ == "__main__":
    unittest.main()

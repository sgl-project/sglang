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
from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_mega_backend as mega_backend_module
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.deepseek_v4_mega_backend import (
    DeepseekV4MegaAttnBackend,
    interleave_output_columns,
    interleave_query_rows,
    pack_query,
    quantize_wo_a,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.deepseek_v4 import MQALayer, _fp8_wo_a_einsum
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.dsv41_kv_quant_reference import (
    interleave_pages,
    quantize_k_cache_v41,
    quantize_k_cache_v41_fp4,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


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
            ((16, 64), (64, 64), (128, 128)),
            (True, False),
            ("decode", "decode_swa", "prefill"),
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
                if op != "prefill":
                    main_kv, extra_kv = decode_caches[use_legacy]
                    extra = {}
                    if op == "decode":
                        extra = dict(
                            extra_kv=extra_kv,
                            extra_indices=extra_indices,
                            extra_topk_length=extra_lengths,
                        )
                    flash_mla_mega_decode(
                        q,
                        main_kv,
                        indices,
                        positions,
                        table,
                        512**-0.5,
                        **extra,
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


def pack_query_reference(q, padded_heads):
    # pack_query without a buffer: fresh zeros every call.
    tokens, heads, _ = q.shape
    out = q.new_zeros(tokens, 32, padded_heads, 16)
    out[:, :, :heads] = q.view(tokens, heads, 32, 16).transpose(1, 2)
    return out.view(tokens, padded_heads, 512)


class _MegaQueryLayer:
    head_dim = 512
    mega_wq_b_permuted = False
    _forward_mega = MQALayer._forward_mega

    def __init__(self):
        self.packed = []

    def _mega_core(
        self, q, attn_sink, forward_batch, attn_backend, allow_mxfp8_output=False
    ):
        self.packed.append(q.clone())
        return q


def mega_backend(max_rows):
    # DeepseekV4MegaAttnBackend.__init__ over a base class that sets only what it reads.
    def base_init(self, model_runner, **kwargs):
        self.device = torch.device("cuda")
        self.speculative_num_draft_tokens = None

    with (
        patch.object(DeepseekV4AttnBackend, "__init__", base_init),
        patch.object(
            mega_backend_module, "max_packed_query_rows", return_value=max_rows
        ),
    ):
        return DeepseekV4MegaAttnBackend(SimpleNamespace(max_running_requests=1))


def poison_allocator(rows, padded_heads):
    # The caching allocator hands this freed block to the next allocation of its size.
    torch.full(
        (rows, padded_heads, 512), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    torch.cuda.synchronize()


def forward(layer, backend, q, padded_heads, mode=ForwardMode.DECODE):
    forward_batch = SimpleNamespace(forward_mode=mode)
    return layer._forward_mega(q, padded_heads, None, forward_batch, backend)


def padding_nonzero(backend, heads, padded_heads):
    padding = backend._packed_query.view(-1, 32, padded_heads, 16)[:, :, heads:]
    return torch.count_nonzero(padding).item()


class TestMegaPackedQueryBuffer(CustomTestCase):
    def test_packs_match_fresh_zeros(self):
        torch.manual_seed(0)
        decode_tokens = (8, 48, 3, 64, 1)
        cases = [
            (heads, padded, ForwardMode.DECODE, decode_tokens)
            for heads, padded in ((8, 64), (16, 64), (32, 64), (64, 64), (128, 128))
        ]
        cases.append((16, 64, ForwardMode.EXTEND, (8192, 1000, 4096)))
        for heads, padded, mode, token_counts in cases:
            with self.subTest(heads=heads, padded=padded, mode=mode):
                max_rows = max(token_counts)
                backend = mega_backend(max_rows)
                layer = _MegaQueryLayer()
                for tokens in token_counts:
                    for _ in range(3):  # layers of one forward
                        q = torch.randn(
                            tokens, heads, 512, device="cuda", dtype=torch.bfloat16
                        )
                        if backend._packed_query is None:
                            poison_allocator(max_rows, padded)
                        out = forward(layer, backend, q, padded, mode)
                        self.assertTrue(
                            torch.equal(out, pack_query_reference(q, padded))
                        )
                        self.assertEqual(
                            out.data_ptr(), backend._packed_query.data_ptr()
                        )
                        layer.packed.clear()
                self.assertEqual(padding_nonzero(backend, heads, padded), 0)

    def test_cuda_graphs_and_eager_forwards_share_the_buffer(self):
        torch.manual_seed(1)
        layer = _MegaQueryLayer()
        backend = mega_backend(64)
        static = {
            rows: [
                torch.randn(rows, 16, 512, device="cuda", dtype=torch.bfloat16)
                for _ in range(4)
            ]
            for rows in (64, 40)
        }
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # A freed block is reused only by allocations on its own stream.
            poison_allocator(64, 64)
            for q in static[64]:
                forward(layer, backend, q, 64)
        torch.cuda.current_stream().wait_stream(stream)
        ptr = backend._packed_query.data_ptr()
        graphs = {}
        for rows in (64, 40):
            layer.packed = []
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for q in static[rows]:
                    forward(layer, backend, q, 64)
            graphs[rows] = (graph, layer.packed)
        layer.packed = []
        self.assertEqual(backend._packed_query.data_ptr(), ptr)
        for _ in range(3):
            for rows in (40, 64):
                for q in static[rows]:
                    q.copy_(torch.randn_like(q))
                graph, packed = graphs[rows]
                graph.replay()
                torch.cuda.synchronize()
                for q, p in zip(static[rows], packed, strict=True):
                    self.assertTrue(torch.equal(p, pack_query_reference(q, 64)))
                q = torch.randn(17, 16, 512, device="cuda", dtype=torch.bfloat16)
                out = forward(layer, backend, q, 64)
                self.assertTrue(torch.equal(out, pack_query_reference(q, 64)))
        self.assertEqual(padding_nonzero(backend, 16, 64), 0)

    def test_forward_beyond_the_buffer_packs_into_fresh_zeros(self):
        torch.manual_seed(3)
        layer = _MegaQueryLayer()
        backend = mega_backend(48)
        poison_allocator(48, 64)
        q = torch.randn(48, 16, 512, device="cuda", dtype=torch.bfloat16)
        forward(layer, backend, q, 64)
        ptr = backend._packed_query.data_ptr()
        before = backend._packed_query.clone()
        q = torch.randn(49, 16, 512, device="cuda", dtype=torch.bfloat16)
        poison_allocator(49, 64)
        out = forward(layer, backend, q, 64)
        self.assertTrue(torch.equal(out, pack_query_reference(q, 64)))
        self.assertNotEqual(
            out.untyped_storage().data_ptr(),
            backend._packed_query.untyped_storage().data_ptr(),
        )
        self.assertEqual(backend._packed_query.data_ptr(), ptr)
        self.assertTrue(torch.equal(backend._packed_query, before))
        self.assertEqual(padding_nonzero(backend, 16, 64), 0)
        q = torch.randn(48, 16, 512, device="cuda", dtype=torch.bfloat16)
        out = forward(layer, backend, q, 64)
        self.assertTrue(torch.equal(out, pack_query_reference(q, 64)))
        self.assertEqual(out.data_ptr(), ptr)


class TestMaxPackedQueryRows(CustomTestCase):
    def _rows(self, *, mode, chunk, prefill_graph, decode_graph, requests, align):
        graphs = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL, max_bs=decode_graph),
            prefill=PhaseConfig(backend=Backend.BREAKABLE, max_bs=prefill_graph),
        )
        with (
            get_context().override_server_args(
                cuda_graph_config=graphs,
                chunked_prefill_size=chunk,
                disaggregation_mode=mode,
            ),
            envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS.override(2048),
            # MLP sync pads the request count to the attention TP size.
            patch.object(
                mega_backend_module,
                "get_eager_max_batch_size",
                side_effect=lambda bs: -(-bs // align) * align,
            ),
        ):
            return mega_backend_module.max_packed_query_rows(
                max_running_requests=requests, rows_per_request=4
            )

    def test_bound_covers_every_forward(self):
        # Expected values derived by hand; every case has 4 rows per request.
        cases = (
            # 64 graph requests x 4 = 256 rows, under the 8192-token chunk.
            (8192, "null", 8192, 2048, 64, 48, 1),
            # The 2048-row decoder-tail bucket outgrows a 1024-token chunk.
            (2048, "null", 1024, 512, 64, 48, 1),
            # An 8192-token prefill-graph bucket outgrows a 4096-token chunk.
            (8192, "null", 4096, 8192, 64, 48, 1),
            # Chunked prefill disabled: no chunk term, so the 2048-row tail bucket.
            (2048, "null", -1, 512, 64, 48, 1),
            # A PD decode server: 1000 eager requests (over the 256 of the graphs) x 4.
            (4000, "decode", 16384, 2048, 256, 1000, 1),
            # 50 requests padded to 56 for MLP sync, x 4.
            (224, "decode", 16384, 2048, None, 50, 8),
        )
        for (
            expected,
            mode,
            chunk,
            prefill_graph,
            decode_graph,
            requests,
            align,
        ) in cases:
            with self.subTest(mode=mode, chunk=chunk, requests=requests):
                rows = self._rows(
                    mode=mode,
                    chunk=chunk,
                    prefill_graph=prefill_graph,
                    decode_graph=decode_graph,
                    requests=requests,
                    align=align,
                )
                self.assertEqual(rows, expected)


if __name__ == "__main__":
    unittest.main()

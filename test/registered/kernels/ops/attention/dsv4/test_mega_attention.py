import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sgl_kernel.flash_mla_mega import flash_mla_mega_decode, flash_mla_mega_prefill

from sglang.kernels.ops.attention.dsv4.mega_attention import pack_mega_query
from sglang.kernels.ops.attention.dsv4.torch_quant import (
    dequantize_k_cache_v41,
    quantize_k_cache_v41,
)
from sglang.srt.layers.attention.dsv4.mega_attention import (
    apply_mega_wo_a,
    prepare_mega_wo_a,
    prepare_mega_wq_b,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.dsv41_kv_quant_reference import (
    dequantize_k_cache_v41_fp4,
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


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "MegaAttention requires SM100/SM103",
)
class TestMegaAttention(CustomTestCase):
    def test_pack_mega_query_layout(self):
        for heads in (8, 16, 32, 64, 128):
            with self.subTest(heads=heads):
                q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
                padded = 64 if heads <= 64 else 128
                expected = q.new_zeros(3, padded, 512)
                expected[:, :heads] = q
                expected = (
                    expected.view(3, padded, 32, 16)
                    .transpose(1, 2)
                    .reshape(3, padded, 512)
                )
                torch.testing.assert_close(
                    pack_mega_query(q, padded), expected, rtol=0, atol=0
                )

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
            for heads in (64, 128):
                with self.subTest(backend=backend, heads=heads):
                    linear = torch.nn.Module()
                    linear.quant_method = method
                    linear.weight = torch.randn(heads * 512, 1280, device="cuda").to(
                        torch.float8_e4m3fn
                    )
                    linear.weight_scale_inv = torch.randint(
                        119, 129, (heads * 512, 40), device="cuda", dtype=torch.uint8
                    )
                    method._process_mxfp8_linear_weight_scale(linear)
                    projection = prepare_mega_wq_b(linear, heads)
                    self.assertIsNotNone(projection)
                    for tokens in (3, 96, 1536):
                        x = torch.randn(
                            tokens, 1280, device="cuda", dtype=torch.bfloat16
                        )
                        expected = pack_mega_query(
                            method.apply(linear, x).view(tokens, heads, 512), heads
                        )
                        torch.testing.assert_close(
                            method.apply(projection, x).view(tokens, heads, 512),
                            expected,
                            rtol=0,
                            atol=0,
                        )
                    # The fused-RMSNorm decode path hands apply() a pre-quantized input.
                    x = torch.randn(3, 1280, device="cuda", dtype=torch.bfloat16)
                    quantized = Mxfp8SwizzledInput(
                        *fp8_utils.flashinfer_mxfp8_quantize(
                            x, is_sf_swizzled_layout=True, alignment=32
                        )
                    )
                    torch.testing.assert_close(
                        method.apply(projection, quantized),
                        pack_mega_query(
                            method.apply(linear, quantized).view(3, heads, 512), heads
                        ).view(3, -1),
                        rtol=0,
                        atol=0,
                    )
                    # Padded head counts keep the runtime pack.
                    self.assertIsNone(prepare_mega_wq_b(linear, 16))

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
        # 16 heads exercises the padded head groups the projection must drop.
        for heads in (16, 64):
            with self.subTest(heads=heads):
                q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
                data, scale, _ = flash_mla_mega_decode(
                    pack_mega_query(q, 64),
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
                actual = apply_mega_wo_a(data, scale, prepare_mega_wo_a(weight))
                expected = torch.einsum(
                    "tgd,grd->tgr",
                    unpack_output(data, scale)[:, :groups],
                    weight.float(),
                )
                error = (actual.float() - expected).square().mean().sqrt()
                self.assertLess(
                    error.item() / expected.square().mean().sqrt().item(), 0.04
                )

    def test_paged_causal_prefill_matches_gathered_nvfp4(self):
        """Paged prefill preserves per-query bounds across fragmented C1/C2 pages."""
        for heads in (64, 128):
            for ratio in (1, 2):
                with self.subTest(heads=heads, ratio=ratio):
                    self._check_paged_causal_prefill(heads, ratio)

    def _check_paged_causal_prefill(self, heads, ratio):
        torch.manual_seed(456)
        query_lens, seq_lens = (129, 257, 3), (131072, 65536, 3)
        tokens = sum(query_lens)
        main_page, extra_page = 256, 256 // ratio
        main_offsets, extra_offsets = [0], [0]
        for length in seq_lens:
            main_offsets.append(
                main_offsets[-1] + ((length + main_page - 1) // main_page * main_page)
            )
            extra_offsets.append(
                extra_offsets[-1]
                + (
                    (max(length // ratio, 1) + extra_page - 1)
                    // extra_page
                    * extra_page
                )
            )
        main_perm = torch.randperm(main_offsets[-1] // main_page, device="cuda")
        extra_perm = torch.randperm(extra_offsets[-1] // extra_page, device="cuda")

        def physical(ids, page_size, permutation):
            ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
            safe = ids.clamp_min(0)
            remapped = (
                permutation[(safe // page_size).long()].to(torch.int32) * page_size
                + safe % page_size
            )
            return torch.where(ids >= 0, remapped, -1)

        rng = random.Random(456)
        positions, main_ids, extra_ids, main_lens, extra_lens = [], [], [], [], []
        for req, (ql, length) in enumerate(zip(query_lens, seq_lens, strict=True)):
            for pos in range(length - ql, length):
                ml, el = min(pos + 1, 128), min((pos + 1) // ratio, 512)
                positions.append(pos)
                main_lens.append(ml)
                extra_lens.append(el)
                main_ids.append(
                    [main_offsets[req] + i for i in range(pos - ml + 1, pos + 1)]
                    + [main_offsets[req] + length - 1] * (128 - ml)
                )
                selected = rng.sample(range((pos + 1) // ratio), el)
                extra_ids.append(
                    [extra_offsets[req] + i for i in selected]
                    + [extra_offsets[req] + max(length // ratio - 1, 0)] * (512 - el)
                )
        main_ids = physical(main_ids, main_page, main_perm)
        extra_ids = physical(extra_ids, extra_page, extra_perm)
        # Holes inside a valid span and valid-but-future entries beyond it are ignored.
        main_ids[0, 4] = -1
        extra_ids[0, 7] = -1
        positions = torch.tensor(positions, device="cuda", dtype=torch.int32)
        main_lens = torch.tensor(main_lens, device="cuda", dtype=torch.int32)
        extra_lens = torch.tensor(extra_lens, device="cuda", dtype=torch.int32)
        main = quantize_k_cache_v41(
            torch.randn(
                len(main_perm), main_page, 512, device="cuda", dtype=torch.bfloat16
            )
        )
        extra = quantize_k_cache_v41_fp4(
            torch.randn(
                len(extra_perm), extra_page, 512, device="cuda", dtype=torch.bfloat16
            )
        )
        main_values = dequantize_k_cache_v41(main, main_page).flatten(0, 1)
        extra_values = dequantize_k_cache_v41_fp4(extra, extra_page).flatten(0, 1)
        kv = torch.cat((extra_values, main_values)).unsqueeze(1)
        combined = torch.full((tokens, 640), -1, device="cuda", dtype=torch.int32)
        for row, (ml, el) in enumerate(zip(main_lens.tolist(), extra_lens.tolist())):
            combined[row, :el] = extra_ids[row, :el]
            combined[row, el : el + ml] = torch.where(
                main_ids[row, :ml] >= 0,
                main_ids[row, :ml] + len(extra_values),
                -1,
            )
        q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
        packed = pack_mega_query(q, heads)
        angle = (
            torch.arange(max(seq_lens) + 64, device="cuda")[:, None]
            / torch.arange(1, 33, device="cuda")[None]
        )
        table = torch.cat((angle.cos(), angle.sin()), dim=-1).float()
        sink = torch.randn(heads, device="cuda", dtype=torch.float32)

        def paged():
            return flash_mla_mega_decode(
                packed,
                paged_view(main, main_page, 528),
                main_ids,
                positions,
                table,
                softmax_scale=512**-0.5,
                attn_sink=sink,
                topk_length=main_lens,
                extra_kv=paged_view(extra, extra_page, 288),
                extra_indices=extra_ids,
                extra_topk_length=extra_lens,
            )

        actual = unpack_output(*paged()[:2])
        if heads == 64:
            self._check_paged_prefill_dispatch(
                packed,
                main,
                extra,
                main_page,
                extra_page,
                main_ids,
                extra_ids,
                main_lens,
                extra_lens,
                positions,
                table,
                sink,
                seq_lens,
                ratio,
            )
        gathered = flash_mla_mega_prefill(
            packed,
            kv,
            combined[:, None],
            positions,
            table,
            softmax_scale=512**-0.5,
            attn_sink=sink,
            topk_length=main_lens + extra_lens,
        )
        expected = unpack_output(*gathered[:2])
        error = (
            actual - expected
        ).square().mean().sqrt() / expected.square().mean().sqrt()
        self.assertLess(error.item(), 0.01)
        self.assertTrue(torch.isfinite(actual).all().item())

        # Independent FP32 reference also covers the fresh request's zero-C2 row.
        for row in (0, query_lens[0], tokens - 3, tokens - 1):
            ids = combined[row].long()
            valid = (ids >= 0) & (
                torch.arange(640, device="cuda") < main_lens[row] + extra_lens[row]
            )
            values = kv[ids.clamp_min(0), 0].float()
            query = q[row].float().clone()

            def rotate(tensor, inverse=False):
                pairs = tensor[..., -64:].view(heads, 32, 2)
                a, b = pairs[..., 0].clone(), pairs[..., 1].clone()
                cos, sin = table[positions[row], :32], table[positions[row], 32:]
                if inverse:
                    sin = -sin
                pairs[..., 0] = a * cos - b * sin
                pairs[..., 1] = a * sin + b * cos

            rotate(query)
            logits = query @ values.T * 512**-0.5
            logits[:, ~valid] = -float("inf")
            probs = torch.cat((logits, sink[:, None]), dim=-1).softmax(-1)[:, :-1]
            reference = probs @ values
            rotate(reference, inverse=True)
            error = (
                actual[row].view(heads, 512) - reference
            ).square().mean().sqrt() / reference.square().mean().sqrt()
            self.assertLess(error.item(), 0.05)

        weight = torch.randn(heads // 8, 64, 4096, device="cuda", dtype=torch.bfloat16)
        weight_fp8, weight_scale = prepare_mega_wo_a(weight)

        def run():
            out = paged()
            return apply_mega_wo_a(out[0], out[1], (weight_fp8, weight_scale))

        for _ in range(3):
            run()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        packed.mul_(0.75)
        positions.add_(13)
        graph.replay()
        torch.testing.assert_close(captured, run(), rtol=0, atol=0)

    def test_prefill_indexer_publishes_physical_slots(self):
        """Paged prefill must receive the indexer's physical selection, not stale slots."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )
        from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
            DeepGEMMPrefillData,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        raw = torch.full((2, 3), -1, device="cuda", dtype=torch.int32)
        physical = torch.full_like(raw, -1)
        core = SimpleNamespace(
            page_table=torch.tensor([[2, 0, 1]], device="cuda", dtype=torch.int32),
            sparse_raw_indices=lambda _: raw,
            sparse_page_indices=lambda _: physical,
        )
        backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
        backend.trtllm_attn, backend.mega_attn = False, True
        backend.tail_forward_metadata = None
        backend.forward_metadata = SimpleNamespace(
            core_metadata=core, late_layer_tail=None
        )
        backend.token_to_kv_pool = SimpleNamespace(request_window=None)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            seq_lens_cpu=torch.tensor([131072]),
            extend_seq_lens_cpu=[2],
            extend_seq_lens=torch.tensor([2], device="cuda", dtype=torch.int32),
            req_pool_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
        )
        x = torch.zeros((2, 1), device="cuda", dtype=torch.bfloat16)
        starts = torch.zeros(2, device="cuda", dtype=torch.int32)
        selected = torch.tensor(
            [[0, 2, -1], [3, -1, 1]], device="cuda", dtype=torch.int32
        )
        slots = torch.tensor([17, 9, 28, 3], device="cuda", dtype=torch.int32)
        scoring = DeepGEMMPrefillData(
            k_slots=slots,
            request_starts=starts,
            lens_per_request=[4],
            rows_per_request=[2],
            compress_lens=torch.full_like(starts, 4),
            q_fp4=x,
            q_sf=x,
            weights=x,
        )
        for ratio in (1, 2):
            physical.fill_(-1)
            layer = SimpleNamespace(
                indexer=None,
                layer_id=0,
                compress_ratio=ratio,
                freqs_cis=x,
                n_local_heads=64,
            )
            with envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True):
                inputs = backend._make_low_ratio_prefill_indexer_inputs(
                    layer, x, x, starts, starts, batch, None
                )
            scoring.write_selection(selected, inputs)
            torch.testing.assert_close(
                physical,
                torch.tensor(
                    [[17, 28, -1], [3, -1, 9]], device="cuda", dtype=torch.int32
                ),
            )
            torch.testing.assert_close(raw, selected)

    def _check_paged_prefill_dispatch(
        self,
        q,
        main,
        extra,
        main_page,
        extra_page,
        main_ids,
        extra_ids,
        main_lens,
        extra_lens,
        positions,
        table,
        sink,
        seq_lens,
        ratio,
    ):
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
            DSV4AttnMetadata,
        )
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4SingleKVPool,
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        def cache(raw, page_size, token_bytes):
            resource = DeepSeekV4SingleKVPool.__new__(DeepSeekV4SingleKVPool)
            resource.dtype = resource.store_dtype = torch.uint8
            resource.page_size = page_size
            resource.kv_cache_total_dim = token_bytes
            resource.kv_buffer = [raw]
            return resource

        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.layer_transfer_counter = pool.request_window = None
        pool._stage_start = 0
        pool.uniform_fp8 = False
        pool.kv_layout = KVLayout.V41
        pool.swa_kv_pool = cache(main, main_page, 528)
        pool.layer_mapping = [(ratio, 0, cache(extra, extra_page, 288))]
        metadata = DSV4AttnMetadata.__new__(DSV4AttnMetadata)
        metadata.positions_casual = positions
        metadata.swa_page_indices = main_ids
        metadata.swa_topk_lengths = main_lens
        metadata.c1_flashmla_metadata = metadata.c2_flashmla_metadata = None
        metadata.set_sparse_topk(ratio, page_indices=extra_ids, topk_lengths=extra_lens)
        backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
        backend.mtp_enabled = backend.trtllm_attn = False
        backend.mega_attn = True
        backend.token_to_kv_pool = pool
        backend.softmax_scale = 512**-0.5
        backend.forward_metadata = SimpleNamespace(
            core_attn_metadata=metadata, late_layer_tail=None
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            seq_lens_cpu=torch.tensor(seq_lens),
            attn_cp_metadata=None,
        )
        expected = flash_mla_mega_decode(
            q,
            paged_view(main, main_page, 528),
            main_ids,
            positions,
            table,
            softmax_scale=512**-0.5,
            attn_sink=sink,
            topk_length=main_lens,
            extra_kv=paged_view(extra, extra_page, 288),
            extra_indices=extra_ids,
            extra_topk_length=extra_lens,
        )
        with envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                actual = backend._forward_attention(
                    q,
                    None,
                    None,
                    SimpleNamespace(layer_id=0),
                    batch,
                    compress_ratio=ratio,
                    save_kv_cache=False,
                    attn_sink=sink,
                    cos_sin_cache=table,
                )
                torch.cuda.synchronize()
        torch.testing.assert_close(
            actual[0].float(), expected[0].float(), rtol=0, atol=0
        )
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        names = [
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        self.assertTrue(any("core_attn::fwd_kernel" in name for name in names))
        self.assertFalse(any("_dequantize_k_cache_paged" in name for name in names))

        from sglang.srt.models.deepseek_v4 import deepseek_v4_mega_attention_with_output

        weight = torch.randn(8, 64, 4096, device="cuda", dtype=torch.bfloat16)
        wo_a = prepare_mega_wo_a(weight)
        model_layer = SimpleNamespace(
            n_local_groups=8,
            o_lora_rank=64,
            attn_mqa=SimpleNamespace(layer_id=0),
            compress_ratio=ratio,
            mega_cos_sin_cache=table,
            mega_wo_a=wo_a,
        )
        batch.global_num_token_non_padded_cpu = q.shape[0]
        padded = torch.cat((q, torch.randn_like(q[:17])))
        with (
            patch(
                "sglang.srt.models.deepseek_v4.get_attn_backend", return_value=backend
            ),
            patch(
                "sglang.srt.models.deepseek_v4.get_tc_piecewise_forward_context",
                return_value=SimpleNamespace(forward_batch=batch),
            ),
        ):
            bridge = padded.new_empty((padded.shape[0], 8, 64))
            deepseek_v4_mega_attention_with_output(model_layer, padded, bridge, sink)
            torch.testing.assert_close(
                bridge[: q.shape[0]],
                apply_mega_wo_a(expected[0], expected[1], wo_a),
                rtol=0,
                atol=0,
            )
            self.assertEqual(torch.count_nonzero(bridge[q.shape[0] :]).item(), 0)
            batch.global_num_token_non_padded_cpu = 0
            bridge.fill_(1)
            deepseek_v4_mega_attention_with_output(model_layer, padded, bridge, sink)
            self.assertEqual(torch.count_nonzero(bridge).item(), 0)
            self.assertEqual(bridge.dtype, torch.bfloat16)

        # DSpark's draft uses ordinary queries and supplies no fused RoPE table,
        # even when the target's launch configuration selects MegaAttention.
        from sgl_kernel.flash_mla import FlashMLASchedMeta, flash_mla_with_kvcache

        backend.is_dsv41, backend.head_dim_v = True, 512
        metadata.c1_flashmla_metadata = metadata.c2_flashmla_metadata = (
            FlashMLASchedMeta()
        )
        batch.forward_mode = ForwardMode.DECODE
        native_q = torch.randn_like(q)
        draft = backend._forward_attention(
            native_q,
            None,
            None,
            SimpleNamespace(layer_id=0, tp_q_head_num=64),
            batch,
            compress_ratio=ratio,
            save_kv_cache=False,
            attn_sink=sink,
        )
        reference = flash_mla_with_kvcache(
            q=native_q[:, None],
            k_cache=paged_view(main, main_page, 528),
            head_dim_v=512,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=metadata.get_flashmla_metadata(ratio),
            softmax_scale=512**-0.5,
            is_fp8_kvcache=True,
            indices=main_ids[:, None],
            topk_length=main_lens,
            attn_sink=sink,
            extra_k_cache=paged_view(extra, extra_page, 288),
            extra_indices_in_kvcache=extra_ids[:, None],
            extra_topk_length=extra_lens,
        )[0].squeeze(1)
        self.assertIsInstance(draft, torch.Tensor)
        torch.testing.assert_close(draft, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

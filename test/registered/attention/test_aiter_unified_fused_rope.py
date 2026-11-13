"""The real HIP fused RoPE producer writes arena full/window physical slots."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd-mi35x")

_HEADS, _KV_HEADS, _DIM, _PAGE = 64, 8, 64, 16
_DTYPE = torch.bfloat16


class _UnifiedFusedRopeMixin:
    backend_name = "aiter"
    window = None

    def setUp(self):
        super().setUp()
        from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
        from sglang.srt.configs.model_config import AttentionArch
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.layers.rotary_embedding import RotaryEmbedding
        from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.runtime_context import get_parallel, publish, reset_context
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.test.unified_allocator_fixtures import (
            UnifiedGeometry,
            build_unified_bundle,
        )

        reset_context()
        self.addCleanup(reset_context)
        args = ServerArgs(model_path="dummy", device="cuda")
        args.enable_unified_memory = True
        args.attention_backend = self.backend_name
        args.page_size = _PAGE
        args._mamba_cache_chunk_size = 64
        publish(args, role="test")
        self.enterContext(get_parallel().override(attn_dcp_size=1, dcp_enabled=False))
        self.enterContext(envs.SGLANG_USE_AITER_UNIFIED_ATTN.override(False))
        self.cache_dtype = fp8_dtype
        bundle = build_unified_bundle(
            UnifiedGeometry(
                chain="swa" if self.window else "mamba",
                full_layer_ids=(0,) if self.window else (0, 2),
                swa_layer_ids=(2,) if self.window else (),
                state_layer_ids=() if self.window else (1,),
                head_num=_KV_HEADS,
                head_dim=_DIM,
                swa_head_num=_KV_HEADS,
                swa_head_dim=_DIM,
                kv_cache_dtype=fp8_dtype,
                page_size=_PAGE,
                window=self.window or 32,
                full_tokens=2048,
                swa_tokens=2048,
                state_slots=64,
                context_len=256,
                max_num_reqs=4,
                device="cuda",
            )
        )
        self.pool, self.alloc = (
            bundle.token_to_kv_pool,
            bundle.token_to_kv_pool_allocator,
        )
        self.requests = bundle.req_to_token_pool.req_to_token
        self.ids = RunnerIds(
            req_to_token=self.requests,
            token_to_kv_pool_allocator=self.alloc,
            token_to_kv_pool=self.pool,
            page_size=_PAGE,
            device="cuda",
        )
        runner = SimpleNamespace(
            model_config=SimpleNamespace(
                dtype=_DTYPE,
                is_multimodal=False,
                is_encoder_decoder=False,
                num_attention_heads=_HEADS,
                head_dim=_DIM,
                get_max_num_attention_heads=lambda: _HEADS,
                get_num_kv_heads=lambda tp, dcp=1: _KV_HEADS // tp,
                attention_arch=AttentionArch.MHA,
                context_len=256,
                hf_config=SimpleNamespace(model_type="gpt_oss"),
            ),
            server_args=args,
            dtype=_DTYPE,
            kv_cache_dtype=fp8_dtype,
            device="cuda",
            gpu_id=0,
            page_size=_PAGE,
            sliding_window_size=self.window,
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            decode_num_tokens_per_req=lambda: 1,
            runner_ids=self.ids,
            token_to_kv_pool=self.pool,
            token_to_kv_pool_allocator=self.alloc,
            req_to_token_pool=bundle.req_to_token_pool,
        )
        if self.backend_name == "aiter":
            from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

            self.backend = AiterAttnBackend(runner)
        else:
            from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

            self.backend = TritonAttnBackend(runner)
        self.enterContext(forward_context(ForwardContext(attn_backend=self.backend)))
        self.layers = [
            RadixAttention(
                _HEADS,
                _DIM,
                _DIM**-0.5,
                _KV_HEADS,
                layer_id,
                sliding_window_size=self.window if self.window and i == 1 else -1,
            )
            for i, layer_id in enumerate((0, 2))
        ]
        self.rope = RotaryEmbedding(_DIM, _DIM, 256, 150000, True, _DTYPE).cuda()
        self.assertTrue(self.rope.use_fallback_kernel)
        self.assertEqual(self.rope.cos_sin_cache.dtype, torch.float32)
        self.groups = [self.alloc.alloc(32) for _ in range(4)]
        self.assertTrue(all(group is not None for group in self.groups))
        self.live = [self.groups[i] for i in (0, 2, 3)]
        self.rows = [3, 1, 2]
        for row, slots in zip(self.rows, self.live):
            self.requests[row, :32] = slots
        self.bands = [self.alloc.full_attn_allocator]
        if self.window:
            self.bands.append(self.alloc.swa_attn_allocator)
        for layer in self.layers:
            for cache in self.pool.layer_router.read(layer.layer_id).kv():
                cache.view(torch.uint8).zero_()
        self.positions = torch.tensor([11, 23, 31], device="cuda")
        torch.manual_seed(317)
        self.sources = [
            tuple(
                torch.randn(3, heads * _DIM, dtype=_DTYPE, device="cuda")
                for heads in (_HEADS, _KV_HEADS, _KV_HEADS)
            )
            for _ in self.layers
        ]

    def _batch(self, real_rows=3):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.model_executor.forward_mode import ForwardMode

        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=real_rows,
            input_ids=torch.zeros(real_rows, dtype=torch.int64, device="cuda"),
            req_pool_indices=torch.tensor(self.rows[:real_rows], device="cuda"),
            seq_lens=torch.full((real_rows,), 32, dtype=torch.int64, device="cuda"),
            seq_lens_cpu=torch.full((real_rows,), 32, dtype=torch.int64),
            seq_lens_sum=32 * real_rows,
            out_cache_loc=torch.stack([slots[-1] for slots in self.live[:real_rows]]),
        )
        self.ids.bind_own_plan(fb)
        return fb

    def _physical(self, layer_index, real_rows=3):
        band = self.bands[1 if self.window and layer_index == 1 else 0]
        virtual = torch.stack([slots[-1] for slots in self.live[:real_rows]])
        return band.ids.virtual_to_physical[virtual // _PAGE] * _PAGE + virtual % _PAGE

    def _produce(self, fb):
        from sglang.srt.models.utils import create_fused_set_kv_buffer_arg

        outputs = []
        for layer, (source_q, source_k, source_v) in zip(self.layers, self.sources):
            # A captured QKV producer refreshes these in-place RoPE inputs.
            q, k = source_q.clone(), source_k.clone()
            args = create_fused_set_kv_buffer_arg(source_v, layer, fb)
            self.assertEqual(args["reserved_skip_index"], 0)
            q, k = self.rope.forward_cuda(
                self.positions, q, k, fused_set_kv_buffer_arg=args
            )
            outputs.append((q, k, args["slot_mapping"]))
        return outputs

    def _packed_reference(self, layer, sources, real_rows):
        q, k, v = [value[:real_rows].clone() for value in sources]
        caches = [
            torch.zeros(
                (real_rows + 1, _PAGE, _KV_HEADS, _DIM),
                dtype=self.cache_dtype,
                device="cuda",
            )
            for _ in range(2)
        ]
        # Static pools permit slot zero; their default skip index must remain -1.
        slots = torch.arange(real_rows, device="cuda", dtype=torch.int64) * _PAGE
        q, k = self.rope.forward_cuda(
            self.positions[:real_rows],
            q,
            k,
            fused_set_kv_buffer_arg={
                "v": v.view(real_rows, _KV_HEADS, _DIM),
                "key_cache": caches[0],
                "value_cache": caches[1],
                "slot_mapping": slots,
                "swa_slot_mapping": None,
                "k_scale": layer.k_scale,
                "v_scale": layer.v_scale,
            },
        )
        stored = [
            cache.view(-1, _KV_HEADS, _DIM).view(torch.uint8)[slots] for cache in caches
        ]
        return q, k, stored

    def _rotated_fp32(self, source, real_rows):
        source = source[:real_rows].view(real_rows, -1, _DIM).float()
        cos, sin = self.rope.cos_sin_cache[self.positions[:real_rows]].chunk(2, dim=-1)
        first, second = source.chunk(2, dim=-1)
        return torch.cat(
            (
                first * cos[:, None] - second * sin[:, None],
                second * cos[:, None] + first * sin[:, None],
            ),
            dim=-1,
        )

    def _check(self, outputs, real_rows=3):
        for i, (layer, sources, (q, k, _)) in enumerate(
            zip(self.layers, self.sources, outputs)
        ):
            ref_q, ref_k, stored = self._packed_reference(layer, sources, real_rows)
            torch.testing.assert_close(q[:real_rows], ref_q, atol=0, rtol=0)
            torch.testing.assert_close(k[:real_rows], ref_k, atol=0, rtol=0)
            math_q, math_k = [
                self._rotated_fp32(value, real_rows) for value in sources[:2]
            ]
            for actual, expected in ((q, math_q), (k, math_k)):
                torch.testing.assert_close(
                    actual[:real_rows].reshape_as(expected).float(),
                    expected,
                    atol=1e-5,
                    rtol=0.004,
                )
            physical = self._physical(i, real_rows)
            ideal = (
                math_k / layer.k_scale,
                sources[2][:real_rows].view(real_rows, _KV_HEADS, _DIM).float()
                / layer.v_scale,
            )
            for cache, expected_bytes, expected_float in zip(
                self.pool.layer_router.read(layer.layer_id).kv(), stored, ideal
            ):
                actual_bytes = cache.view(torch.uint8)[physical]
                torch.testing.assert_close(actual_bytes, expected_bytes, atol=0, rtol=0)
                # Fused K is FP32 RoPE / scale -> FP8, without an intermediate
                # BF16 cast. Bound nearest-FP8 rounding plus FP32 arithmetic;
                # exact packed-producer parity above is the layout regression.
                limits = torch.finfo(self.cache_dtype)
                bound = expected_float.abs().clamp_min(limits.tiny) * limits.eps / 2
                bound += expected_float.abs().clamp_min(1) * 2e-6
                delta = (
                    actual_bytes.view(self.cache_dtype).float() - expected_float
                ).abs()
                self.assertTrue(
                    (delta <= bound).all().item(), (delta - bound).max().item()
                )
                self.assertTrue((cache[0].view(torch.uint8) == 0).all().item())

    def _run_replay(self, k_scale, v_scale):
        from sglang.srt.models.utils import enable_fused_set_kv_buffer

        for layer in self.layers:
            layer.k_scale = torch.tensor(k_scale, device="cuda")
            layer.v_scale = torch.tensor(v_scale, device="cuda")
        fb = self._batch()
        self.assertTrue(enable_fused_set_kv_buffer(fb))
        self.backend.init_forward_metadata(fb)
        self._check(self._produce(fb))
        self.backend.init_cuda_graph_state(3, 3)
        stable_slots = fb.out_cache_loc.clone()
        fb.iteration_ids.bind_replay(fb, self.ids, slots=stable_slots)
        if self.backend_name == "triton":
            # Triton binds named rails to persistent buffers at graph setup;
            # replay only refills those buffers. Borrow no eager plan tensor.
            self.backend.init_forward_metadata_out_graph(fb, in_capture=True)
        self.backend.init_forward_metadata_out_graph(fb)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._produce(fb)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = self._produce(fb)
        pointers = [slots.data_ptr() for _, _, slots in outputs]
        graph.replay()
        self._check(outputs)

        before = [self._physical(i).clone() for i in range(len(self.layers))]
        torch.cuda.synchronize()
        self.alloc.free(self.groups[1])
        for band in self.bands:
            self.assertGreater(band.flush_for_allocation(), 0)
        after = [self._physical(i) for i in range(len(self.layers))]
        for old, new in zip(before, after):
            self.assertFalse(torch.equal(old, new))
        self.assertIsNotNone(self.alloc.alloc(32))
        stale = [old[~torch.isin(old, new)] for old, new in zip(before, after)]
        for layer, old in zip(self.layers, stale):
            for cache in self.pool.layer_router.read(layer.layer_id).kv():
                cache.view(torch.uint8)[old] = 42
        for sources in self.sources:
            for source in sources:
                source.add_(0.25)

        fresh = self._batch()
        # bind_replay borrows the runner's buffer after it has been refreshed;
        # it does not copy the newly translated full-attention ids into it.
        stable_slots.copy_(fresh.out_cache_loc)
        fresh.iteration_ids.bind_replay(fb, self.ids, slots=stable_slots)
        self.backend.init_forward_metadata_out_graph(fb)
        graph.replay()
        self._check(outputs)
        for layer, old in zip(self.layers, stale):
            for cache in self.pool.layer_router.read(layer.layer_id).kv():
                self.assertTrue((cache.view(torch.uint8)[old] == 42).all().item())

        # One live request in the original three-row graph: NaN producers must
        # not write either band's reserved physical slot zero.
        fb.seq_lens.copy_(torch.tensor([32, 1, 1], device="cuda"))
        fb.seq_lens_cpu = torch.tensor([32, 1, 1])
        fb.seq_lens_sum = 34
        fb.req_pool_indices[1:] = 0
        for sources in self.sources:
            for source in sources:
                source[0].add_(0.25)
                source[1:].fill_(torch.nan)
        fresh = self._batch(1)
        stable_slots.zero_()
        stable_slots[:1].copy_(fresh.out_cache_loc)
        fresh.iteration_ids.bind_replay(fb, self.ids, slots=stable_slots)
        self.backend.init_forward_metadata_out_graph(fb)
        graph.replay()
        self._check(outputs, real_rows=1)
        self.assertEqual([slots.data_ptr() for _, _, slots in outputs], pointers)
        for _, _, slots in outputs:
            self.assertTrue((slots[1:] == 0).all().item())

    def test_unit_scale_relocation_and_padded_replay(self):
        self._run_replay(1.0, 1.0)

    def test_nonunit_scale_relocation_and_padded_replay(self):
        self._run_replay(0.3, 0.7)


@unittest.skipUnless(
    envs.SGLANG_ENABLE_AITER_UNIFIED_MEMORY.get(),
    "requires SGLANG_ENABLE_AITER_UNIFIED_MEMORY=1 and the documented AITER fixes",
)
@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "requires ROCm")
class TestAiterUnifiedFusedRope(_UnifiedFusedRopeMixin, CustomTestCase):
    pass


class TestAiterUnifiedWindowFusedRope(TestAiterUnifiedFusedRope):
    window = 127


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "requires ROCm")
class TestTritonUnifiedWindowFusedRope(_UnifiedFusedRopeMixin, CustomTestCase):
    # Exercise the actual named-window metadata producer only. Attention
    # kernels are not invoked or qualified by these fused-store regressions.
    backend_name = "triton"
    window = 127


if __name__ == "__main__":
    unittest.main()

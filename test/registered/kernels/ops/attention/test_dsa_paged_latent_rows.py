"""Paged latent rows of the unified memory pool, on GPU.

The unified pool lays a sparse-attention page out layer by layer, so a layer's
latent rows are a paged buffer ``(pages, 64, 1, 656)`` whose pages lie a page
entry apart. The latent writers, the prefix dequant gathers and FlashMLA's
sparse decode address token ``t``'s row through the buffer's page and row
strides, with the token id itself:

  - each kernel writes and reads the same bytes on the paged buffer as on a
    static pool's flat run of rows, and touches nothing else of the page;
  - on a flat run of rows the kernels write exactly what they always did;
  - a decode step launches no more kernels on the unified pool than on a
    static one: metadata (eager and graph replay) and every layer's latent
    write, index-key write and attention.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    dequantize_k_cache_paged_dsa,
    gather_dequant_requant_fp8_paged_dsa,
)
from sglang.kernels.ops.kvcache.dsa_fused_store_index_cache import (
    fused_store_index_k_cache,
)
from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton
from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache_separate
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import (
    FP8_DS_MLA,
    KV_PART,
    FullLayerParts,
    IndexKeyGeometry,
    MLAGeometry,
    MLARowFormat,
    kv_layer_decls,
)
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, KVWriteLoc
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE, ROW, LAYERS, PAGES, TOPK = 64, 656, 4, 8, 2048
_FLASHMLA = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (
    9,
    10,
)


def _sub_pool_decls(layers=LAYERS):
    return kv_layer_decls(
        full_layer_ids=tuple(range(layers)),
        full_geometry=MLAGeometry(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            kv_cache_dtype=torch.float8_e4m3fn,
            row_format=MLARowFormat(name=FP8_DS_MLA, row_bytes=656, scale_ue8m0=True),
            layer_major=True,
        ),
        full_layer_parts=(
            FullLayerParts(
                layer_ids=tuple(range(layers)),
                geometry=IndexKeyGeometry(index_head_dim=128),
            ),
        ),
    )


def _sub_pool(layers=LAYERS):
    return plan_kv_storage(_sub_pool_decls(layers), page_size=PAGE).owner()


def _spec(layers=LAYERS):
    return _sub_pool(layers).spec


def _rows_views(spec, arena):
    return spec.part_views(arena, KV_PART)


def _arena(spec, pages=PAGES):
    raw = torch.zeros(pages * spec.page_bytes(), dtype=torch.uint8, device="cuda")
    return SimpleNamespace(
        raw=raw,
        device=raw.device,
        band=lambda name: SimpleNamespace(unit_entries=PAGE),
        extent_entries=lambda name: pages * PAGE,
    )


def _token_ids(n, seed=0, *, with_sink=False):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randperm((PAGES - 1) * PAGE, generator=g)[:n] + PAGE
    if with_sink:
        ids[0] = 0  # the padding sink: writes skip it
    return ids.to(torch.int64).cuda()


def _packed_latents(n, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    nope = torch.randn(n, 1, 512, dtype=torch.bfloat16, device="cuda", generator=g)
    rope = torch.randn(n, 1, 64, dtype=torch.bfloat16, device="cuda", generator=g)
    return quantize_k_cache_separate(nope, rope, scale_ue8m0=True)


def _rows(paged, ids):
    return paged[ids // PAGE, ids % PAGE, 0]


class TestPagedWrites(CustomTestCase):
    def _check_paged_write(self, write):
        spec = _spec()
        arena = _arena(spec)
        paged = _rows_views(spec, arena)[2]
        flat = torch.zeros(PAGES * PAGE, 1, ROW, dtype=torch.uint8, device="cuda")
        ids = _token_ids(300, with_sink=True)
        nope, rope = _packed_latents(300)
        write(paged, ids, nope, rope)
        write(flat, ids, nope, rope)
        # A flat run of rows gets exactly the reference scatter.
        expected = torch.zeros_like(flat)
        packed = torch.cat([nope.reshape(300, -1), rope.reshape(300, -1)], dim=-1)
        expected[ids[1:], 0] = packed[1:]
        torch.testing.assert_close(flat, expected, rtol=0, atol=0)
        # The paged buffer holds the same rows at the same token ids ...
        torch.testing.assert_close(_rows(paged, ids), flat[ids, 0], rtol=0, atol=0)
        # ... and nothing else of the arena changed.
        rest = arena.raw.clone()
        rest_arena = SimpleNamespace(**{**vars(arena), "raw": rest})
        rest_view = _rows_views(spec, rest_arena)[2]
        rest_view[ids // PAGE, ids % PAGE, 0] = 0
        self.assertEqual(int(rest.count_nonzero()), 0)

    def test_jit_writer(self):
        self._check_paged_write(
            lambda buf, ids, nope, rope: set_mla_kv_buffer_triton(buf, ids, nope, rope)
        )

    def test_triton_writer(self):
        with patch(
            "sglang.kernels.ops.kvcache.mla_buffer.is_arch_support_pdl",
            return_value=False,
        ):
            self._check_paged_write(
                lambda buf, ids, nope, rope: set_mla_kv_buffer_triton(
                    buf, ids, nope, rope
                )
            )

    def test_norope_writer(self):
        """NoPE-only rows (no RoPE tail) through the paged addressing."""
        pages, row, page_stride = PAGES, 512, 3 * PAGE * 512
        raw = torch.zeros(pages * page_stride, dtype=torch.bfloat16, device="cuda")
        paged = torch.as_strided(
            raw, (pages, PAGE, 1, row), (page_stride, row, row, 1), PAGE * row
        )
        flat = torch.zeros(pages * PAGE, 1, row, dtype=torch.bfloat16, device="cuda")
        ids = _token_ids(200, seed=1)
        nope = torch.randn(200, 1, row, dtype=torch.bfloat16, device="cuda")
        set_mla_kv_buffer_triton(paged, ids, nope, None)
        set_mla_kv_buffer_triton(flat, ids, nope, None)
        torch.testing.assert_close(flat[ids], nope, rtol=0, atol=0)
        torch.testing.assert_close(_rows(paged, ids), flat[ids, 0], rtol=0, atol=0)
        self.assertEqual(int(raw.count_nonzero()), int(nope.count_nonzero()))


class TestPagedGathers(CustomTestCase):
    def setUp(self):
        spec = _spec()
        self.paged = _rows_views(spec, _arena(spec))[1]
        self.flat = torch.zeros(PAGES * PAGE, 1, ROW, dtype=torch.uint8, device="cuda")
        ids = _token_ids(300, seed=2)
        nope, rope = _packed_latents(300, seed=2)
        set_mla_kv_buffer_triton(self.paged, ids, nope, rope)
        set_mla_kv_buffer_triton(self.flat, ids, nope, rope)
        self.gather = ids[torch.randperm(300, device="cuda")[:200]].to(torch.int32)

    def test_dequant(self):
        actual = dequantize_k_cache_paged_dsa(
            self.paged.view(torch.float8_e4m3fn), self.gather
        )
        expected = dequantize_k_cache_paged_dsa(
            self.flat.view(torch.float8_e4m3fn), self.gather
        )
        self.assertTrue(torch.isfinite(expected).all())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dequant_requant(self):
        actual = gather_dequant_requant_fp8_paged_dsa(
            self.paged.view(torch.float8_e4m3fn), self.gather, extra_rows=16
        )
        expected = gather_dequant_requant_fp8_paged_dsa(
            self.flat.view(torch.float8_e4m3fn), self.gather, extra_rows=16
        )
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        )


def _flashmla_backend(pool):
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.__dict__.update(
        token_to_kv_pool=pool,
        flashmla_kv_num_q_heads=64,
        dsa_shards_index_k=False,
        physical_page_size=PAGE,
        kv_cache_dim=ROW,
        dsa_kv_cache_store_fp8=True,
        dsa_index_topk=TOPK,
        flashmla_kv_format="V32",
    )
    return backend


def _flashmla_decode(backend, kv_cache, q, indices, lengths):
    metadata = SimpleNamespace(
        dsa_cache_seqlens_int32=lengths,
        flashmla_metadata=backend._compute_flashmla_metadata(
            cache_seqlens=lengths, seq_len_q=1
        ),
    )
    return backend._forward_flashmla_kv(
        q_all=q,
        kv_cache=kv_cache.view(torch.float8_e4m3fn),
        v_head_dim=512,
        sm_scale=576**-0.5,
        layer=SimpleNamespace(tp_q_head_num=64, head_dim=576),
        metadata=metadata,
        page_table_1=indices,
        forward_batch=SimpleNamespace(),
    )


def _decode_inputs(ids, rows=4, seed=3):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(rows, 64, 576, dtype=torch.bfloat16, device="cuda", generator=g)
    lengths = torch.tensor([300, 17, 150, 1], dtype=torch.int32, device="cuda")[:rows]
    indices = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
    for row, length in enumerate(lengths.tolist()):
        pick = torch.randperm(ids.numel(), device="cuda", generator=g)[:length]
        indices[row, :length] = ids[pick].to(torch.int32)
    return q / 4, indices, lengths


@unittest.skipUnless(_FLASHMLA, "FlashMLA sparse decode needs SM90 or SM100")
class TestFlashMLAPagedDecode(CustomTestCase):
    def test_paged_rows_match_static_rows(self):
        spec = _spec()
        paged = _rows_views(spec, _arena(spec))[3]
        flat = torch.zeros(PAGES * PAGE, 1, ROW, dtype=torch.uint8, device="cuda")
        ids = _token_ids(300, seed=4)
        nope, rope = _packed_latents(300, seed=4)
        set_mla_kv_buffer_triton(paged, ids, nope, rope)
        set_mla_kv_buffer_triton(flat, ids, nope, rope)
        q, indices, lengths = _decode_inputs(ids)
        backend = _flashmla_backend(SimpleNamespace())
        expected = _flashmla_decode(backend, flat, q, indices, lengths)
        actual = _flashmla_decode(backend, paged, q, indices, lengths)
        self.assertTrue(torch.isfinite(expected).all())
        self.assertGreater(expected.abs().sum().item(), 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _gpu_ops(fn) -> list:
    """The GPU kernels and memory ops ``fn`` launches, by name."""
    fn()  # warm up: compile and cache
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]


def _metadata_backend(req_to_token, runner_ids, bs):
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.__dict__.update(
        device=torch.device("cuda"),
        runner_ids=runner_ids,
        req_to_token=req_to_token,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool=SimpleNamespace(
            size=PAGES * PAGE, page_size=PAGE, dtype=torch.float8_e4m3fn
        ),
        _kv_shard_pool=None,
        physical_page_size=PAGE,
        real_page_span=PAGE,
        dsa_index_kpool=1,
        dsa_index_topk=TOPK,
        dsa_shards_index_k=False,
        attn_dcp_size=1,
        attn_dcp_rank=0,
        dsa_kv_cache_store_fp8=True,
        dsa_prefill_impl="flashmla_kv",
        dsa_decode_impl="flashmla_kv",
        enable_auto_select_prefill_impl=False,
        supports_mha_one_shot=False,
        use_mha=False,
        hisparse_coordinator=None,
        speculative_num_draft_tokens=0,
        dsa_drop_wide_page_table=False,
        dsa_topk_backend=DSATopKBackend.TORCH,
        flashmla_kv_num_q_heads=64,
        flashmla_kv_num_q_heads_prefill=64,
        _arange_buf=torch.arange(bs + 1, dtype=torch.int32, device="cuda"),
        _is_in_breakable_cuda_graph=lambda: False,
        _is_in_tc_piecewise_cuda_graph=lambda: False,
        _get_device_sm=lambda: torch.cuda.get_device_capability()[0] * 10,
        _is_blackwell=lambda: torch.cuda.get_device_capability()[0] == 10,
    )
    backend.decode_cuda_graph_metadata = {
        "page_table": torch.zeros(
            bs, req_to_token.shape[1], dtype=torch.int32, device="cuda"
        ),
        "cu_seqlens_q": backend._arange_buf,
    }
    backend._init_kpool_metadata_fusion()
    return backend


@unittest.skipUnless(_FLASHMLA, "FlashMLA sparse decode needs SM90 or SM100")
class TestDecodeLaunchCounts(CustomTestCase):
    """A decode step on the unified pool launches no more GPU work than on a
    static pool: the page tables translate inside the gathers and builders a
    static pool runs anyway, and the paged rows need no remapping."""

    BS = 4

    def setUp(self):
        setup_allocator_context()
        # Four requests of four virtual pages each, over 32 virtual pages.
        g = torch.Generator().manual_seed(6)
        virtual_pages = 32
        pages = torch.randperm(virtual_pages, generator=g)[: self.BS * 4]
        tokens = pages.view(self.BS, 4, 1) * PAGE + torch.arange(PAGE)
        self.req_to_token = torch.nn.functional.pad(
            tokens.view(self.BS, 4 * PAGE).to(torch.int32), (0, 4 * PAGE)
        ).cuda()
        v2p = torch.full((virtual_pages + 1,), -1, dtype=torch.int64, device="cuda")
        v2p[:virtual_pages] = torch.randperm(virtual_pages, generator=g).cuda()
        space = IdSpace(key=("full", 0), read_v2p=v2p, table=v2p)
        self.unified_ids = SimpleNamespace(
            reads_are_translated=True,
            is_translating=True,
            page_size=PAGE,
            space=lambda kind: space if kind == IdSpaceKind.FULL else None,
        )
        self.req = torch.tensor([3, 0, 2, 1], device="cuda")
        self.seq = torch.tensor([200, 17, 256, 64], device="cuda")
        self.counts = {}

    def tearDown(self):
        for phase, (static, unified) in self.counts.items():
            print(f"{phase}: static {len(static)} GPU ops, unified {len(unified)}")

    def _compare(self, phase, static_fn, unified_fn):
        static, unified = _gpu_ops(static_fn), _gpu_ops(unified_fn)
        self.counts[phase] = (static, unified)
        self.assertLessEqual(len(unified), len(static), (phase, static, unified))

    def test_eager_metadata(self):
        def step(runner_ids):
            backend = _metadata_backend(self.req_to_token, runner_ids, self.BS)
            batch = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                batch_size=self.BS,
                req_pool_indices=self.req,
                seq_lens=self.seq,
                seq_lens_cpu=self.seq.cpu(),
                seq_lens_sum=int(self.seq.sum()),
                global_num_tokens_cpu=None,
                attn_cp_metadata=None,
            )
            return lambda: backend.init_forward_metadata(batch)

        self._compare("eager metadata", step(None), step(self.unified_ids))

    def test_graph_replay_metadata(self):
        def step(runner_ids):
            backend = _metadata_backend(self.req_to_token, runner_ids, self.BS)

            def replay():
                backend._apply_cuda_graph_metadata(
                    bs=self.BS,
                    req_pool_indices=self.req,
                    seq_lens=self.seq,
                    seq_lens_cpu=self.seq.cpu(),
                    forward_mode=ForwardMode.DECODE,
                    spec_info=None,
                )

            replay()  # the first call captures the graph's buffers
            return replay

        self._compare("graph replay metadata", step(None), step(self.unified_ids))

    def test_layer_forward(self):
        """Every layer's latent write, index-key write and sparse attention."""
        # The unified pool as a server builds it: the plan, its sizing, then
        # the arena, views and router; layers reach their rows and keys
        # through the routed pool.
        setup_allocator_context()
        plan = plan_kv_storage(_sub_pool_decls(), page_size=PAGE)
        sized = size_plan(
            plan,
            KVSizingInputs(
                full_tokens=PAGES * PAGE,
                max_running_requests=self.BS,
                context_len=PAGES * PAGE,
                alignment=1,
            ),
        )
        unified = build_kv_storage(
            plan,
            sized,
            UnifiedBuildContext(
                device="cuda",
                start_layer=0,
                end_layer=LAYERS,
                enable_memory_saver=False,
                need_sort=False,
                max_num_reqs=self.BS,
                model_context_len=PAGES * PAGE,
            ),
        ).allocator.get_kvcache()
        static = DSATokenToKVPool(
            size=PAGES * PAGE - PAGE,
            page_size=PAGE,
            kv_lora_rank=512,
            dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=64,
            layer_num=LAYERS,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=ROW,
            kv_scale_ue8m0=True,
        )
        loc = _token_ids(self.BS, seed=7)
        k = torch.randn(self.BS, 1, 512, dtype=torch.bfloat16, device="cuda")
        k_rope = torch.randn(self.BS, 1, 64, dtype=torch.bfloat16, device="cuda")
        index_k = torch.randn(self.BS, 128, dtype=torch.bfloat16, device="cuda")
        q, indices, lengths = _decode_inputs(_token_ids(300, seed=8), rows=self.BS)

        def step(pool, write_loc):
            backend = _flashmla_backend(pool)
            metadata = SimpleNamespace(
                dsa_cache_seqlens_int32=lengths,
                flashmla_metadata=backend._compute_flashmla_metadata(
                    cache_seqlens=lengths, seq_len_q=1
                ),
            )

            def forward():
                for layer_id in range(LAYERS):
                    layer = SimpleNamespace(layer_id=layer_id)
                    pool.set_mla_kv_buffer(layer, write_loc, k, k_rope)
                    fused_store_index_k_cache(
                        index_k,
                        pool.get_index_k_with_scale_buffer(layer_id),
                        loc,
                        PAGE,
                    )
                    backend._forward_flashmla_kv(
                        q_all=q,
                        kv_cache=pool.get_key_buffer(layer_id),
                        v_head_dim=512,
                        sm_scale=576**-0.5,
                        layer=SimpleNamespace(tp_q_head_num=64, head_dim=576),
                        metadata=metadata,
                        page_table_1=indices,
                        forward_batch=SimpleNamespace(),
                    )

            return forward

        self._compare(
            f"{LAYERS}-layer forward",
            step(static, loc),
            step(unified, KVWriteLoc(loc, physical=True)),
        )


if __name__ == "__main__":
    unittest.main()

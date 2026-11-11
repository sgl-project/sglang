"""DSA attention on the unified memory pool, on CPU.

`req_to_token` holds virtual token ids there, which the backend's page tables
translate once, as they gather them, into the physical ids of the full
sub-pool: the indexer, the top-k transforms and attention then all read
physical ids. A layer's latent rows are a paged buffer the kernels address by
the physical token id itself; the backend hands them its writes' ids marked
physical. A static pool keeps the ids as `req_to_token` holds them.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
)
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _validate_unified_pool_backend,
)
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
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
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, write_loc_is_physical
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 4
REQ_ROWS, WIDTH, VIRTUAL_PAGES = 4, 32, 40


def _virtual_batch():
    """`req_to_token` over shuffled virtual pages, and a v2p that scatters them
    over physical pages (virtual page 0 and the trailing row stay the sink and
    the -1 sentinel)."""
    g = torch.Generator().manual_seed(7)
    pages = torch.randperm(VIRTUAL_PAGES - 1, generator=g)[: REQ_ROWS * WIDTH // PAGE]
    pages = (pages + 1).view(REQ_ROWS, WIDTH // PAGE)
    req_to_token = (
        (pages[:, :, None] * PAGE + torch.arange(PAGE)).view(REQ_ROWS, WIDTH)
    ).to(torch.int32)
    v2p = torch.full((VIRTUAL_PAGES + 1,), -1, dtype=torch.int64)
    v2p[0] = 0
    v2p[1:VIRTUAL_PAGES] = torch.randperm(VIRTUAL_PAGES - 1, generator=g) + 100
    return req_to_token, v2p


def _reference(ids, v2p):
    ids = ids.to(torch.int64)
    return (v2p[ids // PAGE] * PAGE + ids % PAGE).to(torch.int32)


def _runner_ids(v2p):
    """A unified target's RunnerIds, as the DSA backend reads it."""
    space = IdSpace(key=("full", 0), read_v2p=v2p, table=v2p)
    return SimpleNamespace(
        reads_are_translated=True,
        is_translating=True,
        page_size=PAGE,
        space=lambda kind: space if kind == IdSpaceKind.FULL else None,
    )


def _routed(pool):
    pool.start_layer, pool.layer_transfer_counter = 0, None
    pool.layer_router = PlainPoolRouter(pool, IdSpaceKind.FULL)
    return pool


def _k31_sub_pool(layers=24):
    """The full sub-pool a K3.1-shaped model plans on the unified pool: FP8
    latent rows with block scales laid out layer-major, and the indexer keys
    every layer keeps beside them."""
    decls = kv_layer_decls(
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
    return plan_kv_storage(decls, page_size=64).owner()


def _k31_layer_rows(pages=3):
    """One layer's paged latent rows as the unified pool lays K3.1 out: 64
    656-byte rows per page, pages a 1,212,288-byte unit apart."""
    spec = _k31_sub_pool().spec
    arena = SimpleNamespace(
        raw=torch.zeros(pages * spec.page_bytes(), dtype=torch.uint8),
        band=lambda name: SimpleNamespace(unit_entries=64),
        extent_entries=lambda name: pages * 64,
    )
    return spec.part_views(arena, KV_PART)[5]


def _metadata_backend(req_to_token, runner_ids):
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.__dict__.update(
        device=torch.device("cpu"),
        runner_ids=runner_ids,
        req_to_token=req_to_token,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool=SimpleNamespace(
            size=8, page_size=PAGE, dtype=torch.float8_e4m3fn
        ),
        _kv_shard_pool=None,
        physical_page_size=PAGE,
        real_page_span=PAGE,
        dsa_index_kpool=1,
        dsa_index_topk=16,
        dsa_shards_index_k=False,
        attn_dcp_size=1,
        attn_dcp_rank=0,
        dsa_kv_cache_store_fp8=True,
        dsa_prefill_impl="flashmla_sparse",
        dsa_decode_impl="tilelang",
        enable_auto_select_prefill_impl=False,
        supports_mha_one_shot=False,
        use_mha=False,
        hisparse_coordinator=None,
        speculative_num_draft_tokens=0,
        dsa_topk_backend=DSATopKBackend.TORCH,
        _arange_buf=torch.arange(64, dtype=torch.int32),
        _is_in_breakable_cuda_graph=lambda: False,
        _is_in_tc_piecewise_cuda_graph=lambda: False,
        _get_device_sm=lambda: 100,
        _is_blackwell=lambda: True,
    )
    return backend


def _extend_batch(req_pool_indices, seq_lens, prefix_lens):
    extend = [s - p for s, p in zip(seq_lens, prefix_lens)]
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(seq_lens),
        req_pool_indices=torch.tensor(req_pool_indices),
        seq_lens=torch.tensor(seq_lens),
        seq_lens_cpu=torch.tensor(seq_lens),
        seq_lens_sum=sum(seq_lens),
        extend_seq_lens=torch.tensor(extend, dtype=torch.int32),
        extend_seq_lens_cpu=extend,
        extend_prefix_lens_cpu=prefix_lens,
        global_num_tokens_cpu=None,
        attn_cp_metadata=None,
    )


class TestPhysicalPageTables(CustomTestCase):
    def setUp(self):
        setup_allocator_context()

    def _extend(self, runner_ids):
        req_to_token, v2p = _virtual_batch()
        backend = _metadata_backend(req_to_token, runner_ids(v2p))
        rows, seq_lens = [2, 0, 3], [13, 7, 21]
        backend.init_forward_metadata(_extend_batch(rows, seq_lens, [5, 3, 9]))
        metadata = backend.forward_metadata
        virtual = req_to_token[torch.tensor(rows), : max(seq_lens)]
        return metadata, virtual, v2p, rows, seq_lens

    def test_eager_extend_tables_hold_physical_ids(self):
        """page_table_1, real_page_table and page_table_1_flattened (the prefix
        gather) all name physical ids; the capacity check, which bounds the
        static pool's ids, does not reject them though the batch outgrows the
        pool's `size`."""
        metadata, virtual, v2p, rows, seq_lens = self._extend(_runner_ids)
        physical = _reference(virtual, v2p)
        self.assertTrue(torch.equal(metadata.page_table_1, physical))
        self.assertTrue(
            torch.equal(metadata.real_page_table, physical[:, ::PAGE] // PAGE)
        )
        flat = torch.cat([physical[i, :n] for i, n in enumerate(seq_lens)])
        self.assertTrue(torch.equal(metadata.page_table_1_flattened, flat))

    def test_static_pool_tables_hold_req_to_token_ids(self):
        req_to_token = torch.arange(REQ_ROWS * WIDTH, dtype=torch.int32).view(
            REQ_ROWS, WIDTH
        )
        backend = _metadata_backend(req_to_token, None)
        backend.token_to_kv_pool.size = REQ_ROWS * WIDTH
        rows, seq_lens = [2, 0], [13, 7]
        backend.init_forward_metadata(_extend_batch(rows, seq_lens, [5, 3]))
        metadata = backend.forward_metadata
        expected = req_to_token[torch.tensor(rows), :13]
        self.assertTrue(torch.equal(metadata.page_table_1, expected))
        self.assertTrue(
            torch.equal(metadata.real_page_table, expected[:, ::PAGE] // PAGE)
        )

    def test_eager_decode_tables_hold_physical_ids(self):
        req_to_token, v2p = _virtual_batch()
        backend = _metadata_backend(req_to_token, _runner_ids(v2p))
        rows, seq_lens = [3, 1], [17, 6]
        # The paged logits schedule is a GPU kernel's; these tables are not.
        with patch(
            "sglang.srt.layers.attention.dsa_backend.is_cuda", return_value=False
        ):
            backend.init_forward_metadata(
                SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    batch_size=2,
                    req_pool_indices=torch.tensor(rows),
                    seq_lens=torch.tensor(seq_lens),
                    seq_lens_cpu=torch.tensor(seq_lens),
                    seq_lens_sum=sum(seq_lens),
                    global_num_tokens_cpu=None,
                    attn_cp_metadata=None,
                )
            )
        metadata = backend.forward_metadata
        physical = _reference(req_to_token[torch.tensor(rows), :17], v2p)
        self.assertTrue(torch.equal(metadata.page_table_1, physical))
        self.assertTrue(
            torch.equal(metadata.real_page_table, physical[:, ::PAGE] // PAGE)
        )

    def test_unmapped_pages_read_the_sink(self):
        req_to_token, v2p = _virtual_batch()
        backend = _metadata_backend(req_to_token, _runner_ids(v2p))
        dropped = int(req_to_token[1, 0]) // PAGE
        v2p[dropped] = -1
        table = backend._gather_token_table(req_to_token, torch.tensor([1]), PAGE)
        self.assertEqual(table.tolist(), [[0] * PAGE])


class _Written(Exception):
    """Stops a forward once its latent write has been seen."""


class TestLatentWrites(CustomTestCase):
    """The backend hands its latent writes the batch's write ids under the
    batch's physical mark: the unified pool's views refuse unmarked ids."""

    def _backend(self, seen):
        def store(layer, loc_info, k, k_rope):
            seen.append(loc_info)
            raise _Written

        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.__dict__.update(
            token_to_kv_pool=_routed(SimpleNamespace(set_mla_kv_buffer=store)),
            dsa_prefill_impl="flashmla_kv",
            dsa_decode_impl="flashmla_kv",
            use_mha=False,
            forward_metadata=None,
            _resolve_kpool_tail_backend=lambda indices, impl: impl,
            _check_kpool_tail_backend=lambda *args: None,
        )
        return backend

    def _write_loc(self, forward, mode, physical):
        seen = []
        backend = self._backend(seen)
        loc = torch.tensor([64, 65, 130])
        batch = SimpleNamespace(
            forward_mode=mode,
            out_cache_loc=loc,
            out_cache_loc_is_physical=physical,
        )
        layer = SimpleNamespace(layer_id=0, is_cross_attention=False)
        with self.assertRaises(_Written):
            getattr(backend, forward)(
                q=torch.zeros(3, 8),
                k=torch.zeros(3, 1, 512),
                v=torch.zeros(3, 1, 512),
                layer=layer,
                forward_batch=batch,
                save_kv_cache=True,
                k_rope=torch.zeros(3, 1, 64),
            )
        (loc_info,) = seen
        self.assertIsInstance(loc_info, KVWriteLoc)
        self.assertIs(loc_info.loc, loc)
        return loc_info

    def test_writes_carry_the_physical_mark(self):
        for forward, mode in (
            ("forward_extend", ForwardMode.EXTEND),
            ("forward_decode", ForwardMode.DECODE),
        ):
            with self.subTest(forward=forward):
                loc_info = self._write_loc(forward, mode, physical=True)
                self.assertTrue(write_loc_is_physical(loc_info))

    def test_static_writes_stay_unmarked(self):
        loc_info = self._write_loc("forward_decode", ForwardMode.DECODE, False)
        self.assertFalse(write_loc_is_physical(loc_info))


class TestFlashMLAReadsPagedRows(CustomTestCase):
    def test_indices_are_token_ids_and_the_cache_keeps_its_pages(self):
        """FlashMLA takes the layer's paged rows as they are, with the top-k's
        physical token ids: no remapping per layer."""
        rows = _k31_layer_rows()
        topk = torch.tensor([[0, 65, 130, -1], [3, 64, -1, -1]], dtype=torch.int32)
        seen = {}

        def fake_flash_mla(**kwargs):
            seen.update(kwargs)
            q = kwargs["q"]
            return q.new_zeros(*q.shape[:3], 512), q.new_zeros(q.shape[0], 1)

        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.__dict__.update(
            flashmla_kv_num_q_heads=2,
            dsa_shards_index_k=False,
            physical_page_size=64,
            kv_cache_dim=656,
            dsa_kv_cache_store_fp8=True,
            dsa_index_topk=topk.shape[1],
            flashmla_kv_format="V32",
            _call_flash_mla_with_kvcache=fake_flash_mla,
        )
        backend._forward_flashmla_kv(
            q_all=torch.zeros(2, 2, 576),
            kv_cache=rows,
            v_head_dim=512,
            sm_scale=1.0,
            layer=SimpleNamespace(tp_q_head_num=2, head_dim=576),
            metadata=SimpleNamespace(
                dsa_cache_seqlens_int32=torch.full((2,), 4, dtype=torch.int32),
                flashmla_metadata=SimpleNamespace(
                    flashmla_metadata=None, num_splits=None
                ),
            ),
            page_table_1=topk,
            forward_batch=SimpleNamespace(),
        )
        self.assertTrue(torch.equal(seen["indices"].squeeze(1), topk))
        k_cache = seen["k_cache"]
        self.assertEqual(k_cache.data_ptr(), rows.data_ptr())
        self.assertEqual(k_cache.stride(), (1_212_288, 656, 656, 1))


class TestTorchWriterPagedRows(CustomTestCase):
    def test_paged_rows_match_flat_rows(self):
        """The torch-native writer (platforms without Triton) addresses a paged
        buffer's rows as the kernels do, and a flat run as it always did."""
        from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_naive

        rows = _k31_layer_rows()
        flat = torch.zeros(3 * 64, 1, 656, dtype=torch.uint8)
        loc = torch.tensor([0, 65, 130, 7])
        nope = torch.randint(1, 255, (4, 1, 528), dtype=torch.uint8)
        rope = torch.randint(1, 255, (4, 1, 128), dtype=torch.uint8)
        for buf in (rows, flat):
            set_mla_kv_buffer_naive(
                buf,
                loc,
                nope,
                rope,
                reserved_skip_index=0,
                dcp_world_size=1,
                dcp_rank=0,
            )
        packed = torch.cat([nope, rope], dim=-1)
        self.assertTrue(torch.equal(flat[loc[1:]], packed[1:]))
        self.assertEqual(int(flat[0].count_nonzero()), 0)
        paged = rows[loc // 64, loc % 64]
        self.assertTrue(torch.equal(paged, flat[loc]))


class TestUnifiedPoolRefusals(CustomTestCase):
    SERVED = dict(
        translates_ids=True,
        prefill_impl="flashmla_kv",
        decode_impl="flashmla_kv",
        index_kpool=1,
        dcp_size=1,
        speculative_algorithm=None,
        return_indexer_topk=False,
        hisparse_enabled=False,
    )

    def test_served_configurations_pass(self):
        _validate_unified_pool_backend(**self.SERVED)
        for prefill in ("flashmla_sparse", "flashmla_auto", "triton_sparse_mla"):
            _validate_unified_pool_backend(**{**self.SERVED, "prefill_impl": prefill})

    def test_static_pool_is_not_checked(self):
        _validate_unified_pool_backend(
            **{
                **self.SERVED,
                "translates_ids": False,
                "decode_impl": "trtllm",
                "prefill_impl": "trtllm",
                "dcp_size": 8,
                "index_kpool": 4,
                "speculative_algorithm": "EAGLE",
                "return_indexer_topk": True,
            }
        )

    def test_refusals(self):
        cases = (
            ({"decode_impl": "trtllm", "prefill_impl": "trtllm"}, "trtllm"),
            ({"decode_impl": "fa3"}, "--dsa-decode-backend"),
            ({"prefill_impl": "tilelang"}, "--dsa-prefill-backend"),
            ({"index_kpool": 4}, "pooled indexer keys"),
            ({"dcp_size": 2}, "decode context parallelism"),
            ({"hisparse_enabled": True}, "hisparse"),
            ({"return_indexer_topk": True}, "return-indexer-topk"),
            ({"speculative_algorithm": "EAGLE"}, "speculative decoding"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_unified_pool_backend(**{**self.SERVED, **overrides})


if __name__ == "__main__":
    unittest.main()

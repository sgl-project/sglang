"""CPU unit test for the decode-context-parallel (DCP) per-rank KV-length math.

Pins ``get_dcp_lens`` (the single, superset implementation in
``layers/dcp/layout.py``) to a brute-force owner-count reference, and proves
it is bit-identical to the legacy in-place formula that
``update_local_kv_lens_for_dcp`` used before it was collapsed into a wrapper:

    floor((len - rank - 1) / N) + 1   ==   len // N + (rank < len % N)   (len >= 0)

Usage:
    python -m pytest test_dcp_layout_unit.py -v
    python test_dcp_layout_unit.py
"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt import runtime_context as rc
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.dcp.comm import all_gather_q_for_mla_decode
from sglang.srt.layers.dcp.layout import (
    filter_dcp_local_chunk_kv_indices,
    get_dcp_lens,
    remap_dcp_write_locations_fixed_shape,
)
from sglang.srt.layers.dcp.planner import prepare_decode_context_parallel_metadata
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

DCP_SIZES = [1, 2, 3, 4, 8]
LENS = list(range(0, 41))
STARTS = [0, 1, 2, 5, 7, 13, 31]


def _owner_count(length: int, n: int, rank: int, start: int) -> int:
    """Ground truth: # of absolute positions p in [start, start+length) with p % n == rank."""
    return sum(1 for p in range(start, start + length) if p % n == rank)


def _legacy_inplace_formula(length: int, n: int, rank: int) -> int:
    """The pre-refactor update_local_kv_lens_for_dcp body (start == 0 case)."""
    return (length - rank - 1) // n + 1


class TestFilterDcpLocalChunkKvIndices(CustomTestCase):
    PAGE = 64

    def _build_chunk(self, starts, lens, dcp_size, seed=0):
        g = torch.Generator().manual_seed(seed)
        widened = self.PAGE * dcp_size
        runs = []
        for start, length in zip(starts, lens):
            if length == 0:
                runs.append(torch.empty(0, dtype=torch.int64))
                continue
            pos = torch.arange(start, start + length)
            page_of = pos // widened
            bases = (
                torch.randint(0, 64, (int(page_of.max()) + 1,), generator=g) * widened
            )
            runs.append(bases[page_of] + pos % widened)
        return torch.cat(runs) if runs else torch.empty(0, dtype=torch.int64)

    def _owner_rule(self, kv, dcp_size, dcp_rank):
        # Selection only: the filters leave ids WIDENED and the collapse now
        # happens once, in KVIndexTranslator.translate_dcp_read_ids.
        return kv[kv % dcp_size == dcp_rank]

    def _run(self, starts, lens, dcp_size, dcp_rank, seed=0):
        kv = self._build_chunk(starts, lens, dcp_size, seed)
        with rc.get_parallel().override(
            dcp_enabled=dcp_size > 1, dcp_size=dcp_size, dcp_rank=dcp_rank
        ):
            got = filter_dcp_local_chunk_kv_indices(
                kv, torch.tensor(starts), torch.tensor(lens)
            )
        return kv, got

    def test_matches_owner_rule_on_unaligned_runs(self):
        cases = [
            ([0], [1]),
            ([0, 0], [2048, 2048]),
            ([8192], [3]),
            ([5, 13, 31], [7, 0, 19]),
            ([1365, 1365, 1365], [1365, 1365, 1365]),
            ([43690, 43690], [43690, 17]),
            ([7, 7, 7, 7, 7], [11, 13, 0, 1, 40]),
        ]
        for dcp_size in [2, 3, 4, 8]:
            for dcp_rank in range(dcp_size):
                for i, (starts, lens) in enumerate(cases):
                    kv, got = self._run(starts, lens, dcp_size, dcp_rank, seed=i)
                    torch.testing.assert_close(
                        got,
                        self._owner_rule(kv, dcp_size, dcp_rank),
                        msg=f"size={dcp_size} rank={dcp_rank} "
                        f"starts={starts} lens={lens}",
                    )

    def test_row_count_matches_get_dcp_lens(self):
        starts, lens = [1365, 2730, 0], [1365, 900, 37]
        for dcp_size in [2, 3, 4, 8]:
            for dcp_rank in range(dcp_size):
                _, got = self._run(starts, lens, dcp_size, dcp_rank)
                expected = get_dcp_lens(
                    torch.tensor(lens), dcp_size, dcp_rank, start=torch.tensor(starts)
                )
                self.assertEqual(
                    got.numel(),
                    int(expected.sum()),
                    f"size={dcp_size} rank={dcp_rank}",
                )

    def test_second_chunk_at_batch_three(self):
        starts, lens = [2730, 2730, 2730], [1365, 1365, 1365]
        for dcp_rank in range(8):
            kv, got = self._run(starts, lens, 8, dcp_rank)
            torch.testing.assert_close(got, self._owner_rule(kv, 8, dcp_rank))
            naive = kv[dcp_rank::8] // 8
            self.assertNotEqual(
                got.numel(),
                naive.numel(),
                f"rank={dcp_rank}: phase-free stride happens to match here, "
                f"so this case no longer guards the phase term",
            )

    def test_identity_without_dcp(self):
        kv = torch.arange(37)
        with rc.get_parallel().override(dcp_enabled=False, dcp_size=1, dcp_rank=0):
            self.assertIs(
                filter_dcp_local_chunk_kv_indices(
                    kv, torch.tensor([0]), torch.tensor([37])
                ),
                kv,
            )


class TestGetDcpLens(CustomTestCase):
    def test_shared_mla_query_gather_preserves_head_order(self):
        q_nope = torch.arange(24, dtype=torch.bfloat16).view(2, 3, 4)
        q_rope = torch.arange(12, dtype=torch.bfloat16).view(2, 3, 2)
        group = MagicMock()
        group.all_gather.side_effect = lambda tensor, dim: torch.cat(
            [tensor, tensor + 100], dim=dim
        )
        with (
            rc.get_parallel().override(dcp_group=group),
            patch("sglang.srt.layers.dcp.comm.use_symmetric_memory") as symmetric,
        ):
            nope, rope = all_gather_q_for_mla_decode(q_nope, q_rope)
        symmetric.assert_called_once_with(group, disabled=True)
        torch.testing.assert_close(nope, torch.cat([q_nope, q_nope + 100], dim=1))
        torch.testing.assert_close(rope, torch.cat([q_rope, q_rope + 100], dim=1))

    def test_fixed_shape_write_remap_uses_reserved_dummy_slot(self):
        virtual = torch.tensor([256, 257, 258, 259, 512, 513], dtype=torch.int32)
        rank0 = remap_dcp_write_locations_fixed_shape(virtual, 2, 0)
        rank1 = remap_dcp_write_locations_fixed_shape(virtual, 2, 1)

        self.assertEqual(rank0.tolist(), [128, 0, 129, 0, 256, 0])
        self.assertEqual(rank1.tolist(), [0, 128, 0, 129, 0, 256])
        self.assertEqual(rank0.shape, virtual.shape)
        self.assertEqual(rank1.shape, virtual.shape)

    def test_fixed_shape_write_remap_rejects_invalid_topology(self):
        with self.assertRaises(ValueError):
            remap_dcp_write_locations_fixed_shape(torch.arange(4), 0, 0)
        with self.assertRaises(ValueError):
            remap_dcp_write_locations_fixed_shape(torch.arange(4), 2, 2)

    def test_start_none_matches_owner_count(self):
        for n in DCP_SIZES:
            for rank in range(n):
                lens = torch.tensor(LENS, dtype=torch.int32)
                got = get_dcp_lens(lens, n, rank)
                expected = torch.tensor(
                    [_owner_count(L, n, rank, 0) for L in LENS], dtype=torch.int32
                )
                self.assertTrue(
                    torch.equal(got.to(torch.int32), expected),
                    f"start=None mismatch at n={n}, rank={rank}: {got.tolist()} != {expected.tolist()}",
                )

    def test_start_none_matches_legacy_inplace_formula(self):
        # The collapse claim: get_dcp_lens (start=None) == legacy floor((L-rank-1)/N)+1.
        for n in DCP_SIZES:
            for rank in range(n):
                lens = torch.tensor(LENS, dtype=torch.int64)
                got = get_dcp_lens(lens, n, rank)
                legacy = torch.tensor(
                    [_legacy_inplace_formula(L, n, rank) for L in LENS],
                    dtype=torch.int64,
                )
                self.assertTrue(
                    torch.equal(got.to(torch.int64), legacy),
                    f"legacy-formula mismatch at n={n}, rank={rank}",
                )

    def test_start_tensor_matches_owner_count(self):
        for n in DCP_SIZES:
            for rank in range(n):
                for start in STARTS:
                    lens = torch.tensor(LENS, dtype=torch.int64)
                    start_t = torch.full_like(lens, start)
                    got = get_dcp_lens(lens, n, rank, start=start_t)
                    expected = torch.tensor(
                        [_owner_count(L, n, rank, start) for L in LENS],
                        dtype=torch.int64,
                    )
                    self.assertTrue(
                        torch.equal(got.to(torch.int64), expected),
                        f"start={start} mismatch at n={n}, rank={rank}: "
                        f"{got.tolist()} != {expected.tolist()}",
                    )

    def test_dcp_size_one_is_identity(self):
        lens = torch.tensor(LENS, dtype=torch.int32)
        self.assertTrue(torch.equal(get_dcp_lens(lens, 1, 0), lens))

    def test_metadata_planner_slices_prefix_indices_without_packed_kv(self):
        translator = KVIndexTranslator.__new__(KVIndexTranslator)
        translator.is_translating = False
        backend = SimpleNamespace(
            dcp_use_packed_kv=False, kv_index_translator=translator
        )
        req_to_token = torch.arange(40, 64, dtype=torch.int32).view(3, 8)
        req_indices = torch.tensor([2, 0])
        extend_lens = torch.tensor([2, 3], dtype=torch.int32)

        kernel = MagicMock()
        for prefix_lengths in ([4, 8], [0, 4], [0, 0]):
            prefix_lens = torch.tensor(prefix_lengths, dtype=torch.int32)
            seq_lens = prefix_lens + extend_lens
            all_prefix = torch.cat(
                [req_to_token[r, :n] for r, n in zip(req_indices, prefix_lengths)]
            )
            for rank in range(4):
                with (
                    self.subTest(prefix_lengths=prefix_lengths, rank=rank),
                    rc.get_parallel().override(
                        dcp_enabled=True, dcp_size=4, dcp_rank=rank, attn_dcp_size=4
                    ),
                    patch(
                        "sglang.srt.layers.dcp.planner.get_attn_backend",
                        return_value=backend,
                    ),
                    patch(
                        "sglang.srt.layers.dcp.planner.get_device",
                        return_value=SimpleNamespace(device="cpu"),
                    ),
                    patch(
                        "sglang.srt.layers.dcp.planner.create_dcp_kv_indices"
                    ) as packed,
                ):
                    result = prepare_decode_context_parallel_metadata(
                        seq_lens=seq_lens,
                        extend_prefix_lens=prefix_lens,
                        extend_prefix_lens_cpu=prefix_lens,
                        extend_seq_lens=extend_lens,
                        req_pool_indices=req_indices,
                        req_to_token=req_to_token,
                        seq_lens_sum=int(seq_lens.sum()),
                        kv_buffer_shape=torch.Size([32, 1]),
                        kv_cache_dtype=torch.bfloat16,
                        kv_cache_device="cpu",
                        create_chunked_prefix_cache_kv_indices_fn=kernel,
                    )
                    torch.testing.assert_close(
                        result.dcp_local_prefix_kv_indices, all_prefix[rank::4] // 4
                    )
                    kernel.__getitem__.assert_not_called()
                    packed.__getitem__.assert_not_called()
                    self.assertIsNone(result.dcp_kv_indptr)
                    self.assertIsNone(result.dcp_kv_indices)
                    self.assertIsNone(result.dcp_kv_buffer)
                    self.assertIsNone(result.dcp_extend_prefix_lens_sum)

    def test_metadata_planner_keeps_prefix_kernel_for_packed_kv(self):
        translator = KVIndexTranslator.__new__(KVIndexTranslator)
        translator.is_translating = False
        backend = SimpleNamespace(
            dcp_use_packed_kv=True, kv_index_translator=translator
        )
        table = torch.arange(40, 64, dtype=torch.int32).view(3, 8)
        prefixes = torch.tensor([4, 8], dtype=torch.int32)
        extend = torch.tensor([2, 3], dtype=torch.int32)

        def cpu_prefix_kernel(table, reqs, starts, lens, cu_lens, out, stride):
            for i in range(reqs.numel()):
                out[cu_lens[i] : cu_lens[i] + lens[i]] = table[
                    reqs[i], starts[i] : starts[i] + lens[i]
                ]

        kernel = MagicMock()
        kernel.__getitem__.return_value.side_effect = cpu_prefix_kernel
        with (
            rc.get_parallel().override(
                dcp_enabled=True, dcp_size=4, dcp_rank=1, attn_dcp_size=4
            ),
            patch(
                "sglang.srt.layers.dcp.planner.get_attn_backend", return_value=backend
            ),
            patch(
                "sglang.srt.layers.dcp.planner.get_device",
                return_value=SimpleNamespace(device="cpu"),
            ),
            patch("sglang.srt.layers.dcp.planner.create_dcp_kv_indices") as packed,
        ):
            result = prepare_decode_context_parallel_metadata(
                seq_lens=prefixes + extend,
                extend_prefix_lens=prefixes,
                extend_prefix_lens_cpu=[4, 8],
                extend_seq_lens=extend,
                req_pool_indices=torch.tensor([2, 0]),
                req_to_token=table,
                seq_lens_sum=17,
                kv_buffer_shape=torch.Size([32, 1]),
                kv_cache_dtype=torch.bfloat16,
                kv_cache_device="cpu",
                create_chunked_prefix_cache_kv_indices_fn=kernel,
            )
        kernel.__getitem__.assert_called_once_with((2,))
        kernel.__getitem__.return_value.assert_called_once()
        packed.__getitem__.return_value.assert_called_once()
        expected = torch.cat([table[2, :4], table[0, :8]])[1::4] // 4
        torch.testing.assert_close(result.dcp_local_prefix_kv_indices, expected)
        self.assertEqual(result.dcp_kv_buffer.shape, (17, 1))
        self.assertEqual(result.dcp_extend_prefix_lens_sum, 12)

    def test_gqa_current_chunk_selects_kv_for_the_global_dcp_head_layout(self):
        """A local Q shard must not restart GQA mapping at KV head zero."""

        class FakeDcpGroup:
            world_size = 4
            rank_in_group = 1

            def __init__(self):
                self.all_gather_calls = 0

            def all_gather(self, tensor, dim):
                self.all_gather_calls += 1
                return torch.cat((tensor, tensor + 10), dim=dim)

        group = FakeDcpGroup()
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.forward_metadata = SimpleNamespace(
            custom_mask=None,
            kv_indptr=torch.zeros(2, dtype=torch.int32),
            kv_indices=torch.empty(0, dtype=torch.int64),
            max_extend_len=1,
            qo_indptr=torch.tensor([0, 1], dtype=torch.int64),
        )
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _layer_id: torch.empty(0),
            get_value_buffer=lambda _layer_id: torch.empty(0),
        )

        kernel_q_shapes = []
        kernel_k = []

        def fake_extend_attention(q, k, _v, out, *_args, lse_extend, **_kwargs):
            kernel_q_shapes.append(q.shape)
            kernel_k.append(k.clone())
            out.copy_(q.float())
            lse_extend.zero_()

        backend.extend_attention_fwd = fake_extend_attention
        layer = SimpleNamespace(
            sliding_window_size=-1,
            tp_q_head_num=2,
            tp_k_head_num=2,
            qk_head_dim=2,
            v_head_dim=2,
            k_scale=None,
            v_scale=None,
            layer_id=0,
            scaling=1.0,
            xai_temperature_len=-1,
        )
        q = torch.arange(4, dtype=torch.bfloat16).view(1, 4)
        k = torch.tensor([[[0.0, 1.0], [10.0, 11.0]]])

        with rc.get_parallel().override(dcp_group=group):
            out = backend._forward_extend_dcp(
                q=q,
                k=k,
                v=k.clone(),
                layer=layer,
                forward_batch=SimpleNamespace(),
                causal=True,
                logits_soft_cap=0.0,
                sinks=None,
            )

        self.assertEqual(group.all_gather_calls, 0)
        self.assertEqual(kernel_q_shapes, [torch.Size([1, 2, 2])])
        # In TP4/DCP4 with two KV heads, ranks 0 and 1 both belong to KV head 0.
        self.assertTrue(torch.equal(kernel_k[0], k[:, 0:1]))
        self.assertTrue(torch.equal(out, q))

    def test_dense_q_indptr_matches_the_arange_it_replaces(self):
        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

        max_bs = 16
        for num_draft_tokens in (1, 2, 8):
            backend = object.__new__(TRTLLMMLABackend)
            backend.q_indptr_decode = torch.arange(0, max_bs + 1, dtype=torch.int32)
            backend.num_draft_tokens = num_draft_tokens
            backend.dense_q_indptr_verify = backend.q_indptr_decode * num_draft_tokens
            # Equal hits the precomputed buffer, +1 hits the fallback.
            for draft_token_num in (num_draft_tokens, num_draft_tokens + 1):
                for bs in (1, 3, max_bs):
                    with self.subTest(
                        num_draft_tokens=num_draft_tokens,
                        draft_token_num=draft_token_num,
                        bs=bs,
                    ):
                        got = backend._dense_q_indptr(bs, draft_token_num)
                        expected = torch.arange(
                            0,
                            (bs + 1) * draft_token_num,
                            draft_token_num,
                            dtype=torch.int32,
                        )
                        self.assertEqual(got.dtype, torch.int32)
                        self.assertTrue(
                            torch.equal(got, expected),
                            f"{got.tolist()} != {expected.tolist()}",
                        )

    def test_paged_allocator_exposes_dcp_virtual_capacity(self):
        real_kv_size = 1024
        dcp_size = 4
        physical_page_size = 64
        allocator = PagedTokenToKVPoolAllocator(
            size=real_kv_size * dcp_size,
            page_size=physical_page_size * dcp_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=object(),
            need_sort=False,
        )

        allocations = [allocator.alloc(physical_page_size * dcp_size) for _ in range(4)]
        self.assertTrue(all(indices is not None for indices in allocations))
        virtual_indices = torch.cat(allocations)

        self.assertEqual(allocator.size, real_kv_size * dcp_size)
        self.assertEqual(allocator.page_size, physical_page_size * dcp_size)
        self.assertEqual(allocator.num_pages, real_kv_size // physical_page_size)
        self.assertEqual(
            len(torch.unique(virtual_indices // dcp_size)),
            len(virtual_indices) // dcp_size,
        )
        self.assertLess(
            int((virtual_indices // dcp_size).max()),
            real_kv_size + physical_page_size,
        )

    @staticmethod
    def _kv_head_config(*, is_draft_model: bool):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(model_type="qwen3_5_text")
        model_config.hf_text_config = SimpleNamespace(num_key_value_heads=8)
        model_config.is_draft_model = is_draft_model
        return model_config

    def test_model_config_uses_non_dcp_tp_size_for_kv_heads(self):
        model_config = self._kv_head_config(is_draft_model=False)

        self.assertEqual(model_config.get_num_kv_heads(16), 1)
        self.assertEqual(model_config.get_num_kv_heads(16, dcp_size=4), 2)

    def test_a_draft_keeps_kv_heads_tp_sharded_under_dcp(self):
        """The draft pool must match what a TP-sharded draft builds; sizing it
        with the target's dcp_size over-allocates by that factor."""
        model_config = self._kv_head_config(is_draft_model=True)

        self.assertEqual(model_config.get_num_kv_heads(16, dcp_size=4), 1)
        self.assertEqual(model_config.get_num_kv_heads(16), 1)

    def test_draft_pool_capacity_still_covers_dcp_virtual_addresses(self):
        configurator = KVCacheConfigurator.__new__(KVCacheConfigurator)
        override = rc.get_context().override_server_args(page_size=128)
        override.install()
        self.addCleanup(override.restore)
        for dcp_size in (1, 2, 4):
            for is_draft in (False, True):
                configurator.is_draft_worker = is_draft
                with (
                    self.subTest(dcp_size=dcp_size, is_draft=is_draft),
                    rc.get_parallel().override(attn_dcp_size=dcp_size),
                ):
                    expected_scale = dcp_size if is_draft else 1
                    self.assertEqual(configurator.loc_space_scale, expected_scale)
                    self.assertEqual(configurator.pool_page_size, 128 * expected_scale)

    def test_gqa_qkv_loader_replicates_kv_within_dcp_group(self):
        hidden_size = 4
        head_size = 2
        q_weight = torch.arange(64, dtype=torch.float32).view(16, hidden_size)
        k_weight = torch.arange(16, dtype=torch.float32).view(4, hidden_size) + 100
        v_weight = torch.arange(16, dtype=torch.float32).view(4, hidden_size) + 200

        for tp_rank in range(4):
            layer = QKVParallelLinear(
                hidden_size=hidden_size,
                head_size=head_size,
                total_num_heads=8,
                total_num_kv_heads=2,
                bias=False,
                params_dtype=torch.float32,
                tp_rank=tp_rank,
                tp_size=4,
                kv_tp_rank=tp_rank // 2,
                kv_tp_size=2,
            )
            layer.weight_loader(layer.weight, q_weight, "q")
            layer.weight_loader(layer.weight, k_weight, "k")
            layer.weight_loader(layer.weight, v_weight, "v")

            q, k, v = layer.weight.split([4, 2, 2], dim=0)
            kv_start = (tp_rank // 2) * 2
            self.assertTrue(torch.equal(q, q_weight[tp_rank * 4 : (tp_rank + 1) * 4]))
            self.assertTrue(torch.equal(k, k_weight[kv_start : kv_start + 2]))
            self.assertTrue(torch.equal(v, v_weight[kv_start : kv_start + 2]))

    def test_configurator_scales_only_the_virtual_dcp_allocator(self):
        physical_kv_size = 1024
        physical_page_size = 64
        physical_kv_cache = SimpleNamespace(
            size=physical_kv_size,
            page_size=physical_page_size,
        )
        sizes = SimpleNamespace(
            max_total_num_tokens=physical_kv_size,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
        )
        allocators = {}

        # The configurator's own inputs are published leaves now, so the case
        # publishes them once. The DCP *scale* is not one of them: the allocator
        # widens from the live get_parallel().attn_dcp_size, which the per-size
        # override inside the loop drives.
        override = rc.get_context().override_server_args(
            disaggregation_mode="null",
            page_size=physical_page_size,
            enable_hisparse=False,
        )
        override.install()
        self.addCleanup(override.restore)
        for dcp_size in (1, 4):
            configurator = SimpleNamespace(
                server_args=SimpleNamespace(),
                hybrid_gdn_config=None,
                is_hybrid_swa=False,
                kv_cache_dtype=torch.bfloat16,
                device="cpu",
                is_draft_worker=False,
            )
            # The allocator widens from get_parallel(), not from the injected
            # server_args stand-in -- drive the cause, not the effect.
            with (
                patch(
                    "sglang.srt.mem_cache.kv_cache_configurator.current_platform.is_out_of_tree",
                    return_value=False,
                ),
                rc.get_parallel().override(attn_dcp_size=dcp_size),
            ):
                allocators[dcp_size] = (
                    KVCacheConfigurator._build_token_to_kv_pool_allocator(
                        configurator,
                        sizes=sizes,
                        token_to_kv_pool=physical_kv_cache,
                        is_dsv4_model=False,
                        req_to_token_pool=object(),
                        token_to_kv_pool_allocator=None,
                    )
                )

        dcp1_allocator = allocators[1]
        dcp4_allocator = allocators[4]
        self.assertIs(dcp1_allocator.get_kvcache(), physical_kv_cache)
        self.assertIs(dcp4_allocator.get_kvcache(), physical_kv_cache)
        self.assertEqual(dcp1_allocator.size, 1024)
        self.assertEqual(dcp1_allocator.page_size, 64)
        self.assertEqual(dcp1_allocator.num_pages, 16)
        self.assertEqual(dcp4_allocator.size, 4096)
        self.assertEqual(dcp4_allocator.page_size, 256)
        self.assertEqual(dcp4_allocator.num_pages, 16)

    def test_live_cell_and_page_ownership_formulas(self):
        dcp_size = 4
        physical_page_size = 64
        ragged_lengths = (0, 1, 2, 3, 4, 63, 64, 65, 255, 256, 257, 515)

        per_rank_counts = []
        for rank in range(dcp_size):
            expected_counts = [
                length // dcp_size + int(rank < length % dcp_size)
                for length in ragged_lengths
            ]
            actual_counts = [
                _owner_count(length, dcp_size, rank, 0) for length in ragged_lengths
            ]
            self.assertEqual(actual_counts, expected_counts)
            per_rank_counts.append(sum(actual_counts))

            allocated_pages = [
                math.ceil(length / (physical_page_size * dcp_size))
                for length in ragged_lengths
            ]
            active_pages = [
                math.ceil(count / physical_page_size) for count in actual_counts
            ]
            self.assertTrue(
                all(
                    active <= allocated
                    for active, allocated in zip(active_pages, allocated_pages)
                )
            )
            self.assertTrue(
                all(
                    allocated - active <= 1
                    for active, allocated in zip(active_pages, allocated_pages)
                )
            )

        self.assertEqual(sum(per_rank_counts), sum(ragged_lengths))

        aligned_lengths = (256, 512, 768, 1024)
        full_replica_cells = sum(aligned_lengths)
        full_replica_pages = sum(
            length // physical_page_size for length in aligned_lengths
        )
        for rank in range(dcp_size):
            local_cells = sum(
                _owner_count(length, dcp_size, rank, 0) for length in aligned_lengths
            )
            local_pages = sum(
                math.ceil(_owner_count(length, dcp_size, rank, 0) / physical_page_size)
                for length in aligned_lengths
            )
            self.assertEqual(local_cells * dcp_size, full_replica_cells)
            self.assertEqual(local_pages * dcp_size, full_replica_pages)

    def test_hybrid_pool_reports_the_backing_attention_shape(self):
        pool = object.__new__(HybridLinearKVPool)
        pool.start_layer = 0
        pool.layer_transfer_counter = None
        pool.full_attention_layer_id_mapping = {3: 0, 7: 1}
        pool.full_kv_pool = MagicMock()
        expected = (torch.Size([1024, 1, 576]), torch.Size([1024, 1, 576]))
        pool.full_kv_pool.get_kv_buffer_shape.return_value = expected

        self.assertEqual(pool.get_kv_buffer_shape(), expected)
        pool.full_kv_pool.get_kv_buffer_shape.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

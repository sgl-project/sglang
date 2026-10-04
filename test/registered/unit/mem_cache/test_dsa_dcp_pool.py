"""Replicated DSA index keys must cover the allocator's entire virtual space."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dcp.layout import remap_dcp_sparse_indices
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_cache_configurator import (
    KVCacheConfigurator,
    dsa_dcp_head_groups,
    dsa_dcp_max_query_rows,
    dsa_dcp_merge_size_bytes,
    dsa_dcp_runtime_reservation_bytes,
    dsa_dcp_workspace_size_bytes,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_context, get_parallel, get_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@contextmanager
def _configuration(dcp_size, rank=0, **fields):
    with (
        get_context().override_server_args(
            **{
                "page_size": 64,
                "tp_size": 4,
                "dsa_prefill_backend": "trtllm",
                "dsa_decode_backend": "trtllm",
                "enable_dsa_cache_layer_split": False,
                **fields,
            },
        ),
        get_parallel().override(
            attn_dcp_size=dcp_size,
            attn_dcp_rank=rank,
            attn_tp_size=fields.get("tp_size", 4) // fields.get("attn_dp_size", 1),
            attn_dp_size=fields.get("attn_dp_size", 1),
            dcp_enabled=dcp_size > 1,
        ),
    ):
        yield


def _configurator(dtype=torch.bfloat16, *, share_topk=False):
    kvc = object.__new__(KVCacheConfigurator)
    kvc.device = "cuda"
    kvc.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            model_type="glm_moe_dsa",
            index_topk=2048,
            index_head_dim=128,
            indexer_types=["full", "shared" if share_topk else "full"],
        ),
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        num_attention_heads=64,
        context_len=8192,
        is_draft_model=False,
        linear_attn_registry_result=None,
    )
    kvc.layer_info = SimpleNamespace(start_layer=0, end_layer=2, num_effective_layers=2)
    kvc.model_config.hf_config.get_text_config = lambda: kvc.model_config.hf_config
    kvc.is_draft_worker = False
    kvc.server_args = get_server_args()
    kvc.kv_cache_dtype = dtype
    kvc.kv_cache_dtype_str = str(dtype)
    kvc.use_mla_backend = True
    kvc.page_size = 64
    kvc.pp_size = 1
    kvc.attn_dp_size = get_parallel().attn_dp_size
    kvc.mambaish_config = None
    kvc.is_hybrid_swa = False
    kvc.spec_algorithm = SpeculativeAlgorithm.NONE
    return kvc


def _build_cpu_pool(kvc, size):
    def allocate_cpu(*args, **kwargs):
        kwargs["device"] = "cpu"
        return DSATokenToKVPool(*args, **kwargs)

    # Only replace the allocation device: exercise the real pool factory and
    # the real storage shapes that CUDA kernels will see.
    with patch(
        "sglang.srt.mem_cache.kv_cache_configurator.DSATokenToKVPool",
        side_effect=allocate_cpu,
    ):
        return kvc._build_dsa_kv_pool(max_total_num_tokens=size, max_running_requests=8)


class TestDSADCPPool(CustomTestCase):
    def test_dynamic_profile_workspace_covers_first_125_percent_probe(self):
        with (
            _configuration(
                2,
                chunked_prefill_size=13056,
                max_prefill_tokens=13056,
                enable_dynamic_chunking=True,
                pp_size=2,
                max_running_requests=64,
            ),
            get_parallel().override(pp_size=2),
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.override(384 << 20),
        ):
            rows = dsa_dcp_max_query_rows(_configurator().model_config)
            probe_rows = 16320
            workspace = dsa_dcp_workspace_size_bytes(
                num_q_heads=64, dcp_size=2, max_query_rows=rows
            )
            stats = 8 * 128 * probe_rows * 256 + (1 << 20)
            self.assertGreaterEqual(workspace - stats, 768 << 20)
            self.assertEqual(rows, probe_rows)

    def test_attention_dp_reservation_uses_worker_verification_rows(self):
        with (
            _configuration(
                2,
                chunked_prefill_size=16384,
                max_running_requests=8192,
                tp_size=8,
                attn_dp_size=2,
                speculative_algorithm="EAGLE",
                speculative_num_draft_tokens=6,
            ),
            get_parallel().override(tp_size=8, attn_dp_size=2, attn_tp_size=4),
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.override(384 << 20),
        ):
            kvc = _configurator()
            kvc.attn_dp_size = 2
            worker_requests = kvc.resolve_max_num_reqs(1 << 20)
            self.assertEqual(worker_requests, 4096)
            self.assertEqual(
                dsa_dcp_max_query_rows(kvc.model_config), worker_requests * 6
            )
            self.assertEqual(dsa_dcp_runtime_reservation_bytes(kvc), 5910396928)

    def test_reservation_honors_automatic_and_capacity_limited_request_pools(self):
        for requested, token_cap in ((None, 512), (8192, 512), (None, 1 << 20)):
            with (
                self.subTest(requested=requested, token_cap=token_cap),
                _configuration(
                    2,
                    chunked_prefill_size=64,
                    max_running_requests=requested,
                    max_total_tokens=token_cap,
                    attn_dp_size=2,
                    speculative_algorithm="EAGLE",
                    speculative_num_draft_tokens=6,
                ),
            ):
                kvc = _configurator()
                budget = 32 << 30
                capacity = kvc.config_from_budget(budget).max_total_num_tokens
                requests = kvc.resolve_max_num_reqs(capacity)
                self.assertEqual(requests, min(token_cap // 2, 4096))
                bounded = dsa_dcp_runtime_reservation_bytes(kvc, available_bytes=budget)
                final_capacity = kvc.config_from_budget(
                    budget - bounded
                ).max_total_num_tokens
                self.assertGreaterEqual(
                    requests, kvc.resolve_max_num_reqs(final_capacity)
                )
            with _configuration(
                2,
                chunked_prefill_size=64,
                max_running_requests=requests * 2,
                attn_dp_size=2,
                speculative_algorithm="EAGLE",
                speculative_num_draft_tokens=6,
            ):
                self.assertEqual(
                    bounded, dsa_dcp_runtime_reservation_bytes(_configurator())
                )

    def test_attention_dp_row_bound_preserves_alignment(self):
        with _configuration(2, attn_dp_size=2, chunked_prefill_size=1):
            self.assertEqual(
                dsa_dcp_max_query_rows(
                    _configurator().model_config, max_running_requests=3
                ),
                4,
            )

    def test_draft_prefill_skips_dense_context_metadata(self):
        from sglang.srt.models import deepseek_v2
        from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLMNextN

        # NextN initializes nn.Module directly and has no target-only use_dsa
        # attribute. Its inherited metadata hook must classify the config.
        draft = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
        torch.nn.Module.__init__(draft)
        draft.config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLMNextN"],
            index_topk=2048,
            qk_rope_head_dim=64,
        )
        with (
            patch.object(deepseek_v2, "_is_cuda", True),
            patch.object(deepseek_v2, "_is_npu", False),
            patch.object(
                deepseek_v2,
                "prepare_decode_context_parallel_metadata",
                side_effect=AssertionError("RoPE DSA must not gather dense prefix KV"),
            ),
        ):
            self.assertIsNone(
                draft.prepare_context_parallel_metadata_for_dcp(
                    seq_lens=None,
                    extend_prefix_lens=None,
                    extend_prefix_lens_cpu=None,
                    extend_seq_lens=None,
                    req_pool_indices=None,
                    req_to_token=None,
                    seq_lens_sum=0,
                    kv_buffer_shape=None,
                    kv_cache_dtype=None,
                    kv_cache_device=None,
                    create_chunked_prefix_cache_kv_indices_fn=None,
                )
            )

    def test_replicated_draft_write_preserves_global_slots(self):
        for dcp_size in (2, 4):
            for rank in range(dcp_size):
                with (
                    self.subTest(dcp=dcp_size, rank=rank),
                    _configuration(dcp_size, rank),
                ):
                    kvc = _configurator()
                    kvc.is_draft_worker = True
                    pool = _build_cpu_pool(kvc, 256 * dcp_size)
                    slots = torch.tensor(
                        [64 * dcp_size + i for i in range(dcp_size)]
                        + [320 * dcp_size - 1]
                    )
                    nope = (
                        torch.arange(len(slots), dtype=torch.bfloat16)
                        .view(-1, 1, 1)
                        .expand(-1, 1, 512)
                    )
                    rope = nope[..., :64] + 10

                    def scatter(dst, loc, cache_nope, cache_rope):
                        dst[loc] = torch.cat((cache_nope, cache_rope), dim=-1)

                    with (
                        patch(
                            "sglang.srt.mem_cache.memory_pool.set_mla_kv_buffer_triton",
                            side_effect=scatter,
                        ),
                        patch(
                            "sglang.srt.mem_cache.memory_pool.set_mla_kv_buffer_dcp_sharded_triton",
                            side_effect=AssertionError(
                                "replicated draft must not owner-filter"
                            ),
                        ),
                    ):
                        pool.set_mla_kv_buffer(
                            SimpleNamespace(layer_id=0), slots, nope, rope
                        )
                    torch.testing.assert_close(
                        pool.kv_buffer[0][slots], torch.cat((nope, rope), dim=-1)
                    )

    def test_draft_attention_skips_collective_dispatch(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.models.deepseek_common.attention_forward_methods import (
            forward_mla,
        )

        module = SimpleNamespace(use_dsa=True, qk_rope_head_dim=64)
        for dcp_size in (2, 4):
            for is_draft in (False, True):
                backend = SimpleNamespace(
                    use_dsa=True, qk_rope_head_dim=64, dcp_enabled=not is_draft
                )
                with (
                    self.subTest(dcp=dcp_size, draft=is_draft),
                    _configuration(dcp_size),
                    patch.object(forward_mla, "_is_cuda", True),
                    forward_context(ForwardContext(attn_backend=backend)),
                ):
                    self.assertEqual(forward_mla.is_dcp_mla_enabled(), not is_draft)
                    for mode in (
                        ForwardMode.DECODE,
                        ForwardMode.TARGET_VERIFY,
                        ForwardMode.EXTEND,
                        ForwardMode.DRAFT_EXTEND_V2,
                    ):
                        batch = SimpleNamespace(forward_mode=mode)
                        self.assertEqual(
                            forward_mla.is_dcp_mla_decode_phase(batch),
                            not is_draft
                            and mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY),
                        )
                        self.assertEqual(
                            forward_mla.is_dcp_dsa_extend_phase(module, batch),
                            not is_draft and mode.is_extend(),
                        )
        # Preserve the existing dense, NoPE and ROCm/NPU dispatch contracts.
        for cuda, use_dsa, rope in (
            (True, False, 64),
            (True, True, 0),
            (False, True, 64),
        ):
            backend = SimpleNamespace(
                use_dsa=use_dsa, qk_rope_head_dim=rope, dcp_enabled=False
            )
            with (
                self.subTest(cuda=cuda, dsa=use_dsa, rope=rope),
                _configuration(2),
                patch.object(forward_mla, "_is_cuda", cuda),
                forward_context(ForwardContext(attn_backend=backend)),
            ):
                self.assertTrue(forward_mla.is_dcp_mla_enabled())

    def test_replicated_draft_covers_last_allocator_page(self):
        for dcp_size in (2, 4):
            for rank in range(dcp_size):
                with (
                    self.subTest(dcp=dcp_size, rank=rank),
                    _configuration(dcp_size, rank),
                ):
                    kvc = _configurator()
                    kvc.is_draft_worker = True
                    pool = _build_cpu_pool(kvc, 256 * dcp_size)
                    allocator = PagedTokenToKVPoolAllocator(
                        256 * dcp_size,
                        page_size=64 * dcp_size,
                        dtype=torch.bfloat16,
                        device="cpu",
                        kvcache=pool,
                        need_sort=False,
                    )
                    slots = allocator.alloc(allocator.available_size())
                    self.assertEqual(pool.page_size, 64)
                    self.assertEqual(pool._write_loc_dcp_span, 1)
                    self.assertEqual(pool.kv_buffer[0].shape[0], 320 * dcp_size)
                    self.assertEqual(pool.index_buf_size, 256 * dcp_size)
                    index = pool.get_index_k_with_scale_buffer(0)
                    self.assertEqual(index.shape, (5 * dcp_size, 64 * 132))
                    pool.kv_buffer[0][slots[-1]].fill_(37)
                    self.assertEqual(int(pool.kv_buffer[0][-1, 0, 0]), 37)

    def test_budget_includes_target_and_replicated_draft(self):
        budget = 4 << 20
        for dcp_size in (2, 4):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                for share_topk in (False, True):
                    with (
                        self.subTest(dcp=dcp_size, dtype=dtype, shared=share_topk),
                        _configuration(dcp_size),
                    ):
                        target = _configurator(dtype, share_topk=share_topk)
                        target.spec_algorithm = SpeculativeAlgorithm.EAGLE
                        target.spec_aux_config = SimpleNamespace(
                            eagle_draft_num_layers=1
                        )
                        sizing = DefaultPoolConfigurator(target)
                        size = sizing.calculate_pool_sizes(
                            budget, 64
                        ).max_total_num_tokens
                        draft = _configurator(dtype)
                        draft.is_draft_worker = True
                        draft.layer_info = SimpleNamespace(
                            start_layer=0, end_layer=1, num_effective_layers=1
                        )
                        allocated = sum(
                            pool.get_kv_size_bytes()
                            for pool in (
                                _build_cpu_pool(target, size),
                                _build_cpu_pool(draft, size * dcp_size),
                            )
                        )
                        next_page = sum(
                            pool.get_kv_size_bytes()
                            for pool in (
                                _build_cpu_pool(target, size + 64),
                                _build_cpu_pool(draft, (size + 64) * dcp_size),
                            )
                        )
                        self.assertLessEqual(allocated, budget)
                        self.assertGreater(next_page, budget)

    def test_replicated_draft_retraction_keeps_every_rank_slot(self):
        for dcp_size in (2, 4):
            for rank in range(dcp_size):
                with (
                    self.subTest(dcp=dcp_size, rank=rank),
                    _configuration(dcp_size, rank),
                    patch("torch.cuda.synchronize"),
                ):
                    kvc = _configurator()
                    kvc.is_draft_worker = True
                    pool = _build_cpu_pool(kvc, 256 * dcp_size)
                    page = 64 * dcp_size
                    old, new = (
                        torch.arange(page, 2 * page),
                        torch.arange(4 * page, 5 * page),
                    )
                    for layer, buffer in enumerate(pool.kv_buffer):
                        buffer[old] = (
                            torch.arange(page, dtype=buffer.dtype).view(page, 1, 1)
                            + layer * 100
                        )
                    for layer, index in enumerate(pool.index_k_with_scale_buffer):
                        for subpage in range(dcp_size):
                            index[dcp_size + subpage].fill_(layer * 10 + subpage + 1)
                    expected = [buffer[old].clone() for buffer in pool.kv_buffer]
                    expected_index = [
                        index[dcp_size : 2 * dcp_size].clone()
                        for index in pool.index_k_with_scale_buffer
                    ]
                    backup = pool.get_cpu_copy(old)
                    for buffer in pool.kv_buffer + pool.index_k_with_scale_buffer:
                        buffer.zero_()
                    pool.load_cpu_copy(backup, new)
                    for actual, reference in zip(pool.kv_buffer, expected):
                        torch.testing.assert_close(actual[new], reference)
                    for actual, reference in zip(
                        pool.index_k_with_scale_buffer, expected_index
                    ):
                        torch.testing.assert_close(
                            actual[4 * dcp_size : 5 * dcp_size], reference
                        )

    def test_query_row_bound_covers_target_verification(self):
        with _configuration(
            2,
            chunked_prefill_size=8192,
            speculative_algorithm="EAGLE",
            speculative_num_draft_tokens=6,
        ):
            self.assertEqual(
                dsa_dcp_max_query_rows(_configurator().model_config), 4096 * 6
            )
            self.assertEqual(
                dsa_dcp_max_query_rows(
                    _configurator().model_config, max_running_requests=2048
                ),
                2048 * 6,
            )

    def test_last_virtual_page_is_addressable(self):
        """#36886: allocator locs beyond per-rank capacity must fit index-K."""
        for dcp_size in (1, 2, 4):
            with self.subTest(dcp_size=dcp_size), _configuration(dcp_size):
                pool = _build_cpu_pool(_configurator(), 256)
                allocator = PagedTokenToKVPoolAllocator(
                    256 * dcp_size,
                    page_size=64 * dcp_size,
                    dtype=torch.bfloat16,
                    device="cpu",
                    kvcache=pool,
                    need_sort=False,
                )
                virtual = allocator.alloc(allocator.available_size())
                self.assertEqual(pool.index_buf_size, allocator.size)
                self.assertEqual(pool.kv_buffer[0].shape[0], 320)
                index = pool.get_index_k_with_scale_buffer(0)
                self.assertEqual(index.shape[1], 64 * 132)
                self.assertGreater(index.shape[0] * 64, int(virtual.max()))
                # Touch the final key and FP32 scale bytes, including the
                # widened padding page missing from a size*dcp-only fix.
                last = int(virtual.max())
                page, offset = divmod(last, 64)
                index[page, offset * 128 : (offset + 1) * 128] = 17
                index[page, 64 * 128 + offset * 4 : 64 * 128 + (offset + 1) * 4] = 23
                self.assertEqual(int(index[page, -1]), 23)

    def test_budget_matches_allocated_kv_and_replicated_index_bytes(self):
        """Accounting must leave the padding page and every index replica in budget."""
        budget = 2 << 20
        for dcp_size in (2, 4):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                for share_topk in (False, True):
                    with (
                        self.subTest(dcp=dcp_size, dtype=dtype, share_topk=share_topk),
                        _configuration(dcp_size),
                    ):
                        kvc = _configurator(dtype, share_topk=share_topk)
                        sizing = DefaultPoolConfigurator(kvc)
                        size = sizing.calculate_pool_sizes(
                            budget, 64
                        ).max_total_num_tokens
                        pool = _build_cpu_pool(kvc, size)
                        self.assertLessEqual(pool.get_kv_size_bytes(), budget)
                        larger = _build_cpu_pool(kvc, size + 64)
                        self.assertGreater(larger.get_kv_size_bytes(), budget)

    def test_retraction_restores_virtual_index_keys_and_local_kv(self):
        """A resumed request may receive new pages beyond physical KV capacity."""
        for dcp_size in (2, 4):
            for rank in range(dcp_size):
                with (
                    self.subTest(dcp=dcp_size, rank=rank),
                    _configuration(dcp_size, rank),
                    patch("torch.cuda.synchronize"),
                ):
                    pool = _build_cpu_pool(_configurator(share_topk=True), 256)
                    page = 64 * dcp_size
                    old = torch.arange(page, 2 * page)
                    new = torch.arange(4 * page, 5 * page)
                    source_rows = old[rank::dcp_size] // dcp_size
                    target_rows = new[rank::dcp_size] // dcp_size
                    for layer, buffer in enumerate(pool.kv_buffer):
                        buffer[source_rows] = (
                            torch.arange(64, dtype=buffer.dtype).view(64, 1, 1)
                            + layer * 100
                        )
                    index = pool.index_k_with_scale_buffer[0]
                    for subpage in range(dcp_size):
                        index[dcp_size + subpage].fill_(subpage + 1)
                    expected_kv = [
                        buffer[source_rows].clone() for buffer in pool.kv_buffer
                    ]
                    expected_index = index[dcp_size : 2 * dcp_size].clone()
                    backup = pool.get_cpu_copy(old)
                    for buffer in pool.kv_buffer:
                        buffer.zero_()
                    index.zero_()
                    pool.load_cpu_copy(backup, new)
                    for buffer, expected in zip(pool.kv_buffer, expected_kv):
                        torch.testing.assert_close(buffer[target_rows], expected)
                    torch.testing.assert_close(
                        index[4 * dcp_size : 5 * dcp_size], expected_index
                    )
                    self.assertEqual(pool.index_k_with_scale_buffer[1].numel(), 0)

    def test_runtime_reservation_grows_with_topology_and_chunk_size(self):
        with envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.override(384 << 20):
            reservations = []
            for dcp_size in (1, 2, 4):
                with _configuration(dcp_size, chunked_prefill_size=8192):
                    reservations.append(
                        dsa_dcp_runtime_reservation_bytes(_configurator())
                    )
            self.assertEqual(reservations[0], 0)
            self.assertGreater(reservations[1], 384 << 20)
            self.assertGreater(reservations[2], reservations[1])
            with _configuration(2, chunked_prefill_size=16384):
                kvc = _configurator()
                self.assertGreater(
                    dsa_dcp_runtime_reservation_bytes(kvc), reservations[1]
                )
                kvc.model_config.qk_rope_head_dim = 0
                self.assertEqual(dsa_dcp_runtime_reservation_bytes(kvc), 0)

    def test_runtime_reservation_covers_folded_sparse_metadata(self):
        """A larger sparse table must reduce the KV budget by its actual bytes."""
        rows = 8192
        for dcp_size, num_heads, expected_groups in (
            (2, 64, 1),
            (4, 64, 2),
            (4, 128, 4),
        ):
            with (
                self.subTest(dcp=dcp_size, heads=num_heads),
                _configuration(dcp_size, chunked_prefill_size=rows),
            ):
                kvc = _configurator()
                kvc.model_config.num_attention_heads = num_heads
                groups = dsa_dcp_head_groups(num_heads // 4 * dcp_size)
                self.assertEqual(groups, expected_groups)
                reservations = []
                storage_bytes = []
                for topk in (37, 101):
                    kvc.model_config.hf_config.index_topk = topk
                    reservations.append(dsa_dcp_runtime_reservation_bytes(kvc))
                    source = torch.arange(
                        ((topk + 3) // 4) * 4, dtype=torch.int32
                    ).expand(rows, -1)
                    table, counts = remap_dcp_sparse_indices(
                        source, dcp_size, 0, return_counts=True, repeat_rows=groups
                    )
                    storage_bytes.append(
                        table.numel() * table.element_size()
                        + counts.numel() * counts.element_size()
                    )
                self.assertEqual(
                    reservations[1] - reservations[0],
                    storage_bytes[1] - storage_bytes[0],
                )
                # Remove the known query/output, workspace and counter growth:
                # the remaining reservation must cover every remap output byte.
                heads = num_heads // 4
                runtime_growth = (
                    rows * heads * (dcp_size - 1) * (576 * 2 + 512 * 2 + 4)
                    + rows * heads * dcp_size * 4
                    + dsa_dcp_merge_size_bytes(
                        num_q_heads=heads,
                        dcp_size=dcp_size,
                        max_query_rows=rows,
                        kv_lora_rank=512,
                    )
                )
                workspace_growth = (
                    dsa_dcp_workspace_size_bytes(
                        num_q_heads=heads, dcp_size=dcp_size, max_query_rows=rows
                    )
                    - envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
                )
                self.assertEqual(
                    reservations[0] - runtime_growth - workspace_growth,
                    storage_bytes[0],
                )

    def test_runtime_reservation_covers_live_merge_buffers(self):
        """Merge intermediates remain live alongside the partial output and LSE."""
        rows = 16384
        local_heads = 16
        for dcp_size in (2, 4):
            with (
                self.subTest(dcp_size=dcp_size),
                _configuration(dcp_size, chunked_prefill_size=rows),
            ):
                kvc = _configurator()
                heads = local_heads * dcp_size
                partial = torch.empty(
                    (rows, heads, 512), dtype=torch.bfloat16, device="meta"
                )
                packed = torch.empty(
                    (dcp_size, rows, local_heads, 514),
                    dtype=torch.bfloat16,
                    device="meta",
                )
                result = torch.empty(
                    (rows, local_heads, 512), dtype=torch.bfloat16, device="meta"
                )
                lse = torch.empty((rows, heads), dtype=torch.float32, device="meta")
                corrected = torch.empty_like(partial, dtype=torch.float32)
                reduced = torch.empty_like(result, dtype=torch.float32)
                gathered_lse = torch.empty(
                    (dcp_size, rows, heads), dtype=torch.float32, device="meta"
                )
                workspace = (
                    dsa_dcp_workspace_size_bytes(
                        num_q_heads=local_heads, dcp_size=dcp_size, max_query_rows=rows
                    )
                    - envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
                )
                query_growth = rows * local_heads * (dcp_size - 1) * 576 * 2
                counters = rows * local_heads * (dcp_size - 1) * 4
                groups = dsa_dcp_head_groups(heads)
                metadata = rows * groups * (2048 + 1) * 4
                budget = (
                    dsa_dcp_runtime_reservation_bytes(kvc)
                    - workspace
                    - query_growth
                    - counters
                    - metadata
                )
                for backend, live in (
                    ("a2a", (partial, packed, packed, result, lse)),
                    (
                        "ag_rs",
                        (partial, corrected, reduced, result, lse, gathered_lse, lse),
                    ),
                ):
                    with self.subTest(backend=backend):
                        live_merge = sum(t.numel() * t.element_size() for t in live)
                        # Non-DCP has a local output, but no partial/result LSE.
                        additional_merge = (
                            live_merge - result.numel() * result.element_size()
                        )
                        self.assertGreaterEqual(budget, additional_merge)

    def test_head_groups_preserve_every_head_for_non_power_of_two_counts(self):
        for heads, expected in (
            (16, 1),
            (32, 1),
            (33, 3),
            (40, 2),
            (64, 2),
            (96, 3),
            (128, 4),
        ):
            with self.subTest(heads=heads):
                groups = dsa_dcp_head_groups(heads)
                self.assertEqual(groups, expected)
                self.assertEqual(heads % groups, 0)
                self.assertLessEqual(heads // groups, 32)

    def test_runtime_reservation_covers_chunk_output_concatenation(self):
        """Crossing vendor grid Z must reserve a second live output tensor."""
        for num_heads, last_single_call in ((64, 32767), (128, 16383)):
            reservations = []
            for rows in range(last_single_call, last_single_call + 3):
                with _configuration(4, chunked_prefill_size=rows):
                    kvc = _configurator()
                    kvc.model_config.num_attention_heads = num_heads
                    reservations.append(dsa_dcp_runtime_reservation_bytes(kvc))
            first_jump = reservations[1] - reservations[0]
            later_growth = reservations[2] - reservations[1]
            output = torch.empty(
                (last_single_call, num_heads, 512), dtype=torch.bfloat16, device="meta"
            )
            self.assertEqual(
                first_jump - later_growth, output.numel() * output.element_size()
            )

    def test_workspace_preserves_scratch_after_vendor_lse_allocation(self):
        """16K prefill needs a 256-slot LSE slab per query row, not per request."""
        with envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.override(384 << 20):
            for dcp_size, expected in (
                (1, 402653184),
                (2, 1880096768),
                (4, 3759144960),
            ):
                with self.subTest(dcp=dcp_size):
                    actual = dsa_dcp_workspace_size_bytes(
                        num_q_heads=16, dcp_size=dcp_size, max_query_rows=16384
                    )
                    self.assertEqual(actual, expected)
            # A partial 16K prefill must leave the full kernel scratch budget
            # after the vendor's 256-slot-per-row float2 LSE allocation.
            stats_bytes = 15702 * 32 * 256 * 8 + (1 << 20)
            self.assertGreaterEqual(
                dsa_dcp_workspace_size_bytes(
                    num_q_heads=16, dcp_size=2, max_query_rows=16384
                )
                - stats_bytes,
                768 << 20,
            )

    def test_workspace_is_reused_across_bounded_prefill_calls(self):
        for local_heads, dcp_size in ((16, 2), (16, 4), (32, 4)):
            limit = 65535 // dsa_dcp_head_groups(local_heads * dcp_size)
            with self.subTest(local_heads=local_heads, dcp=dcp_size):
                at_limit = dsa_dcp_workspace_size_bytes(
                    num_q_heads=local_heads, dcp_size=dcp_size, max_query_rows=limit
                )
                for rows in (65536, 131072):
                    self.assertEqual(
                        dsa_dcp_workspace_size_bytes(
                            num_q_heads=local_heads,
                            dcp_size=dcp_size,
                            max_query_rows=rows,
                        ),
                        at_limit,
                    )

    def test_query_row_bound_covers_unchunked_and_dynamic_prefill(self):
        for chunk, dynamic, expected in (
            (8192, False, 8192),
            (8192, True, 16384),
            (-1, False, 1048576),
        ):
            with (
                self.subTest(chunk=chunk, dynamic=dynamic),
                _configuration(
                    2,
                    chunked_prefill_size=chunk,
                    enable_dynamic_chunking=dynamic,
                    pp_size=2,
                ),
                get_parallel().override(pp_size=2),
            ):
                model_config = _configurator().model_config
                model_config.context_len = 1048576
                self.assertEqual(
                    dsa_dcp_max_query_rows(model_config, max_running_requests=64),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()

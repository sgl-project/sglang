"""Unit tests for MiniMax-M3's direct-NHD SAIL MSA path."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.test_utils import CustomTestCase

try:
    from sglang.srt.layers.attention.minimax_sparse_ops.msa_ppu import (
        _allocate_eager_plan_workspace,
        _decode_layer,
        _extend_build,
        _extend_reuse_key,
        _force_init_local_scores_into,
        _is_nhd_layout_unambiguous,
        _LockedJITEntry,
        _pack_len_meta_into,
        _paged_nhd_view,
        _PpuMsaDecodeState,
        _prefill_chunk_token_cap,
        compute_msa_ppu_gate,
        msa_ppu_forward_extend,
        ppu_msa_available,
    )

    _HAS_MODULE = True
    _IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    _HAS_MODULE = False
    _IMPORT_ERROR = e

HAS_GPU = torch.cuda.is_available()

try:
    from sglang.srt.utils.common import get_device_sm
except Exception:  # pragma: no cover
    get_device_sm = None
PAGE = 128
HEAD_DIM = 128


class TestModuleImport(CustomTestCase):
    def test_module_importable(self):
        self.assertTrue(_HAS_MODULE, f"msa_ppu import failed: {_IMPORT_ERROR}")

    @unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
    def test_ppu_msa_available_returns_bool(self):
        self.assertIsInstance(ppu_msa_available(), bool)

    @unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
    def test_ppu_msa_available_is_cached(self):
        self.assertIs(ppu_msa_available(), ppu_msa_available())


@unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
class TestInputValidation(CustomTestCase):
    def test_prefill_chunk_cap_requires_positive_dimensions(self):
        for args in ((0, 128, 4), (4, 0, 4), (4, 128, 0)):
            with (
                self.subTest(args=args),
                self.assertRaisesRegex(ValueError, "must all be positive"),
            ):
                _prefill_chunk_token_cap(*args)

    def test_extend_build_rejects_an_empty_batch_before_importing_fmha(self):
        with self.assertRaisesRegex(ValueError, "non-empty batch"):
            _extend_build(
                SimpleNamespace(),
                torch.empty(1),
                None,
                None,
                torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.int32),
                [],
            )


@unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
class TestLockedJITEntry(CustomTestCase):
    def test_kwargs_order_does_not_duplicate_memo_entries(self):
        result = object()
        fn = mock.Mock(return_value=result)
        with tempfile.TemporaryDirectory() as cache_dir:
            lock_fd = os.open(str(Path(cache_dir) / "lock"), os.O_CREAT | os.O_RDWR)
            try:
                entry = _LockedJITEntry("test", fn, lock_fd, Path(cache_dir), "entry")
                self.assertIs(entry(first=1, second=2), result)
                self.assertIs(entry(second=2, first=1), result)
            finally:
                os.close(lock_fd)
        fn.assert_called_once()

    def test_unhashable_arguments_skip_memoization(self):
        fn = mock.Mock(return_value=object())
        with tempfile.TemporaryDirectory() as cache_dir:
            lock_fd = os.open(str(Path(cache_dir) / "lock"), os.O_CREAT | os.O_RDWR)
            try:
                entry = _LockedJITEntry("test", fn, lock_fd, Path(cache_dir), "entry")
                entry([1])
                entry([1])
            finally:
                os.close(lock_fd)
        self.assertEqual(fn.call_count, 2)


@unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
class TestPpuMsaGate(CustomTestCase):
    def test_fp8_main_kv_accepts_only_e4m3(self):
        if get_device_sm is None:
            self.skipTest("Cannot import get_device_sm")
        sm = get_device_sm()
        if sm < 80:
            self.skipTest(f"MSA PPU gate requires SM>=80, got SM{sm}")
        is_sm89_plus = sm >= 89
        expected_index_dtype = torch.float8_e4m3fn if is_sm89_plus else torch.bfloat16
        expected_score_dtype = torch.bfloat16 if is_sm89_plus else torch.float32

        main_pool = SimpleNamespace(
            head_num=4,
            head_dim=128,
            dtype=torch.float8_e4m3fn,
        )
        backend = SimpleNamespace(
            kv_pool=SimpleNamespace(
                main_pool=main_pool,
                index_kv_pool=None,
                index_k_pool=SimpleNamespace(dtype=expected_index_dtype),
            ),
            block_size_k=PAGE,
            page_size=PAGE,
            topk_blocks=16,
            idx_head_dim=HEAD_DIM,
            sparse_layer_ids={1},
            disable_value_layer_ids={1},
            score_type="max",
            max_context_len=4096,
            init_blocks=1,
            local_blocks=1,
        )
        runner = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=object(), num_attention_heads=8),
            dtype=torch.bfloat16,
            max_running_requests=16,
            server_args=SimpleNamespace(speculative_algorithm=None),
        )
        with (
            mock.patch(
                "sglang.srt.configs.model_config.get_minimax_sparse_attention_config",
                return_value={"sparse_num_index_heads": 4},
            ),
            mock.patch("sglang.srt.utils.common.is_ppu", return_value=True),
            mock.patch(
                "sglang.srt.layers.attention.minimax_sparse_ops.msa_ppu._sail_msa_bool_with_arch_default",
                side_effect=lambda _name, _sm_default: is_sm89_plus,
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1),
            ),
            mock.patch(
                "sglang.srt.layers.attention.minimax_sparse_ops.msa_ppu.ppu_msa_available",
                return_value=True,
            ),
            mock.patch(
                "sglang.srt.layers.attention.minimax_sparse_ops.msa_ppu.envs.SGLANG_SAIL_MINIMAX_M3_MSA.get",
                return_value=True,
            ),
            mock.patch(
                "sglang.srt.layers.attention.minimax_sparse_ops.msa_ppu.envs.SGLANG_SAIL_MINIMAX_M3_MSA_ATTEND.get",
                return_value=True,
            ),
        ):
            ok, use_attend, reasons = compute_msa_ppu_gate(backend, runner)

            self.assertTrue(ok, reasons)
            self.assertTrue(use_attend)
            self.assertEqual(backend._ppu_msa_use_fp8_kvcache, is_sm89_plus)
            self.assertEqual(backend._ppu_msa_indexer_fp8, is_sm89_plus)
            self.assertEqual(backend._ppu_msa_score_dtype, expected_score_dtype)

            main_pool.dtype = torch.float8_e5m2
            ok, _, reasons = compute_msa_ppu_gate(backend, runner)

        self.assertFalse(ok)
        self.assertIn("main KV dtype must be bf16 or fp8_e4m3", reasons)


@unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
class TestPagedNhdView(CustomTestCase):
    def test_view_mapping_is_zero_copy(self):
        pages, heads, dim = 3, 4, 64
        cache = torch.arange(pages * PAGE * heads * dim, dtype=torch.float32).reshape(
            pages * PAGE, heads, dim
        )

        view = _paged_nhd_view(cache)

        self.assertEqual(view.shape, (pages, PAGE, heads, dim))
        self.assertEqual(view.data_ptr(), cache.data_ptr())
        self.assertEqual(view.stride(), (PAGE * heads * dim, heads * dim, dim, 1))
        for page in range(pages):
            self.assertTrue(
                torch.equal(view[page], cache[page * PAGE : (page + 1) * PAGE])
            )

    def test_view_requires_page_multiple(self):
        with self.assertRaises(AssertionError):
            _paged_nhd_view(torch.empty(PAGE + 1, 1, 8))

    def test_hkv_128_is_rejected_as_layout_ambiguous(self):
        self.assertTrue(_is_nhd_layout_unambiguous(1))
        self.assertTrue(_is_nhd_layout_unambiguous(64))
        self.assertFalse(_is_nhd_layout_unambiguous(PAGE))


@unittest.skipUnless(
    _HAS_MODULE and HAS_GPU and ppu_msa_available(),
    "msa_ppu module or fmha_sm100/CUDA unavailable",
)
class TestPlanWorkspaceIsolation(CustomTestCase):
    def test_dedicated_workspaces_preserve_plan_metadata(self):
        from fmha_sm100 import allocate_graph_workspace, fmha_sm100_plan

        indexer_workspace = allocate_graph_workspace(
            1,
            4,
            num_kv_heads=1,
            max_qo_len=186,
            page_size=PAGE,
            num_kv_splits=-1,
            device="cuda",
            output_maxscore=True,
        )
        indexer_plan = fmha_sm100_plan(
            torch.tensor([186], dtype=torch.int32),
            torch.tensor([186], dtype=torch.int32),
            4,
            num_kv_heads=1,
            page_size=PAGE,
            num_kv_splits=-1,
            output_maxscore=True,
            causal=True,
            given_workspace=indexer_workspace,
        )
        before = indexer_plan[3]["qo_segment_lens"].cpu().tolist()

        attend_workspace = allocate_graph_workspace(
            1,
            4,
            num_kv_heads=4,
            max_qo_len=16,
            page_size=PAGE,
            kv_block_num=16,
            num_kv_splits=-1,
            device="cuda",
        )
        fmha_sm100_plan(
            torch.tensor([16], dtype=torch.int32),
            torch.tensor([16], dtype=torch.int32),
            4,
            num_kv_heads=4,
            page_size=PAGE,
            kv_block_num=16,
            num_kv_splits=-1,
            causal=True,
            given_workspace=attend_workspace,
        )

        self.assertEqual(before, [186])
        self.assertEqual(indexer_plan[3]["qo_segment_lens"].cpu().tolist(), [186])

    def test_exact_workspace_handles_skewed_prefill_batch(self):
        from fmha_sm100 import fmha_sm100_plan

        qo_lens = torch.tensor([1024] + [1] * 63, dtype=torch.int32)
        kv_lens = qo_lens + 4096
        indexer_workspace = _allocate_eager_plan_workspace(
            qo_lens,
            4,
            num_kv_heads=1,
            page_size=PAGE,
            kv_block_num=-1,
            num_kv_splits=-1,
            output_maxscore=True,
            use_fp8_kvcache=False,
            device=torch.device("cuda"),
        )
        indexer_plan = fmha_sm100_plan(
            qo_lens,
            kv_lens,
            4,
            num_kv_heads=1,
            page_size=PAGE,
            num_kv_splits=-1,
            output_maxscore=True,
            causal=True,
            given_workspace=indexer_workspace,
        )
        self.assertEqual(
            indexer_plan[3]["qo_segment_lens"].cpu().tolist(), qo_lens.tolist()
        )

        attend_workspace = _allocate_eager_plan_workspace(
            qo_lens,
            8,
            num_kv_heads=8,
            page_size=PAGE,
            kv_block_num=16,
            num_kv_splits=-1,
            output_maxscore=False,
            use_fp8_kvcache=False,
            device=torch.device("cuda"),
        )
        attend_plan = fmha_sm100_plan(
            qo_lens,
            kv_lens,
            8,
            num_kv_heads=8,
            page_size=PAGE,
            kv_block_num=16,
            num_kv_splits=-1,
            causal=True,
            given_workspace=attend_workspace,
        )
        self.assertEqual(
            attend_plan[3]["qo_segment_lens"].sum().item(), qo_lens.sum().item()
        )


@unittest.skipUnless(
    _HAS_MODULE and HAS_GPU, "msa_ppu module or CUDA/PPU device unavailable"
)
class TestGpuMetadataKernels(CustomTestCase):
    def test_pack_len_meta_matches_reference(self):
        page_counts = [2, 3]
        seq_lens = torch.tensor(
            [PAGE + 1, 2 * PAGE + 1], dtype=torch.int32, device="cuda"
        )
        req_to_token = torch.zeros(2, 3 * PAGE, dtype=torch.int64, device="cuda")
        physical_pages = [[4, 1], [2, 5, 3]]
        for request, pages in enumerate(physical_pages):
            for logical_page, physical_page in enumerate(pages):
                tokens = torch.arange(
                    physical_page * PAGE,
                    (physical_page + 1) * PAGE,
                    dtype=torch.int64,
                    device="cuda",
                )
                req_to_token[
                    request, logical_page * PAGE : (logical_page + 1) * PAGE
                ] = tokens

        out = torch.full((sum(page_counts),), -1, dtype=torch.int32, device="cuda")
        kv_lens = torch.empty(2, dtype=torch.int32, device="cuda")
        tok_pages = torch.empty(2, dtype=torch.int32, device="cuda")
        page_starts = torch.empty(2, dtype=torch.int32, device="cuda")
        _pack_len_meta_into(
            out,
            req_to_token,
            torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
            seq_lens,
            kv_lens,
            tok_pages,
            page_starts,
            len(page_counts),
            max(page_counts),
            8,
        )
        torch.cuda.synchronize()

        self.assertEqual(out.cpu().tolist(), [4, 1, 2, 5, 3])
        self.assertEqual(kv_lens.cpu().tolist(), [PAGE + 1, 2 * PAGE + 1])
        self.assertEqual(tok_pages.cpu().tolist(), page_counts)
        self.assertEqual(page_starts.cpu().tolist(), [0, 2])

        # Page ids past the end of the pool clamp to the last physical page.
        out.fill_(-1)
        _pack_len_meta_into(
            out,
            req_to_token,
            torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
            seq_lens,
            kv_lens,
            tok_pages,
            page_starts,
            len(page_counts),
            max(page_counts),
            4,
        )
        torch.cuda.synchronize()
        self.assertEqual(out.cpu().tolist(), [3, 1, 2, 3, 3])

    def test_force_init_local_scores_matches_reference(self):
        heads, k_tiles, tokens = 4, 160, 7
        max_score = torch.full(
            (heads, k_tiles, tokens), -float("inf"), dtype=torch.float32, device="cuda"
        )
        tok_pages = torch.tensor(
            [1, 5, 40, 3, 100, 64, 160], dtype=torch.int32, device="cuda"
        )

        _force_init_local_scores_into(
            max_score, tok_pages, init_blocks=2, local_blocks=3
        )
        torch.cuda.synchronize()

        expected = torch.full(
            (heads, k_tiles, tokens), -float("inf"), dtype=torch.float32
        )
        for token, pages in enumerate(tok_pages.cpu().tolist()):
            expected[:, : min(2, pages), token] = 1.0e30
            expected[:, max(pages - 3, 0) : pages, token] = 1.0e29
        self.assertTrue(torch.equal(max_score.cpu(), expected))


@unittest.skipUnless(
    _HAS_MODULE and HAS_GPU and ppu_msa_available(),
    "msa_ppu module, fmha_sm100, or CUDA/PPU device unavailable",
)
class TestDecodeFp8AttendStaging(CustomTestCase):
    def test_direct_attend_stages_bf16_q_to_fixed_fp8_buffer(self):
        import fmha_sm100

        device = torch.device("cuda")
        bs, max_bs, pages, h_idx, h_q, h_kv = 2, 4, 2, 4, 8, 4
        backend = SimpleNamespace(
            _ppu_msa_max_bs=max_bs,
            max_context_len=pages * PAGE,
            kv_pool=SimpleNamespace(main_pool=SimpleNamespace(size=pages * PAGE)),
            _ppu_msa_num_idx_heads=h_idx,
            _ppu_msa_num_q_heads=h_q,
            topk_blocks=16,
            init_blocks=1,
            local_blocks=1,
            _ppu_msa_score_dtype=torch.float32,
            _ppu_msa_indexer_fp8=False,
            _ppu_msa_attend=True,
            _ppu_msa_use_fp8_kvcache=True,
            idx_head_dim=HEAD_DIM,
            _ppu_msa_idx_scale=HEAD_DIM**-0.5,
            _ppu_msa_scale=HEAD_DIM**-0.5,
        )
        state = _PpuMsaDecodeState(backend, device, torch.bfloat16)
        state.indexer_plan = object()
        state.attend_plan = object()
        state.max_score_active = torch.full(
            (h_idx, state.k_tiles, bs),
            -float("inf"),
            dtype=backend._ppu_msa_score_dtype,
            device=device,
        )
        q = torch.randn(bs, h_q, HEAD_DIM, dtype=torch.bfloat16, device=device)
        idx_q = torch.randn(bs, h_idx, HEAD_DIM, dtype=torch.bfloat16, device=device)
        idx_k = torch.randn(
            pages * PAGE, 1, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        k_cache = torch.randn(
            pages * PAGE, h_kv, HEAD_DIM, dtype=torch.bfloat16, device=device
        ).to(torch.float8_e4m3fn)
        v_cache = torch.randn(
            pages * PAGE, h_kv, HEAD_DIM, dtype=torch.bfloat16, device=device
        ).to(torch.float8_e4m3fn)
        run_mock = mock.MagicMock(return_value=(None, None))
        with (
            mock.patch.object(fmha_sm100, "fmha_sm100", run_mock),
            mock.patch.object(fmha_sm100, "sparse_topk_select", mock.MagicMock()),
            mock.patch(
                "sglang.srt.layers.attention.minimax_sparse_ops.msa_ppu"
                "._force_init_local_scores_into"
            ),
        ):
            out = _decode_layer(backend, state, bs, q, idx_q, idx_k, k_cache, v_cache)

        self.assertEqual(run_mock.call_count, 2)
        self.assertIs(run_mock.call_args_list[0].args[0], idx_q)
        attend_call = run_mock.call_args_list[1]
        attend_q = attend_call.args[0]
        self.assertEqual(attend_q.dtype, torch.float8_e4m3fn)
        self.assertEqual(attend_q.data_ptr(), state.attend_q_fp8.data_ptr())
        self.assertTrue(torch.equal(attend_q, q.to(torch.float8_e4m3fn)))
        self.assertEqual(attend_call.args[1].dtype, torch.float8_e4m3fn)
        self.assertEqual(attend_call.args[2].dtype, torch.float8_e4m3fn)
        self.assertEqual(attend_call.kwargs["out"].dtype, torch.bfloat16)
        self.assertEqual(out.dtype, torch.bfloat16)


@unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
class TestExtendReuseKey(CustomTestCase):
    """The prefill state key must be stable across the layers of one forward.

    ``MiniMaxSparseAttnBackend.forward_extend`` re-allocates ``cu_seqlens`` on
    every layer, so the key must not depend on any per-layer tensor address.
    """

    @staticmethod
    def _make_forward_batch(seq_lens, seq_lens_cpu):
        return SimpleNamespace(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            seq_lens_cpu=torch.tensor(seq_lens_cpu, dtype=torch.int64),
        )

    def test_key_is_stable_across_layers_of_one_forward(self):
        forward_batch = self._make_forward_batch([300], [300])
        first = _extend_reuse_key(forward_batch, [300], 300)
        for _ in range(8):
            self.assertEqual(_extend_reuse_key(forward_batch, [300], 300), first)

    def test_key_discriminates_different_batches(self):
        batch_a = self._make_forward_batch([300], [300])
        batch_b = self._make_forward_batch([300], [300])
        self.assertNotEqual(
            _extend_reuse_key(batch_a, [300], 300),
            _extend_reuse_key(batch_b, [300], 300),
        )

    def test_key_tracks_extend_and_kv_lengths(self):
        forward_batch = self._make_forward_batch([300], [300])
        base = _extend_reuse_key(forward_batch, [300], 300)
        self.assertNotEqual(_extend_reuse_key(forward_batch, [200], 200), base)

        # Same object identity and seq_lens buffer, different kv lengths: the
        # content tuples must still force a rebuild.
        other = self._make_forward_batch([300], [300])
        other.seq_lens = forward_batch.seq_lens
        other.seq_lens_cpu = torch.tensor([400], dtype=torch.int64)
        self.assertNotEqual(_extend_reuse_key(other, [300], 300), base)

    def test_key_handles_missing_cpu_mirrors(self):
        forward_batch = SimpleNamespace(seq_lens=torch.tensor([300], dtype=torch.int32))
        self.assertEqual(
            _extend_reuse_key(forward_batch, None, 300),
            _extend_reuse_key(forward_batch, None, 300),
        )


@unittest.skipUnless(
    _HAS_MODULE and HAS_GPU and ppu_msa_available(),
    "msa_ppu module or fmha_sm100/CUDA unavailable",
)
class TestPrefillPlanReuse(CustomTestCase):
    """All sparse layers of one prefill forward share one set of plans.

    Simulates a multi-layer forward: each layer calls ``msa_ppu_forward_extend``
    with the same forward batch but a freshly allocated ``cu_seqlens`` (what
    ``MiniMaxSparseAttnBackend.forward_extend`` does). Only the indexer chunk
    plans and the whole-batch attend plan may run ``fmha_sm100_plan`` — once
    per forward, never per layer.
    """

    def test_fmha_sm100_plan_runs_once_per_forward_not_per_layer(self):
        import fmha_sm100

        device = torch.device("cuda")
        h_idx, h_q, h_kv, topk = 4, 8, 4, 16
        total_q, kv_len, pages = 64, 2 * PAGE, 4

        backend = SimpleNamespace(
            _ppu_msa_state=None,
            _ppu_msa_attend=True,
            _ppu_msa_use_fp8_kvcache=False,
            _ppu_msa_indexer_fp8=False,
            _ppu_msa_score_dtype=torch.float32,
            _ppu_msa_num_idx_heads=h_idx,
            _ppu_msa_num_q_heads=h_q,
            num_kv_heads=h_kv,
            topk_blocks=topk,
            max_context_len=pages * PAGE,
            init_blocks=1,
            local_blocks=2,
            req_to_token=torch.arange(
                pages * PAGE, dtype=torch.int64, device=device
            ).unsqueeze(0),
            kv_pool=SimpleNamespace(main_pool=SimpleNamespace(size=pages * PAGE)),
            _ppu_msa_idx_scale=128**-0.5,
            _ppu_msa_scale=128**-0.5,
        )
        forward_batch = SimpleNamespace(
            seq_lens=torch.tensor([kv_len], dtype=torch.int32, device=device),
            seq_lens_cpu=torch.tensor([kv_len], dtype=torch.int64),
            req_pool_indices=torch.zeros(1, dtype=torch.int64, device=device),
        )
        q = torch.randn(total_q, h_q, 128, dtype=torch.bfloat16, device=device)
        idx_q = torch.randn(total_q, h_idx, 128, dtype=torch.bfloat16, device=device)
        idx_k_cache = torch.randn(
            pages * PAGE, 1, 128, dtype=torch.bfloat16, device=device
        )
        k_cache = torch.randn(
            pages * PAGE, h_kv, 128, dtype=torch.bfloat16, device=device
        )
        v_cache = torch.randn_like(k_cache)
        extend_seq_lens = torch.tensor([total_q], dtype=torch.int32, device=device)
        prefix_lens = torch.tensor([kv_len - total_q], dtype=torch.int32, device=device)

        plan_mock = mock.MagicMock(return_value=None)
        run_mock = mock.MagicMock(return_value=(None, None))
        select_mock = mock.MagicMock(return_value=None)
        with (
            mock.patch.object(fmha_sm100, "fmha_sm100_plan", plan_mock),
            mock.patch.object(fmha_sm100, "fmha_sm100", run_mock),
            mock.patch.object(fmha_sm100, "sparse_topk_select", select_mock),
        ):
            for layer_id in range(3):
                # The backend allocates cu_seqlens inside forward_extend, so
                # every layer sees a different tensor/address.
                cu_seqlens = torch.tensor(
                    [0, total_q], dtype=torch.int32, device=device
                )
                msa_ppu_forward_extend(
                    backend,
                    q,
                    idx_q,
                    idx_k_cache,
                    k_cache,
                    v_cache,
                    forward_batch,
                    cu_seqlens,
                    forward_batch.seq_lens,
                    prefix_lens,
                    extend_seq_lens,
                    [total_q],
                    max_seqlen_q=total_q,
                    layer_id=layer_id,
                )

        # 1 indexer chunk plan + 1 whole-batch attend plan, shared by 3 layers.
        self.assertEqual(plan_mock.call_count, 2)
        self.assertNotIn("use_fp8_kvcache", plan_mock.call_args_list[0].kwargs)
        self.assertFalse(plan_mock.call_args_list[1].kwargs["use_fp8_kvcache"])
        self.assertGreaterEqual(
            run_mock.call_count, 3 * 2
        )  # indexer + attend per layer
        self.assertIsNotNone(backend._ppu_msa_state)
        self.assertNotIn("attend_q_fp8", backend._ppu_msa_state)

        # A new forward (init_forward_metadata_out_graph clears the state and a
        # new ForwardBatch arrives) rebuilds: plan count grows again.
        forward_batch2 = SimpleNamespace(
            seq_lens=torch.tensor([kv_len + PAGE], dtype=torch.int32, device=device),
            seq_lens_cpu=torch.tensor([kv_len + PAGE], dtype=torch.int64),
            req_pool_indices=torch.zeros(1, dtype=torch.int64, device=device),
        )
        backend._ppu_msa_state = None
        backend._ppu_msa_use_fp8_kvcache = True
        k_cache_fp8 = k_cache.to(torch.float8_e4m3fn)
        v_cache_fp8 = v_cache.to(torch.float8_e4m3fn)
        with (
            mock.patch.object(fmha_sm100, "fmha_sm100_plan", plan_mock),
            mock.patch.object(fmha_sm100, "fmha_sm100", run_mock),
            mock.patch.object(fmha_sm100, "sparse_topk_select", select_mock),
        ):
            cu_seqlens = torch.tensor([0, total_q], dtype=torch.int32, device=device)
            msa_ppu_forward_extend(
                backend,
                q,
                idx_q,
                idx_k_cache,
                k_cache_fp8,
                v_cache_fp8,
                forward_batch2,
                cu_seqlens,
                forward_batch2.seq_lens,
                prefix_lens,
                extend_seq_lens,
                [total_q],
                max_seqlen_q=total_q,
                layer_id=0,
            )
        self.assertEqual(plan_mock.call_count, 4)
        self.assertNotIn("use_fp8_kvcache", plan_mock.call_args_list[-2].kwargs)
        self.assertTrue(plan_mock.call_args_list[-1].kwargs["use_fp8_kvcache"])
        indexer_call, attend_call = run_mock.call_args_list[-2:]
        self.assertTrue(torch.equal(indexer_call.args[0], idx_q))
        attend_q = attend_call.args[0]
        self.assertEqual(attend_q.dtype, torch.float8_e4m3fn)
        self.assertEqual(
            attend_q.data_ptr(), backend._ppu_msa_state["attend_q_fp8"].data_ptr()
        )
        self.assertTrue(torch.equal(attend_q, q.to(torch.float8_e4m3fn)))
        self.assertEqual(attend_call.args[1].dtype, torch.float8_e4m3fn)
        self.assertEqual(attend_call.args[2].dtype, torch.float8_e4m3fn)
        self.assertEqual(attend_call.kwargs["out"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()

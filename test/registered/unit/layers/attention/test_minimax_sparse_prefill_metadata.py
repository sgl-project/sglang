"""CPU contracts for request-level MiniMax sparse-prefill metadata.

The production sparse kernels require a GPU.  These tests stop at the Python
kernel boundary and verify that request-invariant metadata is prepared once,
then reused by every sparse layer in the same forward.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import torch

from sglang.srt.layers import radix_attention as radix_attention_module
from sglang.srt.layers.attention import minimax_sparse_backend as backend_module
from sglang.srt.layers.attention.minimax_sparse_backend import MiniMaxSparseAttnBackend
from sglang.srt.layers.attention.minimax_sparse_ops import msa
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ExtendMode:
    def is_extend(self, include_draft_extend_v2=False):
        return True

    def is_decode_or_idle(self):
        return False


def _make_forward_batch(extend_lens=(2, 1)):
    prefix_lens = (128, 128)
    seq_lens = tuple(
        prefix + extend for prefix, extend in zip(prefix_lens, extend_lens)
    )
    num_tokens = sum(extend_lens)
    return SimpleNamespace(
        forward_mode=_ExtendMode(),
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int64),
        extend_seq_lens_cpu=list(extend_lens),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        seq_lens_cpu=list(seq_lens),
        extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int64),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        out_cache_loc=torch.arange(num_tokens, dtype=torch.int64),
        minimax_m3_precached_sparse_layers=None,
    )


def _make_sparse_layer(layer_id: int):
    return SimpleNamespace(
        layer_id=layer_id,
        q_scale_float=None,
        k_scale_float=None,
        v_scale_float=None,
        idx_q_scale_float=None,
        idx_k_scale_float=None,
        idx_v_scale_float=None,
    )


def _make_backend(*, direct_msa=True):
    """Construct only the backend state exercised by prefill CPU tests."""
    backend = object.__new__(MiniMaxSparseAttnBackend)
    backend._msa_dec_meta = None
    backend._prefill_meta = None
    backend._cache_prefill_metadata = True
    backend._use_msa_prefill = True
    backend._use_msa_direct_prefill = direct_msa
    backend._msa_owns_decode = False
    backend.fp8_attn_gemm = False
    backend.max_context_len = 4096
    backend.req_to_token = torch.arange(2 * 256, dtype=torch.int64).view(2, 256)
    backend.block_size_q = 1
    backend.block_size_k = 128
    backend.topk_blocks = 4
    backend.init_blocks = 1
    backend.local_blocks = 1
    backend.num_q_heads = 4
    backend.num_kv_heads = 1
    backend.disable_value_layer_ids = set()
    backend.score_type = "max"

    k_caches = {layer_id: torch.empty((256, 1, 8)) for layer_id in (3, 7)}
    v_caches = {layer_id: torch.empty((256, 1, 8)) for layer_id in (3, 7)}
    idx_k_caches = {layer_id: torch.empty((256, 1, 4)) for layer_id in (3, 7)}
    idx_v_caches = {layer_id: torch.empty((256, 1, 4)) for layer_id in (3, 7)}
    backend.kv_pool = SimpleNamespace(
        set_fused_kv_index_buffer=MagicMock(),
        get_kv_buffer=lambda layer_id: (k_caches[layer_id], v_caches[layer_id]),
        get_index_kv_buffer=lambda layer_id: (
            idx_k_caches[layer_id],
            idx_v_caches[layer_id],
        ),
    )
    return backend


class TestMiniMaxSparsePrefillMetadata(unittest.TestCase):
    def test_breakable_sparse_attention_uses_eager_metadata_break(self):
        layer = RadixAttention(
            num_heads=4,
            head_dim=8,
            scaling=8**-0.5,
            num_kv_heads=1,
            layer_id=3,
        )
        forward_batch = SimpleNamespace(forward_mode=_ExtendMode())
        q = torch.empty((3, 4, 8))
        k = torch.empty((3, 1, 8))
        v = torch.empty((3, 1, 8))
        idx_q = torch.empty((3, 1, 4))
        idx_k = torch.empty((3, 1, 4))
        idx_v = torch.empty((3, 1, 4))

        with (
            patch.object(
                radix_attention_module,
                "get_tc_piecewise_forward_context",
                return_value=sentinel.context,
            ),
            patch.object(
                radix_attention_module,
                "is_in_breakable_cuda_graph",
                return_value=True,
            ),
            patch.object(
                radix_attention_module,
                "breakable_unified_sparse_attention_with_output",
            ) as breakable_sparse_attention,
            patch.object(
                radix_attention_module,
                "unified_sparse_attention_with_output",
            ) as captured_sparse_attention,
            patch.object(radix_attention_module, "get_attn_backend") as get_backend,
        ):
            idx_out, attn_out = layer(
                q,
                k,
                v,
                forward_batch,
                idx_q=idx_q,
                idx_k=idx_k,
                idx_v=idx_v,
            )

        breakable_sparse_attention.assert_called_once()
        captured_sparse_attention.assert_not_called()
        get_backend.assert_not_called()
        self.assertEqual(idx_out.shape, (3, 4))
        self.assertEqual(attn_out.shape, (3, 32))

    def test_direct_metadata_builder_materializes_final_page_table(self):
        req_to_token = torch.tensor(
            [
                [0, 1, 4, 5, 8, 9],
                [12, 13, 16, 17, 20, 21],
            ],
            dtype=torch.int64,
        )
        cu_seqlens_q = torch.tensor([0, 2, 4], dtype=torch.int32)

        metadata = msa.build_msa_prefill_metadata(
            req_to_token=req_to_token,
            slot_ids=torch.tensor([1, 0], dtype=torch.int32),
            cu_seqlens_q=cu_seqlens_q,
            qo_segment_lens=torch.tensor([2, 2], dtype=torch.int32),
            seq_lens=torch.tensor([3, 5], dtype=torch.int32),
            prefix_lens=torch.tensor([1, 3], dtype=torch.int32),
            block_size_k=2,
            max_seqlen_q=2,
            max_seqlen_k=5,
            seq_lens_cpu=[3, 5],
        )

        self.assertIs(metadata.cu_seqlens_q, cu_seqlens_q)
        torch.testing.assert_close(
            metadata.cu_seqlens_k, torch.tensor([0, 3, 8], dtype=torch.int32)
        )
        torch.testing.assert_close(
            metadata.seqused_k, torch.tensor([3, 5], dtype=torch.int32)
        )
        torch.testing.assert_close(
            metadata.page_table,
            torch.tensor([[6, 8, 0], [0, 2, 4]], dtype=torch.int32),
        )
        self.assertEqual(metadata.total_k, 8)
        self.assertEqual(metadata.total_rows, 5)

    def test_replay_replaces_capture_dummy_metadata(self):
        backend = _make_backend()
        backend._use_msa_prefill = False

        backend.init_forward_metadata_out_graph(_make_forward_batch((64, 0)))
        capture_metadata = backend._prefill_meta
        backend.init_forward_metadata_out_graph(_make_forward_batch((1, 33)))
        replay_metadata = backend._prefill_meta

        self.assertIsNot(capture_metadata, replay_metadata)
        self.assertIsNot(capture_metadata.cu_seqlens, replay_metadata.cu_seqlens)
        torch.testing.assert_close(
            replay_metadata.cu_seqlens,
            torch.tensor([0, 1, 34], dtype=torch.int32),
        )

    def test_direct_metadata_is_built_once_and_shared_by_layers(self):
        backend = _make_backend()
        # MSA selects sparse prefill when the batch contains any >32-token row;
        # short rows in the same varlen batch must not force the bridge path.
        forward_batch = _make_forward_batch((1, 33))
        page_table = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)
        msa_metadata = msa.MSAPrefillMetadata(
            cu_seqlens_q=torch.tensor([0, 1, 34], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 129, 290], dtype=torch.int32),
            seqused_k=torch.tensor([129, 161], dtype=torch.int32),
            page_table=page_table,
            max_seqlen_q=33,
            max_seqlen_k=161,
            total_k=290,
            total_rows=4,
        )

        def fake_sparse_prefill(q, *_args, **_kwargs):
            idx_o = torch.zeros((q.shape[0], 1, 4), dtype=q.dtype)
            o = torch.zeros((q.shape[0], 4, 8), dtype=q.dtype)
            return idx_o, o

        with (
            patch.object(
                msa,
                "build_msa_prefill_metadata",
                return_value=msa_metadata,
            ) as build_metadata,
            patch.object(
                backend_module,
                "minimax_sparse_prefill",
                side_effect=fake_sparse_prefill,
            ) as sparse_prefill,
        ):
            backend.init_forward_metadata_out_graph(forward_batch)
            prepared = backend._prefill_meta
            self.assertIsNotNone(prepared)
            build_metadata.assert_called_once()

            for layer_id in (3, 7):
                backend.forward_extend(
                    q=torch.empty((34, 4, 8)),
                    k=torch.empty((34, 1, 8)),
                    v=torch.empty((34, 1, 8)),
                    layer=_make_sparse_layer(layer_id),
                    forward_batch=forward_batch,
                    idx_q=torch.empty((34, 1, 4)),
                    idx_k=torch.empty((34, 1, 4)),
                    idx_v=torch.empty((34, 1, 4)),
                )

        # Entering two layers must not rebuild request-level direct-MSA metadata.
        build_metadata.assert_called_once()
        self.assertEqual(sparse_prefill.call_count, 2)
        first, second = sparse_prefill.call_args_list
        self.assertIs(first.args[10], second.args[10])  # cu_seqlens
        self.assertIs(first.args[10], prepared.cu_seqlens)
        self.assertIs(first.args[11], second.args[11])  # seq_lens
        self.assertIs(first.args[12], second.args[12])  # prefix_lens
        self.assertIs(first.kwargs["cu_seqblocks_q"], second.kwargs["cu_seqblocks_q"])
        self.assertIs(
            first.kwargs["msa_prefill_metadata"],
            second.kwargs["msa_prefill_metadata"],
        )
        self.assertIs(first.kwargs["msa_prefill_metadata"].page_table, page_table)

    def test_bridge_plan_and_flat_page_table_are_forward_scoped(self):
        backend = _make_backend(direct_msa=False)
        forward_batch = _make_forward_batch()
        kv_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

        def fake_sparse_prefill(q, *_args, **_kwargs):
            return (
                torch.zeros((q.shape[0], 1, 4), dtype=q.dtype),
                torch.zeros((q.shape[0], 4, 8), dtype=q.dtype),
            )

        with (
            patch.object(
                msa,
                "build_msa_prefill_bridge_meta",
                return_value=(kv_indices, sentinel.plan),
            ) as build_bridge,
            patch.object(
                backend_module,
                "minimax_sparse_prefill",
                side_effect=fake_sparse_prefill,
            ) as sparse_prefill,
        ):
            backend.init_forward_metadata_out_graph(forward_batch)
            for layer_id in (3, 7):
                backend.forward_extend(
                    q=torch.empty((3, 4, 8)),
                    k=torch.empty((3, 1, 8)),
                    v=torch.empty((3, 1, 8)),
                    layer=_make_sparse_layer(layer_id),
                    forward_batch=forward_batch,
                    idx_q=torch.empty((3, 1, 4)),
                    idx_k=torch.empty((3, 1, 4)),
                    idx_v=torch.empty((3, 1, 4)),
                )

        build_bridge.assert_called_once()
        self.assertEqual(sparse_prefill.call_count, 2)
        for call in sparse_prefill.call_args_list:
            self.assertIs(call.kwargs["msa_kv_indices"], kv_indices)
            self.assertIs(call.kwargs["msa_plan"], sentinel.plan)

    def test_direct_msa_main_only_runs_layer_dependent_csr_and_attention(self):
        block_size = 2
        metadata = msa.MSAPrefillMetadata(
            cu_seqlens_q=torch.tensor([0, 2, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 4, 7], dtype=torch.int32),
            seqused_k=torch.tensor([4, 3], dtype=torch.int32),
            page_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            max_seqlen_q=2,
            max_seqlen_k=4,
            total_k=7,
            total_rows=4,
        )
        q = torch.randn(3, 4, 8, dtype=torch.bfloat16)
        k_cache = torch.randn(8, 2, 8, dtype=torch.bfloat16)
        v_cache = torch.randn(8, 2, 8, dtype=torch.bfloat16)
        topk_per_layer = (
            torch.zeros((2, 3, 2), dtype=torch.int32),
            torch.ones((2, 3, 2), dtype=torch.int32),
        )
        csr_row_ptr = torch.tensor([0, 1], dtype=torch.int32)
        csr_q_indices = torch.tensor([0], dtype=torch.int32)
        output = torch.randn_like(q)
        build_k2q_csr = MagicMock(
            return_value=(csr_row_ptr, csr_q_indices, sentinel.schedule)
        )
        sparse_atten_func = MagicMock(return_value=output)

        with (
            patch.object(
                msa,
                "_load_msa_sparse",
                return_value=(build_k2q_csr, sparse_atten_func),
            ),
            patch.object(msa, "_load_fmha_sm100") as load_bridge,
            patch.object(msa, "build_msa_prefill_bridge_meta") as build_bridge,
            patch.object(msa, "_build_page_table") as build_flat_page_table,
            patch.object(msa, "_run_fmha_sm100_plan") as run_bridge_plan,
        ):
            results = [
                msa.msa_sparse_prefill_main(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    topk_idx=topk_idx,
                    req_to_token=torch.empty((2, 8), dtype=torch.int64),
                    slot_ids=torch.tensor([0, 1], dtype=torch.int32),
                    cu_seqlens=metadata.cu_seqlens_q,
                    seq_lens=torch.tensor([4, 3], dtype=torch.int32),
                    prefix_lens=torch.tensor([2, 2], dtype=torch.int32),
                    block_size_k=block_size,
                    prefill_metadata=metadata,
                )
                for topk_idx in topk_per_layer
            ]

        self.assertEqual(build_k2q_csr.call_count, 2)
        self.assertEqual(sparse_atten_func.call_count, 2)
        self.assertTrue(all(result is output for result in results))
        for call in build_k2q_csr.call_args_list:
            self.assertIs(call.args[1], metadata.cu_seqlens_q)
            self.assertIs(call.args[2], metadata.cu_seqlens_k)
        for call in sparse_atten_func.call_args_list:
            self.assertIs(call.kwargs["page_table"], metadata.page_table)
            self.assertIs(call.kwargs["seqused_k"], metadata.seqused_k)
            self.assertIs(call.kwargs["schedule"], sentinel.schedule)
        load_bridge.assert_not_called()
        build_bridge.assert_not_called()
        build_flat_page_table.assert_not_called()
        run_bridge_plan.assert_not_called()

    def test_block_size_q_one_aliases_cu_seqlens_without_generic_helper(self):
        backend = _make_backend()
        backend._use_msa_prefill = False
        backend._max_seqlen_q = 2
        backend._max_seqlen_k = 130

        with patch.object(backend_module, "get_cu_seqblocks") as get_blocks:
            metadata = backend._build_prefill_metadata(
                _make_forward_batch(), use_host_lengths=True
            )

        get_blocks.assert_not_called()
        self.assertIs(metadata.cu_seqblocks_q, metadata.cu_seqlens)
        torch.testing.assert_close(
            metadata.cu_seqlens, torch.tensor([0, 2, 3], dtype=torch.int32)
        )
        self.assertEqual(metadata.max_seqblock_q, 2)
        self.assertEqual(metadata.all_seqblock_q, 3)

    def test_bridge_plan_failure_falls_back_for_the_whole_forward(self):
        backend = _make_backend(direct_msa=False)
        backend._max_seqlen_q = 2
        backend._max_seqlen_k = 130
        unavailable = msa.MSAUnavailableError("plan API mismatch")

        with (
            patch.object(
                msa,
                "build_msa_prefill_bridge_meta",
                side_effect=unavailable,
            ) as build_bridge,
            patch.object(backend_module, "_warn_msa_fallback") as warn,
        ):
            metadata = backend._build_prefill_metadata(
                _make_forward_batch(), use_host_lengths=True
            )

        build_bridge.assert_called_once()
        warn.assert_called_once_with(unavailable)
        self.assertFalse(metadata.use_msa)
        self.assertIsNone(metadata.msa_kv_indices)
        self.assertIsNone(metadata.msa_plan)


if __name__ == "__main__":
    unittest.main()

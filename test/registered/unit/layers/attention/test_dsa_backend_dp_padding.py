import unittest
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import graph_utils
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _apply_q_row_plan,
    _plan_live_rows,
    _restore_live_rows,
    _RowPlan,
)
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _routed(pool):
    """A stand-in pool whose layers its own plain router hands out."""
    pool.start_layer, pool.layer_transfer_counter = 0, None
    pool.layer_router = PlainPoolRouter(pool, IdSpaceKind.FULL)
    return pool


class TestDSABackendDPPadding(CustomTestCase):
    def test_occupied_uneven_decode_with_idle_rank_selects_max_len(self):
        with (
            patch(
                "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            ),
            patch(
                "sglang.srt.layers.dp_attention.dp_gather_width",
                return_value=4,
            ),
        ):
            mode = DpPaddingMode.get_dp_padding_mode(
                is_extend_in_batch=False,
                global_num_tokens=[8, 7, 6, 0],
            )
        self.assertEqual(mode, DpPaddingMode.MAX_LEN)

    def test_plan_live_rows(self):
        cases = (
            # (q rows, metadata rows, metadata may be padded, expected plan)
            (4, 4, False, _RowPlan(4, 0, 0)),
            (8, 6, False, _RowPlan(6, 2, 0)),  # DP / MLP-sync padding of q
            (2, 4, True, _RowPlan(2, 0, 2)),  # CP physical-row metadata padding
        )
        for q_rows, meta_rows, padded, expected in cases:
            with (
                self.subTest(q_rows=q_rows, meta_rows=meta_rows),
                patch(
                    "sglang.srt.layers.attention.dsa_backend."
                    "dsa_cache_seqlens_may_be_padded",
                    return_value=padded,
                ),
            ):
                self.assertEqual(
                    _plan_live_rows(q_rows, meta_rows, SimpleNamespace()), expected
                )

    def test_trim_and_restore_active_eager_decode_padding(self):
        q_all = torch.arange(8 * 2 * 3).view(8, 2, 3)
        topk_indices = torch.arange(8 * 4, dtype=torch.int32).view(8, 4)
        plan = _RowPlan(6, 2, 0)

        real_q, real_topk = _apply_q_row_plan(plan, q_all, topk_indices)

        self.assertTrue(torch.equal(real_q, q_all[:6]))
        self.assertTrue(torch.equal(real_topk, topk_indices[:6]))

        real_output = torch.ones((6, 1, 2, 5), dtype=torch.bfloat16)
        output = _restore_live_rows(real_output, plan)
        self.assertEqual(output.shape, (8, 1, 2, 5))
        self.assertTrue(torch.equal(output[:6], real_output))
        self.assertTrue(torch.all(output[6:] == 0))

    def test_pad_and_trim_metadata_padding(self):
        """FlashMLA takes one q row per metadata row: CP metadata padding is
        fed as zero queries with -1 indices and trimmed from the output."""
        q = torch.ones(2, 1, 2, 3)
        indices = torch.arange(2 * 1 * 4, dtype=torch.int32).view(2, 1, 4)
        plan = _RowPlan(2, 0, 3)

        padded_q, padded_indices = _apply_q_row_plan(plan, q, indices)

        self.assertEqual(padded_q.shape, (5, 1, 2, 3))
        self.assertTrue(torch.equal(padded_q[:2], q))
        self.assertTrue(torch.all(padded_q[2:] == 0))
        self.assertTrue(torch.equal(padded_indices[:2], indices))
        self.assertTrue(torch.all(padded_indices[2:] == -1))
        output = _restore_live_rows(torch.ones(5, 1, 2, 2), plan)
        self.assertEqual(output.shape, (2, 1, 2, 2))

    def test_sparse_prefill_restores_tp_padding(self):
        """Sparse prefill must preserve real rows and zero-pad to the TP extent."""
        for implementation in ("tilelang", "triton"):
            for real_rows in (1, 34):
                with self.subTest(implementation=implementation, real_rows=real_rows):
                    self._check_sparse_prefill_padding(implementation, real_rows)

    def _check_sparse_prefill_padding(self, implementation, real_rows):
        rows = (real_rows + 3) // 4 * 4
        q = torch.arange(rows * 2 * 3, dtype=torch.float32).view(rows, 2, 3)
        layer = SimpleNamespace(
            layer_id=0,
            head_dim=3,
            qk_head_dim=3,
            v_head_dim=2,
            tp_q_head_num=2,
            is_cross_attention=False,
            scaling=1.0,
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            out_cache_loc=torch.arange(rows),
            positions=torch.arange(rows),
            mha_return_lse=False,
            _attn_output=None,
        )
        values = torch.arange(real_rows * 2 * 2, dtype=torch.float32).view(
            1, real_rows, 2, 2
        )
        kernel = ModuleType(
            "sglang.kernels.ops.attention.dsa."
            + (
                "tilelang_kernel"
                if implementation == "tilelang"
                else "triton_sparse_mla"
            )
        )

        def fake_sparse_fwd(**kwargs):
            query = kwargs.get("q", kwargs.get("q_nope"))
            self.assertEqual(query.shape[0], real_rows)
            return values

        setattr(
            kernel,
            "tilelang_sparse_fwd"
            if implementation == "tilelang"
            else "triton_sparse_mla_fwd",
            fake_sparse_fwd,
        )
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.dsa_shards_index_k = False
        backend.attn_dcp_size, backend.attn_dcp_rank = 1, 0
        backend.__dict__.update(
            dsa_prefill_impl=implementation,
            use_mha=False,
            use_fused_topk=True,
            hisparse_coordinator=None,
            dsa_index_kpool=1,
            forward_metadata=SimpleNamespace(max_seq_len_k=64),
            token_to_kv_pool=_routed(
                SimpleNamespace(
                    get_key_buffer=lambda _: torch.zeros(64, 1, 3),
                    set_mla_kv_buffer=lambda *args: None,
                )
            ),
            _resolve_kpool_tail_backend=lambda indices, impl: impl,
            _check_kpool_tail_backend=lambda *args: None,
            get_topk_transform_method=lambda _: TopkTransformMethod.PAGED,
            _pad_topk_indices=lambda indices, n: indices[:n],
            _get_fused_topk_page_table=lambda indices: indices,
        )
        backend._forward_tilelang = MethodType(
            DeepseekSparseAttnBackend._forward_tilelang, backend
        )
        backend.forward = MethodType(DeepseekSparseAttnBackend.forward_extend, backend)
        with (
            patch.dict("sys.modules", {kernel.__name__: kernel}),
            patch.object(graph_utils, "get_attn_backend", return_value=backend),
            patch(
                "sglang.srt.layers.attention.dsa_backend.concat_mla_absorb_q_general",
                side_effect=lambda q, rope: torch.cat((q, rope), dim=-1),
            ),
        ):
            output = graph_utils._attention_on_real_rows(
                layer,
                real_rows,
                q,
                torch.zeros(rows, 1, 3),
                torch.zeros(rows, 1, 2),
                batch,
                True,
                topk_indices=torch.zeros(rows, 64, dtype=torch.int32),
            )

        self.assertEqual(output.shape, (rows, 2, 2))
        torch.testing.assert_close(output[:real_rows], values[0])
        self.assertTrue(torch.all(output[real_rows:] == 0))
        self.assertEqual(batch.out_cache_loc.shape[0], rows)

    def test_rejects_metadata_larger_than_physical_batch(self):
        with (
            patch(
                "sglang.srt.layers.attention.dsa_backend."
                "dsa_cache_seqlens_may_be_padded",
                return_value=False,
            ),
            self.assertRaisesRegex(
                AssertionError, "metadata rows \\(3\\) exceed q rows \\(2\\)"
            ),
        ):
            _plan_live_rows(2, 3, SimpleNamespace())

    def test_accepts_metadata_padding_inside_breakable_graph(self):
        # The replay view lacks global_num_tokens_cpu; the bridge narrowed q.
        with (
            patch(
                "sglang.srt.layers.attention.dsa_backend."
                "dsa_cache_seqlens_may_be_padded",
                return_value=False,
            ),
            patch(
                "sglang.srt.model_executor.runner_backend_utils."
                "breakable_cuda_graph.context._in_breakable_cuda_graph",
                True,
            ),
        ):
            plan = _plan_live_rows(34, 48, SimpleNamespace())
        self.assertEqual(
            (plan.live_rows, plan.q_pad_rows, plan.meta_pad_rows), (34, 0, 14)
        )

    def test_trtllm_decode_runs_real_rows_then_restores_physical_batch(self):
        self._check_trtllm_padding(is_prefill=False)

    def test_trtllm_prefill_narrows_metadata_to_live_cp_queries(self):
        self._check_trtllm_padding(is_prefill=True)

    def test_trtllm_prefill_plans_on_per_token_seq_lens(self):
        # One request of four tokens: cache_seqlens_int32 has one row, but the
        # prefill caller passes dsa_cache_seqlens_int32 with one row per token.
        self._check_trtllm_padding(is_prefill=True, per_token_seq_lens=True)

    def _check_trtllm_padding(self, *, is_prefill, per_token_seq_lens=False):
        metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([8, 12], dtype=torch.int32),
            page_table_1=torch.zeros((2, 12), dtype=torch.int32),
            max_seq_len_k=12,
        )
        backend = SimpleNamespace(
            forward_metadata=metadata,
            kv_cache_dtype=torch.bfloat16,
            kv_layers=_routed(
                SimpleNamespace(get_key_buffer=lambda _layer_id: torch.zeros((24, 3)))
            ).layer_router,
            physical_page_size=1,
            kv_cache_dim=3,
            use_fused_topk=False,
            qk_nope_head_dim=2,
            kv_lora_rank=2,
            qk_rope_head_dim=1,
            workspace_buffer=None,
            dsa_index_topk=2,
            _multi_ctas_kv_counter_buffer=None,
            device="cpu",
            num_q_heads=2,
            num_dcp_q_heads=2,
            attn_dcp_size=1,
            dsa_shards_index_k=False,
        )
        backend._pad_topk_indices = MethodType(
            DeepseekSparseAttnBackend._pad_topk_indices, backend
        )
        backend._pad_trtllm_sparse_page_table = MethodType(
            DeepseekSparseAttnBackend._pad_trtllm_sparse_page_table, backend
        )
        backend._multi_ctas_kv_counter_for = MethodType(
            DeepseekSparseAttnBackend._multi_ctas_kv_counter_for, backend
        )

        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=2,
            head_dim=3,
            k_scale_float=None,
            scaling=1.0,
        )
        forward_batch = SimpleNamespace()
        q = torch.arange(4 * 2 * 3, dtype=torch.float32).view(4, 2, 3)
        topk_indices = torch.arange(4 * 2, dtype=torch.int32).view(4, 2)
        seq_lens = None
        if per_token_seq_lens:
            metadata.cache_seqlens_int32 = torch.tensor([12], dtype=torch.int32)
            seq_lens = torch.tensor([9, 10, 11, 12], dtype=torch.int32)
            backend.use_fused_topk = True
            backend._get_fused_topk_page_table = lambda indices: indices
        elif is_prefill:
            # The CP graph bridge has removed padded Q rows; metadata was
            # planned for the physical shard before entering the eager break.
            q = q[:2]
            topk_indices = topk_indices[:2]
            metadata.cache_seqlens_int32 = torch.tensor(
                [8, 12, 0, 0], dtype=torch.int32
            )
            backend.use_fused_topk = True
            backend._get_fused_topk_page_table = lambda indices: indices

        flashinfer = ModuleType("flashinfer")
        flashinfer_decode = ModuleType("flashinfer.decode")
        captured = {}

        def fake_decode(**kwargs):
            captured.update(kwargs)
            rows = kwargs["query"].shape[0]
            return torch.ones((rows, 1, 2, 2), dtype=torch.bfloat16)

        flashinfer_decode.trtllm_batch_decode_with_kv_cache_mla = fake_decode
        flashinfer.decode = flashinfer_decode

        def fake_transform(*, page_table, topk_indices, page_size):
            self.assertEqual(page_table.shape[0], 2)
            self.assertEqual(topk_indices.shape[0], 2)
            self.assertEqual(page_size, 1)
            return torch.zeros((2, 2), dtype=torch.int32)

        with (
            patch.dict(
                "sys.modules",
                {
                    "flashinfer": flashinfer,
                    "flashinfer.decode": flashinfer_decode,
                },
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.transform_index_page_table_decode",
                side_effect=fake_transform,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.dsa_use_prefill_cp",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.grow_multi_ctas_kv_counter_buffer_if_needed",
                return_value=None,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend."
                "dsa_cache_seqlens_may_be_padded",
                return_value=is_prefill and not per_token_seq_lens,
            ),
        ):
            output = DeepseekSparseAttnBackend._forward_trtllm(
                backend,
                q=q,
                k=torch.empty((4, 1, 2)),
                v=torch.empty((4, 1, 2)),
                layer=layer,
                forward_batch=forward_batch,
                seq_lens=(
                    metadata.cache_seqlens_int32 if seq_lens is None else seq_lens
                ),
                save_kv_cache=False,
                topk_indices=topk_indices,
                is_prefill=is_prefill,
            )

        if per_token_seq_lens:
            self.assertEqual(captured["query"].shape, (4, 1, 2, 3))
            self.assertEqual(captured["seq_lens"].tolist(), [9, 10, 11, 12])
            self.assertEqual(output.shape, (4, 1, 2, 2))
            self.assertTrue(torch.all(output == 1))
            return
        self.assertEqual(captured["query"].shape, (2, 1, 2, 3))
        self.assertEqual(captured["seq_lens"].tolist(), [8, 12])
        self.assertEqual(output.shape, (2 if is_prefill else 4, 1, 2, 2))
        self.assertTrue(torch.all(output[:2] == 1))
        self.assertTrue(torch.all(output[2:] == 0))


if __name__ == "__main__":
    unittest.main()

"""CPU regressions for Kimi K3.1's Ascend sparse-attention PCP integration.

Run the real attention/indexer wrappers and CP metadata preparation, replacing
only vendor operators and device communication/storage. This checks tensor and
metadata contracts, not NPU kernel numerics.
"""

import importlib
import unittest
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.dsa import dsa_npu_indexer
from sglang.srt.layers.cp.interleave import (
    InterleaveContextParallelMetadata,
    InterleaveCPStrategy,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=10, suite="extra-a-test-npu")


class _Projection(torch.nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class TestKimiK31PrefillCP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        modules = {
            name: ModuleType(name)
            for name in (
                "torch_npu",
                "sgl_kernel_npu",
                "sgl_kernel_npu.operators",
                "sgl_kernel_npu.operators.attention",
            )
        }
        for name in (
            "attention_sinks",
            "attention_sinks_prefill",
            "sparse_flash_attention",
        ):
            setattr(
                modules["sgl_kernel_npu.operators.attention"],
                name,
                Mock(side_effect=AssertionError("Unexpected vendor operator")),
            )
        with patch.dict("sys.modules", modules):
            cls.attention = importlib.import_module(
                "sglang.srt.hardware_backend.npu.attention.ascend_backend"
            )

    def test_sparse_attention_preserves_query_heads_with_and_without_cp(self):
        """Legacy NPU kernels reject 96 heads; arch35 must receive them natively.

        Both paths must preserve every real head's output and the caller's
        inputs. Extra kernel heads must be zero-filled and removed from output.
        """
        for cp_enabled in (False, True):
            for arch35 in (False, True):
                for heads in (96, 128):
                    with self.subTest(cp=cp_enabled, arch35=arch35, heads=heads):
                        self._check_sparse_attention(cp_enabled, arch35, heads)

    def _check_sparse_attention(self, cp_enabled, arch35, heads):
        module = self.attention
        backend = module.AscendAttnBackend.__new__(module.AscendAttnBackend)
        backend.qk_rope_head_dim = 2
        backend.kv_cache_dtype = torch.bfloat16
        backend.dsa_dcp_target = False
        backend.forward_metadata = module.ForwardMetadata(
            actual_seq_lengths_q=torch.tensor([1, 2], dtype=torch.int32),
            actual_seq_lengths_kv=torch.tensor([2, 3], dtype=torch.int32),
            block_tables=torch.tensor([[0], [0]], dtype=torch.int32),
        )
        key = torch.zeros(1, 4, 1, 4)
        key_rope = torch.zeros(1, 4, 1, 2)
        backend.token_to_kv_pool = SimpleNamespace(
            dsa_kv_cache_store_fp8=False,
            get_kv_buffer=lambda _: (key, key_rope),
        )
        query = torch.arange(2 * heads * 4, dtype=torch.float32).view(2, heads, 4)
        query_rope = torch.arange(2 * heads * 2, dtype=torch.float32).view(2, heads, 2)
        original_query, original_rope = query.clone(), query_rope.clone()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            attn_cp_metadata=object() if cp_enabled else None,
            attn_dcp_metadata=None,
        )
        layer = SimpleNamespace(layer_id=0, tp_q_head_num=heads, scaling=0.5)

        def sparse_attention(**kwargs):
            # The vendor's input contract, independent of the padding strategy.
            self.assertEqual(kwargs["query"].shape[1], heads if arch35 else 128)
            self.assertEqual(kwargs["query_rope"].shape[1], kwargs["query"].shape[1])
            if arch35 or heads == 128:
                self.assertIs(kwargs["query"], query)
                self.assertIs(kwargs["query_rope"], query_rope)
            else:
                self.assertEqual(
                    torch.count_nonzero(kwargs["query"][:, heads:]).item(), 0
                )
                self.assertEqual(
                    torch.count_nonzero(kwargs["query_rope"][:, heads:]).item(), 0
                )
            torch.testing.assert_close(kwargs["query"][:, :heads], original_query)
            torch.testing.assert_close(kwargs["query_rope"][:, :heads], original_rope)
            return (
                kwargs["query"] * 2 + kwargs["query_rope"].sum(-1, keepdim=True),
                None,
                None,
            )

        with (
            get_parallel().override(attn_tp_size=1),
            patch.object(module, "is_npu_arch35", return_value=arch35),
            patch.object(module, "is_dsa_enable_prefill_cp", return_value=cp_enabled),
            patch.object(
                module,
                "torch_npu",
                SimpleNamespace(npu_sparse_flash_attention=sparse_attention),
            ),
        ):
            result = backend.forward_sparse(
                query,
                None,
                None,
                layer,
                batch,
                save_kv_cache=False,
                q_rope=query_rope,
                topk_indices=torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
            )
        torch.testing.assert_close(
            result, original_query * 2 + original_rope.sum(-1, True)
        )
        torch.testing.assert_close(query, original_query)
        torch.testing.assert_close(query_rope, original_rope)

    def test_indexer_prepares_full_attention_metadata_before_scoring(self):
        """A hybrid wrapper must not leave the indexer using global CP lengths."""
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                self._check_indexer_metadata(hybrid)

    def _check_indexer_metadata(self, hybrid):
        module = self.attention
        backend = module.AscendAttnBackend.__new__(module.AscendAttnBackend)
        backend.use_dsa = True
        backend.use_decode_indexer_request_shard = False
        backend.forward_metadata = module.ForwardMetadata(
            seq_lens_cpu_int=torch.tensor([7, 3], dtype=torch.int32),
            block_tables=torch.tensor([[10], [20]], dtype=torch.int32),
        )
        exposed_backend = (
            SimpleNamespace(full_attn_backend=backend) if hybrid else backend
        )
        metadata = InterleaveContextParallelMetadata(
            total_seq_lens=4, per_rank_actual_token=[2, 2]
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            attn_cp_metadata=metadata,
            positions=torch.tensor([4, 5, 6, 2]),
            extend_seq_lens_cpu=[3, 1],
            out_cache_loc=torch.arange(4),
        )
        indexer = dsa_npu_indexer.DSANPUIndexerMixin()
        indexer.hidden_size = 4
        indexer.n_heads = 2
        indexer.head_dim = 4
        indexer.index_topk = 2
        indexer.rotary_emb = None
        indexer.dsa_enable_prefill_cp = True
        indexer.num_init_tokens = indexer.num_local_tokens = 0
        indexer.weights_proj = _Projection(4, 2)
        indexer.wq_b = _Projection(4, 8)
        indexer.wk = _Projection(4, 4)
        indexer.k_norm = torch.nn.Identity()
        cache = torch.zeros(4, 1, 4)
        pool = SimpleNamespace(
            index_k_scale_buffer=None,
            set_index_k_buffer=lambda layer, loc, k: cache.index_copy_(
                0, loc, k.view(-1, 1, 4)
            ),
            get_index_k_buffer=lambda layer: cache,
        )

        def score(**kwargs):
            torch.testing.assert_close(
                kwargs["actual_seq_lengths_query"],
                torch.tensor([1, 2], dtype=torch.int32),
            )
            torch.testing.assert_close(
                kwargs["block_table"], torch.tensor([[10], [20]], dtype=torch.int32)
            )
            lengths = kwargs["actual_seq_lengths_key"]
            return torch.stack((lengths, lengths - 1), -1).unsqueeze(1), None

        with ExitStack() as stack:
            stack.enter_context(
                get_parallel().override(
                    tp_size=2,
                    tp_rank=1,
                    attn_tp_size=1,
                    attn_dp_size=1,
                    attn_cp_size=2,
                    attn_cp_rank=1,
                    moe_tp_size=2,
                    dcp_enabled=False,
                )
            )
            for name, value in (
                ("get_attn_backend", exposed_backend),
                ("get_token_to_kv_pool", pool),
            ):
                stack.enter_context(
                    patch.object(dsa_npu_indexer, name, return_value=value)
                )
            stack.enter_context(
                patch.object(
                    module, "get_cp_strategy", return_value=InterleaveCPStrategy(2)
                )
            )
            stack.enter_context(
                patch.object(
                    dsa_npu_indexer,
                    "cp_gather_full_sequence_states",
                    side_effect=lambda k, *args: k.repeat(2, 1),
                )
            )
            stack.enter_context(
                patch.object(
                    dsa_npu_indexer.envs.SGLANG_NPU_USE_MULTI_STREAM,
                    "get",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch.object(
                    dsa_npu_indexer,
                    "torch_npu",
                    SimpleNamespace(npu_lightning_indexer=score),
                    create=True,
                )
            )
            stack.enter_context(
                patch.object(
                    torch,
                    "npu",
                    SimpleNamespace(current_stream=lambda: None),
                    create=True,
                )
            )
            result = indexer.forward_npu(
                torch.ones(2, 4), torch.ones(2, 4), batch.positions, batch, layer_id=0
            )

        self.assertTrue(batch.dsa_cp_metadata_prepared)
        torch.testing.assert_close(batch.positions, torch.tensor([5, 2]))
        torch.testing.assert_close(
            result, torch.tensor([[6, 5], [3, 2]], dtype=torch.int32)
        )


if __name__ == "__main__":
    unittest.main()

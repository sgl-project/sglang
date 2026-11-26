# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``sglang.srt.layers.attention.minimax_sparse`` subpackage."""

import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.minimax_sparse import (
    MiniMaxBF16SparseAttnBackend,
    MiniMaxHybridAttnBackend,
    MiniMaxMSAV1AttnBackend,
    MiniMaxNPUSparseAttnBackend,
    MiniMaxROCmSparseAttnBackend,
    MiniMaxSparseAttnBackend,
)
from sglang.srt.layers.attention.minimax_sparse import base as base_module
from sglang.srt.layers.attention.minimax_sparse import (
    build_minimax_attn_backend,
    resolve_minimax_sparse_backend_cls,
)
from sglang.srt.layers.attention.minimax_sparse_ops import indexer_cp as indexer_cp_mod
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DummyDenseBackend(AttentionBackend):
    def __init__(self):
        self.runner_ids = object()
        self.token_to_kv_pool = None
        self.req_to_token_pool = None
        self.needs_cpu_seq_lens = False


class TestMiniMaxSparseSubpackageDispatch(CustomTestCase):
    def _runner(self, quantization=None):
        return SimpleNamespace(
            model_config=SimpleNamespace(quantization=quantization),
        )

    def test_package_import_keeps_concrete_backends_lazy(self):
        """Importing ``minimax_sparse`` must not eagerly load ``msa_v1_backend`` /
        ``msa_plan`` (which mutates ``FMHA_SM100_ALLOW_JIT`` and scans package metadata)."""
        code = "\n".join(
            [
                "import os, sys",
                "os.environ.pop('FMHA_SM100_ALLOW_JIT', None)",
                "import sglang.srt.layers.attention.minimax_sparse",
                "assert 'FMHA_SM100_ALLOW_JIT' not in os.environ",
                "for mod in ('msa_v1_backend', 'bf16_backend', 'rocm_backend', 'npu_backend'):",
                "    assert f'sglang.srt.layers.attention.minimax_sparse.{mod}' not in sys.modules",
                "assert 'sglang.srt.layers.attention.minimax_sparse_ops.msa_plan' not in sys.modules",
            ]
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_indexer_cp_reads_subclass_fields_after_subclass_init(self):
        """``make_indexer_cp`` inspects ``use_dense_sparse_decode`` and ``kv_is_nvfp4``,
        which subclasses set after ``super().__init__()``."""
        hf_config = SimpleNamespace(
            text_config=None,
            sparse_attention_config={
                "sparse_index_dim": 128,
                "sparse_block_size": 128,
                "sparse_topk_blocks": 16,
                "sparse_num_index_heads": 4,
                "sparse_init_block": 0,
                "sparse_local_block": 1,
                "sparse_attention_freq": [1, 1, 1, 1],
                "sparse_disable_index_value": [1, 1, 1, 1],
                "sparse_score_type": "max",
            },
        )
        seen = []

        def record_make_indexer_cp(backend, runner, sparse_cfg):
            seen.append((backend.use_dense_sparse_decode, backend.kv_is_nvfp4))
            return "cp-sentinel"

        for pool_cls, backend_cls, expect_nvfp4 in (
            (MiniMaxSparseKVPool, MiniMaxBF16SparseAttnBackend, False),
            (MiniMaxM3Nvfp4KVPool, MiniMaxROCmSparseAttnBackend, True),
        ):
            pool = object.__new__(pool_cls)
            pool.page_size = 128
            pool.main_pool = SimpleNamespace(dtype=torch.bfloat16, head_num=1)
            runner = SimpleNamespace(
                token_to_kv_pool=pool,
                hisparse_coordinator=None,
                req_to_token_pool=SimpleNamespace(
                    row_width=1024, req_to_token=torch.zeros(1, 128)
                ),
                model_config=SimpleNamespace(
                    context_len=1024,
                    num_attention_heads=64,
                    hf_config=hf_config,
                    quantization=None,
                    get_total_num_kv_heads=lambda: 4,
                ),
                server_args=SimpleNamespace(
                    m3_fp8_attn_gemm=False, kv_cache_dtype="auto"
                ),
            )
            with (
                get_context().override_server_args(
                    speculative_algorithm=None,
                    speculative_eagle_topk=None,
                    speculative_num_draft_tokens=None,
                ),
                envs.SGLANG_DISABLE_MSA.override(True),
                envs.SGLANG_MINIMAX_M3_INDEXER_CP.override(True),
                patch.object(
                    indexer_cp_mod,
                    "make_indexer_cp",
                    side_effect=record_make_indexer_cp,
                ),
            ):
                backend = backend_cls(runner)
            self.assertEqual(backend.indexer_cp, "cp-sentinel")
            self.assertEqual(seen[-1], (False, expect_nvfp4))

    def test_resolve_backend_cls_by_platform_and_quantization(self):
        with (
            patch.object(base_module, "is_npu", return_value=True),
            patch.object(base_module, "is_hip", return_value=False),
        ):
            self.assertIs(
                resolve_minimax_sparse_backend_cls(self._runner()),
                MiniMaxNPUSparseAttnBackend,
            )

        with (
            patch.object(base_module, "is_npu", return_value=False),
            patch.object(base_module, "is_hip", return_value=True),
        ):
            self.assertIs(
                resolve_minimax_sparse_backend_cls(self._runner("minimax_m3_nvfp4")),
                MiniMaxROCmSparseAttnBackend,
            )

        with (
            patch.object(base_module, "is_npu", return_value=False),
            patch.object(base_module, "is_hip", return_value=False),
            patch.object(
                base_module, "uses_minimax_m3_nvfp4_native_stack", return_value=True
            ),
        ):
            self.assertIs(
                resolve_minimax_sparse_backend_cls(self._runner("minimax_m3_nvfp4")),
                MiniMaxMSAV1AttnBackend,
            )

        with (
            patch.object(base_module, "is_npu", return_value=False),
            patch.object(base_module, "is_hip", return_value=False),
            patch.object(
                base_module, "uses_minimax_m3_nvfp4_native_stack", return_value=False
            ),
        ):
            self.assertIs(
                resolve_minimax_sparse_backend_cls(self._runner()),
                MiniMaxBF16SparseAttnBackend,
            )

    def test_build_minimax_attn_backend_wraps_hybrid_unless_native_nvfp4(self):
        from sglang.srt.layers.attention import minimax_sparse as pkg

        dense = _DummyDenseBackend()
        fake_sparse = SimpleNamespace(
            sparse_layer_ids=[1, 3],
            dense_layer_ids=[0, 2],
            skips_unused_dense_metadata=lambda: False,
        )
        runner = self._runner()
        with (
            patch.object(pkg, "uses_minimax_m3_nvfp4_native_stack", return_value=False),
            patch.object(
                pkg, "create_minimax_sparse_backend", return_value=fake_sparse
            ),
        ):
            hybrid = build_minimax_attn_backend(runner, dense)
        self.assertIsInstance(hybrid, MiniMaxHybridAttnBackend)
        self.assertIs(hybrid.dense, dense)
        self.assertIs(hybrid.sparse, fake_sparse)
        self.assertEqual(hybrid.sparse_layer_ids, [1, 3])

        sentinel = object()
        with (
            patch.object(pkg, "uses_minimax_m3_nvfp4_native_stack", return_value=True),
            patch.object(
                sys.modules[
                    "sglang.srt.layers.attention.minimax_sparse.msa_v1_backend"
                ],
                "MiniMaxMSAV1AttnBackend",
                return_value=sentinel,
            ),
        ):
            self.assertIs(build_minimax_attn_backend(runner, dense), sentinel)

    def test_linear_verify_funnel_preserves_batch_and_expands_queries(self):
        backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
        backend._linear_verify = True
        backend.speculative_num_draft_tokens = 3
        backend._linear_verify_meta = None

        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            req_pool_indices=torch.tensor([5, 2], dtype=torch.int64),
            out_cache_loc=torch.arange(6, dtype=torch.int64),
        )
        q = torch.zeros(6, 1, 128, dtype=torch.bfloat16)
        seen = []

        def fake_decode(q, k, v, layer, verify_batch, save_kv_cache, **kwargs):
            seen.append(verify_batch)
            return None, torch.stack(
                (verify_batch.req_pool_indices, verify_batch.seq_lens.to(torch.int64)),
                dim=-1,
            )

        backend.forward_decode = fake_decode
        backend._init_linear_verify_metadata(batch)
        idx_o, out = backend.forward_extend(
            q, q, q, None, batch, idx_q=q, idx_k=q, idx_v=None
        )
        self.assertIsNone(idx_o)
        self.assertEqual(
            out.tolist(),
            [[5, 11], [5, 12], [5, 13], [2, 21], [2, 22], [2, 23]],
        )
        self.assertIsNot(seen[0], batch)
        self.assertIs(seen[0].out_cache_loc, batch.out_cache_loc)
        self.assertEqual(batch.seq_lens.tolist(), [10, 20])


if __name__ == "__main__":
    unittest.main()

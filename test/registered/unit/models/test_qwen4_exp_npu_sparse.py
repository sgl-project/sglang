"""NPU QSA dispatch, cache-view and replay regressions without an accelerator.

Kernel calls are external boundaries. These checks do not certify NPU numerics.
"""

import ast
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.qwen4_exp_cpu_test_utils import ROOT, forbidden, load
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

QSA = "srt/layers/attention/qsa/"
BACKEND = "srt/layers/attention/qwen_sparse_attn_backend.py"
KERNELS = "sgl_kernel_npu.qwen3_8_flash_next"


class DeviceAlias:
    """Model the misleading CUDA flag installed by transfer_to_npu."""

    is_cuda = True

    def __init__(self, shape=(2, 4, 128), dtype=torch.bfloat16):
        self.device = NS(type="npu")
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype

    def to(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

    def __sub__(self, other):
        return self

    def numel(self):
        return self.shape[0]


class CudaAliasedCPUTensor(torch.Tensor):
    """Exercise non-CUDA tensor math with transfer_to_npu's CUDA alias."""

    @property
    def is_cuda(self):
        return True


class TestNPUQSADispatch(CustomTestCase):
    def test_prefill_relative_scoring_keeps_non_cuda_on_packed_path(self):
        expected = torch.tensor([[1, 3], [2, 4]], dtype=torch.int32)
        code = load(
            QSA + "qsa_indexer.py",
            {"QSAIndexer": {"select_prefill_tokens"}},
            {
                "HAS_TILELANG": True,
                "_qsa_prefill_row_chunk_size": lambda rows, *_: rows,
                "qsa_mqa_prefill_relative": forbidden,
                "qsa_mqa_prefill": lambda *args: torch.zeros((2, 8)),
                "qsa_fast_topk": lambda *args, **kwargs: expected,
                "expand_qsa_block_indices": lambda indices, *args, **kwargs: indices,
            },
        )
        q = torch.zeros(2, 4, 128).as_subclass(CudaAliasedCPUTensor)
        actual = code.QSAIndexer.select_prefill_tokens(
            NS(token_topk=2, block_topk=2, compress_ratio=1),
            q,
            torch.zeros(8, 1, 128),
            torch.tensor([0, 4]),
            torch.tensor([4, 8]),
            torch.tensor([3, 3]),
            torch.tensor([4, 4]),
            max_row_keys=4,
        )
        torch.testing.assert_close(actual, expected)

    def test_prefill_key_gather_preserves_request_order_without_cuda_kernel(self):
        code = load(
            QSA + "metadata.py",
            {
                "build_qsa_row_ranges": None,
                "QSAIndexerMetadata": {"get_prefill_mqa_inputs"},
            },
            {"gather_qsa_compressed_keys": forbidden},
        )
        buffer = torch.arange(4 * 128).reshape(4, 1, 128)
        buffer = buffer.as_subclass(CudaAliasedCPUTensor)
        owner = NS(
            token_to_kv_pool=NS(get_qsa_compressed_k_buffer=lambda layer: buffer),
            compress_ratio=4,
            sequence_lengths=torch.tensor([8, 2, 4]),
            prefill_sequence_lengths_cpu=(8, 2, 4),
            token_to_batch_idx=torch.tensor([0, 0, 1, 2]),
            token_slot_table=torch.tensor(
                [[12, 13, 14, 15, 4, 5, 6, 7], list(range(8)), list(range(8, 16))]
            ),
        )
        keys, starts, ends, lengths = code.QSAIndexerMetadata.get_prefill_mqa_inputs(
            owner, 0, torch.tensor([3, 7, 1, 3])
        )
        torch.testing.assert_close(keys, buffer[[3, 1, 2]])
        self.assertEqual(starts.tolist(), [0, 0, 2, 2])
        self.assertEqual(ends.tolist(), [1, 2, 2, 3])
        self.assertEqual(lengths.tolist(), [8, 2, 4])
        owner.sequence_lengths = torch.tensor([0, 0, 0])
        owner.prefill_sequence_lengths_cpu = (0, 0, 0)
        owner.token_to_batch_idx = torch.empty(0, dtype=torch.long)
        keys, starts, ends, _ = code.QSAIndexerMetadata.get_prefill_mqa_inputs(
            owner, 0, torch.empty(0, dtype=torch.long)
        )
        self.assertEqual(keys.shape, (0, 1, 128))
        self.assertEqual(starts.numel(), 0)
        self.assertEqual(ends.numel(), 0)

    def test_topk_routes_npu_before_cuda_alias(self):
        code = load(
            QSA + "kernel.py",
            {"qsa_fast_topk": None},
            {"is_batch_invariant_mode_enabled": lambda: False},
        )
        logits, bounds = (
            DeviceAlias((2, 1024), torch.float32),
            DeviceAlias((2,), torch.int32),
        )
        expected = torch.tensor([[5, 2], [7, 1]], dtype=torch.int32)

        def topk(score, lengths, width, starts):
            self.assertIs(score, logits)
            self.assertEqual(width, 512)
            self.assertIs(starts, bounds)
            return expected

        with patch.dict(
            sys.modules,
            {
                KERNELS + ".qsa_topk": NS(fast_topk=topk),
                "sglang.kernels.ops.attention.fast_topk": NS(fast_topk=forbidden),
            },
        ):
            self.assertIs(code.qsa_fast_topk(logits, bounds, bounds, 512), expected)

    def test_mqa_routes_both_modes_before_tilelang(self):
        code = load(
            QSA + "mqa.py",
            {"qsa_mqa_prefill": None, "qsa_mqa_decode": None},
            {
                "HAS_TILELANG": True,
                "tilelang_qsa_mqa_prefill": forbidden,
                "tilelang_qsa_mqa_decode": forbidden,
                "torch_qsa_mqa_prefill": forbidden,
                "torch_qsa_mqa_decode": forbidden,
            },
        )
        q, k, bounds = DeviceAlias(), object(), object()
        expected = torch.tensor([[1.0, 2.0]])
        mqa = NS(packed=lambda *args: expected, paged=lambda *args: expected)
        with patch.dict(sys.modules, {KERNELS: NS(qsa_mqa=mqa)}):
            self.assertIs(code.qsa_mqa_prefill(q, k, bounds, bounds), expected)
            self.assertIs(code.qsa_mqa_decode(q, k, bounds, bounds, 2048), expected)
            with self.assertRaises(ValueError):
                code.qsa_mqa_prefill(q, k, bounds, bounds, score_scale=1.0)

    def test_expansion_routes_npu_before_cuda_alias(self):
        code = load(
            QSA + "kernel.py",
            {"expand_qsa_block_indices": None},
            {"triton_expand_qsa_block_indices": forbidden},
        )
        blocks, bounds = (
            DeviceAlias((2, 512), torch.int32),
            DeviceAlias((2,), torch.int32),
        )
        expected = torch.arange(8).reshape(2, 4)
        with patch.dict(
            sys.modules,
            {KERNELS + ".qsa_expansion": NS(expand_blocks=lambda *a: expected)},
        ):
            self.assertIs(
                code.expand_qsa_block_indices(blocks, bounds, bounds, 4, 2048), expected
            )

    def test_indexer_disables_cuda_fused_preparation(self):
        code = load(QSA + "qsa_indexer.py", {"QSAIndexer": {"_use_fused_prep"}})
        owner = NS(
            index_head_dim=128,
            rotary_emb=NS(
                rotary_dim=64,
                mrope_interleaved_glm=False,
                mrope_section=[11, 11, 10],
                cos_sin_cache=DeviceAlias(dtype=torch.float32),
            ),
        )
        self.assertFalse(code.QSAIndexer._use_fused_prep(owner, DeviceAlias()))

    def test_indexer_npu_uses_shared_forward(self):
        code = load(
            QSA + "qsa_indexer.py", {"QSAIndexer": {"forward_cuda", "forward_npu"}}
        )
        self.assertTrue(
            hasattr(code.QSAIndexer, "forward_npu"), "QSA indexer has no NPU forward"
        )
        args = tuple(object() for _ in range(4))
        owner = NS(_forward_impl=lambda *values: values)
        self.assertEqual(code.QSAIndexer.forward_npu(owner, *args), args + (None,))
        positions = object()
        self.assertEqual(
            code.QSAIndexer.forward_npu(
                owner, *args, cp_global_rope_positions=positions
            ),
            args + (positions,),
        )

    def test_npu_replay_does_not_import_cuda_metadata(self):
        code = load(
            BACKEND, {"QwenSparseAttnBackend": {"_can_replay_with_gpu_kernels"}}
        )
        owner = NS(req_to_token=object())
        metadata = NS(indexer_metadata=NS(token_to_kv_pool=object()))
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.layers.attention.qsa.graph_metadata": NS(
                    supports_graph_metadata_kernels=forbidden
                )
            },
        ):
            self.assertFalse(
                code.QwenSparseAttnBackend._can_replay_with_gpu_kernels(
                    owner, metadata, DeviceAlias()
                )
            )

    def test_paged_attention_uses_npu_with_cuda_alias(self):
        expected = torch.arange(2 * 3 * 256).reshape(2, 3, 256)
        code = load(
            BACKEND,
            {"QwenSparseAttnBackend": {"_forward_paged_attention"}},
            {
                "_resolve_trtllm_sparse_decode": forbidden,
                "_npu_sparse_attention": lambda *args: expected,
            },
        )
        pool = NS(
            layer_router=NS(
                read=lambda layer: NS(key=lambda: object(), value=lambda: object())
            )
        )
        owner = NS(
            token_to_kv_pool=pool,
            _resolve_metadata=lambda batch: object(),
            _expand_block_indices=lambda x, m: x,
            _logical_to_physical=lambda x, m: x,
        )
        actual = code.QwenSparseAttnBackend._forward_paged_attention(
            owner,
            DeviceAlias((2, 3, 256)),
            NS(layer_id=0, scaling=0.0625),
            object(),
            torch.zeros((2, 2051), dtype=torch.int32),
        )
        torch.testing.assert_close(actual, expected.reshape(2, -1))


class TestQSAReplayMapping(CustomTestCase):
    def test_npu_graph_update_keeps_dense_fallback_on_fia_path(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.qsa import config as qsa_config

        code = load(
            "srt/hardware_backend/npu/graph_runner/npu_graph_runner.py",
            {"NPUGraphRunner": {"execute", "_requires_attention_input_update"}},
            {
                "is_deepseek_dsa": lambda c: False,
                "is_deepseek_v4": lambda c: False,
                "get_active_qsa_profile": qsa_config.get_active_qsa_profile,
            },
        )
        self.assertTrue(
            hasattr(code.NPUGraphRunner, "_requires_attention_input_update"),
            "NPU graph runner does not distinguish active QSA from dense fallback",
        )
        config = NS(
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
        )
        owner = NS(model_runner=NS(model_config=NS(hf_config=config)))
        with patch.object(qsa_config, "is_npu", return_value=True):
            for fallback in (False, True):
                with envs.SGLANG_ENABLE_QWEN4_NPU_DENSE_FALLBACK.override(fallback):
                    self.assertEqual(
                        code.NPUGraphRunner._requires_attention_input_update(owner),
                        fallback,
                    )
        owner.model_runner.model_config.hf_config = NS()
        self.assertTrue(code.NPUGraphRunner._requires_attention_input_update(owner))

    def test_graph_mapping_reads_live_request_ids_and_slots(self):
        code = load(BACKEND, {"QwenSparseAttnBackend": {"_logical_to_physical"}})
        metadata = NS(
            sequence_lengths=torch.tensor([3, 2]),
            token_to_batch_idx=torch.tensor([0, 1]),
            token_slot_table=torch.zeros((2, 1), dtype=torch.int32),
            is_cuda_graph=True,
            row_req_pool_indices=torch.tensor([2, 0]),
            req_to_token=torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
            ),
        )
        indices = torch.tensor([[0, 2, -1], [1, 2, -1]])
        actual = code.QwenSparseAttnBackend._logical_to_physical(indices, metadata)
        torch.testing.assert_close(
            actual, torch.tensor([[30, 32, -1], [11, -1, -1]], dtype=torch.int32)
        )
        metadata.row_req_pool_indices.copy_(torch.tensor([1, 2]))
        metadata.req_to_token[1, 2] = 99
        actual = code.QwenSparseAttnBackend._logical_to_physical(indices, metadata)
        torch.testing.assert_close(
            actual, torch.tensor([[20, 99, -1], [31, -1, -1]], dtype=torch.int32)
        )

    def test_eager_mapping_retains_per_batch_table(self):
        code = load(BACKEND, {"QwenSparseAttnBackend": {"_logical_to_physical"}})
        metadata = NS(
            sequence_lengths=torch.tensor([2]),
            token_to_batch_idx=torch.tensor([0]),
            token_slot_table=torch.tensor([[7, 9, 11]]),
            is_cuda_graph=False,
        )
        result = code.QwenSparseAttnBackend._logical_to_physical(
            torch.tensor([[1, 2, -1]]), metadata
        )
        torch.testing.assert_close(
            result, torch.tensor([[9, -1, -1]], dtype=torch.int32)
        )

    def test_npu_cache_view_preserves_storage_and_rejects_copy(self):
        tree = ast.parse((ROOT / BACKEND).read_text())
        self.assertTrue(
            any(getattr(n, "name", None) == "_flatten_qsa_kv_cache" for n in tree.body),
            "NPU cache layout adapter is missing",
        )
        code = load(BACKEND, {"_flatten_qsa_kv_cache": None})
        cache = torch.arange(2 * 3 * 1 * 8).reshape(2, 3, 1, 8)
        flat = code._flatten_qsa_kv_cache(cache, "K")
        self.assertEqual(flat.data_ptr(), cache.data_ptr())
        cache[1, 2, 0, 3] = 1234
        self.assertEqual(flat[5, 0, 3].item(), 1234)
        with self.assertRaises(ValueError):
            code._flatten_qsa_kv_cache(flat, "K")
        with self.assertRaises(RuntimeError):
            code._flatten_qsa_kv_cache(cache.transpose(0, 1), "K")


if __name__ == "__main__":
    unittest.main()

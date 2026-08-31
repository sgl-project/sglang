import subprocess
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

with patch.dict(
    sys.modules,
    {
        module: MagicMock()
        for module in (
            "sgl_kernel",
            "sgl_kernel.quantization",
            "sgl_kernel.scalar_type",
        )
    },
):
    from sglang.srt.layers.attention import attention_registry
    from sglang.srt.layers.attention.minicpm import backend as backend_module
    from sglang.srt.layers.attention.minicpm import sparse_utils
    from sglang.srt.layers.attention.minicpm.attention_adapter import (
        MiniCPMFlashAttentionAdapter,
    )
    from sglang.srt.layers.attention.minicpm.backend import (
        MiniCPMSparseBackend,
        _gather_compressed_keys,
        _transpose_head_group_layout,
    )
    from sglang.srt.layers.attention.minicpm.sparse_utils import (
        CompressionLevelMetadata,
    )
    from sglang.srt.runtime_context import get_context, get_schedule

register_cpu_ci(est_time=24, suite="base-a-test-cpu")


def _compression_layout():
    return SimpleNamespace(
        k1_kernel_size=32,
        k1_kernel_stride=16,
        k2_kernel_size=128,
        k2_kernel_stride=64,
    )


def _construct_sparse_backend(
    *,
    max_context_len=256,
    chunked_prefill_size=64,
    max_running_requests=1,
    use_flashinfer=False,
    blackwell=False,
):
    req_pool = SimpleNamespace(
        req_to_sparse_k1_token=torch.empty(0),
        req_to_sparse_k2_token=torch.empty(0),
    )
    flash_attn_backend = SimpleNamespace(
        max_context_len=max_context_len,
        device="cpu",
        decode_cuda_graph_metadata={},
        req_to_token_pool=req_pool,
        token_to_kv_pool=SimpleNamespace(),
        page_size=1,
    )
    model_runner = SimpleNamespace(
        dtype=torch.float16,
        max_running_requests=max_running_requests,
        token_to_kv_pool_allocator=SimpleNamespace(),
        server_args=SimpleNamespace(
            enable_memory_saver=False,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                has_minicpm_sparse_attention=True,
                sparse_config={
                    "kernel_size": 32,
                    "kernel_stride": 16,
                    "init_blocks": 1,
                    "block_size": 64,
                    "window_size": 64,
                    "dense_len": 128,
                    "topk": 1,
                },
            ),
            num_attention_heads=16,
            head_dim=128,
            get_num_kv_heads=lambda _tp: 1,
        ),
    )
    with (
        get_schedule().override(chunked_prefill_size=chunked_prefill_size),
        patch.object(backend_module, "MiniCPMHybridConfig", SimpleNamespace),
        override_platform(is_blackwell=blackwell),
        patch.object(
            backend_module,
            "FlashAttentionBackend",
            return_value=flash_attn_backend,
        ) as flash_attention,
        patch.object(
            backend_module,
            "MiniCPMFlashInferAdapter",
            return_value=object(),
        ),
        patch.object(
            backend_module,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_size=1),
        ),
        patch.object(backend_module, "attach_compressed_cache"),
    ):
        backend = MiniCPMSparseBackend(model_runner, use_flashinfer=use_flashinfer)
    return backend, model_runner, flash_attn_backend, flash_attention


class _DeviceOffsetsMustNotBeRead:
    def __getitem__(self, _index):
        raise AssertionError("prefill layers must use scheduler-derived CPU offsets")


class _GraphTensorMustNotUseHostListIndex:
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        if isinstance(index, list):
            raise AssertionError("CUDA graph tensors must not use host list indices")
        return self.tensor[index]


class _SingleTensorConversion:
    def __init__(self, values):
        self.values = values
        self.first_item_reads = 0

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if index == 0:
            self.first_item_reads += 1
            if self.first_item_reads > 2:
                raise AssertionError("sequence lengths were converted more than once")
        if index >= len(self.values):
            raise IndexError
        return self.values[index]


class TestMiniCPMSparseMetadata(CustomTestCase):
    def setUp(self):
        super().setUp()
        # The backend reads chunked_prefill_size off the schedule bag, so the
        # context has to be published before any construction; the helper
        # scopes a different value on top of this one where a case needs it.
        override = get_context().override_server_args(chunked_prefill_size=64)
        override.install()
        self.addCleanup(override.restore)

    def test_sparse_backend_rejects_context_too_short_for_layout(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires context_length >= 128, got 64",
        ):
            _construct_sparse_backend(max_context_len=64)

    def test_fused_topk_rejects_disabled_chunked_prefill(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires a positive --chunked-prefill-size",
        ):
            _construct_sparse_backend(
                chunked_prefill_size=-1,
                use_flashinfer=True,
                blackwell=True,
            )

    def test_gathered_compressed_offsets_stay_int32(self):
        compressed = torch.arange(5).reshape(5, 1, 1)
        level = SimpleNamespace(cu_seqlens_cpu=[0, 2, 5])

        _, cu_seqlens = _gather_compressed_keys(compressed, level, [1])

        self.assertEqual(cu_seqlens.dtype, torch.int32)
        self.assertEqual(cu_seqlens.tolist(), [0, 3])

    def test_registered_variants_select_adapter_explicitly(self):
        runner = object()

        def build(_runner, *, use_flashinfer):
            self.assertIs(_runner, runner)
            return use_flashinfer

        with (
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.layers.attention.minicpm.backend": backend_module,
                },
            ),
            patch.object(
                backend_module,
                "MiniCPMSparseBackend",
                side_effect=build,
            ),
        ):
            flashattn = attention_registry.ATTENTION_BACKENDS["minicpm_flashattn"](
                runner
            )
            flashinfer = attention_registry.ATTENTION_BACKENDS["minicpm_flashinfer"](
                runner
            )

        self.assertFalse(flashattn)
        self.assertTrue(flashinfer)

    def test_sparse_metadata_does_not_patch_base_metadata(self):
        base_metadata = SimpleNamespace()
        metadata_type = getattr(sparse_utils, "MiniCPMSparseMetadata")

        metadata = metadata_type(base=base_metadata)
        metadata.sparse_bs_list = [0]

        self.assertEqual(metadata.sparse_bs_list, [0])
        self.assertFalse(hasattr(base_metadata, "sparse_bs_list"))

    def test_head_group_layout_round_trip(self):
        tensor = torch.arange(10).reshape(5, 2, 1)
        original = tensor.clone()

        _transpose_head_group_layout(
            tensor,
            [(1, 2)],
            head_group_num=2,
            heads_per_group=2,
            to_group_major=True,
        )
        self.assertEqual(
            tensor.squeeze(-1).tolist(),
            [[0, 1], [2, 3], [6, 7], [4, 5], [8, 9]],
        )

        _transpose_head_group_layout(
            tensor,
            [(1, 2)],
            head_group_num=2,
            heads_per_group=2,
            to_group_major=False,
        )
        self.assertTrue(torch.equal(tensor, original))

    def test_flashattn_variant_uses_fa4_on_blackwell(self):
        """Blackwell must select FA4 because FA3 binaries cannot execute there."""
        backend, model_runner, flash_attn_backend, flash_attention = (
            _construct_sparse_backend(blackwell=True)
        )
        model_config = model_runner.model_config

        flash_attention.assert_called_once_with(
            model_runner,
            skip_prefill=False,
            fa_impl_ver=4,
        )
        self.assertIs(backend.flash_attn_backend, flash_attn_backend)
        self.assertIs(
            backend.token_to_kv_pool,
            flash_attn_backend.token_to_kv_pool,
        )
        self.assertIsInstance(
            backend.attention_adapter,
            MiniCPMFlashAttentionAdapter,
        )
        self.assertEqual(backend.fused_kernel_kwargs["dtype_str"], "float16")
        self.assertEqual(backend.fused_kernel_kwargs["kernel_stride"], 16)

        model_runner.server_args.attention_backend = "minicpm_flashinfer"
        flashinfer_adapter = object()
        fake_fuse_kernel = ModuleType("sglang.srt.layers.attention.minicpm.fuse_kernel")
        fake_fuse_kernel.fused_attn_pooling_online_topk_prefill = Mock(
            return_value="prefill"
        )
        fake_fuse_kernel.fused_attn_pooling_online_topk_decode = Mock(
            return_value="decode"
        )
        with (
            patch.dict(
                sys.modules,
                {"sglang.srt.layers.attention.minicpm.fuse_kernel": fake_fuse_kernel},
            ),
            patch.object(backend_module, "MiniCPMHybridConfig", SimpleNamespace),
            override_platform(is_blackwell=True),
            patch.object(
                backend_module,
                "FlashAttentionBackend",
                return_value=flash_attn_backend,
            ),
            patch.object(
                backend_module,
                "MiniCPMFlashInferAdapter",
                return_value=flashinfer_adapter,
            ),
            patch.object(
                backend_module,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1),
            ),
            patch.object(backend_module, "attach_compressed_cache"),
        ):
            backend = MiniCPMSparseBackend(model_runner, use_flashinfer=True)

        self.assertIs(backend.flash_attn_backend, flash_attn_backend)
        self.assertIs(backend.attention_adapter, flashinfer_adapter)

        model_config.num_attention_heads = 8
        with (
            patch.object(backend_module, "MiniCPMHybridConfig", SimpleNamespace),
            override_platform(is_blackwell=True),
            patch.object(
                backend_module,
                "FlashAttentionBackend",
                return_value=flash_attn_backend,
            ),
            patch.object(
                backend_module,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1),
            ),
            patch.object(backend_module, "attach_compressed_cache"),
            self.assertRaisesRegex(ValueError, "16 query heads per KV head"),
        ):
            MiniCPMSparseBackend(model_runner, use_flashinfer=True)

        model_runner.server_args.attention_backend = "minicpm_flashattn"
        with (
            patch.object(backend_module, "MiniCPMHybridConfig", SimpleNamespace),
            override_platform(is_blackwell=False),
            patch.object(
                backend_module,
                "FlashAttentionBackend",
                return_value=flash_attn_backend,
            ),
            patch.object(
                backend_module,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1),
            ),
            patch.object(backend_module, "attach_compressed_cache"),
            self.assertRaisesRegex(ValueError, "16 query heads per KV head"),
        ):
            MiniCPMSparseBackend(model_runner, use_flashinfer=False)

    def test_dense_as_sparse_routes_short_prefill(self):
        req_pool = SimpleNamespace(
            req_to_sparse_k1_token=torch.empty(0),
            req_to_sparse_k2_token=torch.empty(0),
        )
        flash_attn_backend = SimpleNamespace(
            max_context_len=256,
            device="cpu",
            decode_cuda_graph_metadata={},
            req_to_token_pool=req_pool,
            token_to_kv_pool=SimpleNamespace(),
            page_size=1,
        )
        hf_config = SimpleNamespace(
            has_minicpm_sparse_attention=True,
            sparse_config={
                "kernel_size": 32,
                "kernel_stride": 16,
                "init_blocks": 1,
                "block_size": 64,
                "window_size": 64,
                "dense_len": 128,
                "topk": 1,
            },
        )
        model_runner = SimpleNamespace(
            dtype=torch.float16,
            token_to_kv_pool_allocator=SimpleNamespace(),
            server_args=SimpleNamespace(
                attention_backend="minicpm_flashattn",
                disable_cuda_graph=False,
                enable_memory_saver=False,
            ),
            model_config=SimpleNamespace(
                hf_config=hf_config,
                num_attention_heads=16,
                head_dim=128,
                get_num_kv_heads=lambda _tp: 1,
            ),
        )

        with (
            backend_module.envs.SGLANG_MINICPM_DENSE_AS_SPARSE.override(True),
            patch.object(backend_module, "MiniCPMHybridConfig", SimpleNamespace),
            override_platform(is_blackwell=False),
            patch.object(
                backend_module,
                "FlashAttentionBackend",
                return_value=flash_attn_backend,
            ),
            patch.object(
                backend_module,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1),
            ),
            patch.object(backend_module, "attach_compressed_cache"),
        ):
            backend = MiniCPMSparseBackend(model_runner, use_flashinfer=False)

        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
            seq_lens=torch.tensor([1], dtype=torch.int32),
            extend_seq_lens_cpu=[1],
            extend_prefix_lens_cpu=[0],
            forward_mode=SimpleNamespace(
                is_extend_or_draft_extend_or_mixed=lambda: True
            ),
        )
        metadata = sparse_utils.MiniCPMSparseMetadata(
            base=SimpleNamespace(
                cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
                cache_seqlens_int32=torch.tensor([1], dtype=torch.int32),
                page_table=torch.zeros((1, 1), dtype=torch.int32),
                max_seq_len_q=1,
            )
        )
        level = CompressionLevelMetadata()
        with patch.object(
            backend_module,
            "_build_k1_k2_compression_metadata",
            return_value=(level, level),
        ):
            backend.update_batch_for_sparse(forward_batch, metadata)

        self.assertEqual(backend.dense_len, 0)
        self.assertEqual(metadata.sparse_bs_list, [0])

    def test_dense_prefill_page_table_covers_total_sequence(self):
        """Dense prefill must retain page-table coverage for the full sequence."""
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=torch.tensor([7000], dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([2904], dtype=torch.int32),
            extend_prefix_lens_cpu=[4096],
        )
        metadata = sparse_utils.MiniCPMSparseMetadata(
            base=SimpleNamespace(
                cu_seqlens_q=torch.tensor([0, 2904], dtype=torch.int32),
                cache_seqlens_int32=torch.tensor([7000], dtype=torch.int32),
                page_table=torch.zeros((1, 7000), dtype=torch.int32),
                max_seq_len_q=2904,
            )
        )

        sparse_utils._plan_sparse_prefill(
            forward_batch,
            metadata,
            head_group_num=2,
            heads_per_group=16,
            dense_len=8192,
            sparse_topk=96,
            block_size=64,
        )

        self.assertEqual(metadata.sparse_page_table.shape, (2, 7000))

    def test_prefill_metadata_builds_layer_invariant_cache_lengths(self):
        """Sparse cache lengths must not be inferred from zero-valued table entries."""
        forward_batch = SimpleNamespace(
            batch_size=2,
            seq_lens_cpu=torch.tensor([200, 64], dtype=torch.int32),
            extend_seq_lens_cpu=[2, 3],
            extend_prefix_lens_cpu=[198, 61],
        )
        metadata = sparse_utils.MiniCPMSparseMetadata(
            base=SimpleNamespace(
                cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
                cache_seqlens_int32=torch.tensor([200, 64], dtype=torch.int32),
                page_table=torch.zeros((2, 200), dtype=torch.int32),
                max_seq_len_q=3,
            )
        )

        sparse_utils._plan_sparse_prefill(
            forward_batch,
            metadata,
            head_group_num=2,
            heads_per_group=16,
            dense_len=100,
            sparse_topk=2,
            block_size=64,
        )

        self.assertEqual(
            metadata.sparse_cache_seqlens_int32.tolist(),
            [71, 71, 72, 72, 64, 64],
        )
        self.assertEqual(
            metadata.sparse_cu_seqlens_k.tolist(),
            [0, 71, 142, 214, 286, 350, 414],
        )

    def test_prefill_planning_builds_mixed_batch_layout(self):
        forward_batch = SimpleNamespace(
            batch_size=2,
            seq_lens_cpu=torch.tensor([200, 64], dtype=torch.int32),
            extend_seq_lens_cpu=[2, 3],
            extend_prefix_lens_cpu=[198, 61],
        )
        metadata = sparse_utils.MiniCPMSparseMetadata(
            base=SimpleNamespace(
                cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
                cache_seqlens_int32=torch.tensor([200, 64], dtype=torch.int32),
                page_table=torch.zeros((2, 200), dtype=torch.int32),
                max_seq_len_q=3,
            )
        )

        sparse_utils._plan_sparse_prefill(
            forward_batch,
            metadata,
            head_group_num=2,
            heads_per_group=16,
            dense_len=100,
            sparse_topk=2,
            block_size=64,
        )

        self.assertEqual(metadata.sparse_bs_list, [0])
        self.assertEqual(metadata.sparse_idx, [0, 1, 2, 3])
        self.assertEqual(metadata.dense_layout, [(1, 4, 4, 3)])
        self.assertEqual(metadata.token_to_bs.tolist(), [0, 0])
        self.assertEqual(metadata.token_pos_in_bs.tolist(), [199, 200])
        self.assertEqual(
            metadata.sparse_cu_seqlens_q.tolist(),
            [0, 1, 2, 3, 4, 7, 10],
        )
        self.assertEqual(
            metadata.sparse_cache_seqlens_int32.tolist(),
            [71, 71, 72, 72, 64, 64],
        )
        self.assertEqual(metadata.topk_cu_seqlens_q.tolist(), [0, 2])
        self.assertEqual(metadata.topk_cu_seqlens_k.tolist(), [0, 200])

    def test_mixed_prefill_compacts_stage1_cache_lengths(self):
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        backend.req_to_sparse_k1_token = torch.empty(0)
        backend.req_to_sparse_k2_token = torch.empty(0)
        backend.k1_kernel_size = 32
        backend.k1_kernel_stride = 16
        backend.k2_kernel_size = 128
        backend.k2_kernel_stride = 64
        backend.dense_len = 100
        backend.head_group_num = 1
        backend.sparse_topk = 2
        backend.block_size = 64
        backend.heads_per_group = 16

        forward_batch = SimpleNamespace(
            batch_size=2,
            seq_lens_cpu=torch.tensor([50, 200], dtype=torch.int32),
            seq_lens=torch.tensor([50, 200], dtype=torch.int32),
            extend_seq_lens_cpu=[1, 1],
            extend_prefix_lens_cpu=[49, 199],
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            forward_mode=SimpleNamespace(
                is_extend_or_draft_extend_or_mixed=lambda: True
            ),
        )
        metadata = sparse_utils.MiniCPMSparseMetadata(
            base=SimpleNamespace(
                cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
                cache_seqlens_int32=torch.tensor([50, 200], dtype=torch.int32),
                page_table=torch.arange(1, 401, dtype=torch.int32).reshape(2, 200),
                max_seq_len_q=1,
            )
        )
        level = CompressionLevelMetadata()

        with patch.object(
            backend_module,
            "_build_k1_k2_compression_metadata",
            return_value=(level, level),
        ):
            backend.update_batch_for_sparse(forward_batch, metadata)

        self.assertEqual(metadata.sparse_bs_list, [1])
        self.assertEqual(metadata.cache_seqlens_int32_stage1.tolist(), [199])
        self.assertEqual(
            metadata.sparse_page_table[0, :50].tolist(),
            metadata.base.page_table[0, :50].tolist(),
        )

    def test_dense_decode_page_table_matches_batch_length(self):
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=torch.tensor([7000], dtype=torch.int32),
        )
        base_metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([7000], dtype=torch.int32),
            page_table=torch.empty((1, 7000), dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        )

        metadata = sparse_utils.MiniCPMSparseMetadata(base=base_metadata)
        sparse_utils._plan_sparse_decode(
            forward_batch=forward_batch,
            metadata=metadata,
            head_group_num=2,
            dense_len=8192,
            sparse_topk=96,
            block_size=64,
        )

        self.assertEqual(metadata.sparse_page_table.shape, (2, 7000))

    def test_mixed_prefill_uses_compact_sparse_page_table(self):
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        q = torch.ones(2, 1)
        k = torch.ones(2, 1, 1)
        v = torch.ones(2, 1, 1)
        key_cache = torch.ones(4, 1, 1, 1)
        value_cache = torch.ones(4, 1, 1, 1)
        backend.flash_attn_backend = SimpleNamespace(
            prepare_paged_mha_query=Mock(return_value=(q, None, None, None, None)),
            get_paged_mha_kv_cache=Mock(return_value=(key_cache, value_cache)),
        )
        backend.token_to_kv_pool = SimpleNamespace(set_kv_buffer=Mock())
        backend.attention_adapter = SimpleNamespace(
            forward=Mock(return_value=torch.ones(2, 1, 1))
        )
        backend.forward_metadata = sparse_utils.MiniCPMSparseMetadata(
            base=SimpleNamespace(
                page_table=torch.tensor([[10, 0], [20, 21]], dtype=torch.int32),
            ),
            sparse_bs_list=[1],
            sparse_idx=[1],
            dense_layout=[(0, 0, 0, 1)],
            sparse_page_table=torch.tensor([[10, 0], [0, 0]], dtype=torch.int32),
            token_to_bs=torch.tensor([0], dtype=torch.int32),
            token_pos_in_bs=torch.tensor([2], dtype=torch.int32),
            seqlen_k_sparse_bs_tensor=torch.tensor([2], dtype=torch.int32),
        )
        backend.head_group_num = 1
        backend.heads_per_group = 1
        backend.block_size = 1
        backend.num_sparse_topk_tokens = 1
        backend.get_topk_for_sparse = Mock(
            return_value=torch.tensor([[[0]]], dtype=torch.int32)
        )
        layer = SimpleNamespace(
            is_cross_attention=False,
            sliding_window_size=-1,
            tp_q_head_num=1,
            tp_k_head_num=1,
            head_dim=1,
            k_scale=None,
            v_scale=None,
        )
        forward_batch = SimpleNamespace(
            batch_size=2,
            seq_lens_cpu=torch.tensor([1, 2], dtype=torch.int32),
            extend_seq_lens_cpu=[1, 1],
            out_cache_loc=torch.tensor([0, 1], dtype=torch.int64),
            forward_mode=SimpleNamespace(is_draft_extend_v2=lambda: False),
        )

        def get_sparse_page_table(_topk, page_table, *_args, **_kwargs):
            self.assertEqual(page_table.tolist(), [[20, 21]])
            self.assertFalse(_kwargs["elementwise"])
            return torch.tensor([[21]], dtype=torch.int32)

        with patch.object(
            backend_module,
            "get_block_table",
            side_effect=get_sparse_page_table,
        ):
            backend.forward_extend(q, k, v, layer, forward_batch)

        self.assertEqual(
            backend.forward_metadata.sparse_page_table.tolist(),
            [[10, 0], [21, 0]],
        )

    def test_dense_decode_copies_full_page_table(self):
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        q = torch.ones(1, 1)
        k = torch.ones(1, 1, 1)
        v = torch.ones(1, 1, 1)
        key_cache = torch.ones(4, 1, 1, 1)
        value_cache = torch.ones(4, 1, 1, 1)
        backend.flash_attn_backend = SimpleNamespace(
            prepare_paged_mha_query=Mock(return_value=(q, None, None, None, None)),
            get_paged_mha_kv_cache=Mock(return_value=(key_cache, value_cache)),
            forward_decode=Mock(),
        )
        backend.token_to_kv_pool = SimpleNamespace(set_kv_buffer=Mock())
        backend.attention_adapter = SimpleNamespace(
            forward=Mock(return_value=torch.ones(1, 1, 1))
        )
        backend.forward_metadata = sparse_utils.MiniCPMSparseMetadata(
            base=SimpleNamespace(
                page_table=torch.tensor([[5, 6, 7, 0]], dtype=torch.int32),
                cache_seqlens_int32=torch.tensor([3], dtype=torch.int32),
                max_seq_len_q=1,
            ),
            sparse_bs_list=[],
            sparse_idx=[],
            sparse_page_table=torch.tensor([[5, 6, 7, 0]], dtype=torch.int32),
            sparse_cache_seqlens_int32=torch.tensor([3], dtype=torch.int32),
            sparse_cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            sparse_cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32),
            token_to_bs=torch.tensor([0], dtype=torch.int32),
        )
        backend.head_group_num = 1
        backend.heads_per_group = 1
        backend.page_size = 1
        backend.block_size = 1
        backend.num_sparse_topk_tokens = 2
        backend.dense_len = 4
        backend._use_cuda_graph_buffers = False
        backend._compress_decode_keys = Mock()
        backend.get_topk_for_sparse = Mock(return_value=None)
        layer = SimpleNamespace(
            is_cross_attention=False,
            sliding_window_size=-1,
            tp_q_head_num=1,
            tp_k_head_num=1,
            tp_v_head_num=1,
            head_dim=1,
            v_head_dim=1,
            k_scale=None,
            v_scale=None,
        )
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
            out_cache_loc=torch.tensor([1], dtype=torch.int64),
        )

        with patch.object(
            backend_module,
            "get_block_table",
        ) as get_block_table:
            backend.forward_decode(q, k, v, layer, forward_batch)

        backend.get_topk_for_sparse.assert_called_once()
        get_block_table.assert_not_called()
        backend._compress_decode_keys.assert_not_called()
        backend.attention_adapter.forward.assert_called_once()
        backend.flash_attn_backend.forward_decode.assert_not_called()
        self.assertEqual(
            backend.forward_metadata.sparse_page_table[0, :3].tolist(),
            [5, 6, 7],
        )

    def test_graph_decode_preserves_dense_rows_from_sparse_topk(self):
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        q = torch.ones(2, 1)
        key_cache = torch.ones(8, 1, 1, 1)
        value_cache = torch.ones(8, 1, 1, 1)
        backend.flash_attn_backend = SimpleNamespace(
            prepare_paged_mha_query=Mock(return_value=(q, None, None, None, None)),
            get_paged_mha_kv_cache=Mock(return_value=(key_cache, value_cache)),
        )
        backend.token_to_kv_pool = SimpleNamespace(set_kv_buffer=Mock())
        backend.attention_adapter = SimpleNamespace(
            forward=Mock(return_value=torch.ones(2, 1, 1))
        )
        backend.forward_metadata = sparse_utils.MiniCPMSparseMetadata(
            base=SimpleNamespace(
                page_table=_GraphTensorMustNotUseHostListIndex(
                    torch.tensor([[5, 6, 7, 0, 0], [8, 9, 10, 11, 12]])
                ),
                cache_seqlens_int32=torch.tensor([3, 5], dtype=torch.int32),
            ),
            sparse_bs_list=[0, 1],
            sparse_idx=[0, 1],
            token_to_bs=torch.tensor([0, 1], dtype=torch.int32),
            sparse_page_table=torch.tensor([[5, 6, 7, 0, 0], [0, 0, 0, 0, 0]]),
        )
        backend.head_group_num = 1
        backend.heads_per_group = 1
        backend.page_size = 1
        backend.block_size = 1
        backend.num_sparse_topk_tokens = 2
        backend.dense_len = 4
        backend._use_cuda_graph_buffers = True
        backend.get_topk_for_sparse = Mock(
            return_value=torch.tensor([[[0, 1], [0, 1]]], dtype=torch.int32)
        )
        layer = SimpleNamespace(
            is_cross_attention=False,
            sliding_window_size=-1,
            tp_q_head_num=1,
            tp_k_head_num=1,
            tp_v_head_num=1,
            head_dim=1,
            v_head_dim=1,
            k_scale=None,
            v_scale=None,
        )
        forward_batch = SimpleNamespace(
            batch_size=2,
            out_cache_loc=torch.tensor([1, 2], dtype=torch.int64),
        )

        with patch.object(
            backend_module,
            "get_block_table",
            return_value=torch.tensor([[30, 31], [40, 41]], dtype=torch.int32),
        ):
            backend.forward_decode(
                q,
                torch.ones(2, 1, 1),
                torch.ones(2, 1, 1),
                layer,
                forward_batch,
            )

        self.assertEqual(
            backend.forward_metadata.sparse_page_table[:, :3].tolist(),
            [[5, 6, 7], [40, 41, 0]],
        )

    def test_decode_metadata_uses_scheduler_cpu_lengths(self):
        """Decode metadata must not synchronize device offsets to recover lengths."""
        forward_batch = SimpleNamespace(
            batch_size=2,
            seq_lens_cpu=torch.tensor([64, 200], dtype=torch.int32),
        )
        base_metadata = SimpleNamespace(
            cache_seqlens_int32=SimpleNamespace(
                dtype=torch.int32,
                device=torch.device("cpu"),
            ),
            page_table=torch.empty((2, 200), dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        )

        metadata = sparse_utils.MiniCPMSparseMetadata(base=base_metadata)
        sparse_utils._plan_sparse_decode(
            forward_batch=forward_batch,
            metadata=metadata,
            head_group_num=2,
            dense_len=100,
            sparse_topk=2,
            block_size=64,
        )

        self.assertEqual(
            metadata.sparse_cache_seqlens_int32.tolist(),
            [64, 64, 72, 72],
        )
        self.assertEqual(metadata.sparse_bs_list, [1])
        self.assertEqual(metadata.sparse_idx, [2, 3])
        self.assertEqual(metadata.dense_layout, [(0, 0, 0, 1)])
        self.assertEqual(metadata.token_to_bs.tolist(), [0])
        self.assertEqual(metadata.topk_cu_seqlens_q.tolist(), [0, 1])
        self.assertEqual(metadata.sparse_page_table.shape, (4, 128))

    def test_cuda_graph_page_table_covers_dense_decode(self):
        """Captured dense decode must reserve a threshold-sized page table."""
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        backend.flash_attn_backend = SimpleNamespace(
            decode_cuda_graph_metadata={},
            init_cuda_graph_state=lambda *_: None,
        )
        backend.attention_adapter = SimpleNamespace(
            init_cuda_graph_state=lambda *_: None,
        )
        backend.num_sparse_topk_tokens = 6144
        backend.page_size = 1
        backend.head_group_num = 2
        backend.device = "cpu"
        backend.model_dtype = torch.float16
        backend.heads_per_group = 16
        backend.max_context_len = 256
        backend.config_dense_len = 8192
        backend.dense_len = 8192
        backend.head_dim = 128
        backend.k1_kernel_size = 32
        backend.k1_kernel_stride = 16
        backend.k2_kernel_size = 128
        backend.k2_kernel_stride = 64

        backend.init_cuda_graph_state(max_bs=1, max_num_tokens=1)

        self.assertEqual(
            backend.decode_cuda_graph_metadata["sparse_page_table"].shape,
            (2, 8192),
        )
        self.assertEqual(
            backend.decode_cuda_graph_metadata["compress_k1"].dtype,
            torch.float16,
        )
        for level in ("k1", "k2"):
            for field in (
                "new_token_nums",
                "new_compress_token_nums",
                "cu_new_compress_token_nums",
                "total_compress_token_nums",
            ):
                self.assertNotIn(
                    f"{level}.{field}",
                    backend.decode_cuda_graph_metadata,
                )

        base_metadata = SimpleNamespace(
            cu_seqlens_k=torch.zeros(2, dtype=torch.int32),
            max_seq_len_k=0,
            max_seq_len_q=1,
        )
        capture_metadata = sparse_utils.MiniCPMSparseMetadata(base=base_metadata)
        forward_batch = SimpleNamespace(batch_size=1)
        backend._bind_sparse_graph_metadata(
            forward_batch,
            capture_metadata,
            in_capture=True,
        )
        self.assertIsNotNone(capture_metadata.k1)
        self.assertEqual(capture_metadata.k1.cu_seqlens_cpu, [0, 511])
        self.assertEqual(capture_metadata.k2.cu_seqlens_cpu, [0, 127])
        self.assertFalse(hasattr(base_metadata, "k1"))

        base_metadata.cu_seqlens_k.copy_(torch.tensor([0, 7], dtype=torch.int32))
        replay_metadata = sparse_utils.MiniCPMSparseMetadata(base=base_metadata)
        backend._bind_sparse_graph_metadata(
            forward_batch,
            replay_metadata,
            in_capture=False,
        )
        self.assertEqual(base_metadata.cu_seqlens_k.tolist(), [0, 7])

    def test_compression_uses_configured_k1_k2_layout(self):
        """K1/K2 compression must honor checkpoint strides instead of fixed defaults."""
        layer = SimpleNamespace(layer_id=0, tp_k_head_num=1, head_dim=1)
        forward_batch = SimpleNamespace(req_pool_indices=[0])
        level = CompressionLevelMetadata(
            table=torch.empty(0),
            history_compress_token_nums=torch.empty(0),
            cu_new_token_nums=torch.empty(0),
            cu_total_compress_token_nums=torch.empty(0),
        )
        metadata = sparse_utils.MiniCPMSparseMetadata(
            base=SimpleNamespace(page_table=torch.empty(0)),
            k1=level,
            k2=level,
        )
        pool = SimpleNamespace(get_key_buffer=lambda _layer_id: torch.empty(1, 1, 1))

        with (
            patch.object(sparse_utils, "get_token_to_kv_pool", return_value=pool),
            patch.object(sparse_utils, "compress_k_core_new") as compress,
        ):
            sparse_utils.get_compress_k_v2(
                layer,
                forward_batch,
                metadata,
                torch.empty(0),
                torch.empty(0),
                max_context_length=256,
                k1_kernel_size=5,
                k1_kernel_stride=3,
                k2_kernel_size=13,
                k2_kernel_stride=7,
            )

        self.assertEqual(
            [(call.args[8], call.args[9]) for call in compress.call_args_list],
            [(5, 3), (13, 7)],
        )

    def test_decode_compression_uses_compact_layout(self):
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        backend.forward_metadata = SimpleNamespace(
            k1=SimpleNamespace(cu_seqlens_cpu=[0, 2, 3]),
            k2=SimpleNamespace(cu_seqlens_cpu=[0, 1, 1]),
        )
        backend.max_context_len = 8
        backend.k1_kernel_size = 2
        backend.k1_kernel_stride = 2
        backend.k2_kernel_size = 4
        backend.k2_kernel_stride = 4
        backend.device = torch.device("cpu")
        layer = SimpleNamespace(tp_k_head_num=1, head_dim=2)
        forward_batch = SimpleNamespace(batch_size=2)

        for use_graph_buffers in (False, True):
            with self.subTest(use_graph_buffers=use_graph_buffers):
                backend._use_cuda_graph_buffers = use_graph_buffers
                backend.decode_cuda_graph_metadata = {
                    "compress_k1": torch.empty(8, 1, 2),
                    "compress_k2": torch.empty(4, 1, 2),
                }
                with patch.object(backend_module, "get_compress_k_v2") as compress:
                    k1, k2 = backend._compress_decode_keys(
                        torch.empty(1, dtype=torch.float16),
                        layer,
                        forward_batch,
                    )

                self.assertEqual(k1.shape, (3, 1, 2))
                self.assertEqual(k2.shape, (1, 1, 2))
                compress.assert_called_once()
                self.assertNotIn("padded", compress.call_args.kwargs)

    def test_dense_decode_maintains_compressed_cache_without_topk(self):
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        backend.forward_metadata = SimpleNamespace(sparse_bs_list=[])
        backend._compress_decode_keys = Mock(
            return_value=(torch.empty(0), torch.empty(0))
        )
        backend.sparse_get_topk_impl = Mock()
        layer = SimpleNamespace()
        forward_batch = SimpleNamespace(batch_size=1)

        result = backend.get_topk_for_sparse(
            query_states=torch.empty(1, 1, 1),
            key_states=torch.empty(1, 1, 1),
            layer=layer,
            forward_batch=forward_batch,
            is_prefill=False,
        )

        self.assertIsNone(result)
        backend._compress_decode_keys.assert_called_once()
        backend.sparse_get_topk_impl.assert_not_called()

    def test_mixed_decode_runs_topk_for_sparse_requests_only(self):
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        backend.forward_metadata = SimpleNamespace(
            sparse_bs_list=[1],
            base=SimpleNamespace(
                cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
                cu_seqlens_k=torch.tensor([0, 3, 8], dtype=torch.int32),
                max_seq_len_k=5,
            ),
            topk_cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            topk_cu_seqlens_k=torch.tensor([0, 5], dtype=torch.int32),
            topk_max_seqlen_k=5,
            k1=SimpleNamespace(
                cu_seqlens=torch.tensor([0, 1, 3], dtype=torch.int32),
                cu_seqlens_cpu=[0, 1, 3],
            ),
            k2=SimpleNamespace(
                cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32),
                cu_seqlens_cpu=[0, 1, 2],
            ),
        )
        backend._compress_decode_keys = Mock(
            return_value=(
                torch.tensor([[[10.0]], [[20.0]], [[21.0]]]),
                torch.tensor([[[30.0]], [[31.0]]]),
            )
        )
        backend._get_fused_topk_kernel = Mock(return_value="kernel")
        backend.sparse_get_topk_impl = Mock(return_value="topk")
        forward_batch = SimpleNamespace(batch_size=2)

        result = backend.get_topk_for_sparse(
            query_states=torch.tensor([[[1.0]], [[2.0]]]),
            key_states=torch.empty(2, 1, 1),
            layer=SimpleNamespace(),
            forward_batch=forward_batch,
            is_prefill=False,
        )

        self.assertEqual(result, "topk")
        args = backend.sparse_get_topk_impl.call_args.args
        kwargs = backend.sparse_get_topk_impl.call_args.kwargs
        self.assertEqual(args[0].flatten().tolist(), [2.0])
        self.assertEqual(args[1].tolist(), [0, 1])
        self.assertEqual(args[2].tolist(), [0, 3, 8])
        self.assertEqual(kwargs["compressed_k"].flatten().tolist(), [20.0, 21.0])
        self.assertEqual(kwargs["compressed_cu_seqlens"].tolist(), [0, 2])
        self.assertEqual(kwargs["compressed_k2"].flatten().tolist(), [31.0])
        self.assertEqual(kwargs["compressed_cu_seqlens2"].tolist(), [0, 1])
        backend._get_fused_topk_kernel.assert_called_once_with(1, is_prefill=False)

    def test_fused_topk_prefill_kernels_compile_for_all_batches_at_startup(self):
        fake_fuse_kernel = ModuleType("sglang.srt.layers.attention.minicpm.fuse_kernel")
        fake_fuse_kernel.fused_attn_pooling_online_topk_prefill = Mock(
            side_effect=lambda **kwargs: f"prefill-{kwargs['batch_size']}"
        )
        fake_fuse_kernel.fused_attn_pooling_online_topk_decode = Mock(
            side_effect=lambda **kwargs: f"decode-{kwargs['batch_size']}"
        )
        with patch.dict(
            sys.modules,
            {"sglang.srt.layers.attention.minicpm.fuse_kernel": fake_fuse_kernel},
        ):
            backend, *_ = _construct_sparse_backend(
                max_running_requests=3,
                use_flashinfer=True,
                blackwell=True,
            )

        self.assertEqual(
            backend.prefill_fused_kernels,
            {1: "prefill-1", 2: "prefill-2", 3: "prefill-3"},
        )
        self.assertEqual(backend.decode_fused_kernels, {})

    def test_fused_topk_kernels_cache_each_batch_size(self):
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        backend.minicpm_fuse_topk = True
        backend.decode_fused_kernels = {}
        backend.prefill_fused_kernels = {}
        backend.fused_kernel_kwargs = {"topk": 8}
        backend.prefill_kernel_max_seqlen_q_grid = 64

        fake_fuse_kernel = ModuleType("sglang.srt.layers.attention.minicpm.fuse_kernel")
        prefill = Mock(return_value="prefill")
        decode = Mock(return_value="decode")
        fake_fuse_kernel.fused_attn_pooling_online_topk_prefill = prefill
        fake_fuse_kernel.fused_attn_pooling_online_topk_decode = decode
        with patch.dict(
            sys.modules,
            {"sglang.srt.layers.attention.minicpm.fuse_kernel": fake_fuse_kernel},
        ):
            self.assertEqual(
                backend._get_fused_topk_kernel(3, is_prefill=True), "prefill"
            )
            self.assertEqual(
                backend._get_fused_topk_kernel(3, is_prefill=True), "prefill"
            )
            self.assertEqual(
                backend._get_fused_topk_kernel(3, is_prefill=False), "decode"
            )
            self.assertEqual(
                backend._get_fused_topk_kernel(3, is_prefill=False), "decode"
            )

        prefill.assert_called_once_with(
            topk=8,
            batch_size=3,
            max_seqlen_q_grid=64,
        )
        decode.assert_called_once_with(topk=8, batch_size=3)

    def test_backend_import_does_not_require_tilelang(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import builtins
import sys
from unittest.mock import MagicMock

for module in ("sgl_kernel", "sgl_kernel.quantization", "sgl_kernel.scalar_type"):
    sys.modules[module] = MagicMock()

original_import = builtins.__import__

def import_without_tilelang(name, *args, **kwargs):
    if name == "tilelang" or name.startswith("tilelang."):
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_tilelang
import sglang.srt.layers.attention.minicpm.backend
""",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_forward_metadata_tracks_cuda_graph_buffer_ownership(self):
        """Only replay metadata may be marked as backed by CUDA graph buffers."""
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        metadata = SimpleNamespace()
        backend.flash_attn_backend = SimpleNamespace(
            forward_metadata=metadata,
            init_forward_metadata=lambda *_: None,
            init_forward_metadata_out_graph=lambda *_: None,
        )
        backend.update_batch_for_sparse = lambda *_: None
        backend._get_fused_topk_kernel = lambda *_args, **_kwargs: None
        backend._bind_sparse_graph_metadata = lambda *_args, **_kwargs: None
        backend._replay_sparse_graph_metadata = lambda *_: None
        backend.attention_adapter = SimpleNamespace(
            prepare_forward=lambda *_args, **_kwargs: None,
        )
        forward_mode = SimpleNamespace(
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
            is_idle=lambda: False,
            is_decode_or_idle=lambda: True,
        )
        forward_batch = SimpleNamespace(forward_mode=forward_mode, batch_size=1)

        backend._use_cuda_graph_buffers = True
        backend.init_forward_metadata(forward_batch)
        self.assertFalse(backend._use_cuda_graph_buffers)
        self.assertIs(backend.forward_metadata.base, metadata)

        backend.init_forward_metadata_out_graph(forward_batch)
        self.assertTrue(backend._use_cuda_graph_buffers)
        self.assertIs(backend.forward_metadata.base, metadata)

    def test_idle_batch_skips_sparse_metadata(self):
        """An idle DP rank must not attempt sparse metadata construction."""
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        metadata = SimpleNamespace()
        backend.flash_attn_backend = SimpleNamespace(
            forward_metadata=metadata,
            init_forward_metadata=lambda *_: None,
        )
        backend.update_batch_for_sparse = lambda *_: self.fail(
            "idle batches must not build sparse metadata"
        )
        forward_mode = SimpleNamespace(
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
            is_idle=lambda: True,
        )

        backend.init_forward_metadata(SimpleNamespace(forward_mode=forward_mode))

        self.assertIs(backend.forward_metadata.base, metadata)

    def test_mixed_prefill_compiles_fused_topk_for_sparse_batch_only(self):
        """A mixed batch must compile fused top-k for its sparse sub-batch only."""
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        backend.forward_metadata = SimpleNamespace(
            sparse_bs_list=[1],
            base=SimpleNamespace(
                cu_seqlens_q=_DeviceOffsetsMustNotBeRead(),
            ),
            topk_cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            topk_cu_seqlens_k=torch.tensor([0, 1], dtype=torch.int32),
            topk_max_seqlen_q=1,
            topk_max_seqlen_k=1,
            k1=SimpleNamespace(
                cu_seqlens=_DeviceOffsetsMustNotBeRead(),
                cu_seqlens_cpu=[0, 0, 1],
            ),
            k2=SimpleNamespace(
                cu_seqlens=_DeviceOffsetsMustNotBeRead(),
                cu_seqlens_cpu=[0, 0, 1],
            ),
        )
        backend.k1_kernel_size = 1
        backend.k1_kernel_stride = 1
        backend.k2_kernel_size = 1
        backend.k2_kernel_stride = 1
        backend.dense_len = 1
        backend.max_context_len = 1
        layer = SimpleNamespace(tp_q_head_num=1, tp_k_head_num=1, head_dim=1)
        forward_batch = SimpleNamespace(batch_size=2, extend_seq_lens_cpu=[1, 1])

        with (
            patch.object(
                backend_module,
                "allocate_and_compress_keys",
                return_value=(torch.ones(1, 1, 1), torch.ones(1, 1, 1)),
            ) as allocate,
            patch.object(
                backend,
                "_get_fused_topk_kernel",
                return_value="sparse-kernel",
            ) as get_kernel,
            patch.object(
                backend,
                "sparse_get_topk_impl",
                side_effect=lambda *_args, **kwargs: kwargs["fused_kernel"],
            ),
        ):
            result = backend.get_topk_for_sparse(
                query_states=torch.empty(2, 1, 1),
                key_states=torch.empty(2, 1, 1),
                layer=layer,
                forward_batch=forward_batch,
            )

        self.assertEqual(result, "sparse-kernel")
        get_kernel.assert_called_once_with(1, is_prefill=True)

    def test_compression_metadata_ignores_cuda_graph_padding(self):
        """CUDA graph padding rows must not alter offsets for real requests."""
        config = _compression_layout()

        # The graph was captured for batch size 4, but only the first three
        # requests are real during this replay.
        forward_batch = SimpleNamespace(
            batch_size=3,
            seq_lens_cpu=_SingleTensorConversion([100, 200, 300]),
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
        )
        base_metadata = SimpleNamespace(
            cu_seqlens_q=torch.arange(5, dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 100, 200, 300, 400], dtype=torch.int32),
        )
        req_to_sparse_token = torch.arange(4 * 32, dtype=torch.int32).reshape(4, 32)

        k1, k2 = sparse_utils._build_k1_k2_compression_metadata(
            forward_batch=forward_batch,
            base_metadata=base_metadata,
            req_to_sparse_k1_token=req_to_sparse_token,
            req_to_sparse_k2_token=req_to_sparse_token,
            k1_kernel_size=config.k1_kernel_size,
            k1_kernel_stride=config.k1_kernel_stride,
            k2_kernel_size=config.k2_kernel_size,
            k2_kernel_stride=config.k2_kernel_stride,
            cu_seqlens_q=base_metadata.cu_seqlens_q,
        )

        self.assertEqual(k1.cu_seqlens_cpu, [0, 5, 16, 33])
        self.assertEqual(k2.cu_seqlens_cpu, [0, 0, 2, 5])
        for level in (k1, k2):
            self.assertEqual(level.table.shape[0], forward_batch.batch_size)
            self.assertEqual(
                level.history_compress_token_nums.numel(), forward_batch.batch_size
            )
            self.assertEqual(level.cu_new_token_nums.numel(), 4)
            self.assertEqual(level.cu_total_compress_token_nums.numel(), 4)

    def test_sparse_graph_replay_pads_metadata_for_missing_request(self):
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        backend.head_group_num = 2
        backend.k1_kernel_stride = 2
        backend.k2_kernel_stride = 4
        backend.max_context_len = 8
        backend.req_to_sparse_k1_token = torch.arange(8).reshape(4, 2)
        backend.req_to_sparse_k2_token = torch.arange(8).reshape(4, 2)
        backend.decode_cuda_graph_metadata = {
            "compress_k1": torch.empty(16, 1, 1),
            "compress_k2": torch.empty(8, 1, 1),
        }

        def level(history, cumulative):
            return CompressionLevelMetadata(
                history_compress_token_nums=torch.tensor(history),
                cu_seqlens=torch.tensor(cumulative),
                cu_new_token_nums=torch.tensor(cumulative),
                cu_total_compress_token_nums=torch.tensor(cumulative),
            )

        decode_metadata = sparse_utils.MiniCPMSparseMetadata(
            base=SimpleNamespace(),
            sparse_idx=[2, 3, 4, 5],
            dense_layout=[(0, 0, 0, 1)],
            sparse_cache_seqlens_int32=torch.tensor([1, 1, 2, 2, 3, 3]),
            sparse_cu_seqlens_k=torch.tensor([0, 1, 2, 4, 6, 9, 12]),
        )
        compression_metadata = (
            level([1, 2, 3], [0, 1, 3, 6]),
            level([4, 5, 6], [0, 4, 9, 15]),
        )
        backend._build_sparse_decode_replay_metadata = Mock(
            return_value=(decode_metadata, compression_metadata)
        )

        def graph_level():
            return CompressionLevelMetadata(
                table=torch.full((4, 2), -1),
                history_compress_token_nums=torch.full((4,), -1),
                cu_seqlens=torch.full((5,), -1),
                cu_new_token_nums=torch.full((5,), -1),
                cu_total_compress_token_nums=torch.full((5,), -1),
            )

        metadata = sparse_utils.MiniCPMSparseMetadata(
            base=SimpleNamespace(
                cache_seqlens_int32=torch.tensor([2, 3, 4, 0]),
                page_table=torch.tensor(
                    [[5, 6, 0, 0], [7, 8, 9, 0], [10, 11, 12, 13], [0, 0, 0, 0]]
                ),
            ),
            k1=graph_level(),
            k2=graph_level(),
            sparse_page_table=torch.full((8, 4), -1, dtype=torch.int32),
            sparse_cache_seqlens_int32=torch.full((8,), -1),
            sparse_cu_seqlens_k=torch.full((9,), -1),
            cache_seqlens_int32_stage1=torch.full((4,), -1),
        )
        forward_batch = SimpleNamespace(
            batch_size=4,
            num_padding=1,
            req_pool_indices=torch.arange(4),
            seq_lens_cpu=torch.tensor([2, 3, 4, 0]),
        )

        backend._replay_sparse_graph_metadata(forward_batch, metadata)

        self.assertEqual(
            metadata.sparse_cache_seqlens_int32.tolist(), [1, 1, 2, 2, 3, 3, 0, 0]
        )
        self.assertEqual(
            metadata.sparse_cu_seqlens_k.tolist(), [0, 1, 2, 4, 6, 9, 12, 12, 12]
        )
        self.assertEqual(metadata.cache_seqlens_int32_stage1.tolist(), [1, 2, 3, 0])
        self.assertEqual(metadata.sparse_page_table[0, :2].tolist(), [10, 12])
        self.assertEqual(metadata.sparse_page_table[1, :2].tolist(), [11, 13])
        for compression, expected_history, expected_cumulative in (
            (metadata.k1, [1, 2, 3, 0], [0, 1, 3, 6, 6]),
            (metadata.k2, [4, 5, 6, 0], [0, 4, 9, 15, 15]),
        ):
            self.assertEqual(
                compression.history_compress_token_nums.tolist(), expected_history
            )
            self.assertEqual(compression.cu_seqlens.tolist(), expected_cumulative)
            self.assertEqual(
                compression.cu_new_token_nums.tolist(), expected_cumulative
            )
            self.assertEqual(
                compression.cu_total_compress_token_nums.tolist(), expected_cumulative
            )

    def test_idle_sparse_graph_replay_clears_compression_lengths(self):
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)

        def graph_level():
            return CompressionLevelMetadata(
                history_compress_token_nums=torch.ones(2, dtype=torch.int32),
                cu_seqlens=torch.ones(3, dtype=torch.int32),
                cu_new_token_nums=torch.ones(3, dtype=torch.int32),
                cu_total_compress_token_nums=torch.ones(3, dtype=torch.int32),
            )

        metadata = sparse_utils.MiniCPMSparseMetadata(
            base=SimpleNamespace(),
            k1=graph_level(),
            k2=graph_level(),
            sparse_cache_seqlens_int32=torch.ones(2, dtype=torch.int32),
            sparse_cu_seqlens_k=torch.ones(3, dtype=torch.int32),
            cache_seqlens_int32_stage1=torch.ones(2, dtype=torch.int32),
        )

        backend._replay_sparse_graph_metadata(
            SimpleNamespace(batch_size=2, num_padding=2), metadata
        )

        for tensor in (
            metadata.sparse_cache_seqlens_int32,
            metadata.sparse_cu_seqlens_k,
            metadata.cache_seqlens_int32_stage1,
            metadata.k1.history_compress_token_nums,
            metadata.k1.cu_seqlens,
            metadata.k1.cu_new_token_nums,
            metadata.k1.cu_total_compress_token_nums,
            metadata.k2.history_compress_token_nums,
            metadata.k2.cu_seqlens,
            metadata.k2.cu_new_token_nums,
            metadata.k2.cu_total_compress_token_nums,
        ):
            self.assertEqual(torch.count_nonzero(tensor).item(), 0)


if __name__ == "__main__":
    unittest.main()

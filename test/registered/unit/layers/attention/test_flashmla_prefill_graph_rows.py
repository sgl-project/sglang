"""CPU contracts for FlashMLA row counts under prefill graph slicing."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
DSA = dsa_backend.DeepseekSparseAttnBackend


def make_backend():
    backend = DSA.__new__(DSA)
    backend.__dict__.update(
        physical_page_size=64,
        kv_cache_dim=4,
        flashmla_kv_num_q_heads=1,
        dsa_kv_cache_store_fp8=True,
        dsa_index_topk=4,
        kv_scale_ue8m0=False,
        flashmla_kv_format=None,
    )
    return backend


class TestFlashMLAPrefillGraphRows(CustomTestCase):
    def test_launch_trims_only_metadata_padding(self):
        backend = make_backend()
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.tensor([1, 2, 0, 0], dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(
                num_splits=torch.zeros(3, dtype=torch.int32), flashmla_metadata=None
            ),
        )
        observed = {}

        def kernel(**kwargs):
            observed.update(kwargs)
            self.assertEqual(kwargs["cache_seqlens"].shape[0], kwargs["q"].shape[0])
            return torch.ones(2, 1, 1, 2), None

        backend._call_flash_mla_with_kvcache = kernel
        result = backend._forward_flashmla_kv(
            torch.ones(2, 1, 4),
            torch.zeros(64, 4),
            2,
            1.0,
            SimpleNamespace(tp_q_head_num=1, head_dim=4),
            metadata,
            torch.zeros(2, 4, dtype=torch.int32),
        )
        self.assertEqual(result.shape[0], 2)
        torch.testing.assert_close(
            observed["cache_seqlens"], torch.tensor([1, 2], dtype=torch.int32)
        )
        self.assertEqual(
            observed["cache_seqlens"].data_ptr(),
            metadata.dsa_cache_seqlens_int32.data_ptr(),
        )

    def test_inconsistent_metadata_is_rejected_before_native_launch(self):
        for cache_rows, split_rows, index_rows in (
            (1, 3, 2),
            (4, 4, 2),
            (4, 3, 1),
            (4, 0, 2),
        ):
            with self.subTest(cache=cache_rows, splits=split_rows, indices=index_rows):
                backend = make_backend()
                backend._call_flash_mla_with_kvcache = Mock()
                metadata = SimpleNamespace(
                    dsa_cache_seqlens_int32=torch.ones(cache_rows, dtype=torch.int32),
                    flashmla_metadata=SimpleNamespace(
                        num_splits=torch.zeros(split_rows, dtype=torch.int32)
                    ),
                )
                with self.assertRaisesRegex(RuntimeError, "FlashMLA.*row"):
                    backend._forward_flashmla_kv(
                        torch.ones(2, 1, 4),
                        torch.zeros(64, 4),
                        2,
                        1.0,
                        SimpleNamespace(tp_q_head_num=1, head_dim=4),
                        metadata,
                        torch.zeros(index_rows, 4, dtype=torch.int32),
                    )
                backend._call_flash_mla_with_kvcache.assert_not_called()

    def test_metadata_builder_keeps_real_rows_inside_prefill_graph(self):
        for graph_kind in ("eager", "breakable", "piecewise"):
            with self.subTest(graph_kind=graph_kind):
                backend = make_backend()
                backend.__dict__.update(
                    req_to_token_pool=SimpleNamespace(
                        req_to_token=torch.arange(64).view(1, 64)
                    ),
                    _kv_shard_pool=None,
                    use_mha=False,
                    dsa_decode_impl="flashmla_kv",
                    dsa_prefill_impl="flashmla_kv",
                    dsa_index_kpool=1,
                    set_dsa_prefill_impl=lambda fb: None,
                    get_topk_transform_method=lambda mode: TopkTransformMethod.PAGED,
                    _cal_indexer_k_start_end=lambda fb, bs: (None, None),
                    get_device_int32_arange=lambda n: torch.arange(
                        n, dtype=torch.int32
                    ),
                    _transform_table_1_to_real=lambda table: table,
                    _build_topk_v2_plan=lambda seq: None,
                    _init_kpool_metadata=lambda metadata, *args, **kwargs: metadata,
                    init_kernel_metadata=lambda fb: None,
                    _is_in_breakable_cuda_graph=lambda: graph_kind == "breakable",
                    _is_in_tc_piecewise_cuda_graph=lambda: graph_kind == "piecewise",
                )
                captured = []

                def meta(**kwargs):
                    captured.append(kwargs["cache_seqlens"].clone())
                    return SimpleNamespace(
                        num_splits=torch.zeros(len(captured[-1]) + 1, dtype=torch.int32)
                    )

                backend._compute_flashmla_metadata = meta
                fb = SimpleNamespace(
                    batch_size=1,
                    seq_lens=torch.tensor([2]),
                    seq_lens_cpu=torch.tensor([2]),
                    forward_mode=ForwardMode.EXTEND,
                    req_pool_indices=torch.tensor([0]),
                    extend_seq_lens_cpu=[2],
                    extend_seq_lens=torch.tensor([2]),
                    extend_prefix_lens_cpu=[0],
                    seq_lens_sum=2,
                )
                with (
                    patch.object(
                        dsa_backend, "can_dsa_prefill_cp_interleave", return_value=False
                    ),
                    patch.object(
                        dsa_backend,
                        "pad_dsa_cache_seqlens",
                        side_effect=lambda fb, x: torch.cat(
                            [x, torch.zeros(2, dtype=x.dtype)]
                        ),
                    ),
                    patch.object(dsa_backend, "is_cuda", return_value=False),
                ):
                    backend.init_forward_metadata(fb)
                self.assertEqual(len(captured[0]), 4 if graph_kind == "eager" else 2)
                self.assertEqual(
                    len(backend.forward_metadata.dsa_cache_seqlens_int32), 4
                )


if __name__ == "__main__":
    unittest.main()

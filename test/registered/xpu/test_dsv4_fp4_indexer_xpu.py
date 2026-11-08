"""XPU full top-k dispatch and paged-logits argument contracts."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(not is_xpu(), reason="XPU required")


class TestXpuFullTopK:
    def setup_method(self):
        from sglang.srt.layers.attention.dsv4.v41_indexer import full_topk

        self.full_topk = full_topk
        self.pool = SimpleNamespace(
            page_size=256,
            get_index_k_page_size=lambda ratio: 64,
        )
        self.indexer = full_topk.FullTopKIndexer(
            token_to_kv_pool=self.pool,
            req_to_token=torch.empty((1, 1), dtype=torch.int64),
            use_deep_gemm_prefill=True,
            use_deep_gemm_decode=True,
        )
        self.kernel_calls = []

        def fp4_paged_mqa_logits(q, cache, weights, lengths, tables, *args, **kwargs):
            q_fp4, q_scales = q
            self.kernel_calls.append(
                ("logits", q_fp4.shape, q_scales.shape, lengths.clone(), tables.clone())
            )
            return torch.zeros(
                q_fp4.shape[0] * q_fp4.shape[1], args[1], dtype=torch.float32
            )

        def topk_transform(logits, lengths, tables, page_out, page_size, raw_out=None):
            self.kernel_calls.append(
                ("topk", lengths.clone(), tables.clone(), page_size)
            )
            page_out.fill_(0)
            if raw_out is not None:
                raw_out.fill_(0)

        self.fake_sgl_kernel = SimpleNamespace(
            fp4_paged_mqa_logits=fp4_paged_mqa_logits,
            topk_transform=topk_transform,
        )

    def test_prefill_uses_xpu_paged_logits_and_page_topk(self):
        rows = 5
        query = SimpleNamespace(
            x=torch.empty((rows, 4)),
            indexer=object(),
            q_lora=torch.empty((rows, 3)),
            freqs_cis=torch.empty((8, 2)),
            positions=torch.tensor([0, 1, 0, 1, 2]),
            compress_ratio=1,
            seq_lens_cpu=[2, 3],
            rows_per_request=[2, 3],
            req_pool_indices=torch.tensor([10, 11]),
            kv_page_table=torch.tensor([[0], [1]], dtype=torch.int32),
            layer_id=0,
            out_page_indices=torch.full((rows, 4), -1, dtype=torch.int32),
            out_raw_indices=torch.full((rows, 4), -1, dtype=torch.int32),
            reset_outputs=lambda: None,
        )
        packed = torch.empty((rows, 32, 64), dtype=torch.int8)
        scales = torch.empty((rows, 32), dtype=torch.int32)
        weights = torch.empty((rows, 32), dtype=torch.float32)
        with (
            patch.object(self.full_topk, "is_xpu", return_value=True),
            patch.object(
                self.full_topk,
                "_index_q_and_weights",
                return_value=(packed, scales, weights),
            ),
            patch.object(self.full_topk, "get_index_k_cache", return_value=object()),
            patch.dict("sys.modules", {"sgl_kernel": self.fake_sgl_kernel}),
        ):
            self.indexer.topk_prefill(query)

        logits_calls = [call for call in self.kernel_calls if call[0] == "logits"]
        topk_calls = [call for call in self.kernel_calls if call[0] == "topk"]
        assert [call[1][0] for call in logits_calls] == [2, 3]
        assert [call[1].tolist() for call in topk_calls] == [[1, 2], [1, 2, 3]]
        assert topk_calls[0][2][0].tolist() == [0, 1, 2, 3]
        assert topk_calls[1][2][0].tolist() == [4, 5, 6, 7]

    def test_decode_uses_xpu_paged_logits_and_repeats_verify_page_tables(self):
        rows = 4
        data = SimpleNamespace(
            q_fp4=torch.empty((rows, 1, 32, 64), dtype=torch.int8),
            q_sf=torch.empty((rows, 1, 32), dtype=torch.int32),
            weights=torch.empty((rows, 32), dtype=torch.float32),
            k_cache=object(),
        )
        page_tables = torch.tensor([[0, 1], [0, 1], [2, 3], [2, 3]], dtype=torch.int32)
        output = SimpleNamespace(
            paged_metadata=SimpleNamespace(
                rows_per_request=2,
                compressed_seq_lens=torch.tensor([1, 2, 2, 3], dtype=torch.int32),
                page_table=page_tables,
                max_compressed_seq_len=3,
                compressed_page_size=64,
            ),
            out_page_indices=torch.full((rows, 4), -1, dtype=torch.int32),
            reset_outputs=lambda: None,
        )
        with (
            patch.object(self.full_topk, "is_xpu", return_value=True),
            patch.object(
                self.full_topk, "get_deep_gemm_decode_data", return_value=data
            ),
            patch.dict("sys.modules", {"sgl_kernel": self.fake_sgl_kernel}),
        ):
            self.indexer.topk_decode(output)

        logits_call = next(call for call in self.kernel_calls if call[0] == "logits")
        topk_call = next(call for call in self.kernel_calls if call[0] == "topk")
        assert logits_call[3].tolist() == [1, 2, 2, 3]
        assert topk_call[2].tolist() == [[0, 1], [0, 1], [2, 3], [2, 3]]
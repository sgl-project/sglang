"""XPU full top-k and candidate-indexer contracts."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(not is_xpu(), reason="XPU required")

DEVICE = "xpu"


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


def test_xpu_candidate_prefill_and_decode(monkeypatch):
    import sgl_kernel

    from sglang.srt.layers.attention.dsv4.v41_indexer import dense_blocks

    storage = torch.empty((3, 4352), dtype=torch.uint8)
    codes = (torch.arange(64) % 7 + 1) * 17
    storage[:, :4096] = (
        codes.to(torch.uint8).reshape(1, 64, 1).expand(3, 64, 64).reshape(3, 4096)
    )
    storage[:, 4096:] = 127
    cache = storage.to(DEVICE).view(3, 64, 1, 68)
    query = torch.full((2, 1, 32, 64), 0x22, dtype=torch.int8, device=DEVICE)
    scales = torch.full((2, 1, 32), 0x7F7F7F7F, dtype=torch.int32, device=DEVICE)
    weights = torch.ones((2, 32), dtype=torch.float32, device=DEVICE)
    positions = torch.tensor([9, 4], device=DEVICE)
    lengths = torch.tensor([10, 5], dtype=torch.int32, device=DEVICE)
    pages = torch.tensor([[2], [1]], dtype=torch.int32, device=DEVICE)
    out = torch.empty((2, 3), dtype=torch.int32, device=DEVICE)
    raw = torch.empty_like(out)

    def reset_outputs():
        out.fill_(-1)
        raw.fill_(-1)

    pool = SimpleNamespace(page_size=64, get_index_k_page_size=lambda ratio: 64)
    backend = dense_blocks.DenseBlocksBackend(
        token_to_kv_pool=pool,
        req_to_token=None,
        candidate_topk_blocks=2,
        candidate_block_size=4,
        use_deep_gemm_prefill=False,
    )
    prefill = SimpleNamespace(
        positions=positions,
        seq_lens_cpu=[10, 5],
        rows_per_request=[1, 1],
        compress_ratio=1,
        indexer=object(),
        x=torch.empty((2, 1), device=DEVICE),
        q_lora=None,
        freqs_cis=None,
        kv_page_table=pages,
        layer_id=0,
        out_raw_indices=raw,
        out_page_indices=out,
        reset_outputs=reset_outputs,
    )
    metadata = SimpleNamespace(
        rows_per_request=1,
        page_table=pages,
        compressed_seq_lens=lengths,
        max_compressed_seq_len=10,
        compressed_page_size=64,
    )
    decode = SimpleNamespace(
        positions=positions,
        paged_metadata=metadata,
        out_page_indices=out,
        reset_outputs=reset_outputs,
    )
    prepared = SimpleNamespace(q_fp4=query, q_sf=scales, weights=weights, k_cache=cache)
    monkeypatch.setattr(
        dense_blocks,
        "_index_q_and_weights",
        lambda **kwargs: (query.squeeze(1), scales.squeeze(1), weights),
    )
    monkeypatch.setattr(dense_blocks, "get_index_k_cache", lambda **kwargs: cache)
    monkeypatch.setattr(
        dense_blocks, "get_deep_gemm_decode_data", lambda *args: prepared
    )

    def check_output():
        assert sorted(out[0].cpu().tolist()) == [132, 133, 134]
        assert sorted(out[1].cpu().tolist()) == [66, 67, 68]

    with patch.object(
        sgl_kernel, "topk_among_blocks", wraps=sgl_kernel.topk_among_blocks
    ) as topk:
        published = backend.publish_prefill(prefill)
        assert set(published.blocks[0].cpu().tolist()) == {1, 2}
        assert set(published.blocks[1].cpu().tolist()) == {0, 1}
        assert sorted(raw[0].cpu().tolist()) == [4, 5, 6]
        check_output()
        backend.consume_prefill(prefill, published)
        check_output()
        prefill.out_page_indices = None
        backend.consume_prefill(prefill, published)
        assert sorted(raw[0].cpu().tolist()) == [4, 5, 6]
        published = backend.publish_decode(decode)
        assert set(published.blocks[0].cpu().tolist()) == {1, 2}
        check_output()
        backend.consume_decode(decode, published)
        check_output()
        assert topk.call_count >= 8

        out = torch.empty((4, 3), dtype=torch.int32, device=DEVICE)
        raw = torch.empty_like(out)
        prepared.q_fp4 = query.repeat(2, 1, 1, 1)
        prepared.q_sf = scales.repeat(2, 1, 1)
        prepared.weights = weights.repeat(2, 1)
        metadata.rows_per_request = 2
        metadata.page_table = torch.tensor(
            [[2], [2], [1], [1]], dtype=torch.int32, device=DEVICE
        )
        metadata.compressed_seq_lens = torch.tensor(
            [0, 2, 9, 10], dtype=torch.int32, device=DEVICE
        )
        decode.positions = torch.tensor([0, 1, 8, 9], device=DEVICE)
        decode.out_page_indices = out
        published = backend.publish_decode(decode)
        assert out[0].cpu().tolist() == [-1, -1, -1]
        assert sorted(out[1].cpu().tolist()) == [-1, 128, 129]
        backend.consume_decode(decode, published)
        assert out[0].cpu().tolist() == [-1, -1, -1]
        assert sorted(out[1].cpu().tolist()) == [-1, 128, 129]

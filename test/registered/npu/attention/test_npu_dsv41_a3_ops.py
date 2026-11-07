"""A3 V4.1 vendor operators: causal selection and persistent C2 state.

Requires matching regular/experimental ops-transformer packages and their
vendor environment before Python starts. No model weights are needed.
"""

import unittest

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.attention.ascend_dsv41_backend import (
    _dsv41_a3_compressor,
    _dsv41_a3_compressor_kernel,
    _dsv41_a3_ops,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=60,
    suite="nightly-1-npu-a3",
    nightly=True,
    disabled="Requires the DSV4.1 A3 regular and CompressorV2 vendor packages in the CI image",
)


class TestDSV41A3Operators(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if is_npu_arch35():
            raise unittest.SkipTest("BF16 arch22 operator contracts require A3")
        torch_npu.npu.set_device(0)
        cls.ops = _dsv41_a3_ops()
        _dsv41_a3_compressor_kernel()

    def test_smla_low_ratios_match_window_and_compressed_reference(self):
        torch.manual_seed(812)
        tokens, heads, width, length, page = 3, 8, 512, 129, 128
        q = (torch.randn(tokens, heads, width) * 0.1).bfloat16()
        original = (torch.randn(length, width) * 0.1).bfloat16()
        original_cache = torch.zeros(3, page, 1, width, dtype=torch.bfloat16)
        original_cache.view(-1, width)[page : page + length] = original
        sinks = torch.randn(heads) * 0.2
        for ratio in (0, 1, 2):
            with self.subTest(ratio=ratio):
                compressed_len = length // ratio if ratio else 0
                compressed = (torch.randn(compressed_len, width) * 0.1).bfloat16()
                common = {
                    "cu_seqlens_q": torch.tensor([0, tokens], dtype=torch.int32).npu(),
                    "seqused_ori_kv": torch.tensor([length], dtype=torch.int32).npu(),
                    "seqused_cmp_kv": (
                        torch.tensor([compressed_len], dtype=torch.int32).npu()
                        if ratio
                        else None
                    ),
                    "cmp_residual_kv": (
                        torch.tensor([length % ratio], dtype=torch.int32).npu()
                        if ratio > 1
                        else None
                    ),
                    "cmp_ratio": ratio or 1,
                    "ori_mask_mode": 4,
                    "cmp_mask_mode": 3 if ratio else 0,
                    "ori_win_left": 127,
                    "ori_win_right": 0,
                    "layout_q": "TND",
                    "layout_kv": "PA_BBND",
                }
                metadata = self.ops.metadata(
                    heads,
                    1,
                    width,
                    batch_size=1,
                    has_ori_kv=True,
                    has_cmp_kv=bool(ratio),
                    cmp_topk=8 if ratio else 0,
                    **common,
                )
                extra = {}
                if ratio:
                    cmp_page = page // ratio
                    cmp_cache = torch.zeros(3, cmp_page, 1, width, dtype=torch.bfloat16)
                    cmp_cache.view(-1, width)[cmp_page : cmp_page + compressed_len] = (
                        compressed
                    )
                    extra = {
                        "cmp_kv": cmp_cache.npu(),
                        "cmp_sparse_indices": torch.arange(8, dtype=torch.int32)
                        .view(1, 1, 8)
                        .repeat(tokens, 1, 1)
                        .npu(),
                        "cmp_block_table": torch.tensor(
                            [[1, 2]], dtype=torch.int32
                        ).npu(),
                    }
                output, _ = self.ops.attention(
                    q.npu(),
                    ori_kv=original_cache.npu(),
                    ori_block_table=torch.tensor([[1, 2]], dtype=torch.int32).npu(),
                    sinks=sinks.npu(),
                    metadata=metadata,
                    softmax_scale=width**-0.5,
                    **common,
                    **extra,
                )
                expected = []
                for row in range(tokens):
                    position = length - tokens + row
                    keys = original[max(0, position - 127) : position + 1].float()
                    if ratio:
                        keys = torch.cat((keys, compressed[:8].float()))
                    logits = q[row].float() @ keys.T * width**-0.5
                    probabilities = torch.cat((logits, sinks[:, None]), 1).softmax(1)
                    expected.append(probabilities[:, :-1] @ keys)
                torch.testing.assert_close(
                    output.cpu().float(),
                    torch.stack(expected),
                    rtol=0.02,
                    atol=0.0002,
                )

    def test_candidate_consumer_respects_causality_and_candidate_domain(self):
        torch.manual_seed(981)
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                rows, heads, nk, page, topk = 3, 8, 1025, 64, 32
                q = torch.randn(rows, heads, 128).bfloat16() * 0.1
                keys = torch.randn(nk, 128).bfloat16() * 0.1
                weights = torch.rand(rows, heads) / heads
                pages = (nk + page - 1) // page
                cache = torch.zeros(pages + 1, page, 1, 128, dtype=torch.bfloat16)
                cache.view(-1, 128)[page : page + nk] = keys
                common = {
                    "cu_seqlens_q": torch.tensor([0, rows], dtype=torch.int32).npu(),
                    "seqused_k": torch.tensor([nk], dtype=torch.int32).npu(),
                    "cmp_residual_k": (
                        torch.tensor([ratio - 1], dtype=torch.int32).npu()
                        if ratio > 1
                        else None
                    ),
                    "block_table": torch.arange(1, pages + 1, dtype=torch.int32)[
                        None
                    ].npu(),
                    "layout_q": "TND",
                    "layout_k": "PA_BBND",
                    "mask_mode": 3,
                    "cmp_ratio": ratio,
                    "return_value": 1,
                }
                qn, kn, wn = q.npu(), cache.npu(), weights.npu()
                _, values, candidates, lengths = self.ops.candidate(
                    qn,
                    kn,
                    wn,
                    topk,
                    candidate_topk_blocks=64,
                    candidate_block_size=8,
                    **common,
                )
                self.assertEqual(lengths.numel(), 0)
                scores = (
                    torch.einsum("thd,kd->thk", q.float(), keys.float()).relu()
                    * weights[:, :, None]
                ).sum(1)
                original_len = nk * ratio + ratio - 1
                visible = torch.tensor(
                    [(original_len - rows + i + 1) // ratio for i in range(rows)]
                )
                scores.masked_fill_(
                    torch.arange(nk)[None] >= visible[:, None], -torch.inf
                )
                torch.testing.assert_close(
                    values.cpu().reshape(rows, -1).sort(descending=True).values,
                    scores.topk(topk).values,
                    rtol=0.02,
                    atol=0.0005,
                )
                for kind in ("source", "sparse", "empty"):
                    blocks = candidates
                    if kind != "source":
                        blocks = torch.full((rows, 1, 64), -1, dtype=torch.int32)
                        if kind == "sparse":
                            blocks[:, 0, :5] = torch.tensor([0, 2, 2, -1, 99999])
                        blocks = blocks.npu()
                    indices, actual_values = self.ops.sparse_indexer(
                        qn, kn, wn, topk, blocks, candidate_block_size=8, **common
                    )
                    indices = indices.cpu().reshape(rows, -1)
                    actual_values = actual_values.cpu().reshape(rows, -1)
                    blocks = blocks.cpu().reshape(rows, -1)
                    for row in range(rows):
                        allowed = torch.isin(torch.arange(nk) // 8, blocks[row])
                        expected = scores[row].masked_fill(~allowed, -torch.inf)
                        expected = expected.topk(topk).values
                        count = int(torch.isfinite(expected).sum())
                        valid = indices[row] >= 0
                        self.assertEqual(int(valid.sum()), count)
                        ids = indices[row, valid].long()
                        self.assertTrue(bool((ids < visible[row]).all()))
                        self.assertTrue(bool(torch.isin(ids // 8, blocks[row]).all()))
                        self.assertEqual(ids.unique().numel(), ids.numel())
                        torch.testing.assert_close(
                            actual_values[row, valid].sort(descending=True).values,
                            expected[:count],
                            rtol=0.02,
                            atol=0.0005,
                        )

    def test_compressor_pending_pair_and_ring_wrap(self):
        torch.manual_seed(713)
        hidden, width = 5120, 512
        wkv = (torch.randn(width, hidden) * 0.01).bfloat16()
        wgate = (torch.randn(width, hidden) * 0.01).bfloat16()
        device_wkv, device_wgate = wkv.npu(), wgate.npu()
        # Bank zero is padding. Bank one is live; its row stride includes a guard.
        backing = torch.full((2, 3, 2 * width), 17.0, device="npu")
        state = backing[:, :2]
        state[1].zero_()
        expected_state = torch.zeros(2, 2 * width)
        pending = None
        position = 0
        for length in (1, 1, 3, 1, 8, 1):
            x = (torch.randn(length, hidden) * 0.1).bfloat16()
            kv = x.float() @ wkv.float().T
            score = x.float() @ wgate.float().T
            expected = []
            for row in range(length):
                absolute = position + row
                expected_state[absolute % 2] = torch.cat((kv[row], score[row]))
                if absolute % 2 == 0:
                    pending = (kv[row], score[row])
                else:
                    pair_kv = torch.stack((pending[0], kv[row]))
                    pair_score = torch.stack((pending[1], score[row]))
                    expected.append((pair_kv * pair_score.softmax(0)).sum(0))
            actual = _dsv41_a3_compressor(
                x=x.npu(),
                wkv=device_wkv,
                wgate=device_wgate,
                state_cache=state,
                state_block_table=torch.tensor([1], dtype=torch.int32).npu(),
                cu_seqlens=torch.tensor([0, length], dtype=torch.int32).npu(),
                seqused=torch.tensor([length], dtype=torch.int32).npu(),
                start_pos=torch.tensor([position], dtype=torch.int32).npu(),
                cmp_ratio=2,
            ).cpu()
            if expected:
                torch.testing.assert_close(
                    actual[: len(expected)].float(),
                    torch.stack(expected),
                    rtol=0.02,
                    atol=0.002,
                )
            torch.testing.assert_close(
                state[1].cpu(), expected_state, rtol=1e-4, atol=1e-5
            )
            self.assertTrue(bool((backing[:, 2].cpu() == 17).all()))
            self.assertTrue(bool((state[0].cpu() == 17).all()))
            position += length


if __name__ == "__main__":
    unittest.main()

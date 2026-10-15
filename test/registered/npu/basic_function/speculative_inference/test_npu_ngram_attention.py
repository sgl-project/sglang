import os
import unittest
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.server_args import ServerArgs
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=15, suite="nightly-1-npu-a3", nightly=True)


class TestNgramAttention(unittest.TestCase):
    def test_resolved_npu_configuration(self):
        options = dict(
            model_path=os.environ.get(
                "SGLANG_TEST_NGRAM_MODEL", QWEN3_0_6B_WEIGHTS_PATH
            ),
            device="npu",
            attention_backend="ascend",
            speculative_algorithm="NGRAM",
            speculative_ngram_capacity=1000,
        )
        args = ServerArgs(**options)
        args.resolve_once()
        args = resolved_view(args)
        self.assertTrue(args.disable_cuda_graph)
        self.assertTrue(args.disable_overlap_schedule)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.DISABLED)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)
        self.assertEqual(args.speculative_num_draft_tokens, 12)
        self.assertGreaterEqual(args.speculative_num_steps, 1)
        for invalid in (
            dict(speculative_ngram_external_corpus_path="corpus.jsonl"),
            dict(
                speculative_ngram_external_corpus_path="corpus.jsonl",
                speculative_ngram_external_sam_budget=12,
            ),
            dict(
                speculative_ngram_external_corpus_path="corpus.jsonl",
                speculative_ngram_external_sam_budget=4,
                speculative_ngram_external_corpus_max_tokens=0,
            ),
            dict(speculative_use_rejection_sampling=True),
        ):
            with self.subTest(options=invalid), self.assertRaises(ValueError):
                ServerArgs(**dict(options, **invalid)).resolve_once()

    def test_paged_gqa_with_ragged_prefix_and_branch(self):
        torch.manual_seed(42)
        bs, d, nq, nk, hd, page = 2, 4, 4, 2, 128, 128
        prefixes = [125, 129]
        lengths = [n + d for n in prefixes]
        tree = torch.tensor(
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 1]],
            dtype=torch.bool,
        ).repeat(bs, 1, 1)
        mask = AscendAttnBackend._build_ngram_attention_mask(
            tree.flatten().npu(), torch.tensor(prefixes).npu(), d, 2 * page
        )
        expected_mask = torch.ones((bs, 1, d, 2 * page), dtype=torch.bool)
        for b, prefix in enumerate(prefixes):
            expected_mask[b, 0, :, :prefix] = False
            expected_mask[b, 0, :, prefix : prefix + d] = ~tree[b]
        torch.testing.assert_close(mask.cpu(), expected_mask)

        # Non-consecutive physical pages ensure that attention uses block_table.
        table = torch.tensor([[3, 0], [1, 2]], dtype=torch.int32)
        for dtype, tolerance in [(torch.float16, 0.005), (torch.bfloat16, 0.03)]:
            with self.subTest(dtype=dtype):
                q = torch.randn(bs, d, nq, hd, dtype=dtype)
                k = torch.randn(4, page, nk, hd, dtype=dtype)
                v = torch.randn_like(k)
                expected = []
                for b in range(bs):
                    keys = k[table[b].long()].flatten(0, 1)[: lengths[b]].float()
                    values = v[table[b].long()].flatten(0, 1)[: lengths[b]].float()
                    keys = keys.repeat_interleave(nq // nk, dim=1)
                    values = values.repeat_interleave(nq // nk, dim=1)
                    scores = torch.einsum("qhd,khd->hqk", q[b].float(), keys) / hd**0.5
                    scores.masked_fill_(
                        expected_mask[b, 0, :, : lengths[b]], float("-inf")
                    )
                    expected.append(
                        torch.einsum("hqk,khd->qhd", scores.softmax(-1), values)
                    )
                actual, _ = torch_npu.npu_fused_infer_attention_score(
                    q.reshape(bs, d, nq * hd).npu(),
                    k.reshape(4, page, nk * hd).npu(),
                    v.reshape(4, page, nk * hd).npu(),
                    block_table=table.npu(),
                    block_size=page,
                    num_heads=nq,
                    num_key_value_heads=nk,
                    input_layout="BSH",
                    atten_mask=mask,
                    scale=hd**-0.5,
                    actual_seq_lengths=[d] * bs,
                    actual_seq_lengths_kv=lengths,
                    sparse_mode=0,
                    pre_tokens=2147483647,
                    next_tokens=2147483647,
                )
                torch.testing.assert_close(
                    actual.cpu().float().reshape(bs, d, nq, hd),
                    torch.stack(expected),
                    atol=tolerance,
                    rtol=tolerance,
                )

    def test_empty_and_single_root(self):
        mask = AscendAttnBackend._build_ngram_attention_mask(
            torch.empty(0, dtype=torch.bool, device="npu"),
            torch.empty(0, dtype=torch.int64, device="npu"),
            4,
            128,
        )
        self.assertEqual(mask.shape, (0, 1, 4, 128))
        mask = (
            AscendAttnBackend._build_ngram_attention_mask(
                torch.ones(1, dtype=torch.bool, device="npu"),
                torch.tensor([3], device="npu"),
                1,
                128,
            )
            .cpu()
            .flatten()
        )
        self.assertFalse(mask[:4].any())
        self.assertTrue(mask[4:].all())

    def test_kv_compaction_snapshots_overlapping_sources(self):
        # Test both NPU buffer envelopes and asymmetric K/V head dimensions.
        for use_fia in (False, True):
            with (
                self.subTest(use_fia=use_fia),
                patch.dict("os.environ", {"ASCEND_USE_FIA": str(use_fia)}),
            ):
                pool = NPUMHATokenToKVPool(
                    size=256,
                    page_size=128,
                    dtype=torch.float16,
                    head_num=2,
                    head_dim=16,
                    v_head_dim=8,
                    layer_num=2,
                    device="npu",
                    enable_memory_saver=False,
                )
                keys = pool._hicache_k_buffer.flatten(1, 2)
                values = pool._hicache_v_buffer.flatten(1, 2)
                keys.copy_(
                    torch.arange(keys.numel(), device="npu").reshape(keys.shape) % 997
                )
                values.copy_(
                    torch.arange(values.numel(), device="npu").reshape(values.shape)
                    % 991
                )
                before_k, before_v = keys.cpu().clone(), values.cpu().clone()
                src = torch.tensor([127, 128, 130], device="npu")
                dst = torch.tensor([128, 130, 127], device="npu")
                pool.move_kv_cache(dst, src)
                expected_k, expected_v = before_k.clone(), before_v.clone()
                expected_k[:, dst.cpu()] = before_k[:, src.cpu()]
                expected_v[:, dst.cpu()] = before_v[:, src.cpu()]
                torch.testing.assert_close(keys.cpu(), expected_k)
                torch.testing.assert_close(values.cpu(), expected_v)
                pool.move_kv_cache(dst[:0], src[:0])


if __name__ == "__main__":
    unittest.main()

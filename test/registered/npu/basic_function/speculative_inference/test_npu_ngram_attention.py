import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.attention.ngram_attention import build_ngram_attention_mask

from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    ForwardMetadata,
)
from sglang.srt.hardware_backend.npu.attention.mla_cache import gather_mla_cache_pages
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import is_fia_nz
from sglang.srt.hardware_backend.npu.memory_pool_npu import (
    NPUMHATokenToKVPool,
    NPUMLATokenToKVPool,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=15, suite="nightly-1-npu-a3", nightly=True)


class TestNgramAttention(unittest.TestCase):
    @staticmethod
    def make_mla_pool(dtype, use_nz):
        with (
            patch.dict(os.environ, {"SGLANG_USE_FIA_NZ": str(use_nz)}),
            patch(
                "sglang.srt.hardware_backend.npu.memory_pool_npu.get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0),
            ),
        ):
            return NPUMLATokenToKVPool(
                size=512,
                page_size=128,
                dtype=dtype,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                layer_num=2,
                device="npu",
                enable_memory_saver=False,
            )

    def test_mla_forward_matches_cpu_ancestor_attention(self):
        for is_ngram in (False, True):
            with self.subTest(is_ngram=is_ngram):
                self._check_mla_verify_forward(is_ngram)

    def _check_mla_verify_forward(self, is_ngram):
        torch.manual_seed(42)
        bs, d, page, latent, rope = 2, 4, 128, 512, 64
        prefixes = [3, 127]
        lengths = [prefix + d for prefix in prefixes]
        table = torch.tensor([[3, 0], [1, 2]], dtype=torch.int32)
        tree = torch.tensor(
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 1]],
            dtype=torch.bool,
        ).repeat(bs, 1, 1)
        if not is_ngram:
            tree = torch.ones_like(tree).tril()
        blocked = torch.ones(bs, 1, d, 2 * page, dtype=torch.bool)
        for b, prefix in enumerate(prefixes):
            blocked[b, 0, :, :prefix] = False
            blocked[b, 0, :, prefix : prefix + d] = ~tree[b]
        positions = torch.tensor(prefixes)[:, None] + torch.arange(d)
        draft_slots = (
            table.gather(1, positions // page).long() * page + positions % page
        ).flatten()
        layer = SimpleNamespace(
            layer_id=1,
            tp_k_head_num=1,
            tp_v_head_num=1,
            v_head_dim=latent,
            sliding_window_size=-1,
            attn_type=AttentionType.DECODER,
            logit_cap=0,
            scaling=(latent + rope) ** -0.5,
        )
        for dtype, tolerance in ((torch.float16, 0.005), (torch.bfloat16, 0.03)):
            for use_nz in (False, True):
                for heads in (12, 16):
                    with self.subTest(dtype=dtype, use_nz=use_nz, heads=heads):
                        is_fia_nz.cache_clear()
                        with patch.dict(os.environ, {"SGLANG_USE_FIA_NZ": str(use_nz)}):
                            pool = self.make_mla_pool(dtype, use_nz)
                            c = torch.randn(5, page, 1, latent, dtype=dtype)
                            kr = torch.randn(5, page, 1, rope, dtype=dtype)
                            pool.set_kv_buffer(
                                layer, torch.arange(5 * page).npu(), c.npu(), kr.npu()
                            )
                            draft_c = torch.randn(bs * d, 1, latent, dtype=dtype)
                            draft_kr = torch.randn(bs * d, 1, rope, dtype=dtype)
                            c.view(-1, 1, latent)[draft_slots] = draft_c
                            kr.view(-1, 1, rope)[draft_slots] = draft_kr
                            q = torch.randn(bs, d, heads, latent, dtype=dtype)
                            qr = torch.randn(bs, d, heads, rope, dtype=dtype)
                            layer.tp_q_head_num = heads
                            backend = object.__new__(AscendAttnBackend)
                            backend.use_mla = True
                            backend.graph_mode = False
                            backend.speculative_num_draft_tokens = d
                            backend.use_fias_v2_bsnd = False
                            backend.tp_q_head_num = heads
                            backend.model_dtype = dtype
                            backend.mtp_mask = torch.ones(
                                2048, 2048, dtype=torch.bool, device="npu"
                            ).triu(1)
                            backend.kv_lora_rank, backend.qk_rope_head_dim = (
                                latent,
                                rope,
                            )
                            backend.q_head_num_padding = 16
                            backend.page_size, backend.token_to_kv_pool = page, pool
                            backend.forward_metadata = SimpleNamespace(
                                block_tables=table.npu(),
                                ngram_attn_mask=build_ngram_attention_mask(
                                    tree.npu(),
                                    torch.tensor(prefixes).npu(),
                                    d,
                                    2 * page,
                                ),
                                seq_lens_cpu_int=torch.tensor(lengths),
                            )
                            batch = SimpleNamespace(
                                batch_size=bs,
                                spec_info=SimpleNamespace(
                                    spec_input_type=(
                                        SpecInputType.NGRAM_VERIFY
                                        if is_ngram
                                        else SpecInputType.EAGLE_VERIFY
                                    ),
                                    draft_token_num=d,
                                ),
                                forward_mode=ForwardMode.TARGET_VERIFY,
                                global_num_token_non_padded_cpu=bs * d,
                                out_cache_loc=draft_slots.npu(),
                            )
                            actual = (
                                backend.forward_mtp(
                                    q.flatten(0, 1).npu(),
                                    draft_c.npu(),
                                    None,
                                    layer,
                                    batch,
                                    True,
                                    q_rope=qr.flatten(0, 1).npu(),
                                    k_rope=draft_kr.npu(),
                                )
                                .cpu()
                                .float()
                                .reshape(bs, d, heads, latent)
                            )
                            expected = []
                            for b in range(bs):
                                keys = (
                                    c[table[b].long()]
                                    .flatten(0, 1)[: lengths[b]]
                                    .float()
                                )
                                ropes = (
                                    kr[table[b].long()]
                                    .flatten(0, 1)[: lengths[b]]
                                    .float()
                                )
                                scores = (
                                    torch.einsum(
                                        "qhd,kd->hqk", q[b].float(), keys[:, 0]
                                    )
                                    + torch.einsum(
                                        "qhd,kd->hqk", qr[b].float(), ropes[:, 0]
                                    )
                                ) * layer.scaling
                                scores.masked_fill_(
                                    blocked[b, 0, :, : lengths[b]], float("-inf")
                                )
                                expected.append(
                                    torch.einsum(
                                        "hqk,kd->qhd", scores.softmax(-1), keys[:, 0]
                                    )
                                )
                            torch.testing.assert_close(
                                actual,
                                torch.stack(expected),
                                atol=tolerance,
                                rtol=tolerance,
                            )
                            for cache, reference in zip(pool.get_kv_buffer(1), (c, kr)):
                                logical = gather_mla_cache_pages(
                                    cache, torch.arange(5).npu(), is_nz=use_nz
                                )
                                torch.testing.assert_close(logical.cpu(), reference)
        is_fia_nz.cache_clear()

    def test_mla_compaction_preserves_latent_and_rope(self):
        src = torch.tensor([127, 128, 130], device="npu")
        dst = torch.tensor([128, 130, 127], device="npu")
        for use_nz in (False, True):
            with self.subTest(use_nz=use_nz):
                pool = self.make_mla_pool(torch.float16, use_nz)
                reference = []
                for layer_id in range(2):
                    c = (torch.arange(640 * 512).reshape(640, 1, 512) % 997).half()
                    kr = (torch.arange(640 * 64).reshape(640, 1, 64) % 991).half()
                    pool.set_kv_buffer(
                        SimpleNamespace(layer_id=layer_id),
                        torch.arange(640).npu(),
                        c.npu(),
                        kr.npu(),
                    )
                    for values in (c, kr):
                        values[dst.cpu()] = values[src.cpu()].clone()
                    reference.append((c, kr))
                pool.move_kv_cache(dst, src)
                pool.move_kv_cache(dst[:0], src[:0])
                for layer_id in range(2):
                    for cache, expected in zip(
                        pool.get_kv_buffer(layer_id), reference[layer_id]
                    ):
                        logical = gather_mla_cache_pages(
                            cache, torch.arange(5).npu(), is_nz=use_nz
                        )
                        torch.testing.assert_close(
                            logical.cpu().flatten(0, 1), expected
                        )

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
        self.assertFalse(args.disable_cuda_graph)
        self.assertTrue(args.disable_overlap_schedule)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.FULL)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)
        self.assertEqual(args.speculative_num_draft_tokens, 12)
        self.assertGreaterEqual(args.speculative_num_steps, 1)
        disabled = ServerArgs(**options, disable_cuda_graph=True)
        disabled.resolve_once()
        self.assertTrue(resolved_view(disabled).disable_cuda_graph)
        self.assertEqual(
            resolved_view(disabled).cuda_graph_config.decode.backend, Backend.DISABLED
        )
        with patch(
            "sglang.srt.arg_groups.speculative_hook.use_mla_backend", return_value=True
        ):
            mla = ServerArgs(**options)
            mla.resolve_once()
        self.assertTrue(resolved_view(mla).disable_cuda_graph)
        self.assertEqual(
            resolved_view(mla).cuda_graph_config.decode.backend, Backend.DISABLED
        )
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
            dict(tp_size=2, dcp_size=2),
        ):
            with self.subTest(options=invalid), self.assertRaises(ValueError):
                ServerArgs(**dict(options, **invalid)).resolve_once()

    def test_graph_metadata_updates_mask_and_lengths_in_place(self):
        bs, d, page = 2, 4, 128
        backend = object.__new__(AscendAttnBackend)
        backend.device = "npu"
        backend.page_size = page
        backend.speculative_num_draft_tokens = d
        backend.use_sliding_window_kv_pool = False
        backend.is_hybrid_swa = False
        backend.is_draft_worker = False
        backend.req_to_token = torch.arange(bs * 2 * page, device="npu").reshape(
            bs, 2 * page
        )
        metadata = ForwardMetadata(
            block_tables=torch.zeros((bs, 2), dtype=torch.int32, device="npu"),
            seq_lens=torch.empty(bs, dtype=torch.int64, device="npu"),
        )
        backend.graph_metadata = {bs: metadata}
        branch = torch.tensor(
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 1]],
            dtype=torch.bool,
        )
        mask_ptr = None
        # Move between padded and full batches, change the tree, and cross a page.
        for prefixes, tree in (
            ([125, 1], branch),
            ([130, 127], torch.ones_like(branch).tril()),
            ([3, 1], branch),
        ):
            live_bs = 1 if prefixes[1] == 1 else bs
            trees = tree.repeat(live_bs, 1, 1)
            with patch(
                "sglang.srt.hardware_backend.npu.attention.ascend_backend.get_parallel",
                return_value=SimpleNamespace(dcp_enabled=False),
            ):
                backend._apply_cuda_graph_metadata(
                    bs=bs,
                    req_pool_indices=torch.arange(bs, device="npu"),
                    seq_lens=torch.tensor(prefixes, device="npu"),
                    seq_lens_cpu=torch.tensor(prefixes),
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    spec_info=SimpleNamespace(
                        spec_input_type=SpecInputType.NGRAM_VERIFY,
                        draft_token_num=d,
                        custom_mask=trees.flatten().npu(),
                    ),
                )
            current_ptr = metadata.ngram_attn_mask.data_ptr()
            if mask_ptr is not None:
                self.assertEqual(current_ptr, mask_ptr)
            mask_ptr = current_ptr
            expected = torch.ones((bs, 1, d, 2 * page), dtype=torch.bool)
            for b in range(live_bs):
                prefix = prefixes[b]
                expected[b, 0, :, :prefix] = False
                expected[b, 0, :, prefix : prefix + d] = ~tree
            torch.testing.assert_close(metadata.ngram_attn_mask.cpu(), expected)
            self.assertEqual(metadata.seq_lens_cpu_list, [p + d for p in prefixes])
            self.assertIsNone(metadata.seq_lens_cpu_int)

    def test_paged_gqa_with_ragged_prefix_and_branch(self):
        torch.manual_seed(42)
        bs, d, nq, nk, hd, page = 2, 4, 4, 2, 128, 128
        prefixes = [125, 129]
        lengths = [n + d for n in prefixes]
        tree = torch.tensor(
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 1]],
            dtype=torch.bool,
        ).repeat(bs, 1, 1)
        mask = build_ngram_attention_mask(
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
        mask = build_ngram_attention_mask(
            torch.empty(0, dtype=torch.bool, device="npu"),
            torch.empty(0, dtype=torch.int64, device="npu"),
            4,
            128,
        )
        self.assertEqual(mask.shape, (0, 1, 4, 128))
        mask = (
            build_ngram_attention_mask(
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

    def test_shared_verify_forward_preserves_mtp_and_ngram_masks(self):
        torch.manual_seed(42)
        bs, d, nq, nk, hd, page = 2, 4, 4, 2, 128, 128
        prefixes = [125, 129]
        lengths = [p + d for p in prefixes]
        table = torch.tensor([[3, 0], [1, 2]], dtype=torch.int32)
        q = torch.randn(bs, d, nq, hd, dtype=torch.bfloat16)
        k = torch.randn(4, page, nk, hd, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        k_npu, v_npu = k.npu(), v.npu()
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=nq,
            tp_k_head_num=nk,
            tp_v_head_num=nk,
            qk_head_dim=hd,
            v_head_dim=hd,
            scaling=hd**-0.5,
            sliding_window_size=-1,
            attn_type=AttentionType.DECODER,
            logit_cap=0,
        )
        backend = object.__new__(AscendAttnBackend)
        backend.use_mla = False
        backend.is_hybrid_swa = False
        backend.speculative_num_draft_tokens = d
        backend.page_size = page
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _: k_npu,
            get_value_buffer=lambda _: v_npu,
        )
        backend.mtp_mask = torch.ones(2048, 2048, dtype=torch.bool, device="npu").triu(
            1
        )
        for is_ngram in (False, True):
            tree = torch.ones(d, d, dtype=torch.bool).tril()
            if is_ngram:
                tree = torch.tensor(
                    [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 1]],
                    dtype=torch.bool,
                )
            blocked = torch.ones(bs, 1, d, 2 * page, dtype=torch.bool)
            for b, prefix in enumerate(prefixes):
                blocked[b, 0, :, :prefix] = False
                blocked[b, 0, :, prefix : prefix + d] = ~tree
            expected = []
            for b in range(bs):
                keys = k[table[b].long()].flatten(0, 1)[: lengths[b]].float()
                values = v[table[b].long()].flatten(0, 1)[: lengths[b]].float()
                keys = keys.repeat_interleave(nq // nk, dim=1)
                values = values.repeat_interleave(nq // nk, dim=1)
                scores = (
                    torch.einsum("qhd,khd->hqk", q[b].float(), keys) * layer.scaling
                )
                scores.masked_fill_(blocked[b, 0, :, : lengths[b]], float("-inf"))
                expected.append(
                    torch.einsum("hqk,khd->qhd", scores.softmax(-1), values)
                )
            for graph_mode in (False, True):
                with self.subTest(is_ngram=is_ngram, graph_mode=graph_mode):
                    backend.graph_mode = graph_mode
                    backend.forward_metadata = SimpleNamespace(
                        block_tables=table.npu(),
                        ngram_attn_mask=blocked.npu(),
                        seq_lens_cpu_int=None if graph_mode else torch.tensor(lengths),
                        seq_lens_cpu_list=lengths,
                    )
                    batch = SimpleNamespace(
                        spec_info=SimpleNamespace(
                            spec_input_type=(
                                SpecInputType.NGRAM_VERIFY
                                if is_ngram
                                else SpecInputType.EAGLE_VERIFY
                            ),
                            draft_token_num=d,
                        ),
                        forward_mode=ForwardMode.TARGET_VERIFY,
                        global_num_token_non_padded_cpu=bs * d,
                    )
                    actual = backend.forward_mtp(
                        q.flatten(0, 1).npu(), None, None, layer, batch, False
                    )
                    torch.testing.assert_close(
                        actual.cpu().float().reshape(bs, d, nq, hd),
                        torch.stack(expected),
                        atol=0.03,
                        rtol=0.03,
                    )

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

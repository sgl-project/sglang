"""Stable suffix-slot construction from packed verify masks."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.metadata import prepare_tree_spec_metadata
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestTreeSpecMetadata(CustomTestCase):
    def _case(self, batch, width, mask_dtype, pool_dtype, *, qlen_only: bool):
        torch.manual_seed(42)
        capacity = 130
        pool = (
            torch.randperm((batch + 2) * (capacity + width))
            .reshape(batch + 2, capacity + width)
            .to(pool_dtype)
        )
        gpu_pool = pool.cuda()
        req_indices = torch.empty(batch, dtype=torch.int64, device="cuda")
        seq_lens = torch.empty(batch, dtype=torch.int64, device="cuda")
        cu_seqlens = torch.empty(batch + 1, dtype=torch.int32, device="cuda")
        # Match graph replay: the final request may be padding, whose mask
        # indices are redirected to element zero of the packed mask.
        live_rows = (batch - 1) * width if batch > 1 else width
        live_reqs = live_rows // width
        mask_size = (
            live_rows * width if qlen_only else batch * (capacity + width) * width
        )
        mask = torch.empty(mask_size, dtype=mask_dtype, device="cuda")
        table = torch.empty((batch * width, width), dtype=pool_dtype, device="cuda")
        lengths = torch.empty(batch * width, dtype=torch.int32, device="cuda")
        generation = 0

        def update_inputs():
            nonlocal generation
            generation += 1
            reqs = torch.randperm(batch + 2)[:batch]
            lens = torch.randint(0, capacity + 1, (batch,))
            lens[0] = 0
            chunks, trees, expected, expected_lengths = [], [], [], []
            for b, length in enumerate(lens.tolist()):
                tree = torch.rand(width, width) > 0.5
                tree[0].fill_(generation % 2 == 0)
                tree[-1].fill_(True)
                trees.append(tree)
                chunks.append(torch.cat((torch.ones(width, length).bool(), tree), 1))
            if qlen_only:
                packed = torch.cat([tree.flatten() for tree in trees[:live_reqs]])
            else:
                packed = torch.cat([chunk.flatten() for chunk in chunks])
            for b, chunk in enumerate(chunks):
                length = lens[b].item()
                slots = pool[reqs[b], length : length + width]
                for t in range(width):
                    visible = chunk[t, length:]
                    if b * width + t >= live_rows:
                        visible = packed[0].expand(width)
                    expected.append(torch.cat((slots[visible], slots[~visible])))
                    expected_lengths.append(int(visible.sum()))
            req_indices.copy_(reqs)
            seq_lens.copy_(lens)
            if qlen_only:
                # Sentinel values verify QLEN_ONLY never indexes cu_seqlens.
                cu_seqlens.fill_(-999999)
            else:
                cu_seqlens[0] = 0
                cu_seqlens[1:].copy_(lens.cumsum(0))
            mask[: packed.numel()].copy_(packed)
            return torch.stack(expected), torch.tensor(
                expected_lengths, dtype=torch.int32
            )

        def run():
            prepare_tree_spec_metadata(
                gpu_pool,
                req_indices,
                seq_lens,
                cu_seqlens,
                mask,
                table,
                lengths,
                live_rows,
                qlen_only=qlen_only,
            )

        expected_table, expected_lengths = update_inputs()
        run()
        torch.testing.assert_close(table.cpu(), expected_table, rtol=0, atol=0)
        torch.testing.assert_close(lengths.cpu(), expected_lengths, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(2):
            expected_table, expected_lengths = update_inputs()
            graph.replay()
            torch.testing.assert_close(table.cpu(), expected_table, rtol=0, atol=0)
            torch.testing.assert_close(lengths.cpu(), expected_lengths, rtol=0, atol=0)

    def test_shapes_and_replay(self):
        for qlen_only in (False, True):
            for batch, width, mask_dtype, pool_dtype in (
                (1, 1, torch.bool, torch.int32),
                (3, 17, torch.uint8, torch.int64),
                (5, 65, torch.bool, torch.int32),
                (3, 128, torch.bool, torch.int32),
                (2, 257, torch.uint8, torch.int64),
                (1, 1024, torch.bool, torch.int32),
            ):
                with self.subTest(qlen_only=qlen_only, batch=batch, width=width):
                    self._case(
                        batch, width, mask_dtype, pool_dtype, qlen_only=qlen_only
                    )

    def test_qlen_only_rejects_undersized_mask(self):
        width = 8
        pool = torch.zeros((2, 32), dtype=torch.int32, device="cuda")
        req_indices = torch.zeros(2, dtype=torch.int64, device="cuda")
        seq_lens = torch.zeros(2, dtype=torch.int64, device="cuda")
        cu_seqlens = torch.zeros(3, dtype=torch.int32, device="cuda")
        table = torch.empty((2 * width, width), dtype=torch.int32, device="cuda")
        lengths = torch.empty(2 * width, dtype=torch.int32, device="cuda")
        undersized_mask = torch.ones(width * width - 1, dtype=torch.bool, device="cuda")
        with self.assertRaises(AssertionError):
            prepare_tree_spec_metadata(
                pool,
                req_indices,
                seq_lens,
                cu_seqlens,
                undersized_mask,
                table,
                lengths,
                width,
                qlen_only=True,
            )

    def test_flashattention_backend_qlen_only_matches_full_mask(self):
        device = torch.device("cuda")
        d = 8
        raw_bs = 3
        padded_bs = 4
        max_ctx = 256

        be = object.__new__(FlashAttentionBackend)
        be.device = device
        be.topk = 4
        be.speculative_num_draft_tokens = d
        be.max_context_len = max_ctx
        be.page_size = 1
        be.max_num_pages = max_ctx
        be.has_swa = False
        be.use_sliding_window_kv_pool = False
        be.use_mla = False
        be.is_encoder_decoder = False
        be.local_attn_builder = None
        be._draft_kv_ring = None
        be._kv_shard_pool = None
        be.fa_impl_ver = 3
        be.fa3_upgrades_to_fa4 = False
        be.attn_cp_size = 1
        be.is_prefill_aware_swa = False
        be.needs_cpu_seq_lens = False
        be.runner_ids = SimpleNamespace(
            is_translating=False, reads_are_translated=False
        )

        req_to_token = torch.arange(
            8 * (max_ctx + d), dtype=torch.int32, device=device
        ).view(8, max_ctx + d)
        be.req_to_token = req_to_token
        be.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, size=8)

        torch.manual_seed(123)
        trees = torch.rand((raw_bs, d, d), device=device) > 0.5
        for b in range(raw_bs):
            trees[b].fill_diagonal_(True)
        qlen_mask = trees.flatten().contiguous()

        seq_lens_raw = torch.tensor([17, 64, 129], dtype=torch.int64, device=device)
        req_idx_raw = torch.tensor([2, 5, 1], dtype=torch.int64, device=device)
        full_mask = torch.cat(
            [
                torch.cat(
                    (
                        torch.ones(
                            (d, int(seq_lens_raw[b].item())),
                            dtype=torch.bool,
                            device=device,
                        ),
                        trees[b],
                    ),
                    dim=1,
                ).flatten()
                for b in range(raw_bs)
            ],
            dim=0,
        )

        # 1. Eager init_forward_metadata
        def run_eager(cm, qlen_only):
            fb = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=raw_bs,
                req_pool_indices=req_idx_raw,
                seq_lens=seq_lens_raw,
                seq_lens_cpu=seq_lens_raw.cpu(),
                seq_lens_sum=int(seq_lens_raw.sum().item()),
                encoder_lens=None,
                extend_prefix_lens=None,
                extend_prefix_lens_cpu=None,
                extend_seq_lens=None,
                extend_seq_lens_cpu=None,
                extend_num_tokens=None,
                out_cache_loc=None,
                ragged_verify_layout=None,
                iteration_ids=None,
                spec_info=NgramVerifyInput(
                    custom_mask=cm,
                    positions=torch.arange(
                        raw_bs * d, dtype=torch.int64, device=device
                    ),
                    draft_token_num=d,
                    custom_mask_is_qlen_only=qlen_only,
                ),
            )
            be.init_forward_metadata(fb)
            return (
                be.forward_metadata_spec_decode_expand.page_table.clone(),
                be.forward_metadata_spec_decode_expand.cache_seqlens_int32.clone(),
                be.forward_metadata_spec_decode_expand.cu_seqlens_k.clone(),
            )

        pt_f, cs_f, cu_f = run_eager(full_mask, False)
        pt_q, cs_q, cu_q = run_eager(qlen_mask, True)
        torch.testing.assert_close(pt_q, pt_f, rtol=0, atol=0)
        torch.testing.assert_close(cs_q, cs_f, rtol=0, atol=0)
        torch.testing.assert_close(cu_q, cu_f, rtol=0, atol=0)

        # 2. Padded CUDA graph replay (_apply_cuda_graph_metadata)
        seq_lens_pad = torch.tensor([17, 64, 129, 1], dtype=torch.int64, device=device)
        req_idx_pad = torch.tensor([2, 5, 1, 0], dtype=torch.int64, device=device)

        def run_cg(cm, qlen_only):
            m = FlashAttentionMetadata()
            m.cache_seqlens_int32 = torch.zeros(
                padded_bs, dtype=torch.int32, device=device
            )
            m.cu_seqlens_k = torch.zeros(
                padded_bs + 1, dtype=torch.int32, device=device
            )
            m.page_table = torch.zeros(
                (padded_bs, max_ctx), dtype=torch.int32, device=device
            )
            me = FlashAttentionMetadata()
            me.cache_seqlens_int32 = torch.zeros(
                padded_bs * d, dtype=torch.int32, device=device
            )
            me.cu_seqlens_k = torch.zeros(
                padded_bs * d + 1, dtype=torch.int32, device=device
            )
            me.page_table = torch.zeros(
                (padded_bs * d, d), dtype=torch.int32, device=device
            )
            be.target_verify_metadata_topk_normal = {padded_bs: m}
            be.target_verify_metadata_topk_expand = {padded_bs: me}
            spec_info = NgramVerifyInput(
                custom_mask=cm,
                positions=torch.arange(raw_bs * d, dtype=torch.int64, device=device),
                draft_token_num=d,
                custom_mask_is_qlen_only=qlen_only,
            )
            be._apply_cuda_graph_metadata(
                plan=None,
                bs=padded_bs,
                req_pool_indices=req_idx_pad,
                seq_lens=seq_lens_pad,
                encoder_lens=None,
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_info=spec_info,
                ragged_verify_layout=None,
                seq_lens_cpu=None,
                out_cache_loc=None,
            )
            return (
                be.forward_metadata_spec_decode_expand.page_table.clone(),
                be.forward_metadata_spec_decode_expand.cache_seqlens_int32.clone(),
                be.forward_metadata_spec_decode_expand.cu_seqlens_k.clone(),
            )

        pt_cg_f, cs_cg_f, cu_cg_f = run_cg(full_mask, False)
        pt_cg_q, cs_cg_q, cu_cg_q = run_cg(qlen_mask, True)
        torch.testing.assert_close(pt_cg_q, pt_cg_f, rtol=0, atol=0)
        torch.testing.assert_close(cs_cg_q, cs_cg_f, rtol=0, atol=0)
        torch.testing.assert_close(cu_cg_q, cu_cg_f, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

"""DSpark metadata on the HIP DeepSeek-V4 backend: the draft block window and the target-verify indexer rows."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")

SWA_WINDOW = 128
PAGE_SIZE = 256
NUM_REQ_SLOTS = 6
MAX_CONTEXT = 512
# Block slots (out_cache_loc) live past every req_to_token slot.
OUT_LOC_BASE = NUM_REQ_SLOTS * MAX_CONTEXT + 1
NUM_FULL_SLOTS = OUT_LOC_BASE + 4096


def _make_backend(*, block_size, device, is_dspark_draft=True, low_ratios=()):
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DeepseekV4HipRadixBackend,
    )

    backend = object.__new__(DeepseekV4HipRadixBackend)
    backend.device = device
    backend.cuda_int32_kwargs = {"device": device, "dtype": torch.int32}
    backend.swa_page_size = SWA_WINDOW
    backend.page_size = PAGE_SIZE
    g = torch.Generator().manual_seed(0)
    # Distinct full slots per (request slot, position); slot 0 stays the dummy.
    req_to_token = (
        torch.randperm(NUM_REQ_SLOTS * MAX_CONTEXT, generator=g).view(
            NUM_REQ_SLOTS, MAX_CONTEXT
        )
        + 1
    )
    backend.req_to_token = req_to_token.to(device=device, dtype=torch.int32)
    backend.req_to_token_pool = SimpleNamespace(req_to_token=backend.req_to_token)
    backend.MAX_SEQ_LEN_FOR_CAPTURE = MAX_CONTEXT
    # full -> swa: injective and not the identity, so a wrong source shows.
    full_to_swa = (torch.arange(NUM_FULL_SLOTS) * 3 + 7).to(
        device=device, dtype=torch.int64
    )
    backend.token_to_kv_pool = SimpleNamespace(
        full_to_swa_index_mapping=full_to_swa,
        translate_loc_from_full_to_swa=lambda idx: full_to_swa[idx.to(torch.int64)],
        unified_swa_pages=0,
    )
    backend.index_topk = 512
    backend.present_ratios = tuple(sorted(low_ratios))
    backend.low_ratios = tuple(low_ratios)
    backend.has_c4 = False
    backend.has_c128 = False
    backend.candidate_masks = None
    backend.low_ratio_identity_skip = False
    backend.low_ratio_candidate_span = None
    backend.enable_deepseek_v4_fp4_indexer = False
    backend.topk = 1
    backend.mtp_enabled = True
    backend.speculative_num_steps = 0
    backend.speculative_step_id = 0
    backend.speculative_num_draft_tokens = block_size + 1
    backend.is_draft_worker = is_dspark_draft
    backend.is_dspark_draft = is_dspark_draft
    # The draft verifies gamma rows, the target gamma + 1 (bonus included).
    backend.target_verify_num_draft_tokens = (
        block_size if is_dspark_draft else block_size + 1
    )
    backend.forward_metadata = None
    return backend


def _expected_block_row(backend, *, req_slot, prefix, out_loc_block, width):
    """Reference window row: the last min(prefix, SWA_WINDOW) committed tokens,
    then the block's slots, -1 padded; all through the full -> swa map."""
    full_to_swa = backend.token_to_kv_pool.full_to_swa_index_mapping
    ctx = min(prefix, SWA_WINDOW)
    row = torch.full((width,), -1, dtype=torch.int32, device=full_to_swa.device)
    if ctx:
        ctx_full = backend.req_to_token[req_slot, prefix - ctx : prefix].to(torch.int64)
        row[:ctx] = full_to_swa[ctx_full].to(torch.int32)
    row[ctx : ctx + out_loc_block.numel()] = full_to_swa[out_loc_block].to(torch.int32)
    return row, ctx


@unittest.skipUnless(is_hip(), "HIP DeepSeek-V4 backend")
class TestDsparkDraftBlockWindowHip(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda")

    def test_every_block_row_sees_context_plus_whole_block(self):
        block = 5
        backend = _make_backend(block_size=block, device=self.device)
        prefix = torch.tensor(
            [0, 1, 37, 128, 300], dtype=torch.int32, device=self.device
        )
        req_pool = torch.tensor([5, 2, 0, 4, 1], dtype=torch.int32, device=self.device)
        bs = prefix.numel()
        out_loc = (
            torch.arange(bs * block, device=self.device, dtype=torch.int64) * 2
            + OUT_LOC_BASE
        )

        metadata = backend.init_forward_metadata_dspark_draft_block(
            max_seq_len=int(prefix.max()),
            req_pool_indices=req_pool,
            seq_lens=prefix,
            out_cache_loc=out_loc,
            block_size=block,
        )
        core = metadata.core_attn_metadata

        width = core.swa_page_indices.shape[1]
        self.assertEqual(core.swa_page_indices.shape[0], bs * block)
        self.assertEqual(width % 64, 0)
        self.assertGreaterEqual(width, SWA_WINDOW + block)
        for b in range(bs):
            p = int(prefix[b])
            row, ctx = _expected_block_row(
                backend,
                req_slot=int(req_pool[b]),
                prefix=p,
                out_loc_block=out_loc[b * block : (b + 1) * block],
                width=width,
            )
            for j in range(block):
                r = b * block + j
                self.assertTrue(torch.equal(core.swa_page_indices[r], row), (b, j))
                self.assertEqual(int(core.swa_topk_lengths[r]), ctx + block, (b, j))
                self.assertEqual(int(core.seq_lens_casual[r]), p + 1 + j, (b, j))
                self.assertEqual(int(core.positions_casual[r]), p + j, (b, j))
        # SWA-only draft: no compression / indexer metadata is built.
        self.assertIs(core.raw_out_loc, out_loc)
        self.assertIsNone(metadata.indexer_metadata)
        self.assertIsNone(core.c4_sparse_page_indices)
        self.assertEqual(metadata.low_ratio_indexer_metadata_by_ratio(), {})

    def test_block_window_differs_from_the_causal_verify_window(self):
        """The draft block window must differ from the causal target-side verify window."""
        block = 4
        backend = _make_backend(block_size=block, device=self.device)
        prefix = torch.tensor([200], dtype=torch.int32, device=self.device)
        req_pool = torch.tensor([3], dtype=torch.int32, device=self.device)
        out_loc = (
            torch.arange(block, device=self.device, dtype=torch.int64) + OUT_LOC_BASE
        )
        seq_lens_casual, req_rep = backend.expand_extend_with_same_length(
            bs=1, qo_len=block, seq_lens=prefix + block, req_pool_indices=req_pool
        )
        causal = backend.get_swa_page_indices(
            seq_lens_casual=seq_lens_casual, req_pool_indices_repeated=req_rep
        )
        dspark, lens = backend.get_dspark_swa_page_indices(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_rep,
            out_loc=out_loc,
            block_size=block,
        )
        # causal rows differ (one more position each); block rows are identical and read out_loc
        self.assertFalse(torch.equal(causal[0], causal[1]))
        self.assertTrue(torch.equal(dspark[0], dspark[block - 1]))
        self.assertTrue(torch.equal(lens, torch.full_like(lens, SWA_WINDOW + block)))
        block_swa = backend.token_to_kv_pool.full_to_swa_index_mapping[out_loc]
        self.assertTrue(
            torch.equal(
                dspark[0, SWA_WINDOW : SWA_WINDOW + block], block_swa.to(torch.int32)
            )
        )

    def test_block_window_is_refused_off_the_dspark_draft(self):
        block = 4
        backend = _make_backend(
            block_size=block, device=self.device, is_dspark_draft=False
        )
        prefix = torch.tensor([10], dtype=torch.int32, device=self.device)
        req_pool = torch.tensor([1], dtype=torch.int32, device=self.device)
        seq_lens_casual, req_rep = backend.expand_extend_with_same_length(
            bs=1, qo_len=block, seq_lens=prefix + block, req_pool_indices=req_pool
        )
        with self.assertRaises(AssertionError):
            backend.make_core_attn_metadata(
                req_to_token=backend.req_to_token,
                req_pool_indices_repeated=req_rep,
                seq_lens_casual=seq_lens_casual,
                max_seq_len=64,
                out_loc=torch.zeros(block, dtype=torch.int64, device=self.device),
                need_compress=False,
                dspark_block_size=block,
            )

    def test_graph_capture_and_replay_route_the_draft_through_the_block_window(self):
        """A TARGET_VERIFY capture on the DSpark draft must build the block window, and
        a replay must refresh it in place for new lengths and slots."""
        block = 3
        bs = 2
        backend = _make_backend(block_size=block, device=self.device)
        backend.init_cuda_graph_state(max_bs=bs, max_num_tokens=bs * block)

        def make_batch(prefix_list, out_base, cpu_extended):
            prefix = torch.tensor(prefix_list, dtype=torch.int32, device=self.device)
            out_loc = (
                torch.arange(bs * block, device=self.device, dtype=torch.int64)
                + out_base
            )
            seq_lens_cpu = torch.tensor(prefix_list, dtype=torch.int64)
            if cpu_extended:
                # The worker hands the draft a seq_lens_cpu already + block.
                seq_lens_cpu = seq_lens_cpu + block
            return SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                req_pool_indices=torch.tensor(
                    [4, 1], dtype=torch.int32, device=self.device
                ),
                seq_lens=prefix,
                seq_lens_cpu=seq_lens_cpu,
                seq_lens_sum=int(seq_lens_cpu.sum()),
                positions=torch.zeros(
                    bs * block, dtype=torch.int64, device=self.device
                ),
                out_cache_loc=out_loc,
                spec_info=SimpleNamespace(draft_token_num=block),
            )

        capture_batch = make_batch([1, 1], OUT_LOC_BASE, cpu_extended=False)
        backend.init_forward_metadata_out_graph(capture_batch, in_capture=True)
        captured = backend.forward_metadata
        core = captured.core_attn_metadata
        width = core.swa_page_indices.shape[1]

        replay_batch = make_batch([150, 7], OUT_LOC_BASE + 100, cpu_extended=True)
        backend.init_forward_metadata_out_graph(replay_batch)
        self.assertIs(backend.forward_metadata, captured)
        for b, p in enumerate([150, 7]):
            row, ctx = _expected_block_row(
                backend,
                req_slot=int(replay_batch.req_pool_indices[b]),
                prefix=p,
                out_loc_block=replay_batch.out_cache_loc[b * block : (b + 1) * block],
                width=width,
            )
            for j in range(block):
                r = b * block + j
                self.assertTrue(torch.equal(core.swa_page_indices[r], row), (b, j))
                self.assertEqual(int(core.swa_topk_lengths[r]), ctx + block)
                # Derived from the device prefix, not the pre-extended CPU copy.
                self.assertEqual(int(core.positions_casual[r]), p + j)


@unittest.skipUnless(is_hip(), "HIP DeepSeek-V4 backend")
class TestAiterSparseLengthFoldPerStep(CustomTestCase):
    """The length-fold cache must not outlive the forward: a TARGET_VERIFY bucket keeps
    one core object across steps, so the warmup's lists would be replayed."""

    def test_graph_bucket_refold_follows_the_new_lengths(self):
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            _fold_lengths_for_aiter_sparse,
        )

        device = torch.device("cuda")
        block, bs = 3, 2
        backend = _make_backend(block_size=block, device=device)
        backend.init_cuda_graph_state(max_bs=bs, max_num_tokens=bs * block)

        def make_batch(prefix_list, out_base):
            prefix = torch.tensor(prefix_list, dtype=torch.int32, device=device)
            seq_lens_cpu = torch.tensor(prefix_list, dtype=torch.int64)
            return SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                req_pool_indices=torch.tensor([4, 1], dtype=torch.int32, device=device),
                seq_lens=prefix,
                seq_lens_cpu=seq_lens_cpu,
                seq_lens_sum=int(seq_lens_cpu.sum()),
                positions=torch.zeros(bs * block, dtype=torch.int64, device=device),
                out_cache_loc=torch.arange(bs * block, device=device, dtype=torch.int64)
                + out_base,
                spec_info=SimpleNamespace(draft_token_num=block),
            )

        def fold(core):
            masked, _ = _fold_lengths_for_aiter_sparse(
                core,
                0,
                core.swa_page_indices.unsqueeze(1),
                core.swa_topk_lengths,
                None,
                None,
            )
            return masked.squeeze(1)

        # Warmup-like forward at the capture lengths (prefix 1: 1 + block valid).
        backend.init_forward_metadata_out_graph(
            make_batch([1, 1], OUT_LOC_BASE), in_capture=True
        )
        core = backend.forward_metadata.core_attn_metadata
        warm = fold(core).clone()
        self.assertTrue(
            torch.equal(
                (warm >= 0).sum(1), torch.full((bs * block,), 1 + block, device=device)
            )
        )

        # Next step: replay copies longer contexts into the same bucket object.
        replay = make_batch([150, 7], OUT_LOC_BASE + 100)
        backend.init_forward_metadata_out_graph(replay)
        self.assertIs(backend.forward_metadata.core_attn_metadata, core)
        backend.init_forward_metadata_in_graph(replay)
        got = fold(core)
        expect_valid = torch.tensor(
            [SWA_WINDOW + block] * block + [7 + block] * block, device=device
        )
        self.assertTrue(torch.equal((got >= 0).sum(1), expect_valid))
        self.assertTrue(
            torch.equal(
                got,
                core.swa_page_indices.clone().masked_fill(
                    torch.arange(got.shape[1], device=device)[None, :]
                    >= core.swa_topk_lengths[:, None],
                    -1,
                ),
            )
        )
        self.assertFalse(torch.equal(got, warm))


@unittest.skipUnless(is_hip(), "HIP DeepSeek-V4 backend")
class TestLowRatioTargetVerifyHip(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        import sglang.srt.layers.attention.deepseek_v4_backend_hip_radix as module

        self.module = module

    def test_target_verify_indexer_takes_the_decode_body(self):
        backend = _make_backend(
            block_size=4, device=self.device, is_dspark_draft=False, low_ratios=(1, 2)
        )
        calls = []
        saved = self.module.low_ratio_index_topk_hip_decode
        self.module.low_ratio_index_topk_hip_decode = lambda *a, **k: calls.append(
            "decode"
        )
        backend._low_ratio_index_topk_torch = lambda *a, **k: calls.append("torch")
        # every body rewrites the ratio's page indices, so the dispatcher drops their folds first
        dropped = []
        backend.forward_metadata = SimpleNamespace(
            core_metadata=SimpleNamespace(drop_folded_sparse_indices=dropped.append)
        )
        layer = SimpleNamespace(compress_ratio=2)
        try:
            for mode in (
                ForwardMode.TARGET_VERIFY,
                ForwardMode.DECODE,
                ForwardMode.EXTEND,
            ):
                forward_batch = SimpleNamespace(
                    forward_mode=mode, seq_lens_cpu=None, extend_seq_lens_cpu=None
                )
                backend._low_ratio_index_topk(
                    layer, None, None, None, None, forward_batch
                )
        finally:
            self.module.low_ratio_index_topk_hip_decode = saved
        self.assertEqual(dropped, [2, 2, 2])
        # Extend without CPU lengths still falls back to the torch oracle.
        self.assertEqual(calls, ["decode", "decode", "torch"])

    def test_target_verify_builder_self_adds_drafts_and_asks_decode_rows(self):
        backend = _make_backend(
            block_size=4, device=self.device, is_dspark_draft=False, low_ratios=(1, 2)
        )
        num_draft = backend.target_verify_num_draft_tokens
        seen = {}

        def fake_prefill(**kwargs):
            seen.update(kwargs)
            return "metadata"

        backend.init_forward_metadata_prefill = fake_prefill
        prefix = torch.tensor([3, 9], dtype=torch.int32, device=self.device)
        out = backend.init_forward_metadata_target_verify(
            max_seq_len=9,
            req_pool_indices=torch.tensor(
                [1, 2], dtype=torch.int32, device=self.device
            ),
            seq_lens=prefix,
            out_cache_loc=torch.zeros(
                2 * num_draft, dtype=torch.int64, device=self.device
            ),
            seq_lens_cpu=[3, 9],
        )
        self.assertEqual(out, "metadata")
        self.assertTrue(seen["low_ratio_decode_rows"])
        self.assertTrue(seen["attach_decode_streams"])
        self.assertTrue(seen["need_compress"])
        self.assertEqual(seen["seq_lens_cpu"], [3 + num_draft, 9 + num_draft])
        self.assertEqual(seen["extend_seq_lens_cpu"], [num_draft, num_draft])
        self.assertEqual(seen["num_tokens"], 2 * num_draft)
        self.assertEqual(seen["seq_lens"].tolist(), [3 + num_draft, 9 + num_draft])
        self.assertEqual(seen["max_seq_len"], 9 + num_draft)

    def test_low_ratio_indexer_rows_take_the_decode_form_when_asked(self):
        backend = _make_backend(
            block_size=4, device=self.device, is_dspark_draft=False, low_ratios=(2,)
        )
        backend.dsa_topk_backend = SimpleNamespace(should_use_topk_v2=lambda: False)
        backend.token_to_kv_pool.get_index_k_page_size = lambda ratio: 64
        rows = 6
        core = SimpleNamespace(
            page_table=torch.zeros((rows, 2), dtype=torch.int32, device=self.device),
            seq_lens_casual=torch.tensor(
                [1, 2, 3, 4, 5, 6], dtype=torch.int32, device=self.device
            ),
            c2_topk_lengths_clamp1=torch.tensor(
                [1, 1, 1, 2, 2, 3], dtype=torch.int32, device=self.device
            ),
            low_ratios=(2,),
        )
        decode_form = backend._init_low_ratio_indexer_metadata(core, is_prefill=False)
        prefill_form = backend._init_low_ratio_indexer_metadata(core, is_prefill=True)
        self.assertEqual(set(decode_form), {"c2_indexer_metadata"})
        self.assertEqual(
            decode_form["c2_indexer_metadata"].c4_seq_lens.tolist(), [1, 1, 1, 2, 2, 3]
        )
        # Prefill rows keep the raw visible count (a zero row is skipped).
        self.assertEqual(
            prefill_form["c2_indexer_metadata"].c4_seq_lens.tolist(), [0, 1, 1, 2, 2, 3]
        )

    def test_in_graph_hoists_verify_rows_and_builds_decode_workspaces(self):
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            DSV4Metadata,
        )

        backend = _make_backend(
            block_size=4, device=self.device, is_dspark_draft=False, low_ratios=(1, 2)
        )
        block = backend.target_verify_num_draft_tokens
        bs = 3
        req = torch.tensor([4, 0, 2], dtype=torch.int32, device=self.device)
        positions = (
            torch.arange(bs * block, device=self.device, dtype=torch.int64) + 100
        )
        out_cache_loc = (
            torch.arange(bs * block, device=self.device, dtype=torch.int64) + 10
        )
        c2_meta = object()
        sentinel = {2: object()}
        seen = {}
        saved = self.module.build_low_ratio_decode_workspaces

        def fake_builder(by_ratio):
            seen.update(by_ratio)
            return sentinel

        self.module.build_low_ratio_decode_workspaces = fake_builder
        try:
            for mode, hoisted in (
                (ForwardMode.TARGET_VERIFY, True),
                (ForwardMode.DECODE, True),
                (ForwardMode.EXTEND, False),
            ):
                core = SimpleNamespace(low_ratios=(2,))
                metadata = DSV4Metadata(
                    core_attn_metadata=core,
                    indexer_metadata=None,
                    c2_indexer_metadata=c2_meta,
                )
                backend.forward_metadata = metadata
                seen.clear()
                n = bs * block if mode.is_target_verify() else bs
                forward_batch = SimpleNamespace(
                    forward_mode=mode,
                    batch_size=bs,
                    req_pool_indices=req,
                    positions=positions[:n],
                    out_cache_loc=out_cache_loc[:n],
                    spec_info=SimpleNamespace(draft_token_num=block),
                    extend_seq_lens=torch.ones(
                        bs, dtype=torch.int64, device=self.device
                    ),
                )
                backend.init_forward_metadata_in_graph(forward_batch)
                expect_swa = backend.token_to_kv_pool.translate_loc_from_full_to_swa(
                    out_cache_loc[:n]
                ).to(torch.int32)
                self.assertTrue(torch.equal(core.swa_out_cache_loc, expect_swa), mode)
                if not hoisted:
                    self.assertIsNone(metadata.low_ratio_req_indices, mode)
                    self.assertIsNone(metadata.low_ratio_pos_i64, mode)
                    self.assertEqual(metadata.fp4_low_ratio_decode_workspaces, {}, mode)
                    continue
                repeats = block if mode.is_target_verify() else 1
                self.assertTrue(
                    torch.equal(
                        metadata.low_ratio_req_indices,
                        req.to(torch.int64).repeat_interleave(repeats),
                    ),
                    mode,
                )
                self.assertTrue(torch.equal(metadata.low_ratio_pos_i64, positions[:n]))
                self.assertIs(metadata.fp4_low_ratio_decode_workspaces, sentinel, mode)
                self.assertEqual(seen, {2: c2_meta}, mode)
        finally:
            self.module.build_low_ratio_decode_workspaces = saved


if __name__ == "__main__":
    unittest.main()

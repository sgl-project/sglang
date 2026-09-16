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
    backend.is_dspark = True
    backend.needs_cpu_seq_lens = False
    backend._fp4_graph_row_limit = None
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
        self._check_block_window_replay(cpu_mirror=True)

    def test_block_window_replay_without_cpu_lengths(self):
        self._check_block_window_replay(cpu_mirror=False)

    def _check_block_window_replay(self, *, cpu_mirror):
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
        if not cpu_mirror:
            replay_batch.seq_lens_cpu = None
            replay_batch.seq_lens_sum = None
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
class TestEagleDeviceMetadataHip(CustomTestCase):
    def test_draft_extend_eager_and_replay_without_cpu_lengths(self):
        device = torch.device("cuda")
        backend = _make_backend(block_size=4, device=device)
        backend.is_dspark = backend.is_dspark_draft = False
        backend.enable_decoder_swa_bounded_replay = False
        rows = backend.speculative_num_draft_tokens
        bs = 2
        backend.init_cuda_graph_state(max_bs=bs, max_num_tokens=bs * rows)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=bs,
            req_pool_indices=torch.tensor([1, 4], device=device, dtype=torch.int32),
            seq_lens=torch.tensor([130, 255], device=device, dtype=torch.int32),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            # Device-only EAGLE does not publish prefill CPU length lists.
            extend_seq_lens_cpu=None,
            extend_seq_lens=None,
            out_cache_loc=torch.arange(bs * rows, device=device) + OUT_LOC_BASE,
            positions=torch.zeros(bs * rows, device=device, dtype=torch.int64),
            encoder_swa_replay=False,
        )
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        captured = backend.forward_metadata
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            device_built = backend.init_forward_metadata_draft_extend(
                max_seq_len=MAX_CONTEXT,
                req_pool_indices=batch.req_pool_indices,
                seq_lens=batch.seq_lens,
                seq_lens_cpu=None,
                num_tokens_per_req=rows,
                out_cache_loc=batch.out_cache_loc,
            )
        for lengths, slots in (([130, 255], [1, 4]), ([257, 381], [4, 2])):
            batch.seq_lens.copy_(torch.tensor(lengths, device=device))
            batch.req_pool_indices.copy_(torch.tensor(slots, device=device))
            batch.out_cache_loc.add_(bs * rows)
            # Keep the old full-context builder as an independent reference.
            reference = backend.init_forward_metadata_prefill(
                max_seq_len=MAX_CONTEXT,
                req_pool_indices=batch.req_pool_indices,
                seq_lens=batch.seq_lens,
                seq_lens_cpu=lengths,
                extend_seq_lens=torch.full_like(batch.seq_lens, rows),
                extend_seq_lens_cpu=[rows] * bs,
                num_tokens=rows * bs,
                need_compress=False,
                out_cache_loc=batch.out_cache_loc,
            )
            self.assertEqual(reference.core_metadata.page_table.shape[1], 2)
            self.assertEqual(device_built.core_metadata.page_table.shape[1], 1)
            graph.replay()
            for name in ("seq_lens_casual", "swa_page_indices", "swa_topk_lengths"):
                self.assertTrue(
                    torch.equal(
                        getattr(device_built.core_metadata, name),
                        getattr(reference.core_metadata, name),
                    ),
                    name,
                )
            backend.init_forward_metadata_out_graph(batch)
            self.assertIs(backend.forward_metadata, captured)
            for name in ("seq_lens_casual", "swa_page_indices", "swa_topk_lengths"):
                self.assertTrue(
                    torch.equal(
                        getattr(captured.core_metadata, name),
                        getattr(reference.core_metadata, name),
                    ),
                    name,
                )
            backend.init_forward_metadata(batch)
            for name in ("seq_lens_casual", "swa_page_indices", "swa_topk_lengths"):
                self.assertTrue(
                    torch.equal(
                        getattr(backend.forward_metadata.core_metadata, name),
                        getattr(reference.core_metadata, name),
                    ),
                    name,
                )


@unittest.skipUnless(is_hip(), "HIP DeepSeek-V4 backend")
class TestLowRatioTargetVerifyHip(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        import sglang.srt.layers.attention.deepseek_v4_backend_hip_radix as module

        self.module = module

    def test_target_verify_device_lengths_match_cpu_and_replay(self):
        backend = _make_backend(block_size=4, device=self.device, is_dspark_draft=False)
        backend.has_c128 = True
        backend.token_to_kv_pool.swa_page_size = SWA_WINDOW
        backend.token_to_kv_pool._unified_kv = False
        backend.token_to_kv_pool.get_ring_size = lambda **kwargs: 128
        prefix = torch.tensor([125, 255], dtype=torch.int32, device=self.device)
        slots = torch.tensor([1, 4], dtype=torch.int32, device=self.device)
        count = 2 * backend.target_verify_num_draft_tokens
        out_loc = torch.arange(count, device=self.device) + OUT_LOC_BASE

        def build(cpu_lengths):
            return backend.init_forward_metadata_target_verify_old(
                max_seq_len=MAX_CONTEXT - backend.target_verify_num_draft_tokens,
                req_pool_indices=slots,
                seq_lens=prefix,
                seq_lens_cpu=cpu_lengths,
                out_cache_loc=out_loc,
                use_prefill_cuda_graph=True,
            )

        # Build the full compressor plan on device, not just the raw wrapper.
        build(None)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = build(None)
        for lengths, request_slots in (([125, 255], [1, 4]), ([254, 381], [4, 2])):
            prefix.copy_(torch.tensor(lengths, device=self.device))
            slots.copy_(torch.tensor(request_slots, device=self.device))
            out_loc.add_(count)
            graph.replay()
            reference = build(lengths)
            for name in ("seq_lens_casual", "swa_page_indices", "swa_topk_lengths"):
                self.assertTrue(
                    torch.equal(
                        getattr(captured.core_metadata, name),
                        getattr(reference.core_metadata, name),
                    ),
                    name,
                )
            for name in ("plan_c", "plan_w"):
                self.assertTrue(
                    torch.equal(
                        getattr(captured.c128_compress_metadata, name),
                        getattr(reference.c128_compress_metadata, name),
                    ),
                    name,
                )

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
            decode_form["c2_indexer_metadata"].compressed_seq_lens.tolist(),
            [1, 1, 1, 2, 2, 3],
        )
        # Prefill rows keep the raw visible count (a zero row is skipped).
        self.assertEqual(
            prefill_form["c2_indexer_metadata"].compressed_seq_lens.tolist(),
            [0, 1, 1, 2, 2, 3],
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


@unittest.skipUnless(is_hip(), "HIP multi-stream preparation")
class TestLowRatioPrepareStreams(CustomTestCase):
    def test_cp_reference_uses_local_request_row_counts(self):
        from unittest.mock import Mock

        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            DeepseekV4HipRadixBackend,
        )

        backend = SimpleNamespace(_low_ratio_index_topk_torch=Mock())
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor([7, 9, 11], device="cuda")
        )
        x = torch.empty(5, 16, device="cuda")
        pos = torch.tensor([2, 4, 6, 8, 10], device="cuda")
        from sglang.srt.environ import envs

        with envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(True):
            DeepseekV4HipRadixBackend._low_ratio_index_topk_dense(
                backend,
                None,
                x,
                x,
                pos,
                batch,
                torch.tensor([2, 0, 3], device="cuda"),
                [2, 0, 3],
            )
        call = backend._low_ratio_index_topk_torch.call_args.args
        self.assertEqual(call[3].tolist(), [7, 7, 11, 11, 11])
        self.assertIs(call[4], pos)

    def test_graph_replay_joins_kv_and_source_streams(self):
        from unittest.mock import patch

        from sglang.srt.environ import envs
        from sglang.srt.models.deepseek_v4 import MQALayer
        from sglang.srt.runtime_context import get_parallel

        config = SimpleNamespace(
            model_type="deepseek_v41",
            hidden_size=32,
            head_dim=128,
            qk_rope_head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=1,
            o_groups=1,
            q_lora_rank=32,
            o_lora_rank=32,
            max_position_embeddings=128,
            compress_ratios=[2],
            rope_scaling={"original_max_position_embeddings": 128, "factor": 1.0},
            rope_theta=10000,
            compress_rope_theta=40000,
            rms_norm_eps=1e-6,
            q_head_norm=True,
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
        )
        with (
            torch.device("cuda"),
            get_parallel().override(
                tp_size=1, tp_rank=0, attn_tp_rank=0, attn_tp_size=1
            ),
            patch(
                "sglang.srt.models.deepseek_v4.get_device",
                return_value=SimpleNamespace(device="cuda"),
            ),
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.override(True),
            envs.SGLANG_OPT_FUSE_WQA_WKV.override(True),
        ):
            layer = MQALayer(
                config, 0, alt_streams=[torch.cuda.Stream(), torch.cuda.Stream()]
            )

        x = torch.randn(6, 32, device="cuda")
        compressed, indexed, kv = (torch.empty_like(x) for _ in range(3))

        def sources(*, x, q_lora, run_compressor=True, run_indexer=True, **kwargs):
            if run_compressor:
                compressed.copy_(x * 2)
            if run_indexer:
                indexed.copy_(compressed + q_lora)

        layer.compressor = object()
        layer.indexer = object()
        layer.wqkv_a.forward = lambda x: (x * 4, None)
        layer._compute_q_a = lambda x, **kw: (x + 1, x + 1)
        layer._compute_q_b = lambda q, positions, q_out: q * 3
        layer._compute_kv_to_cache = lambda x, positions, batch, backend, qkv_a: (
            kv.copy_(qkv_a + 5)
        )
        backend = SimpleNamespace(forward_low_ratio_sources=sources)

        def run():
            q = MQALayer._forward_prepare_low_ratio_multi_stream(
                layer, x, None, None, backend
            )
            return q + indexed + kv

        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        for _ in range(3):
            x.normal_()
            graph.replay()
            torch.testing.assert_close(
                output, (x + 1) * 3 + (x * 2 + x + 1) + (x * 4 + 5)
            )


@unittest.skipUnless(is_hip(), "HIP context-parallel prefill")
class TestLowRatioContextParallel(CustomTestCase):
    def test_tail_preserves_rank_ownership_and_restores_full_batch(self):
        from unittest.mock import patch

        from sglang.srt.layers.cp import base as cp_base
        from sglang.srt.layers.cp.interleave import (
            InterleaveContextParallelMetadata,
            InterleaveCPStrategy,
            interleave_rows_per_request,
        )
        from sglang.srt.layers.dp_attention import (
            get_local_dp_buffer_len,
            set_local_dp_buffer_len,
        )
        from sglang.srt.runtime_context import get_parallel

        device = "cuda"
        backend = _make_backend(
            block_size=5, device=device, low_ratios=(1, 2), is_dspark_draft=False
        )
        backend.token_to_kv_pool.get_index_k_page_size = lambda ratio: 64
        backend.dsa_topk_backend = SimpleNamespace(should_use_topk_v2=lambda: False)
        # Unequal tail ownership, a request absent on some ranks, and a single request.
        for extend in ([257, 1, 130], [257]):
            with self.subTest(extend=extend):
                req = torch.tensor([4, 1, 3][: len(extend)], device=device)
                seq = [e + 13 for e in extend]
                lengths = torch.tensor(extend, device=device, dtype=torch.int32)
                positions = torch.cat([torch.arange(13, s, device=device) for s in seq])
                requests = req.repeat_interleave(lengths.long())
                out_loc = backend.req_to_token[requests, positions].long()
                n = sum(extend)
                padded = (n + 3) // 4 * 4
                batch = SimpleNamespace(
                    batch_size=len(extend),
                    input_ids=torch.arange(padded, device=device),
                    positions=positions,
                    req_pool_indices=req,
                    seq_lens=torch.tensor(seq, device=device, dtype=torch.int32),
                    seq_lens_cpu=torch.tensor(seq),
                    extend_seq_lens_cpu=extend,
                    out_cache_loc=out_loc,
                    forward_mode=ForwardMode.EXTEND,
                    attn_cp_metadata=None,
                )
                global_tail = backend._build_late_layer_tail_metadata(batch)
                tail_indices = torch.cat(
                    [
                        torch.arange(
                            start + max(e - SWA_WINDOW, 0), start + e, device=device
                        )
                        for start, e in zip(
                            [0, *torch.tensor(extend).cumsum(0).tolist()], extend
                        )
                    ]
                )
                gathered = []
                for rank in range(4):
                    original_cp = InterleaveContextParallelMetadata(
                        per_rank_actual_token=[padded // 4] * 4, total_seq_lens=n
                    )
                    batch.attn_cp_metadata = original_cp
                    with (
                        get_parallel().override(attn_cp_size=4, attn_cp_rank=rank),
                        patch.object(
                            cp_base, "_STRATEGY", InterleaveCPStrategy(cp_size=4)
                        ),
                    ):
                        full = backend.init_forward_metadata_prefill(
                            max_seq_len=max(seq),
                            req_pool_indices=req,
                            seq_lens=batch.seq_lens,
                            seq_lens_cpu=seq,
                            out_cache_loc=out_loc,
                            num_tokens=n,
                            extend_seq_lens=lengths,
                            extend_seq_lens_cpu=extend,
                            cp_metadata=original_cp,
                        )
                        tail_metadata = backend._build_late_layer_tail_metadata(batch)
                        tail = tail_metadata.late_layer_tail
                        selected = tail_indices[tail_indices % 4 == rank]
                        local_rows = torch.arange(rank, padded, 4, device=device)
                        torch.testing.assert_close(tail.real_rows(local_rows), selected)
                        gathered.append(tail.rows(local_rows))
                        torch.testing.assert_close(
                            tail.positions[: len(selected)], positions[selected]
                        )
                        self.assertTrue((tail.positions[len(selected) :] == 0).all())
                        for field in global_tail.core_metadata._CP_REINDEX_FIELDS:
                            expected = getattr(global_tail.core_metadata, field)
                            actual = getattr(tail_metadata.core_metadata, field)
                            torch.testing.assert_close(
                                actual[: len(selected)],
                                expected[tail_indices % 4 == rank],
                            )
                        torch.testing.assert_close(
                            tail.swa_out_cache_loc,
                            global_tail.late_layer_tail.swa_out_cache_loc,
                        )
                        # Last index source published distinct values for each local query.
                        for ratio in (1, 2):
                            for get_buffer in (
                                "sparse_page_indices",
                                "sparse_raw_indices",
                                "sparse_topk_lengths",
                            ):
                                buf = getattr(full.core_metadata, get_buffer)(ratio)
                                buf.copy_(
                                    local_rows.to(buf.dtype)
                                    .view(-1, *([1] * (buf.ndim - 1)))
                                    .expand_as(buf)
                                )
                        local_lens = interleave_rows_per_request(extend, rank, 4)
                        masks = list(local_rows[: sum(local_lens)].split(local_lens))
                        backend.forward_metadata = full
                        backend.candidate_masks = masks
                        backend.tail_forward_metadata = tail_metadata
                        previous_len = get_local_dp_buffer_len()
                        set_local_dp_buffer_len(padded)
                        try:
                            saved = backend.enter_late_layer_tail(batch)
                            self.assertIs(batch.attn_cp_metadata, tail.cp_metadata)
                            self.assertEqual(
                                get_local_dp_buffer_len(),
                                len(tail.rows(local_rows)) * 4,
                            )
                            torch.testing.assert_close(
                                torch.cat(backend.candidate_masks), selected
                            )
                            for ratio in (1, 2):
                                for get_buffer in (
                                    "sparse_page_indices",
                                    "sparse_raw_indices",
                                    "sparse_topk_lengths",
                                ):
                                    buf = getattr(
                                        tail_metadata.core_metadata, get_buffer
                                    )(ratio)
                                    expected = (
                                        selected.to(buf.dtype)
                                        .view(-1, *([1] * (buf.ndim - 1)))
                                        .expand_as(buf[: len(selected)])
                                    )
                                    torch.testing.assert_close(
                                        buf[: len(selected)], expected
                                    )
                                    if get_buffer != "sparse_topk_lengths":
                                        self.assertTrue(
                                            (buf[len(selected) :] == -1).all()
                                        )
                            backend.exit_late_layer_tail(saved, batch)
                            self.assertIs(backend.forward_metadata, full)
                            self.assertIs(backend.candidate_masks, masks)
                            self.assertIs(batch.attn_cp_metadata, original_cp)
                            self.assertEqual(get_local_dp_buffer_len(), padded)
                        finally:
                            set_local_dp_buffer_len(previous_len)
                torch.testing.assert_close(
                    torch.cat(gathered)[tail.cp_metadata.gather_index], tail_indices
                )

    def test_local_indexer_matches_global_rows(self):
        for extend in ([5, 1, 6], [4, 1, 6]):
            with self.subTest(extend=extend):
                self._check_local_indexer(extend)

    def _check_local_indexer(self, extend):
        from unittest.mock import patch

        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            store_fp4_index_k_cache_split,
        )
        from sglang.kernels.ops.attention.dsv4.torch_quant import fake_quant_fp4
        from sglang.srt.environ import envs
        from sglang.srt.layers.cp import base as cp_base
        from sglang.srt.layers.cp.interleave import (
            InterleaveContextParallelMetadata,
            InterleaveCPStrategy,
            interleave_rows_per_request,
        )
        from sglang.srt.runtime_context import get_parallel

        torch.manual_seed(481)
        device = "cuda"
        backend = _make_backend(
            block_size=5, device=device, low_ratios=(1, 2), is_dspark_draft=False
        )
        context = 1536
        slots = NUM_REQ_SLOTS * context
        backend.req_to_token = torch.arange(
            slots, device=device, dtype=torch.int32
        ).view(NUM_REQ_SLOTS, context)
        backend.token_to_kv_pool.full_to_swa_index_mapping = torch.arange(
            slots, device=device, dtype=torch.int64
        )
        backend.token_to_kv_pool.translate_loc_from_full_to_swa = lambda idx: idx
        backend.token_to_kv_pool.get_index_k_page_size = lambda ratio: 64
        backend.dsa_topk_backend = SimpleNamespace(should_use_topk_v2=lambda: False)
        req = torch.tensor([4, 1, 3], device=device, dtype=torch.int32)
        seq = [prefix + length for prefix, length in zip([1100, 1300, 100], extend)]
        positions = torch.cat(
            [torch.arange(s - e, s, device=device) for s, e in zip(seq, extend)]
        )
        repeated = req.repeat_interleave(torch.tensor(extend, device=device))
        out_loc = backend.req_to_token[repeated.long(), positions].long()
        batch = SimpleNamespace(
            seq_lens_cpu=seq,
            extend_seq_lens_cpu=extend,
            req_pool_indices=req,
            input_ids=torch.arange(12, device=device),
            positions=positions,
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor(extend, device=device, dtype=torch.int32),
            out_cache_loc=out_loc,
        )
        kwargs = dict(
            max_seq_len=max(seq),
            req_pool_indices=req,
            seq_lens=torch.tensor(seq, device=device, dtype=torch.int32),
            seq_lens_cpu=seq,
            out_cache_loc=out_loc,
            num_tokens=sum(extend),
            extend_seq_lens=torch.tensor(extend, device=device, dtype=torch.int32),
            extend_seq_lens_cpu=extend,
        )
        queries = fake_quant_fp4(
            torch.randn(sum(extend), 32, 128, device=device, dtype=torch.bfloat16)
        )
        weights = torch.rand(sum(extend), 32, device=device, dtype=torch.bfloat16)
        indexer = SimpleNamespace(
            n_local_heads=32,
            n_heads=32,
            index_topk=512,
            weights_proj_hip_max_tokens=0,
            queries=lambda q, freqs, positions: q,
            head_weights=lambda x: x,
            candidate_topk_blocks=8,
            candidate_block_size=8,
        )
        for ratio in (1, 2):
            k = fake_quant_fp4(
                torch.randn(slots // ratio, 128, device=device, dtype=torch.bfloat16)
            )
            payload = torch.empty(
                slots // ratio // 64, 1, 4, 64, 16, device=device, dtype=torch.uint8
            ).view(torch.float4_e2m1fn_x2)
            scales = torch.empty(
                slots // ratio // 64, 1, 4, 64, device=device, dtype=torch.uint8
            )
            store_fp4_index_k_cache_split(
                k,
                payload,
                scales,
                torch.arange(slots // ratio, device=device, dtype=torch.int32),
                page_size=64,
                rne=True,
            )
            backend.token_to_kv_pool.get_index_k_fp4_payload_buffer = lambda layer: (
                payload
            )
            backend.token_to_kv_pool.get_index_k_fp4_scale_buffer = lambda layer: scales
            layer = SimpleNamespace(
                indexer=indexer, compress_ratio=ratio, layer_id=0, freqs_cis=None
            )
            full = backend.init_forward_metadata_prefill(**kwargs)
            expected = []
            for rank in (None, 0, 1, 2, 3):
                if rank is None:
                    metadata, rows, lens = full, slice(None), extend
                else:
                    rows = slice(rank, None, 4)
                    lens = interleave_rows_per_request(extend, rank, 4)
                    batch.attn_cp_metadata = InterleaveContextParallelMetadata(
                        per_rank_actual_token=[3] * 4, total_seq_lens=sum(extend)
                    )
                    with (
                        get_parallel().override(attn_cp_size=4, attn_cp_rank=rank),
                        patch.object(
                            cp_base, "_STRATEGY", InterleaveCPStrategy(cp_size=4)
                        ),
                    ):
                        metadata = backend._init_forward_metadata_prefill_from_batch(
                            batch,
                            max_seq_len=max(seq),
                            req_pool_indices=req,
                            seq_lens=kwargs["seq_lens"],
                            seq_lens_cpu=torch.tensor(seq),
                        )
                    for field in full.core_metadata._CP_REINDEX_FIELDS:
                        torch.testing.assert_close(
                            getattr(metadata.core_metadata, field)[: sum(lens)],
                            getattr(full.core_metadata, field)[rows],
                            rtol=0,
                            atol=0,
                        )
                    for field in full.core_metadata._CP_GLOBAL_FIELDS:
                        torch.testing.assert_close(
                            getattr(metadata.core_metadata, field),
                            getattr(full.core_metadata, field),
                            rtol=0,
                            atol=0,
                        )
                backend.forward_metadata = metadata
                backend.candidate_masks = None
                for stage in ("source", "consumer"):
                    indexer.is_candidate_source = stage == "source"
                    indexer.uses_candidates = stage == "consumer"
                    with envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(False):
                        if rank is None:
                            backend._low_ratio_index_topk_dense(
                                layer,
                                weights,
                                queries,
                                positions,
                                batch,
                                torch.tensor(lens, device=device),
                                lens,
                            )
                        else:
                            with get_parallel().override(
                                attn_cp_size=4, attn_cp_rank=rank
                            ):
                                backend._forward_low_ratio_sources_cp(
                                    layer=layer,
                                    x=weights[rows],
                                    q_lora=queries[rows],
                                    positions=positions[rows],
                                    forward_batch=batch,
                                    run_compressor=False,
                                    run_indexer=True,
                                )
                    result = (
                        metadata.core_metadata.sparse_page_indices(ratio),
                        metadata.core_metadata.sparse_raw_indices(ratio),
                    )
                    if rank is None:
                        expected.append(tuple(t.clone() for t in result))
                    else:
                        for actual, global_result in zip(
                            result, expected[stage == "consumer"]
                        ):
                            torch.testing.assert_close(
                                actual[: sum(lens)], global_result[rows], rtol=0, atol=0
                            )
                            self.assertTrue((actual[sum(lens) :] == -1).all())


if __name__ == "__main__":
    unittest.main()

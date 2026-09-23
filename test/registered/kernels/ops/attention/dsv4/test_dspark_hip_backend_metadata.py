"""DSpark metadata on the HIP DeepSeek-V4 backend: the draft block window and the target-verify indexer rows."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=15, suite="stage-b-kernel-test-1-gpu-amd-mi35x")

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
        request_window=None,
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
    # main (#40205): compressed-KV metadata is built only on the target worker.
    backend.need_compress = not is_dspark_draft
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

    def test_graph_capture_and_replay_route_the_draft_through_the_block_window(self):
        """A TARGET_VERIFY capture on the DSpark draft must build the block window, and
        a replay must refresh it in place for new lengths and slots, whether or not the
        replay batch carries a CPU mirror of the lengths (the draft-window bucket must
        not read seq_lens_cpu)."""
        for cpu_mirror in (True, False):
            with self.subTest(cpu_mirror=cpu_mirror):
                self._check_block_window_replay(cpu_mirror=cpu_mirror)

    def _check_block_window_replay(self, *, cpu_mirror):
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


if __name__ == "__main__":
    unittest.main()

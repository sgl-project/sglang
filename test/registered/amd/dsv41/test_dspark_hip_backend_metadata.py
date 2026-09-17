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




if __name__ == "__main__":
    unittest.main()

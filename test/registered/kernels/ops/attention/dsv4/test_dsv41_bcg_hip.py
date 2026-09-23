"""The HIP radix backend under the low-ratio (V4.1) fork: breakable prefill graphs keep the captured metadata active across replays, only the SWA store target keeps its storage and every other field is the eager build; CP tail queries read the same packed SWA cache as an unsharded tail. The non-low-ratio copy-in-place fork is pinned by test/registered/amd/test_dsv4_hip_bcg_metadata.py."""

import copy
import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=25, suite="stage-b-kernel-test-1-gpu-amd-mi35x")


def _extend_batch(
    *,
    seq_lens,
    extend_lens,
    req_pool_indices,
    out_cache_loc,
    device,
) -> ForwardBatch:
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    extend_t = torch.tensor(extend_lens, dtype=torch.int32, device=device)
    positions = torch.cat(
        [
            torch.arange(s - e, s, dtype=torch.int64, device=device)
            for s, e in zip(seq_lens, extend_lens)
        ]
    )
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(seq_lens),
        input_ids=torch.zeros(sum(extend_lens), dtype=torch.int64, device=device),
        req_pool_indices=torch.tensor(
            req_pool_indices, dtype=torch.int32, device=device
        ),
        seq_lens=seq_lens_t,
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=sum(seq_lens),
        positions=positions,
    )
    batch.extend_prefix_lens = seq_lens_t - extend_t
    batch.extend_prefix_lens_cpu = [s - e for s, e in zip(seq_lens, extend_lens)]
    batch.extend_seq_lens = extend_t
    batch.extend_seq_lens_cpu = list(extend_lens)
    batch.extend_start_loc = torch.cumsum(extend_t, dim=0) - extend_t
    batch.extend_num_tokens = sum(extend_lens)
    batch.global_num_token_non_padded_cpu = sum(extend_lens)
    return batch


def _tensor_fields(obj):
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name, None)
        if isinstance(value, torch.Tensor):
            yield f.name, value


# torch.empty_like scratch: the index source writes every row before a consumer reads
# one, so two builds agree on shape, not contents
_UNINITIALIZED_SCRATCH_FIELDS = frozenset(
    {"c1_sparse_raw_indices", "c2_sparse_raw_indices", "c4_sparse_raw_indices"}
)


@unittest.skipUnless(
    is_hip() and torch.cuda.is_available(), "the HIP radix backend is ROCm only"
)
class TestHipBreakableGraphCaptureReplay(CustomTestCase):
    """A captured store must read the refreshed target, and break-time metadata and
    attention must equal the eager build for the same batch."""

    BUCKET = 96

    def test_capture_replay_matches_eager(self):
        self._capture_replay_matches_eager(max_context_size=None)

    def test_fixed_context_capture_replay_matches_eager(self):
        self._capture_replay_matches_eager(max_context_size=512)

    def _capture_replay_matches_eager(self, max_context_size):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.test.kits.attention_unittest.attention_methods.dsv4_attention import (
            DSV4_PAGE_SIZE,
            DSV4AttentionCase,
            _populate_swa_kv_cache,
            build_dsv4_attention_fixture,
        )

        device = "cuda"
        case = DSV4AttentionCase(
            name="hip_bcg_extend",
            backend="dsv4",
            forward_mode=ForwardMode.EXTEND,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(256, 256) if max_context_size is not None else (0, 0),
            extend_lens=(40, 24),
        )
        fixture = build_dsv4_attention_fixture(
            self,
            case,
            device=device,
            max_context_len=2 * DSV4_PAGE_SIZE,
            compression_ratios=[0, 2, 1],
        )
        backend = fixture.backend
        pool = fixture.runner.token_to_kv_pool
        self.assertEqual(backend.low_ratios, (1, 2))
        _populate_swa_kv_cache(
            fixture, max_context_len=2 * DSV4_PAGE_SIZE, device=device
        )
        q_input, _ = fixture.actual_module.project(fixture.input_hidden)
        live = fixture.forward_batch
        live.global_num_token_non_padded_cpu = case.num_input_tokens
        num_tokens = case.num_input_tokens

        def attention(forward_batch):
            return backend.forward(
                q=q_input,
                k=q_input,
                v=q_input,
                layer=fixture.actual_module.attn,
                forward_batch=forward_batch,
                compress_ratio=0,
                save_kv_cache=False,
                attn_sink=fixture.actual_module.attn_sink,
            )

        # capture batch of a bucket: one request of BUCKET tokens on the static (all-zero) out_cache_loc slot
        capture_batch = _extend_batch(
            seq_lens=[self.BUCKET],
            extend_lens=[self.BUCKET],
            req_pool_indices=[0],
            out_cache_loc=torch.zeros(self.BUCKET, dtype=torch.int64, device=device),
            device=device,
        )
        capture_batch.max_seq_len_override = max_context_size
        # static view of the live batch: out_cache_loc padded to the bucket with the dummy slot
        static = copy.copy(live)
        static.max_seq_len_override = max_context_size
        static.out_cache_loc = torch.nn.functional.pad(
            live.out_cache_loc, (0, self.BUCKET - num_tokens), value=0
        )

        with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
            backend.init_forward_metadata(live)
            eager = backend.forward_metadata
            eager_out = attention(live)
            if max_context_size is not None:
                eager = backend._prefill_metadata_for_batch(
                    live, max_seq_len_override=max_context_size
                )
                backend.forward_metadata = eager
                backend.init_forward_metadata_in_graph(live)

            captured = backend.init_forward_metadata_for_breakable_cuda_graph_capture(
                capture_batch
            )
            self.assertIs(backend.forward_metadata, captured)
            pinned = captured.core_attn_metadata.swa_out_cache_loc
            self.assertEqual(tuple(pinned.shape), (self.BUCKET,))
            self.assertEqual(
                captured.core_attn_metadata.page_table.shape[1],
                ((max_context_size or self.BUCKET) + DSV4_PAGE_SIZE - 1)
                // DSV4_PAGE_SIZE,
            )

            # A segment's store reads the target by address.
            sink = torch.empty_like(pinned)
            graph = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                sink.copy_(backend.get_swa_out_cache_loc(capture_batch))
            torch.cuda.current_stream().wait_stream(side)
            with torch.cuda.graph(graph):
                sink.copy_(backend.get_swa_out_cache_loc(capture_batch))

            backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
                captured, live, static_forward_batch=static
            )
            self.assertIs(backend.forward_metadata, captured)
            self.assertIs(captured.core_attn_metadata.swa_out_cache_loc, pinned)
            graph.replay()
            torch.cuda.synchronize()
            expected_target = pool.translate_loc_from_full_to_swa(
                static.out_cache_loc
            ).to(torch.int32)
            self.assertTrue(torch.equal(sink, expected_target))
            self.assertTrue(
                torch.equal(
                    pinned[:num_tokens], eager.core_attn_metadata.swa_out_cache_loc
                )
            )

            # Every break-time field is the eager build for the live batch.
            for name, value in _tensor_fields(eager.core_attn_metadata):
                if name == "swa_out_cache_loc":
                    continue
                with self.subTest(field=name):
                    rebound = getattr(captured.core_attn_metadata, name)
                    if name in _UNINITIALIZED_SCRATCH_FIELDS:
                        self.assertEqual(rebound.shape, value.shape)
                        self.assertEqual(rebound.dtype, value.dtype)
                        self.assertEqual(rebound.device, value.device)
                    else:
                        self.assertTrue(torch.equal(rebound, value))
            for ratio in (1, 2):
                for name in ("page_table", "compressed_seq_lens"):
                    with self.subTest(ratio=ratio, field=name):
                        self.assertTrue(
                            torch.equal(
                                getattr(
                                    captured.low_ratio_indexer_metadata(ratio), name
                                ),
                                getattr(eager.low_ratio_indexer_metadata(ratio), name),
                            )
                        )
            self.assertEqual(set(captured.fp4_low_ratio_prefill_workspaces), {1, 2})

            # The attention break on the refreshed metadata is the eager attention.
            replay_out = attention(live)
            self.assertTrue(torch.equal(replay_out, eager_out))

            # A second replay with a different batch shape rebinds again.
            live2 = _extend_batch(
                seq_lens=[24, 40],
                extend_lens=[24, 40],
                req_pool_indices=[1, 0],
                out_cache_loc=torch.flip(live.out_cache_loc, dims=[0]),
                device=device,
            )
            static2 = copy.copy(live2)
            static2.max_seq_len_override = max_context_size
            static2.out_cache_loc = torch.nn.functional.pad(
                live2.out_cache_loc, (0, self.BUCKET - num_tokens), value=0
            )
            backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
                captured, live2, static_forward_batch=static2
            )
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(
                    sink,
                    pool.translate_loc_from_full_to_swa(static2.out_cache_loc).to(
                        torch.int32
                    ),
                )
            )
            backend.init_forward_metadata(live2)
            eager2 = backend.forward_metadata
            self.assertTrue(
                torch.equal(
                    captured.core_attn_metadata.seq_lens_casual,
                    eager2.core_attn_metadata.seq_lens_casual,
                )
            )
            if max_context_size is not None:
                overflow = copy.copy(live2)
                overflow.seq_lens_cpu = torch.tensor([max_context_size + 1, 40])
                with self.assertRaisesRegex(
                    ValueError, "smaller than the live context"
                ):
                    backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
                        captured, overflow, static_forward_batch=static2
                    )


@unittest.skipUnless(is_hip() and torch.cuda.is_available(), "HIP attention")
class TestDecoderSwaContextParallel(CustomTestCase):
    """CP tail queries must read the same packed SWA cache as an unsharded tail."""

    def test_local_tail_attention_matches_unsharded(self):
        from sglang.srt.layers.cp import base as cp_base
        from sglang.srt.layers.cp.interleave import (
            InterleaveContextParallelMetadata,
            InterleaveCPStrategy,
        )
        from sglang.srt.runtime_context import get_parallel
        from sglang.test.kits.attention_unittest.attention_methods.dsv4_attention import (
            DSV4AttentionCase,
            build_dsv4_attention_fixture,
        )

        case = DSV4AttentionCase(
            name="hip_decoder_cp_tail",
            backend="dsv4",
            forward_mode=ForwardMode.EXTEND,
            num_heads=64,
            page_size=256,
            prefix_lens=(0, 0, 0),
            extend_lens=(257, 1, 130),
        )
        fixture = build_dsv4_attention_fixture(
            self, case, max_context_len=1024, swa_size=4096, compression_ratios=[0]
        )
        self.addCleanup(fixture.runner._server_args_override.restore)
        backend = fixture.backend
        lengths = torch.tensor(case.extend_lens, device="cuda", dtype=torch.int32)
        positions = torch.cat(
            [torch.arange(n, device="cuda") for n in case.extend_lens]
        )
        req = torch.arange(3, device="cuda", dtype=torch.int32)
        for slot in range(3):
            backend.req_to_token[slot] = torch.arange(
                1 + slot * 1024, 1 + (slot + 1) * 1024, device="cuda"
            )
        requests = req.repeat_interleave(lengths.long()).long()
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=3,
            input_ids=torch.zeros_like(positions),
            req_pool_indices=req,
            seq_lens=lengths,
            seq_lens_cpu=lengths.cpu(),
            seq_lens_sum=len(positions),
            positions=positions,
            out_cache_loc=backend.req_to_token[requests, positions].long(),
            extend_seq_lens=lengths,
            extend_seq_lens_cpu=list(case.extend_lens),
        )
        global_metadata = backend._build_late_layer_tail_metadata(batch)
        global_tail = global_metadata.late_layer_tail
        torch.manual_seed(581)
        q = torch.randn(len(positions), 64, 512, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(len(positions), 512, device="cuda", dtype=torch.bfloat16)
        global_k = global_tail.rows(k)
        layer = SimpleNamespace(layer_id=0, v_head_dim=512)
        sink = torch.zeros(64, device="cuda", dtype=torch.float32)

        def attention(query):
            return backend.forward(
                query,
                global_k,
                global_k,
                layer,
                batch,
                compress_ratio=0,
                attn_sink=sink,
            )

        backend.forward_metadata = global_metadata
        expected = attention(global_tail.rows(q))
        outputs = []
        for rank in range(4):
            batch.attn_cp_metadata = InterleaveContextParallelMetadata(
                per_rank_actual_token=[len(positions) // 4] * 4,
                total_seq_lens=len(positions),
            )
            with (
                # a consistent four-rank CP topology: the published tp_size and
                # its MoE decomposition must agree with attn_cp_size
                get_parallel().override(
                    tp_size=4,
                    tp_rank=rank,
                    moe_tp_size=4,
                    attn_cp_size=4,
                    attn_cp_rank=rank,
                ),
                patch.object(cp_base, "_STRATEGY", InterleaveCPStrategy(cp_size=4)),
            ):
                metadata = backend._build_late_layer_tail_metadata(batch)
                tail = metadata.late_layer_tail
                backend.forward_metadata = metadata
                outputs.append(attention(tail.rows(q[rank::4])))
        actual = torch.cat(outputs)[tail.cp_metadata.gather_index]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

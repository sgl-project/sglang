"""Request-window replay must preserve packed KV and match paged HIP attention."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip() and torch.cuda.is_available(), "HIP attention")
class TestEncoderSwaReplay(unittest.TestCase):
    def test_rebuilt_window_matches_paged_attention(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
        from sglang.srt.runtime_context import get_context
        from sglang.test.kits.attention_unittest.attention_methods.dsv4_attention import (
            DSV4AttentionCase,
            build_dsv4_attention_fixture,
        )

        case = DSV4AttentionCase(
            name="hip_request_window",
            backend="dsv4",
            forward_mode=ForwardMode.EXTEND,
            num_heads=64,
            page_size=256,
            prefix_lens=(0, 0),
            extend_lens=(1, 1),
        )
        fixture = build_dsv4_attention_fixture(
            self, case, max_context_len=1024, swa_size=4096, compression_ratios=[0]
        )
        self.addCleanup(fixture.runner._server_args_override.restore)
        backend = fixture.backend
        pool = fixture.runner.token_to_kv_pool
        paged_pool = pool.swa_kv_pool
        for slot in range(2):
            backend.req_to_token[slot] = torch.arange(
                1 + slot * 1024, 1 + (slot + 1) * 1024, device="cuda"
            )

        with get_context().override_server_args(
            enable_encoder_swa_bounded_replay=True,
            chunked_prefill_size=1024,
            max_running_requests=2,
            page_size=256,
        ):
            bounded_pool = DeepSeekV4TokenToKVPool(
                max_num_reqs=2,
                num_req_slots=3,
                swa_size=4096,
                c4_size=0,
                c128_size=0,
                c4_state_pool_size=0,
                c128_state_pool_size=0,
                page_size=256,
                swa_page_size=pool.swa_page_size,
                dtype=torch.float8_e4m3fn,
                c4_state_dtype=torch.float32,
                c128_state_dtype=torch.bfloat16,
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                indexer_head_dim=128,
                layer_num=1,
                device="cuda",
                enable_memory_saver=False,
                compression_ratios=[0],
                full_size=4096,
            )
        bounded_pool.register_mapping(pool.full_to_swa_index_mapping)
        pool = bounded_pool
        backend.token_to_kv_pool = pool
        backend.enable_decoder_swa_bounded_replay = True
        window = pool.request_window
        self.assertIsNotNone(window)
        self.assertIsNone(pool.swa_kv_pool)
        self.assertFalse(pool.needs_paged_swa_allocator)
        layer = SimpleNamespace(layer_id=0, v_head_dim=512)
        sink = torch.zeros(64, device="cuda", dtype=torch.float32)

        def batch(starts, lengths, *, decode=False, replay=False):
            pos = torch.cat(
                [torch.arange(s, s + n, device="cuda") for s, n in zip(starts, lengths)]
            )
            req = torch.repeat_interleave(
                torch.arange(2, device="cuda"), torch.tensor(lengths, device="cuda")
            )
            seq_cpu = torch.tensor(
                [s + n for s, n in zip(starts, lengths)], dtype=torch.int32
            )
            fb = ForwardBatch(
                forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
                batch_size=2,
                input_ids=torch.zeros_like(pos),
                req_pool_indices=torch.arange(2, device="cuda", dtype=torch.int32),
                seq_lens=seq_cpu.cuda(),
                seq_lens_cpu=seq_cpu,
                out_cache_loc=backend.req_to_token[req, pos].long(),
                seq_lens_sum=int(seq_cpu.sum()),
                positions=pos,
                encoder_swa_replay=replay,
            )
            if not decode:
                fb.extend_prefix_lens_cpu = list(starts)
                fb.extend_prefix_lens = torch.tensor(starts, device="cuda")
                fb.extend_seq_lens_cpu = list(lengths)
                fb.extend_seq_lens = torch.tensor(
                    lengths, device="cuda", dtype=torch.int32
                )
                fb.extend_num_tokens = sum(lengths)
                fb.extend_start_loc = (
                    torch.cumsum(fb.extend_seq_lens, 0) - fb.extend_seq_lens
                )
            return fb

        def attention(fb, q, k):
            return backend.forward(q, k, k, layer, fb, compress_ratio=0, attn_sink=sink)

        def reference(fb, q, k):
            pool.request_window, pool.swa_kv_pool = None, paged_pool
            backend.init_forward_metadata(fb)
            if fb.encoder_swa_replay:
                # Rebuilding starts with no history before each request's replay chunk.
                core = backend.forward_metadata.core_attn_metadata
                lengths = (
                    torch.cat(
                        [
                            torch.arange(1, n + 1, device="cuda")
                            for n in fb.extend_seq_lens_cpu
                        ]
                    )
                    .clamp_max(128)
                    .int()
                )
                core.swa_topk_lengths = lengths
                cols = torch.arange(core.swa_page_indices.shape[1], device="cuda")
                core.swa_page_indices.masked_fill_(
                    cols[None, :] >= lengths[:, None], -1
                )
            expected = attention(fb, q, k)
            pool.request_window, pool.swa_kv_pool = window, None
            return expected

        torch.manual_seed(41)
        for starts, lengths, replay in (
            ((0, 0), (300, 150), False),
            ((300, 150), (7, 5), False),
            ((179, 27), (128, 128), True),
        ):
            if replay:
                window.reset(torch.arange(2, device="cuda"))
                window.state.kv_buffer[0].fill_(219)
            fb = batch(starts, lengths, replay=replay)
            q = torch.randn(sum(lengths), 64, 512, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(sum(lengths), 512, device="cuda", dtype=torch.bfloat16)
            expected = reference(fb, q, k)
            backend.init_forward_metadata(fb)
            actual = attention(fb, q, k)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if starts == (0, 0):
                full_metadata = backend.forward_metadata
                tail_metadata = backend.tail_forward_metadata
                tail = tail_metadata.late_layer_tail
                tail_q, tail_k = tail.rows(q), tail.rows(k)
                tail_fb = batch(
                    tuple(s + n - 128 for s, n in zip(starts, lengths)),
                    (128, 128),
                    replay=True,
                )
                expected_tail = reference(tail_fb, tail_q, tail_k)
                backend.forward_metadata = full_metadata
                backend.tail_forward_metadata = tail_metadata
                saved = backend.enter_late_layer_tail(fb)
                actual_tail = attention(fb, tail_q, tail_k)
                torch.testing.assert_close(actual_tail, expected_tail, rtol=0, atol=0)
                backend.exit_late_layer_tail(saved, fb)
                self.assertIs(
                    window.layout,
                    full_metadata.core_attn_metadata.request_window_layout,
                )
            self.assertLessEqual(window.layout.size, 2 * 128 + sum(lengths))
            for slot, end in enumerate(s + n for s, n in zip(starts, lengths)):
                last = torch.arange(
                    end - min(lengths[slot], window.capacity), end, device="cuda"
                )
                self.assertTrue(
                    torch.equal(
                        window.tags[0, slot * window.capacity + last % window.capacity],
                        last,
                    )
                )

        def check_graph_replays(make_batch, offsets):
            fb = make_batch(0)
            rows = fb.positions.numel()
            q = torch.randn(rows, 64, 512, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
            expected = reference(fb, q, k)
            backend.init_forward_metadata(fb)
            actual = attention(fb, q, k)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(window.layout.size, 2 * 128 + rows)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                backend.init_forward_metadata(fb)
                output = attention(fb, q, k)
            for offset in offsets:
                fresh = make_batch(offset)
                for name in ("positions", "seq_lens", "out_cache_loc"):
                    getattr(fb, name).copy_(getattr(fresh, name))
                fb.seq_lens_cpu.copy_(fresh.seq_lens_cpu)
                q.normal_()
                k.normal_()
                expected = reference(fb, q, k)
                graph.replay()
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

        check_graph_replays(
            lambda step: batch((307 + step, 155 + step), (1, 1), decode=True),
            range(1, 5),
        )
        backend.target_verify_num_draft_tokens = 5

        def verify_batch(accepted):
            verify = batch((312 + accepted, 160 + accepted), (5, 5))
            verify.forward_mode = ForwardMode.TARGET_VERIFY
            verify.seq_lens.sub_(5)
            verify.seq_lens_cpu.sub_(5)
            verify.seq_lens_sum -= 10
            return verify

        check_graph_replays(verify_batch, (1, 3))
        window.reset(torch.tensor([0], device="cuda"))
        backend.init_forward_metadata(batch((312, 160), (1, 1), decode=True))
        with self.assertRaisesRegex(RuntimeError, "history is missing"):
            pool.get_swa_raw_buffer(0)


if __name__ == "__main__":
    unittest.main()

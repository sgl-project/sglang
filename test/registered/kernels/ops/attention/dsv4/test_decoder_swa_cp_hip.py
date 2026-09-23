"""CP tail queries must read the same packed SWA cache as an unsharded tail."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=25, suite="stage-b-kernel-test-1-gpu-amd-mi35x")


@unittest.skipUnless(is_hip() and torch.cuda.is_available(), "HIP attention")
class TestDecoderSwaContextParallel(CustomTestCase):
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

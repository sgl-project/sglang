"""MLU paged attention against a CPU GQA reference, including cached prefixes."""

import unittest
import weakref
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch_mlu  # noqa: F401

from sglang.srt.hardware_backend.mlu.attention import MLUAttnBackend
from sglang.srt.hardware_backend.mlu.memory_pool import MLUMHATokenToKVPool
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-1-mlu")


class TestAttentionMLU(CustomTestCase):
    def test_eager_metadata_released_and_capture_addresses_preserved(self):
        """Varying eager batch sizes must not retain old page tables."""
        backend = MLUAttnBackend(
            NS(
                device="mlu",
                page_size=16,
                model_config=NS(context_len=128),
                req_to_token_pool=NS(
                    req_to_token=torch.zeros(8, 128, device="mlu", dtype=torch.int32)
                ),
                token_to_kv_pool=None,
                dtype=torch.bfloat16,
                kv_cache_dtype=torch.bfloat16,
            )
        )

        def batch(size):
            return NS(
                forward_mode=ForwardMode.DECODE,
                batch_size=size,
                req_pool_indices=torch.arange(size, device="mlu"),
                seq_lens=torch.full((size,), 7, device="mlu", dtype=torch.int32),
            )

        previous = None
        for size in range(1, 9):
            backend.init_forward_metadata(batch(size))
            if previous is not None:
                self.assertIsNone(previous(), "Eager page tables were retained")
            previous = weakref.ref(backend.forward_metadata.block_tables)

        backend.init_forward_metadata_out_graph(batch(2), in_capture=True)
        captured = backend.forward_metadata
        addresses = (captured.block_tables.data_ptr(), captured.seq_lens.data_ptr())
        backend.init_forward_metadata(batch(3))
        replay = batch(2)
        replay.seq_lens.fill_(11)
        backend.init_forward_metadata_out_graph(replay)
        self.assertEqual(
            (
                backend.forward_metadata.block_tables.data_ptr(),
                backend.forward_metadata.seq_lens.data_ptr(),
            ),
            addresses,
        )
        torch.testing.assert_close(captured.seq_lens, replay.seq_lens)

    def test_paged_gqa(self):
        torch.manual_seed(42)
        pool = MLUMHATokenToKVPool(
            size=256,
            page_size=16,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=64,
            layer_num=1,
            device="mlu",
            enable_memory_saver=False,
        )

        req = torch.zeros(2, 128, device="mlu", dtype=torch.int32)
        req[0, :32] = torch.cat(
            [torch.arange(16, 32, device="mlu"), torch.arange(80, 96, device="mlu")]
        )
        req[1, :48] = torch.cat(
            [
                torch.arange(48, 64, device="mlu"),
                torch.arange(112, 128, device="mlu"),
                torch.arange(160, 176, device="mlu"),
            ]
        )
        backend = MLUAttnBackend(
            NS(
                device="mlu",
                page_size=16,
                model_config=NS(context_len=128),
                req_to_token_pool=NS(req_to_token=req),
                token_to_kv_pool=pool,
                dtype=torch.bfloat16,
                kv_cache_dtype=torch.bfloat16,
            )
        )
        layer = NS(
            layer_id=0,
            tp_q_head_num=4,
            tp_k_head_num=2,
            tp_v_head_num=2,
            qk_head_dim=64,
            v_head_dim=64,
            scaling=64**-0.5,
            sliding_window_size=-1,
            logit_cap=0.0,
            is_cross_attention=False,
            attn_type=AttentionType.DECODER,
        )
        k = torch.randn(56, 2, 64, device="mlu", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        for mode, lens, qlens in [
            (ForwardMode.EXTEND, [21, 35], [21, 35]),
            (ForwardMode.EXTEND, [21, 35], [2, 3]),
            (ForwardMode.MIXED, [21, 35], [2, 1]),
            (ForwardMode.DECODE, [21, 35], [1, 1]),
        ]:
            indices = torch.tensor([0, 1], device="mlu")
            seq = torch.tensor(lens, device="mlu")
            ext = torch.tensor(qlens, device="mlu")
            loc = torch.cat([req[i, lens[i] - qlens[i] : lens[i]] for i in range(2)])
            pool.set_kv_buffer(layer, torch.cat([req[0, :21], req[1, :35]]), k, v)
            q = torch.randn(sum(qlens), 4, 64, device="mlu", dtype=torch.bfloat16)
            fb = NS(
                batch_size=2,
                req_pool_indices=indices,
                seq_lens=seq,
                seq_lens_cpu=seq.cpu(),
                extend_seq_lens=ext,
                extend_seq_lens_cpu=qlens,
                forward_mode=mode,
                out_cache_loc=loc,
            )
            backend.init_forward_metadata(fb)
            fn = backend.forward_decode if mode.is_decode() else backend.forward_extend
            # Unsupported semantics used to silently produce ordinary causal
            # attention. Exercise the real forward paths, not only a validator.
            for field, value, message in (
                ("logit_cap", 0.5, "logit_cap"),
                ("attn_type", AttentionType.ENCODER_ONLY, "causal decoder"),
                ("attn_type", AttentionType.DECODER_BIDIRECTIONAL, "causal decoder"),
                ("is_cross_attention", True, "cross-attention"),
            ):
                with self.subTest(mode=mode, field=field, value=value):
                    with patch.object(layer, field, value):
                        with self.assertRaisesRegex(ValueError, message):
                            fn(q, None, None, layer, fb, save_kv_cache=False)
            actual = (
                fn(q, None, None, layer, fb, save_kv_cache=False)
                .view(-1, 4, 64)
                .cpu()
                .float()
            )
            offset = 0
            ko = 0
            for n, m in zip(lens, qlens):
                qi = q[offset : offset + m].cpu().float().transpose(0, 1)
                ki = (
                    k[ko : ko + n].cpu().float().repeat_interleave(2, 1).transpose(0, 1)
                )
                vi = (
                    v[ko : ko + n].cpu().float().repeat_interleave(2, 1).transpose(0, 1)
                )
                scores = qi @ ki.transpose(-1, -2) * layer.scaling
                mask = torch.arange(n)[None, :] > torch.arange(n - m, n)[:, None]
                expected = (
                    scores.masked_fill(mask, float("-inf")).softmax(-1) @ vi
                ).transpose(0, 1)
                torch.testing.assert_close(
                    actual[offset : offset + m], expected, rtol=0.03, atol=0.03
                )
                offset += m
                ko += n


if __name__ == "__main__":
    unittest.main()

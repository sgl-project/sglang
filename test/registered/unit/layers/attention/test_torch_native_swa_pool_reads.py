"""torch_native reads a sliding-window layer's KV at the slots it wrote.

With a hybrid SWA pool, a sliding-window layer's K/V lives in the SWA sub-pool,
which numbers its slots apart from the full pool. The backend writes there at
the translated ``swa_out_cache_loc``; its reads must translate the full-pool ids
in ``req_to_token`` the same way, or they gather other tokens' rows, and once a
full id passes the SWA pool's size they index past its buffer.
"""

import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEADS, HEAD_DIM, WINDOW = 2, 8, 3
SWA_LAYER, FULL_LAYER = 0, 1


class TestTorchNativeSWAPoolReads(CustomTestCase):
    def _build(self, size, size_swa):
        torch.manual_seed(0)
        self.pool = SWAKVPool(
            size=size,
            size_swa=size_swa,
            page_size=1,
            dtype=torch.float32,
            head_num=HEADS,
            head_dim=HEAD_DIM,
            swa_attention_layer_ids=[SWA_LAYER],
            full_attention_layer_ids=[FULL_LAYER],
            device="cpu",
        )
        self.allocator = SWATokenToKVPoolAllocator(
            size=size,
            size_swa=size_swa,
            page_size=1,
            dtype=torch.float32,
            device="cpu",
            kvcache=self.pool,
            need_sort=True,
        )
        self.req_to_token_pool = ReqToTokenPool(
            size=2, max_context_len=32, device="cpu", enable_memory_saver=False
        )
        runner_ids = RunnerIds(
            req_to_token=self.req_to_token_pool.req_to_token,
            token_to_kv_pool_allocator=self.allocator,
            token_to_kv_pool=self.pool,
            page_size=1,
            device="cpu",
        )
        self.backend = TorchNativeAttnBackend(
            SimpleNamespace(
                device="cpu",
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.pool,
                runner_ids=runner_ids,
            )
        )

    def _layer(self, layer_id, window):
        return RadixAttention(
            num_heads=HEADS,
            head_dim=HEAD_DIM,
            scaling=HEAD_DIM**-0.5,
            num_kv_heads=HEADS,
            layer_id=layer_id,
            sliding_window_size=window,
        )

    def _diverged_slots(self, num_tokens):
        # One request's out-of-window SWA slots are evicted, another request
        # finishes: the next request's full ids and SWA ids now disagree.
        evicted = self.allocator.alloc(12)
        finished = self.allocator.alloc(12)
        self.allocator.free_swa(evicted)
        self.allocator.free(finished)
        full = self.allocator.alloc(num_tokens)
        swa = self.allocator.translate_loc_from_full_to_swa(full)
        self.assertFalse(torch.equal(full, swa))
        return full

    @staticmethod
    def _reference(q, k, v, window):
        # q: [q_len, H, D] at the last q_len positions of k/v: [kv_len, H, D].
        q_len, kv_len = q.shape[0], k.shape[0]
        q_pos = torch.arange(kv_len - q_len, kv_len).unsqueeze(1)
        k_pos = torch.arange(kv_len).unsqueeze(0)
        mask = k_pos <= q_pos
        if window > -1:
            mask &= k_pos >= q_pos - window
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            attn_mask=mask,
            scale=HEAD_DIM**-0.5,
        ).transpose(0, 1)

    def _check(self, layer_id, window):
        layer = self._layer(layer_id, window)
        prompt_len = 7
        slots = self._diverged_slots(prompt_len + 1)
        self.req_to_token_pool.req_to_token[0, : prompt_len + 1] = slots
        q, k, v = (
            torch.randn(prompt_len + 1, HEADS, HEAD_DIM, dtype=torch.float32)
            for _ in range(3)
        )

        extend_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            out_cache_loc=slots[:prompt_len],
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([prompt_len]),
            extend_prefix_lens=torch.tensor([0]),
            extend_seq_lens=torch.tensor([prompt_len]),
            encoder_lens=None,
        )
        self.backend.init_forward_metadata(extend_batch)
        out = self.backend.forward_extend(
            q[:prompt_len].flatten(1),
            k[:prompt_len],
            v[:prompt_len],
            layer,
            extend_batch,
        )
        torch.testing.assert_close(
            out.view(prompt_len, HEADS, HEAD_DIM),
            self._reference(q[:prompt_len], k[:prompt_len], v[:prompt_len], window),
        )

        decode_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            out_cache_loc=slots[prompt_len:],
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([prompt_len + 1]),
            encoder_lens=None,
        )
        self.backend.init_forward_metadata(decode_batch)
        out = self.backend.forward_decode(
            q[prompt_len:].flatten(1),
            k[prompt_len:],
            v[prompt_len:],
            layer,
            decode_batch,
        )
        torch.testing.assert_close(
            out.view(1, HEADS, HEAD_DIM), self._reference(q[prompt_len:], k, v, window)
        )

    def test_sliding_window_layer_reads_its_own_slots(self):
        # Equal pools: the full ids stay inside the SWA buffer but name other
        # rows. A larger full pool: they also run past the SWA buffer's end.
        for size in (24, 64):
            with self.subTest(full_pool_size=size):
                self._build(size=size, size_swa=24)
                self._check(SWA_LAYER, WINDOW)

    def test_full_layer_reads_full_slots(self):
        self._build(size=64, size_swa=24)
        self._check(FULL_LAYER, -1)


if __name__ == "__main__":
    unittest.main()

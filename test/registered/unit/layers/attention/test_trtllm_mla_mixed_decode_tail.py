"""trtllm_mla serves the decode tail of a mixed extend batch with the decode kernel.

With --enable-mixed-chunk, a prefill chunk carries the running requests' decode
tokens at its end, and the batch runs as EXTEND. trtllm_mla used to attend
those one-token rows through the chunked-prefix MHA path, re-reading each
request's whole context per prefix chunk. The backend now finds that tail once
per forward (from host-side lengths, without a device sync), restricts the
causal and prefix passes to the prefill requests, and hands the tail to the
absorbed MLA decode kernel.

Covered here:
- which trailing requests count as the decode tail;
- the request range the causal / prefix passes cover;
- the decode-tail kernel call against a PyTorch reference, for fp8 and bf16 KV
  (Blackwell only: the kernel is trtllm-gen).

    python3 test/registered/unit/layers/attention/test_trtllm_mla_mixed_decode_tail.py
"""

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLAMixedDecodeTail,
    TRTLLMMLAPrefillMetadata,
    find_mixed_decode_tail_start,
    make_persistent_multi_ctas_kv_counter_buffer,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
PAGE_SIZE = 64


def _is_sm100() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and is_flashinfer_available()
    )


class TestFindMixedDecodeTailStart(CustomTestCase):
    def test_cases(self):
        cases = [
            # (extend lens, prefix lens, expected start)
            ([4096, 1, 1, 1], [0, 900, 17, 4000], 1),  # one prefill chunk + 3 decodes
            ([2048, 300, 1, 1], [0, 512, 64, 64], 2),  # two prefill requests
            ([4096, 1], [8192, 100], 1),  # prefill with its own cached prefix
            ([4096, 2048], [0, 0], None),  # pure prefill
            ([1, 1, 1], [10, 20, 30], None),  # decode-shaped only: no prefill part
            ([4096, 1], [0, 0], None),  # trailing 1-token prompt (no cached prefix)
            ([4096, 1, 1], [0, 0, 50], 2),  # the tail stops at the 1-token prompt
            ([1, 4096], [50, 0], None),  # a 1-token extend that is not trailing
        ]
        for extend_lens, prefix_lens, expected in cases:
            with self.subTest(extend_lens=extend_lens, prefix_lens=prefix_lens):
                self.assertEqual(
                    find_mixed_decode_tail_start(
                        extend_seq_lens=extend_lens, extend_prefix_lens=prefix_lens
                    ),
                    expected,
                )


class TestExtendQueryReqs(CustomTestCase):
    """The causal / prefix passes cover only the prefill requests when, and
    only when, the caller hands them just the prefill rows."""

    def _backend(self, decode_tail):
        extend_lens = torch.tensor([300, 200, 1, 1, 1], dtype=torch.int32)
        backend = TRTLLMMLABackend.__new__(TRTLLMMLABackend)
        backend.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(
            max_seq_len=300,
            cum_seq_lens=torch.nn.functional.pad(extend_lens.cumsum(0), (1, 0)).int(),
            seq_lens=extend_lens,
            decode_tail=decode_tail,
        )
        return backend

    def test_request_range(self):
        decode_tail = TRTLLMMLAMixedDecodeTail(
            num_prefill_reqs=2,
            num_prefill_tokens=500,
            block_kv_indices=torch.zeros((3, 2), dtype=torch.int32),
            seq_lens_k=torch.tensor([10, 20, 30], dtype=torch.int32),
            max_seq_len_k=30,
        )
        forward_batch = SimpleNamespace(batch_size=5)

        backend = self._backend(decode_tail)
        bs, cum_seq_lens, seq_lens = backend._extend_query_reqs(500, forward_batch)
        self.assertEqual(bs, 2)
        self.assertEqual(cum_seq_lens.tolist(), [0, 300, 500])
        self.assertEqual(seq_lens.tolist(), [300, 200])

        # Every row passed (a caller that does not split): the whole batch.
        bs, cum_seq_lens, seq_lens = backend._extend_query_reqs(503, forward_batch)
        self.assertEqual(bs, 5)
        self.assertEqual(cum_seq_lens.tolist(), [0, 300, 500, 501, 502, 503])

        # No decode tail: the whole batch regardless of the row count.
        backend = self._backend(None)
        bs, _, _ = backend._extend_query_reqs(500, forward_batch)
        self.assertEqual(bs, 5)


@unittest.skipUnless(_is_sm100(), "trtllm-gen MLA decode needs SM100 and flashinfer")
class TestForwardMixedDecodeTail(CustomTestCase):
    """forward_mixed_decode_tail = absorbed MLA attention of each tail row over
    its cached latent KV, through the paged block table."""

    NUM_HEADS = 16
    SEQ_LENS = [1, 63, 64, 65, 700, 2049]

    def _run(self, kv_dtype: torch.dtype, atol: float):
        torch.manual_seed(0)
        device = torch.device("cuda")
        num_reqs = len(self.SEQ_LENS)
        kv_dim = KV_LORA_RANK + QK_ROPE_HEAD_DIM

        backend = TRTLLMMLABackend.__new__(TRTLLMMLABackend)
        backend.backend = "trtllm-gen"
        backend.data_type = kv_dtype
        backend.page_size = PAGE_SIZE
        backend.kv_lora_rank = KV_LORA_RANK
        backend.qk_nope_head_dim = QK_NOPE_HEAD_DIM
        backend.qk_rope_head_dim = QK_ROPE_HEAD_DIM
        backend.workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.int8, device=device
        )
        backend._multi_ctas_kv_counter_buffer = (
            make_persistent_multi_ctas_kv_counter_buffer(
                device, self.NUM_HEADS, max_batch_size=num_reqs
            )
        )

        # Each request gets its own shuffled pages of one shared latent cache.
        pages_per_req = [math.ceil(n / PAGE_SIZE) for n in self.SEQ_LENS]
        page_ids = torch.randperm(sum(pages_per_req) + 3, device=device)[
            : sum(pages_per_req)
        ]
        max_blocks = backend._calc_padded_blocks(max(self.SEQ_LENS))
        block_table = torch.full(
            (num_reqs, max_blocks), -1, dtype=torch.int32, device=device
        )
        offset = 0
        for i, n in enumerate(pages_per_req):
            block_table[i, :n] = page_ids[offset : offset + n]
            offset += n
        num_slots = (sum(pages_per_req) + 3) * PAGE_SIZE
        kv_cache = (torch.randn(num_slots, 1, kv_dim, device=device) * 0.5).to(kv_dtype)
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda layer_id: kv_cache
        )

        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=self.NUM_HEADS,
            scaling=(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5,
            k_scale_float=None,
        )
        q_nope_out = torch.randn(
            num_reqs, self.NUM_HEADS, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        q_rope = torch.randn(
            num_reqs,
            self.NUM_HEADS,
            QK_ROPE_HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        decode_tail = TRTLLMMLAMixedDecodeTail(
            num_prefill_reqs=1,
            num_prefill_tokens=128,
            block_kv_indices=block_table,
            seq_lens_k=torch.tensor(self.SEQ_LENS, dtype=torch.int32, device=device),
            max_seq_len_k=max(self.SEQ_LENS),
        )

        with get_parallel().override(dcp_enabled=False):
            out = backend.forward_mixed_decode_tail(
                q_nope_out=q_nope_out,
                q_rope=q_rope,
                layer=layer,
                decode_tail=decode_tail,
            )
        self.assertEqual(out.shape, (num_reqs, self.NUM_HEADS, KV_LORA_RANK))

        # Reference: softmax(q . [c, k_pe]) . c over each request's tokens, with
        # the query rounded to the KV dtype as the kernel receives it.
        query = torch.cat([q_nope_out, q_rope], dim=-1).to(kv_dtype).float()
        pages = kv_cache.float().view(-1, PAGE_SIZE, kv_dim)
        for i, n in enumerate(self.SEQ_LENS):
            kv = pages[block_table[i, : pages_per_req[i]].long()].reshape(-1, kv_dim)
            kv = kv[:n]
            scores = query[i] @ kv.T * layer.scaling
            ref = torch.softmax(scores, dim=-1) @ kv[:, :KV_LORA_RANK]
            torch.testing.assert_close(out[i].float(), ref, atol=atol, rtol=0.05)

    def test_fp8_kv(self):
        self._run(torch.float8_e4m3fn, atol=3e-2)

    def test_bf16_kv(self):
        self._run(torch.bfloat16, atol=1e-2)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch
from test_swapab_attention import make_cache, reference

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

if is_hip():
    from sglang.kernels.ops.attention.dsv4.swapab_gluon_hip import swapab_attention

# fp8 e4m3 0xFF and bf16 0xFFFF are NaN, so a read of a poisoned byte shows as NaN
POISON = 0xFF


def poisoned_cache(page, pages=11):
    raw, _, values = make_cache(page, pages)
    stride = raw.shape[1]
    storage = torch.full((pages + 2, stride), POISON, dtype=torch.uint8, device="cuda")
    storage[:pages] = raw
    storage[0, :576] = POISON
    storage[0, page * 576 : page * 576 + 8] = POISON
    values[0] = float("nan")
    cache = storage[:pages].as_strided((pages, page, 1, 584), (stride, 584, 584, 1))
    return cache, values


def cache_at_top(raw, page, pages_total):
    pages, stride = raw.shape
    big = torch.empty((pages_total, stride), dtype=torch.uint8, device="cuda")
    big[-pages:] = raw
    cache = big.as_strided((pages_total, page, 1, 584), (stride, 584, 584, 1))
    return cache, (pages_total - pages) * page


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(), "requires gfx950"
)
class TestSwapABBufferGather(CustomTestCase):
    def test_poisoned_invalid_rows(self):
        torch.manual_seed(7)
        b, nk, ne = 4, 200, 100
        kv, values = poisoned_cache(64)
        extra_kv, extra_values = poisoned_cache(16)
        q = torch.randn(b, 1, 16, 512, device="cuda").bfloat16() * 0.5
        ids = torch.randint(1, values.shape[0], (b, 1, nk), device="cuda").int()
        ei = torch.randint(1, extra_values.shape[0], (b, 1, ne), device="cuda").int()
        ids[..., ::5] = -1
        ids[..., 1::7] = values.shape[0] + 2
        ei[..., ::3] = -2
        ei[..., 1::4] = extra_values.shape[0] + 17
        lengths = torch.tensor([nk, nk - 37, 64, 0], device="cuda", dtype=torch.int32)
        el = torch.tensor([ne, 5, 0, ne], device="cuda", dtype=torch.int32)
        sink = torch.randn(64, device="cuda")
        out = swapab_attention(q, kv, ids, lengths, sink, extra_kv, ei, el)
        gold = reference(q, values, ids, lengths, sink, (extra_values, ei, el))
        self.assertFalse(out.isnan().any().item())
        torch.testing.assert_close(out.float(), gold.float(), atol=2e-5, rtol=0.004)

    def check_top_pages(self, pages_total, past_buffer_range):
        torch.manual_seed(11)
        b, nk, page = 2, 96, 64
        raw, _, values = make_cache(page)
        kv, shift = cache_at_top(raw, page, pages_total)
        self.assertEqual(shift // page * raw.shape[1] > 2**31, past_buffer_range)
        q = torch.randn(b, 1, 16, 512, device="cuda").bfloat16() * 0.5
        ids = torch.randint(0, values.shape[0], (b, 1, nk), device="cuda").int()
        ids[..., ::9] = -1
        lengths = torch.tensor([nk, nk - 21], device="cuda", dtype=torch.int32)
        sink = torch.randn(64, device="cuda")
        shifted = torch.where(ids >= 0, ids + shift, ids)
        out = swapab_attention(q, kv, shifted, lengths, sink)
        gold = reference(q, values, ids, lengths, sink)
        torch.testing.assert_close(out.float(), gold.float(), atol=2e-5, rtol=0.004)

    def test_odd_page_stride(self):
        torch.manual_seed(13)
        b, nk, page, pages = 2, 96, 64, 11
        raw, _, values = make_cache(page, pages)
        stride = raw.shape[1] + 1
        storage = torch.zeros((pages, stride), dtype=torch.uint8, device="cuda")
        storage[:, :-1] = raw
        kv = storage.as_strided((pages, page, 1, 584), (stride, 584, 584, 1))
        q = torch.randn(b, 1, 16, 512, device="cuda").bfloat16() * 0.5
        ids = torch.randint(0, values.shape[0], (b, 1, nk), device="cuda").int()
        ids[..., ::9] = -1
        lengths = torch.tensor([nk, nk - 21], device="cuda", dtype=torch.int32)
        sink = torch.randn(64, device="cuda")
        out = swapab_attention(q, kv, ids, lengths, sink)
        gold = reference(q, values, ids, lengths, sink)
        torch.testing.assert_close(out.float(), gold.float(), atol=2e-5, rtol=0.004)

    def test_top_of_buffer_range(self):
        stride = 64 * 584 + 256
        self.check_top_pages((2**31 - 2) // stride, past_buffer_range=False)

    def test_past_buffer_range(self):
        stride = 64 * 584 + 256
        self.check_top_pages(2**31 // stride + 12, past_buffer_range=True)


if __name__ == "__main__":
    unittest.main()

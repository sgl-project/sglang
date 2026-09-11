import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA only")
class TestSM90FP4IndexerPrefill(CustomTestCase):
    def setUp(self):
        if torch.cuda.get_device_capability()[0] != 9:
            self.skipTest("SM90 only")

    def test_fused_score_matches_torch_reference(self):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            store_fp4_index_k_cache,
        )
        from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer import (
            fp8_index_logits_prefill,
            quantize_bf16_index_queries_fp8,
            unpack_fp4_index_keys_to_fp8,
        )
        from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4

        torch.manual_seed(7)
        device = "cuda"
        rows, heads, head_dim = 7, 32, 128
        width, page_size, pool_size = 193, 64, 256
        slots = torch.randperm(pool_size, device=device)[:width].to(torch.int64)
        table = torch.zeros(
            pool_size // page_size,
            page_size * 68,
            dtype=torch.uint8,
            device=device,
        )
        keys = fake_quant_fp4(
            torch.randn(width, head_dim, dtype=torch.bfloat16, device=device)
        )
        store_fp4_index_k_cache(
            input=keys,
            cache=table,
            loc=slots.to(torch.int32),
            page_size=page_size,
            rne=True,
        )
        queries = torch.randn(
            rows, heads, head_dim, dtype=torch.bfloat16, device=device
        )
        weights = torch.randn(rows, heads, dtype=torch.bfloat16, device=device)
        lens = torch.tensor([1, 17, 64, 65, 129, 192, 193], device=device)

        fp8_keys = unpack_fp4_index_keys_to_fp8(slots, table, page_size)
        torch.testing.assert_close(fp8_keys.bfloat16(), keys)

        fp8_queries = quantize_bf16_index_queries_fp8(queries)
        actual = fp8_index_logits_prefill(
            fp8_queries,
            weights,
            fp8_keys,
            lens,
        )
        fp8_expected = torch.einsum(
            "bhd,nd->bhn", fp8_queries.float(), fp8_keys.float()
        )
        fp8_expected = fp8_expected.bfloat16().float().relu()
        fp8_expected = (fp8_expected * weights.float().unsqueeze(-1)).bfloat16().float()
        fp8_expected = fp8_expected.sum(dim=1).bfloat16().float()
        columns = torch.arange(width, device=device)
        fp8_expected.masked_fill_(columns[None, :] >= lens[:, None], -torch.inf)
        self.assertEqual(actual.shape, (rows, 196))
        torch.testing.assert_close(
            actual[:, :width], fp8_expected, atol=0.125, rtol=0.02
        )
        self.assertTrue(torch.isneginf(actual[:, width:]).all())


if __name__ == "__main__":
    unittest.main()

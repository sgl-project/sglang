import importlib.util
import os
import unittest
from pathlib import Path

import torch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except Exception:
    if os.getenv("SGLANG_IS_IN_CI"):
        raise
    CustomTestCase = unittest.TestCase

    def register_cpu_ci(*args, **kwargs):
        return None

register_cpu_ci(est_time=5, suite="stage-a-test-cpu", disabled="Requires XPU")


def _load_dsv4_sparse_attn():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = (
        repo_root
        / "python"
        / "sglang"
        / "srt"
        / "layers"
        / "attention"
        / "dsv4_sparse_attention.py"
    )
    spec = importlib.util.spec_from_file_location("dsv4_sparse_attention", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dsv4_sparse_attn


dsv4_sparse_attn = _load_dsv4_sparse_attn()


def _test_device() -> torch.device:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise unittest.SkipTest("XPU is not available")
    return torch.device("xpu")


def _reference_dsv4_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    batch_size, query_len, num_query_heads, head_dim = query.shape
    kv_len, num_kv_heads = key.shape[1], key.shape[2]
    value_dim = value.shape[-1]
    output = query.new_empty((batch_size, query_len, num_query_heads, value_dim))
    repeat_factor = num_query_heads // num_kv_heads

    for batch_idx in range(batch_size):
        for query_idx in range(query_len):
            for head_idx in range(num_query_heads):
                kv_head_idx = head_idx // repeat_factor
                logits = (
                    query[batch_idx, query_idx, head_idx]
                    @ key[batch_idx, :, kv_head_idx].transpose(0, 1)
                ) * softmax_scale

                valid_indices = topk_indices[batch_idx, query_idx]
                valid_indices = valid_indices[
                    (valid_indices >= 0) & (valid_indices < kv_len)
                ].long()
                masked_logits = torch.full_like(logits, torch.finfo(torch.float32).min)
                if valid_indices.numel() > 0:
                    masked_logits[valid_indices] = logits[valid_indices]

                combined_logits = torch.cat(
                    [masked_logits, sinks[head_idx : head_idx + 1]]
                )
                probs = torch.softmax(combined_logits, dim=-1)
                output[batch_idx, query_idx, head_idx] = (
                    probs[:-1].to(value.dtype) @ value[batch_idx, :, kv_head_idx]
                )

    return output


class TestXPUDSV4SparseAttention(CustomTestCase):
    def test_sparse_attention_matches_reference_with_gqa_and_invalid_topk(self):
        device = _test_device()
        torch.manual_seed(0)
        query = torch.randn(2, 3, 4, 5, dtype=torch.float32, device=device)
        key = torch.randn(2, 6, 2, 5, dtype=torch.float32, device=device)
        value = torch.randn(2, 6, 2, 7, dtype=torch.float32, device=device)
        sinks = torch.randn(4, dtype=torch.float32, device=device)
        topk_indices = torch.tensor(
            [
                [[0, 3, -1, 99], [1, 2, 5, -1], [4, 4, 0, -1]],
                [[5, 1, -1, -1], [2, 3, 0, 7], [-1, -1, -1, -1]],
            ],
            dtype=torch.int32,
            device=device,
        )
        softmax_scale = 0.31

        expected = _reference_dsv4_sparse_attn(
            query, key, value, sinks, topk_indices, softmax_scale
        )

        actual = dsv4_sparse_attn(query, key, value, sinks, topk_indices, softmax_scale)
        self.assertEqual(actual.device.type, "xpu")
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_all_invalid_topk_returns_zero_value_contribution(self):
        device = _test_device()
        query = torch.randn(1, 2, 2, 4, dtype=torch.float32, device=device)
        key = torch.randn(1, 3, 2, 4, dtype=torch.float32, device=device)
        value = torch.randn(1, 3, 2, 4, dtype=torch.float32, device=device)
        sinks = torch.tensor([2.0, -1.0], dtype=torch.float32, device=device)
        topk_indices = torch.full((1, 2, 3), -1, dtype=torch.int32, device=device)

        actual = dsv4_sparse_attn(query, key, value, sinks, topk_indices, 1.0)

        self.assertEqual(actual.device.type, "xpu")
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)

    def test_rejects_non_divisible_gqa_heads(self):
        device = _test_device()
        query = torch.randn(1, 1, 3, 4, dtype=torch.float32, device=device)
        key = torch.randn(1, 2, 2, 4, dtype=torch.float32, device=device)
        value = torch.randn(1, 2, 2, 4, dtype=torch.float32, device=device)
        sinks = torch.randn(3, dtype=torch.float32, device=device)
        topk_indices = torch.tensor([[[0, 1]]], dtype=torch.int32, device=device)

        with self.assertRaisesRegex(ValueError, "must be divisible by KV heads"):
            dsv4_sparse_attn(query, key, value, sinks, topk_indices, 1.0)


if __name__ == "__main__":
    unittest.main()
    
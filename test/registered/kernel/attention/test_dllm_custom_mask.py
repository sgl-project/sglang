"""Custom masks must override causal loop bounds, including across query tiles."""

import unittest

import torch
from sglang.kernels.ops.attention.extend_attention import (
    extend_attention_fwd,
    extend_attention_fwd_unified,
)

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class TestDllmCustomMask(unittest.TestCase):
    def _run_case(self, mask_kind, dtype, causal):
        import flashinfer

        torch.manual_seed(19)
        device = "cuda"
        prefixes, lengths = [0, 17], [193, 257]
        q_parts, k_parts, v_parts, masks, expected, flashinfer_outputs = (
            [] for _ in range(6)
        )
        for prefix, length in zip(prefixes, lengths):
            q = torch.randn(length, 4, 64, device=device, dtype=dtype)
            k = torch.randn(prefix + length, 2, 64, device=device, dtype=dtype)
            v = torch.randn_like(k)
            q_pos = prefix + torch.arange(length, device=device)
            k_pos = torch.arange(prefix + length, device=device)
            if mask_kind == "block":
                mask = k_pos[None, :] // 128 <= q_pos[:, None] // 128
            else:
                # A non-triangular mask that reaches well beyond the first
                # query tile, and also masks some prefix keys.
                mask = (k_pos[None, :] + q_pos[:, None]) % 3 != 0
            scores = (
                torch.einsum(
                    "qhd,khd->hqk", q.float(), k.repeat_interleave(2, dim=1).float()
                )
                / 8
            )
            scores.masked_fill_(~mask, float("-inf"))
            ref = torch.einsum(
                "hqk,khd->qhd",
                scores.softmax(dim=-1),
                v.repeat_interleave(2, dim=1).float(),
            )
            expected.append(ref)
            flashinfer_outputs.append(
                flashinfer.single_prefill_with_kv_cache(
                    q, k, v, custom_mask=mask, causal=causal, backend="fa2"
                )
            )
            q_parts.append(q)
            k_parts.append(k)
            v_parts.append(v)
            masks.append(mask.flatten())

        q = torch.cat(q_parts)
        k_buffer, v_buffer = torch.cat(k_parts), torch.cat(v_parts)
        k_extend = torch.cat([k[p:] for k, p in zip(k_parts, prefixes)])
        v_extend = torch.cat([v[p:] for v, p in zip(v_parts, prefixes)])
        qo = torch.tensor(
            [0, lengths[0], sum(lengths)], device=device, dtype=torch.int32
        )
        prefix_indptr = torch.tensor([0, 0, 17], device=device, dtype=torch.int32)
        prefix_indices = torch.arange(lengths[0], lengths[0] + 17, device=device)
        kv_indptr = torch.tensor(
            [0, lengths[0], sum(lengths) + sum(prefixes)],
            device=device,
            dtype=torch.int32,
        )
        kv_indices = torch.arange(k_buffer.shape[0], device=device)
        mask_indptr = torch.tensor(
            [0, masks[0].numel(), sum(m.numel() for m in masks)],
            device=device,
            dtype=torch.int64,
        )
        mask = torch.cat(masks)
        prefix_lens = torch.tensor(prefixes, device=device, dtype=torch.int32)
        actual, unified = torch.empty_like(q), torch.empty_like(q)

        def run():
            extend_attention_fwd(
                q,
                k_extend,
                v_extend,
                actual,
                k_buffer,
                v_buffer,
                qo,
                prefix_indptr,
                prefix_indices,
                mask,
                causal,
                mask_indptr,
                max(lengths),
                1.0,
                1.0,
                skip_prefix_custom_mask=False,
            )
            extend_attention_fwd_unified(
                q,
                unified,
                k_buffer,
                v_buffer,
                1.0,
                1.0,
                qo,
                kv_indptr,
                kv_indices,
                prefix_lens,
                max(lengths),
                custom_mask=mask,
                mask_indptr=mask_indptr,
                is_causal=causal,
            )

        run()
        ref = torch.cat(expected)
        tolerance = 0.015 if dtype == torch.bfloat16 else 0.002
        for output in (actual, unified, torch.cat(flashinfer_outputs)):
            torch.testing.assert_close(
                output.float(), ref, atol=tolerance, rtol=tolerance
            )
        # Capture the same dispatch and buffers used in eager execution.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        actual.zero_()
        unified.zero_()
        graph.replay()
        for output in (actual, unified):
            torch.testing.assert_close(
                output.float(), ref, atol=tolerance, rtol=tolerance
            )

    def test_mask_parity_across_backends(self):
        for kind in ("block", "custom"):
            for dtype in (torch.float16, torch.bfloat16):
                for causal in (True, False):
                    with self.subTest(kind=kind, dtype=dtype, causal=causal):
                        self._run_case(kind, dtype, causal)


if __name__ == "__main__":
    unittest.main()

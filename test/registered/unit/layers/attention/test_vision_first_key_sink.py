"""MiMo first-key vision sink regression against an independent fp64 oracle."""

import math
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.attention import vision
from sglang.srt.layers.attention.vision_sink import apply_first_key_sink
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def reference(q, k, v, cu, sinks=None, window=(64, 64), scale=None, mask=None):
    """Independent fp64 eager oracle: bias real key zero before softmax."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    out, lses = [], []
    bounds = cu.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        if start == end:
            continue
        qq = q[start:end].double().transpose(0, 1)
        kk = (
            k[start:end]
            .double()
            .repeat_interleave(q.shape[1] // k.shape[1], 1)
            .transpose(0, 1)
        )
        vv = (
            v[start:end]
            .double()
            .repeat_interleave(q.shape[1] // k.shape[1], 1)
            .transpose(0, 1)
        )
        scores = (qq @ kk.transpose(-1, -2)) * scale
        qi = torch.arange(end - start, device=q.device)[:, None]
        kj = torch.arange(end - start, device=q.device)[None, :]
        allowed = torch.ones_like(qi - kj, dtype=torch.bool)
        if window[0] >= 0:
            allowed &= kj >= qi - window[0]
        if window[1] >= 0:
            allowed &= kj <= qi + window[1]
        scores.masked_fill_(~allowed, -math.inf)
        if mask is not None:
            part = mask[..., start:end, start:end].squeeze(0)
            if part.dtype == torch.bool:
                scores.masked_fill_(~part, -math.inf)
            else:
                scores += part.double()
        if sinks is not None:
            scores[..., 0] += sinks.double()[:, None]
        lses.append(scores.logsumexp(-1))
        out.append((scores.softmax(-1).nan_to_num() @ vv).transpose(0, 1))
    return torch.cat(out), torch.cat(lses, dim=1)


def inputs(lengths, heads=8, kv_heads=2, dim=64, device="cpu", dtype=torch.float64):
    torch.manual_seed(1011)
    # Real QKV projections have strided token dimensions after splitting.
    fused = torch.randn(
        sum(lengths), heads + 2 * kv_heads, dim, device=device, dtype=dtype
    )
    q, k, v = fused.split((heads, kv_heads, kv_heads), dim=1)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32
    )
    sinks = torch.linspace(-0.9, 1.4, heads, device=device, dtype=dtype)
    return q, k, v, cu, sinks


def run_backend(backend, q, k, v, cu, sinks, window=(64, 64), **kwargs):
    metadata = vision.VisionAttentionMetadata(
        cu, cu[1:] - cu[:-1], int((cu[1:] - cu[:-1]).max())
    )
    return backend(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu,
        bsz=1,
        seq_len=q.shape[0],
        forward_metadata=metadata,
        window_size=window,
        s_aux=sinks,
        sink_mode="first_key",
        **kwargs,
    )


class TestVisionSinkCPU(CustomTestCase):
    def test_correction_matches_fp64_bias_with_gqa_and_boundaries(self):
        for heads, kv in ((8, 8), (8, 2)):
            for window in ((64, 64), (0, 0), (-1, -1), (4, 1)):
                with self.subTest(heads=heads, kv=kv, window=window):
                    q, k, v, cu, sinks = inputs([1, 0, 5, 65, 66, 129], heads, kv)
                    expected, _ = reference(q, k, v, cu, sinks, window)
                    out, lse = reference(q, k, v, cu, window=window)
                    got = apply_first_key_sink(
                        q, k, v, out, lse, sinks, cu, window, 0.125
                    )
                    torch.testing.assert_close(got, expected, atol=2e-14, rtol=2e-14)

    def test_sdpa_masks_gqa_and_full_attention(self):
        q, k, v, cu, sinks = inputs([4, 7], dim=16)
        for window, bias in (((2, 2), sinks), ((-1, -1), None)):
            for floating in (False, True):
                mask = torch.ones(1, 1, 11, 11, dtype=torch.bool)
                mask[..., 0] = False
                if floating:
                    mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(
                        ~mask, -math.inf
                    )
                    mask[..., 6] += 0.2
                expected, _ = reference(q, k, v, cu, bias, window, mask=mask)
                backend = vision.VisionSdpaAttention(16, 8, 2, flatten_batch=True)
                got = run_backend(
                    backend, q, k, v, cu, bias, window, attention_mask=mask
                )
                torch.testing.assert_close(got, expected, atol=2e-14, rtol=2e-14)

    def test_single_precision_sdpa_branch(self):
        q, k, v, cu, sinks = inputs([5, 9], dtype=torch.float32)
        expected, _ = reference(q, k, v, cu, sinks)
        backend = vision.VisionSdpaAttention(
            64, 8, 2, flatten_batch=True, softmax_in_single_precision=True
        )
        got = run_backend(backend, q, k, v, cu, sinks)
        torch.testing.assert_close(got.double(), expected, atol=5e-7, rtol=2e-5)

    def test_backend_fallback_only_for_first_key_sinks(self):
        for use_sink, mode, expected in (
            (True, "first_key", "sdpa"),
            (False, "first_key", "triton_attn"),
            (True, "virtual", "triton_attn"),
        ):
            with self.subTest(use_sink=use_sink, mode=mode), ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        vision,
                        "get_parallel",
                        return_value=SimpleNamespace(attn_tp_size=1, attn_tp_rank=0),
                    )
                )
                stack.enter_context(
                    patch.object(
                        vision.VisionAttention,
                        "_determine_attention_backend",
                        return_value="triton_attn",
                    )
                )
                stack.enter_context(
                    patch.object(
                        vision, "QKVParallelLinear", return_value=nn.Identity()
                    )
                )
                stack.enter_context(
                    patch.object(
                        vision, "RowParallelLinear", return_value=nn.Identity()
                    )
                )
                stack.enter_context(
                    patch.object(vision, "reject_attn_tp_shard_with_tp_reduce")
                )
                layer = vision.VisionAttention(
                    embed_dim=32,
                    num_heads=4,
                    projection_size=32,
                    use_qkv_parallel=True,
                    qkv_backend="triton_attn",
                    use_sink=use_sink,
                    sink_mode=mode,
                )
                self.assertEqual(layer.qkv_backend_name, expected)
                self.assertEqual(layer.sinks is not None, use_sink)

    def test_flash_contract_requests_lse_without_virtual_sink(self):
        q, k, v, cu, sinks = inputs([5, 9])
        captured = []

        def fake_flash(q, k, v, **kwargs):
            captured.append(kwargs)
            out, lse = reference(q, k, v, cu, window=kwargs["window_size"])
            return out, lse, None

        with (
            patch.object(vision, "_is_cuda", True),
            patch.object(vision, "flash_attn_func", fake_flash, create=True),
        ):
            for ver in (3, 4):
                backend = getattr(vision, f"VisionFlash{ver}Attention")()
                got = run_backend(backend, q, k, v, cu, sinks)
                expected, _ = reference(q, k, v, cu, sinks)
                torch.testing.assert_close(got, expected, atol=2e-14, rtol=2e-14)
                self.assertTrue(captured[-1]["return_softmax_lse"])
                self.assertNotIn("sinks", captured[-1])
                self.assertEqual(captured[-1]["ver"], ver)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()

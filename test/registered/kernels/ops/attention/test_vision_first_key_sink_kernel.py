"""MiMo first-key vision sink regression against an independent fp64 oracle."""

import math
import unittest

import torch

from sglang.srt.layers.attention import vision
from sglang.srt.layers.attention.vision_sink import apply_first_key_sink
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


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


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9,
    "FA4 requires SM90+",
)
class TestVisionSinkCUDA(CustomTestCase):
    def test_fa4_matches_eager_and_sdpa(self):
        for heads, kv in ((8, 8), (32, 8), (8, 2)):
            with self.subTest(heads=heads, kv=kv):
                q, k, v, cu, sinks = inputs(
                    [1, 5, 65, 66, 129], heads, kv, device="cuda", dtype=torch.bfloat16
                )
                got = run_backend(vision.VisionFlash4Attention(), q, k, v, cu, sinks)
                expected, _ = reference(q, k, v, cu, sinks)
                sdpa = run_backend(
                    vision.VisionSdpaAttention(64, heads, kv, flatten_batch=True),
                    q,
                    k,
                    v,
                    cu,
                    sinks,
                )
                torch.testing.assert_close(
                    got.double(), expected, atol=0.012, rtol=0.02
                )
                torch.testing.assert_close(
                    sdpa.double(), expected, atol=0.012, rtol=0.02
                )
                # Reproduce the old bug with the real kernel, not a kwargs-only mock.
                legacy = vision.flash_attn_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu,
                    cu_seqlens_k=cu,
                    max_seqlen_q=129,
                    max_seqlen_k=129,
                    window_size=(64, 64),
                    sinks=sinks,
                    ver=4,
                )
                rmse = (got.double() - expected).square().mean().sqrt().item()
                legacy_rmse = (legacy.double() - expected).square().mean().sqrt().item()
                print(
                    f"heads={heads}/{kv}: corrected RMSE={rmse:.6g}, old sink RMSE={legacy_rmse:.6g}",
                    flush=True,
                )
                self.assertLess(rmse * 5, legacy_rmse)

    def test_zero_sink_and_invisible_key_are_unchanged(self):
        q, k, v, cu, sinks = inputs([66, 130], device="cuda", dtype=torch.bfloat16)
        q[65].fill_(1000)
        k[0].fill_(1000)  # invisible to query 65; exp(q @ k0 - lse) would overflow
        base, lse = vision.flash_attn_func(
            q,
            k,
            v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=130,
            max_seqlen_k=130,
            window_size=(64, 64),
            return_softmax_lse=True,
            ver=4,
        )[:2]
        got = apply_first_key_sink(
            q, k, v, base.clone(), lse, sinks * 0, cu, (64, 64), 0.125
        )
        torch.testing.assert_close(got, base, atol=0, rtol=0)
        got = apply_first_key_sink(
            q, k, v, base.clone(), lse, sinks, cu, (64, 64), 0.125
        )
        self.assertTrue(torch.isfinite(got).all())
        torch.testing.assert_close(got[65], base[65], atol=0, rtol=0)
        torch.testing.assert_close(got[131:], base[131:], atol=0, rtol=0)

    def test_cuda_graph_replay_with_changed_segment_bounds(self):
        q, k, v, cu, sinks = inputs([70, 90], device="cuda", dtype=torch.bfloat16)
        backend = vision.VisionFlash4Attention()
        metadata = vision.VisionAttentionMetadata(cu, cu[1:] - cu[:-1], 100)

        def run():
            return backend(
                q,
                k,
                v,
                cu,
                1,
                160,
                forward_metadata=metadata,
                window_size=(64, 64),
                s_aux=sinks,
                sink_mode="first_key",
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        cu.copy_(torch.tensor([0, 100, 160], device="cuda", dtype=torch.int32))
        graph.replay()
        expected, _ = reference(q, k, v, cu, sinks)
        torch.testing.assert_close(output.double(), expected, atol=0.012, rtol=0.02)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()

"""Actual SGLang AITER varlen adapter, new/old API and graph parity."""

import json
from unittest.mock import patch

import torch
from aiter.ops.triton.attention import mha

from sglang.multimodal_gen.runtime.layers.attention.backends import aiter as backend

assert backend.USE_AITER_GFX942
assert backend._use_int32_varlen_strides
original = mha.flash_attn_varlen_func
torch.manual_seed(20260913)
impl = backend.AITerImpl(4, 128, 128**-0.5)
with torch.inference_mode():
    for length in (129, 1024):
        q, k, v = [
            torch.randn(length, 4, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        control = original(q, k, v, cu, cu, length, length)
        actual = impl.forward_varlen(q, k, v, cu_seqlens=cu, max_seqlen=length)
        torch.testing.assert_close(actual, control, rtol=0, atol=0)

        # A genuine old callable must never receive the new keyword.
        def old_api(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
        ):
            return original(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        with patch.object(mha, "flash_attn_varlen_func", old_api):
            fallback = impl.forward_varlen(q, k, v, cu_seqlens=cu, max_seqlen=length)
        torch.testing.assert_close(fallback, control, rtol=0, atol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                impl.forward_varlen(q, k, v, cu_seqlens=cu, max_seqlen=length)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = impl.forward_varlen(q, k, v, cu_seqlens=cu, max_seqlen=length)
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, control, rtol=0, atol=0)
        assert mha._USE_INT64_STRIDES is True
        print(
            json.dumps(
                {
                    "length": length,
                    "real_sglang_adapter_new_old_graph": "pass",
                    "source": backend.__file__,
                }
            ),
            flush=True,
        )
print("H3_ENGINE_ADAPTER_PASS_NO_VIDEO_ENDPOINT", flush=True)

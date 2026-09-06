"""GPU check that the GLM-5-Next vision precompile hook loads the real kernel.

Runs ``precompile_kernels_after_loading`` against the real Triton
``context_attention_fwd`` entry point, then issues one call shaped like the
vision tower's own: several variable-length image sequences, ``q``/``k``
contiguous (as ``_apply_qk_norm_head_size`` and RoPE leave them) and ``v`` a
strided view of the fused qkv projection buffer (``pass_strided_qkv``). The
test asserts through Triton's per-device kernel cache on ``_fwd_kernel``
(``JITFunction.device_caches[device] -> (kernel_cache, ...)``, Triton 3.7)
and through ``triton.knobs.runtime.kernel_load_start_hook`` that the
representative call neither compiles nor device-loads a new specialization,
and checks its output against an SDPA reference.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.models.glm5_next as glm5_next
from sglang.kernels.ops.attention import prefill_attention
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_HIDDEN = 1536
_HEADS = 12
_HEAD_DIM = _HIDDEN // _HEADS
_SEQ_LENS = [612, 1035, 288]  # not multiples of the kernel's block size


class _FakeVisionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_backend_name = "triton_attn"


class _FakeVisionTower(nn.Module):
    """Only what the hook reads: hidden_size, num_heads, dtype, device."""

    def __init__(self):
        super().__init__()
        self.hidden_size = _HIDDEN
        self.num_heads = _HEADS
        self.attn = _FakeVisionAttention()
        self.proj = nn.Linear(2, 2, bias=False).to(device="cuda", dtype=torch.bfloat16)

    @property
    def dtype(self) -> torch.dtype:
        return self.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.proj.weight.device


def _make_model():
    model = glm5_next.Glm5NextForConditionalGeneration.__new__(
        glm5_next.Glm5NextForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.visual = _FakeVisionTower()
    model.pp_group = SimpleNamespace(is_first_rank=True)
    return model


def _kernel_cache() -> dict:
    jit_fn = prefill_attention._fwd_kernel
    caches = getattr(jit_fn, "device_caches", None)
    assert caches is not None, (
        "triton JITFunction has no device_caches; the cache layout assumed "
        "here is Triton 3.7's (kernel_cache, kernel_key_cache, target, backend, binder)"
    )
    return caches[torch.cuda.current_device()][0]


def _vision_qkv():
    """Return (q, k, v, cu_seqlens, seq_lens, max_seqlen) in the tower's layout."""
    total = sum(_SEQ_LENS)
    gen = torch.Generator(device="cuda").manual_seed(0)
    qkv = torch.randn(
        (1, total, 3 * _HIDDEN), generator=gen, device="cuda", dtype=torch.bfloat16
    )
    q, k, v = qkv.split([_HIDDEN, _HIDDEN, _HIDDEN], dim=-1)
    q = q.reshape(total, _HEADS, _HEAD_DIM)
    k = k.reshape(total, _HEADS, _HEAD_DIM)
    v = v.reshape(total, _HEADS, _HEAD_DIM)
    # qk-norm and RoPE hand the backend fresh contiguous q/k; v is still the
    # strided view into the fused projection output.
    q = q.contiguous()
    k = k.contiguous()
    assert v.stride(0) == 3 * _HIDDEN and not v.is_contiguous()
    seq_lens = torch.tensor(_SEQ_LENS, device="cuda", dtype=torch.int32)
    cu_seqlens = torch.zeros(len(_SEQ_LENS) + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)
    return q, k, v, cu_seqlens, seq_lens, max(_SEQ_LENS)


def _reference(q, k, v, cu_seqlens):
    out = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        qs = q[start:end].float().transpose(0, 1)
        ks = k[start:end].float().transpose(0, 1)
        vs = v[start:end].float().transpose(0, 1)
        o = torch.nn.functional.scaled_dot_product_attention(qs, ks, vs)
        out[start:end] = o.transpose(0, 1).to(out.dtype)
    return out


class TestGlm5NextVisionPrecompileGpu(CustomTestCase):
    def setUp(self):
        import triton.knobs as knobs

        self._loads = []
        self._hook = knobs.runtime.kernel_load_start_hook
        self._hook.add(self._on_load)

    def tearDown(self):
        self._hook.remove(self._on_load)

    def _on_load(self, module, function, name, metadata_group, hash):
        self._loads.append(name)

    def test_hook_loads_the_kernel_the_vision_call_reuses(self):
        cache = _kernel_cache()
        entries_before = len(cache)

        with patch.object(glm5_next, "VisionAttention", _FakeVisionAttention):
            _make_model().precompile_kernels_after_loading()
        torch.cuda.synchronize()

        entries_after_hook = len(cache)
        self.assertGreaterEqual(entries_after_hook, 1)
        # Another test in this process may already have compiled it.
        self.assertIn(entries_after_hook - entries_before, (0, 1))
        loads_after_hook = len(self._loads)

        q, k, v, cu_seqlens, seq_lens, max_seqlen = _vision_qkv()
        out = torch.empty_like(q)
        prefill_attention.context_attention_fwd(
            q, k, v, out, cu_seqlens, seq_lens, max_seqlen, is_causal=False
        )
        torch.cuda.synchronize()

        self.assertEqual(len(cache), entries_after_hook)
        self.assertEqual(len(self._loads), loads_after_hook)
        torch.testing.assert_close(
            out, _reference(q, k, v, cu_seqlens), atol=2e-2, rtol=2e-2
        )


if __name__ == "__main__":
    unittest.main()

"""The gfx942 MLA v absorb under CUDA graph capture (fp8 Triton BMM) against
the eager path (w_vc dequantized to bf16, then torch.bmm).

The two differ only by the kernel's per-128-group fp8 quantization of the
attention output, so they agree to fp8 rounding.
"""

import os
from types import SimpleNamespace

os.environ.setdefault("SGLANG_USE_AITER", "1")

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="base-b-test-1-gpu-small-amd")

if not torch.cuda.is_available():
    pytest.skip("GPU required", allow_module_level=True)

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)

if not is_fp8_fnuz():
    pytest.skip("gfx942 (fp8 e4m3fnuz) only", allow_module_level=True)

KV_LORA_RANK = 512
FNUZ = torch.float8_e4m3fnuz
# One e4m3 rounding of every activation is about 2.5e-2 relative RMS on normal
# inputs, and the output inherits it; a layout or scale bug is O(1).
MAX_REL_ERR = 5e-2


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _attn(heads: int, v_head_dim: int):
    """A layer as the HIP weight loader leaves it: per-tensor fp8 fnuz w_vc,
    a (heads, kv_lora_rank, v_head_dim) transposed view of a contiguous
    (heads, v_head_dim, kv_lora_rank) tensor."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn(heads, v_head_dim, KV_LORA_RANK, device="cuda", generator=gen)
    w_scale = w.abs().max() / torch.finfo(FNUZ).max
    w_vc = (w / w_scale).to(FNUZ).transpose(1, 2)
    return SimpleNamespace(
        # The absorb gate reads only w_kc's dtype, which matches w_vc's.
        w_kc=w_vc,
        w_vc=w_vc,
        w_scale=w_scale,
        num_local_heads=heads,
        o_proj=SimpleNamespace(weight=torch.empty(1, device="cuda")),
    )


@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 64, 256, 300])
@pytest.mark.parametrize("heads, v_head_dim", [(8, 256), (16, 128)])
def test_capture_matches_eager(tokens, heads, v_head_dim, monkeypatch):
    # The eager path reads one exec flag; a kernel test has no published exec config.
    monkeypatch.setattr(
        forward_mla_rocm,
        "get_exec",
        lambda: SimpleNamespace(kernel=SimpleNamespace(enable_amd_monokernel=False)),
    )
    attn = _attn(heads, v_head_dim)
    assert forward_mla_rocm._use_fp8_absorb_bmm(attn, is_capture_mode=True)
    gen = torch.Generator(device="cuda").manual_seed(1)
    attn_output = torch.randn(
        tokens, heads, KV_LORA_RANK, device="cuda", generator=gen
    ).bfloat16()

    captured = forward_mla_rocm.rocm_absorb_v_bmm(
        attn, attn_output, is_capture_mode=True
    )
    eager = forward_mla_rocm.rocm_absorb_v_bmm(attn, attn_output, is_capture_mode=False)

    assert captured.shape == eager.shape == (tokens, heads * v_head_dim)
    assert captured.dtype == torch.bfloat16
    assert _rel_err(captured, eager) < MAX_REL_ERR


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion import (
    can_use_helios_qk_rope,
    fused_inplace_helios_qk_rope,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(value: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_1, x_2 = value.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs.unsqueeze(-2).chunk(2, dim=-1)
    out = torch.empty_like(value)
    out[..., 0::2] = x_1 * cos[..., 0::2] - x_2 * sin[..., 1::2]
    out[..., 1::2] = x_1 * sin[..., 1::2] + x_2 * cos[..., 0::2]
    return out.type_as(value)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tokens,heads,head_dim",
    [
        (1, 1, 64),
        (17, 8, 128),
        (129, 4, 256),
        (8640, 40, 128),
    ],
)
def test_helios_qk_rope_matches_eager_transposed_path(
    dtype: torch.dtype,
    tokens: int,
    heads: int,
    head_dim: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260826)
    q = torch.randn(
        tokens, heads, head_dim, device="cuda", dtype=dtype, generator=generator
    )
    k = torch.randn_like(q)
    freqs = torch.randn(
        tokens, 2 * head_dim, device="cuda", dtype=torch.float32, generator=generator
    )
    q_ref = _reference(q, freqs)
    k_ref = _reference(k, freqs)
    q_out, k_out = q.clone(), k.clone()
    q_ptr, k_ptr = q_out.data_ptr(), k_out.data_ptr()

    fused_inplace_helios_qk_rope(q_out, k_out, freqs)
    torch.cuda.synchronize()

    assert q_out.data_ptr() == q_ptr
    assert k_out.data_ptr() == k_ptr
    assert torch.equal(q_out, q_ref)
    assert torch.equal(k_out, k_ref)


def test_helios_qk_rope_runtime_guards() -> None:
    q = torch.randn(1, 17, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    freqs = torch.randn(1, 17, 256, device="cuda", dtype=torch.float32)
    assert can_use_helios_qk_rope(q, k, freqs)
    assert not can_use_helios_qk_rope(q.float(), k, freqs)
    assert not can_use_helios_qk_rope(q, k.float(), freqs)
    assert not can_use_helios_qk_rope(q, k, freqs.bfloat16())
    assert not can_use_helios_qk_rope(q.cpu(), k.cpu(), freqs.cpu())
    assert not can_use_helios_qk_rope(q, k, freqs[..., :-2])
    assert not can_use_helios_qk_rope(q[:, :, :, ::2], k, freqs)
    assert not can_use_helios_qk_rope(q[:, :0], k[:, :0], freqs[:, :0])

    q_unaligned = torch.empty(q.numel() + 1, device=q.device, dtype=q.dtype)[
        1:
    ].view_as(q)
    k_unaligned = torch.empty(k.numel() + 1, device=k.device, dtype=k.dtype)[
        1:
    ].view_as(k)
    assert q_unaligned.is_contiguous() and q_unaligned.storage_offset() == 1
    assert k_unaligned.is_contiguous() and k_unaligned.storage_offset() == 1
    assert not can_use_helios_qk_rope(q_unaligned, k, freqs)
    assert not can_use_helios_qk_rope(q, k_unaligned, freqs)


def test_helios_attention_dispatch_and_tp_fallback() -> None:
    import sglang.multimodal_gen.runtime.models.dits.helios as helios

    attention = helios.HeliosSelfAttention.__new__(helios.HeliosSelfAttention)
    nn.Module.__init__(attention)
    attention.tp_rmsnorm = False
    q = torch.randn(1, 17, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    freqs = torch.randn(1, 17, 256, device="cuda", dtype=torch.float32)

    with (
        patch.object(helios, "can_use_helios_qk_rope", return_value=True),
        patch.object(helios, "fused_inplace_helios_qk_rope") as fused,
    ):
        q_out, k_out = attention._apply_rotary_qk(q, k, freqs)
    assert q_out is q
    assert k_out is k
    fused.assert_called_once()
    assert fused.call_args.args[0].shape == (17, 8, 128)
    assert fused.call_args.args[1].shape == (17, 8, 128)
    assert fused.call_args.args[2].shape == (17, 256)

    attention.tp_rmsnorm = True
    with patch.object(helios, "fused_inplace_helios_qk_rope") as fused:
        q_out, k_out = attention._apply_rotary_qk(q, k, freqs)
    fused.assert_not_called()
    assert torch.equal(q_out, _reference(q, freqs))
    assert torch.equal(k_out, _reference(k, freqs))


def test_helios_qk_rope_fullgraph_custom_op() -> None:
    q = torch.randn(1, 17, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    freqs = torch.randn(1, 17, 256, device="cuda", dtype=torch.float32)
    q_ref, k_ref = _reference(q, freqs), _reference(k, freqs)

    @torch.compile(fullgraph=True)
    def compiled(
        q_arg: torch.Tensor, k_arg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused_inplace_helios_qk_rope(
            q_arg.view(-1, q_arg.shape[-2], q_arg.shape[-1]),
            k_arg.view(-1, k_arg.shape[-2], k_arg.shape[-1]),
            freqs.view(-1, freqs.shape[-1]),
        )
        return q_arg, k_arg

    q_out, k_out = compiled(q.clone(), k.clone())
    assert torch.equal(q_out, q_ref)
    assert torch.equal(k_out, k_ref)


def test_helios_attention_fullgraph_dispatch() -> None:
    import sglang.multimodal_gen.runtime.models.dits.helios as helios

    attention = helios.HeliosSelfAttention.__new__(helios.HeliosSelfAttention)
    nn.Module.__init__(attention)
    attention.tp_rmsnorm = False
    q = torch.randn(1, 17, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    freqs = torch.randn(1, 17, 256, device="cuda", dtype=torch.float32)
    q_ref, k_ref = _reference(q, freqs), _reference(k, freqs)

    compiled = torch.compile(attention._apply_rotary_qk, fullgraph=True)
    q_out, k_out = compiled(q.clone(), k.clone(), freqs)

    assert torch.equal(q_out, q_ref)
    assert torch.equal(k_out, k_ref)


def test_helios_qk_rope_rejects_bad_frequency_shape() -> None:
    q = torch.randn(3, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    freqs = torch.randn(3, 128, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="freq_dim"):
        fused_inplace_helios_qk_rope(q, k, freqs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

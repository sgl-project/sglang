import sys
from unittest.mock import patch

import pytest
import torch

import sglang.kernels.ops.diffusion as ops
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def _inputs(shape=(2, 17, 3, 128), dtype=torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(42)
    q, k = [
        torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
        for _ in range(2)
    ]
    angles = torch.randn(shape[1], shape[-1], device="cuda", generator=generator)
    return q, k, angles.cos(), angles.sin()


def _reference(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return (
        x.float() * cos[None, :, None, :] + rotated.float() * sin[None, :, None, :]
    ).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1, 1, 128), (2, 17, 3, 128), (1, 4096, 16, 128)])
def test_anima_rope_bit_exact(shape, dtype):
    q, k, cos, sin = args = _inputs(shape, dtype)
    saved = [t.clone() for t in args]
    outputs = ops.fused_rope_rotate_half_fp32(*args)
    for output, x in zip(outputs, (q, k)):
        assert output.dtype == x.dtype and output.shape == x.shape
        assert torch.equal(output, _reference(x, cos, sin))
        assert output.data_ptr() not in (q.data_ptr(), k.data_ptr())
    assert all(torch.equal(t, original) for t, original in zip(args, saved))


def test_anima_rope_input_guards():
    q, k, cos, sin = args = _inputs()
    assert ops.can_use_fused_rope_rotate_half_fp32(*args)
    unsupported = [
        (q.float(), k.float(), cos, sin),
        (q[:, :, ::2], k[:, :, ::2], cos, sin),
        _inputs((1, 17, 3, 64)),
        (q, k, cos.bfloat16(), sin.bfloat16()),
        (q, k, cos[:-1], sin[:-1]),
        tuple(t.cpu() for t in args),
    ]
    for inputs in unsupported:
        assert not ops.can_use_fused_rope_rotate_half_fp32(*inputs)
        with pytest.raises(ValueError, match="Expected contiguous CUDA"):
            ops.fused_rope_rotate_half_fp32(*inputs)


def test_anima_rope_dispatch_and_fallback():
    import sglang.multimodal_gen.runtime.models.dits.anima as anima

    q, k, cos, sin = args = _inputs()
    gate = ops.BitExactFusionGate("Anima test", per_signature=True)
    with (
        patch.object(anima, "_ANIMA_ROPE", gate),
        patch.object(
            ops, "fused_rope_rotate_half_fp32", wraps=ops.fused_rope_rotate_half_fp32
        ) as fused,
        patch.object(
            anima, "_anima_rope_eager", wraps=anima._anima_rope_eager
        ) as eager,
    ):
        for _ in range(2):
            outputs = anima._anima_rope(*args)
            assert all(
                torch.equal(out, _reference(x, cos, sin))
                for out, x in zip(outputs, (q, k))
            )
        assert fused.call_count == 2
        assert eager.call_count == 1  # Only the first call verifies against eager.
        assert gate.is_verified((q.device, q.dtype, q.shape)) and not gate.disabled

        fused.reset_mock()
        outputs = anima._anima_rope(q.float(), k.float(), cos, sin)
        fused.assert_not_called()
        assert eager.call_count == 2
        assert all(
            torch.equal(out, _reference(x.float(), cos, sin))
            for out, x in zip(outputs, (q, k))
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

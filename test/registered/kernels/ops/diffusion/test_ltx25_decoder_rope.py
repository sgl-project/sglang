import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.diffusion import (
    can_use_ltx25_decoder_rope,
    fused_ltx25_decoder_rope,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DIM_SPLIT = (16, 24, 24)


def make_tables(
    frames: int, height: int, width: int
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    tables = []
    for length, dim in zip((frames, height, width), DIM_SPLIT, strict=True):
        exponents = torch.arange(0, dim, 2, dtype=torch.float64, device="cuda") / dim
        inv_freqs = (1.0 / 10000.0**exponents).to(torch.float32)
        positions = torch.arange(length, dtype=torch.float32, device="cuda")
        angles = positions[:, None] * inv_freqs[None, :]
        tables.append((angles.cos(), angles.sin()))
    return tuple(tables)


def eager_rope(
    hidden_states: torch.Tensor,
    tables: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> torch.Tensor:
    outputs = []
    offset = 0
    for axis, (dim, (cos, sin)) in enumerate(zip(DIM_SPLIT, tables, strict=True), 1):
        chunk = hidden_states[..., offset : offset + dim]
        pairs = chunk.reshape(*chunk.shape[:-1], dim // 2, 2)
        even = pairs[..., 0].float()
        odd = pairs[..., 1].float()
        shape = [1, 1, 1, 1, 1, dim // 2]
        shape[axis] = cos.shape[0]
        cos = cos.reshape(shape)
        sin = sin.reshape(shape)
        rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        outputs.append(rotated.reshape(chunk.shape).to(hidden_states.dtype))
        offset += dim
    return torch.cat(outputs, dim=-1)


CASES = get_ci_test_range(
    [
        (1, 3, 7, 7, 1),
        (1, 18, 17, 30, 32),
        (1, 16, 68, 96, 8),
        (1, 31, 136, 192, 4),
    ],
    [(1, 3, 7, 7, 1), (1, 31, 136, 192, 4)],
)


@pytest.mark.parametrize("batch,frames,height,width,heads", CASES)
def test_ltx25_decoder_rope_is_bit_exact(
    batch: int, frames: int, height: int, width: int, heads: int
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(
        batch,
        frames,
        height,
        width,
        heads,
        64,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn_like(q)
    tables = make_tables(frames, height, width)

    q_out, k_out = fused_ltx25_decoder_rope(
        q, k, *tables[0], *tables[1], *tables[2], DIM_SPLIT[0], DIM_SPLIT[1]
    )

    assert torch.equal(q_out, eager_rope(q, tables))
    assert torch.equal(k_out, eager_rope(k, tables))
    assert q_out.data_ptr() not in (q.data_ptr(), k.data_ptr())
    assert k_out.data_ptr() not in (q.data_ptr(), k.data_ptr())


def test_ltx25_decoder_rope_predicate_rejects_unsupported_inputs() -> None:
    q = torch.empty(1, 3, 7, 7, 1, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.empty_like(q)
    tables = make_tables(3, 7, 7)

    assert can_use_ltx25_decoder_rope(q, k, tables, DIM_SPLIT)
    assert not can_use_ltx25_decoder_rope(q.flatten(), k, tables, DIM_SPLIT)
    assert not can_use_ltx25_decoder_rope(q.float(), k, tables, DIM_SPLIT)
    assert not can_use_ltx25_decoder_rope(q, k[..., ::2], tables, DIM_SPLIT)
    assert not can_use_ltx25_decoder_rope(q, k, tables, (16, 16, 16))
    bad_tables = (tables[0], tables[1], (tables[2][0].double(), tables[2][1]))
    assert not can_use_ltx25_decoder_rope(q, k, bad_tables, DIM_SPLIT)
    unaligned = torch.empty(q.numel() + 1, dtype=torch.bfloat16, device="cuda")[
        1:
    ].view_as(q)
    assert unaligned.is_contiguous()
    assert not can_use_ltx25_decoder_rope(unaligned, k, tables, DIM_SPLIT)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

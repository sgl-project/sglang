import sys

import pytest
import torch

from sglang.kernels.ops.embeddings.engram_gather import (
    engram_gather,
    engram_gather_peer,
    unpack_mxfp4_rows,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_DIM = 256
_BLOCK = 32
# E2M1 magnitudes by 3-bit code, from the OCP Microscaling Formats spec.
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _packed_table(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    packed = torch.randint(0, 256, (rows, _DIM // 2), dtype=torch.uint8, generator=gen)
    packed[:2] = torch.arange(256, dtype=torch.uint8).view(2, _DIM // 2)
    exps = torch.randint(
        120, 135, (rows, _DIM // _BLOCK), dtype=torch.uint8, generator=gen
    )
    return packed.cuda(), exps.cuda()


def _reference(packed: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    nibbles = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2).long()
    magnitude = torch.tensor(_E2M1_VALUES, device=packed.device)[nibbles & 0x7]
    values = torch.where(nibbles >= 8, -magnitude, magnitude)
    scale = torch.exp2(exps.float() - 127).repeat_interleave(_BLOCK, dim=-1)
    return (values * scale).to(torch.bfloat16)


def _assert_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # torch.equal treats -0.0 and +0.0 as equal; the bits do not.
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_engram_gather_decodes_packed_mxfp4_bit_exactly() -> None:
    """Every E2M1 byte, -0 included, gathers to the spec value, packed or widened."""
    packed, exps = _packed_table(rows=64)
    widened = unpack_mxfp4_rows(packed)
    ids = torch.randperm(packed.shape[0], device="cuda")
    expected = _reference(packed, exps)[ids]
    for table, mxfp4 in ((packed, True), (widened, False)):
        out = torch.empty(ids.numel(), _DIM, dtype=torch.bfloat16, device="cuda")
        engram_gather(
            weight_ptr=table.data_ptr(),
            scale_ptr=exps.data_ptr(),
            ids=ids,
            out=out,
            dim=_DIM,
            block_size=_BLOCK,
            mxfp4=mxfp4,
        )
        _assert_bits_equal(out, expected)


def test_engram_gather_peer_decodes_packed_mxfp4() -> None:
    """The peer kernel decodes packed rows as it reads widened ones."""
    packed, exps = _packed_table(rows=64)
    widened = unpack_mxfp4_rows(packed)
    ids = torch.randperm(packed.shape[0], device="cuda").view(4, -1)
    expected = _reference(packed, exps)[ids]
    # The peer kernel writes every zero as +0.
    expected = torch.where(expected == 0, torch.zeros_like(expected), expected)
    starts = torch.zeros(1, dtype=torch.int64, device="cuda")
    s_bias = torch.tensor([exps.data_ptr()], dtype=torch.int64, device="cuda")
    for table, mxfp4 in ((packed, True), (widened, False)):
        w_bias = torch.tensor([table.data_ptr()], dtype=torch.int64, device="cuda")
        out = torch.empty(*ids.shape, _DIM, dtype=torch.bfloat16, device="cuda")
        engram_gather_peer(
            ids=ids,
            out=out,
            w_bias=w_bias,
            s_bias=s_bias,
            starts=starts,
            dim=_DIM,
            block_size=_BLOCK,
            mxfp4=mxfp4,
        )
        _assert_bits_equal(out, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

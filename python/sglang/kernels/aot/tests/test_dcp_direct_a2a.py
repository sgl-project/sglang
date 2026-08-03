import sys
from types import SimpleNamespace

import pytest
import torch


if not torch.cuda.is_available():
    pytest.skip(
        "direct DCP A2A requires the CUDA AOT extension",
        allow_module_level=True,
    )

from sgl_kernel import direct_dcp_a2a_lse_reduce


def _make_args(
    *,
    dtype: torch.dtype = torch.float16,
    num_tokens: int = 2,
    total_heads: int = 4,
    head_dim: int = 8,
    world_size: int = 2,
    max_num_tokens: int = 2,
):
    heads_per_rank = total_heads // world_size if world_size > 0 else 0
    pointers = torch.zeros(world_size, dtype=torch.int64, device="cuda")
    return {
        "partial_output": torch.zeros(
            num_tokens, total_heads, head_dim, dtype=dtype, device="cuda"
        ),
        "partial_lse": torch.zeros(
            num_tokens, total_heads, dtype=torch.float32, device="cuda"
        ),
        "peer_output_ptrs": pointers,
        "peer_lse_ptrs": pointers,
        "peer_signal_ptrs": pointers,
        "received_output": torch.zeros(
            2,
            world_size,
            max_num_tokens,
            heads_per_rank,
            head_dim,
            dtype=dtype,
            device="cuda",
        ),
        "received_lse": torch.zeros(
            2,
            world_size,
            max_num_tokens,
            heads_per_rank,
            dtype=torch.float32,
            device="cuda",
        ),
        "received_signal": torch.zeros(
            2, world_size, dtype=torch.int64, device="cuda"
        ),
        "epoch": torch.zeros(1, dtype=torch.int64, device="cuda"),
        "world_size": world_size,
        "rank": 0,
        "max_num_tokens": max_num_tokens,
        "is_lse_base_on_e": False,
    }


def test_direct_dcp_a2a_is_exported_from_package_root():
    assert callable(direct_dcp_a2a_lse_reduce)


def test_direct_dcp_a2a_allocates_rank_local_output(monkeypatch):
    calls = []
    fake_packet = SimpleNamespace(default=lambda *args: calls.append(args))
    monkeypatch.setattr(
        torch.ops.sgl_kernel,
        "direct_dcp_a2a_lse_reduce",
        fake_packet,
        raising=False,
    )
    args = _make_args()

    output = direct_dcp_a2a_lse_reduce(**args)

    assert output.shape == (2, 2, 8)
    assert output.dtype == torch.float16
    assert calls[0][9] is output


def test_direct_dcp_a2a_rejects_unsupported_output_dtype():
    with pytest.raises(
        RuntimeError,
        match=r"symm_a2a.*FP16 and BF16.*--dcp-comm-backend a2a",
    ):
        direct_dcp_a2a_lse_reduce(**_make_args(dtype=torch.float32))


def test_direct_dcp_a2a_rejects_world_size_one():
    args = _make_args(world_size=1)
    args["combined_output"] = torch.empty(
        2, 4, 8, dtype=torch.float16, device="cuda"
    )
    with pytest.raises(RuntimeError, match="world_size must be greater than 1"):
        direct_dcp_a2a_lse_reduce(**args)


def test_direct_dcp_a2a_default_output_rejects_world_size_zero():
    with pytest.raises(ValueError, match="world_size must be greater than 1"):
        direct_dcp_a2a_lse_reduce(**_make_args(world_size=0))


def test_direct_dcp_a2a_default_output_rejects_nondivisible_heads():
    with pytest.raises(ValueError, match="attention heads must divide evenly"):
        direct_dcp_a2a_lse_reduce(**_make_args(total_heads=3, world_size=2))


def test_direct_dcp_a2a_rejects_mismatched_lse_shape():
    args = _make_args()
    args["partial_lse"] = torch.zeros(2, 3, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="LSE shape must match"):
        direct_dcp_a2a_lse_reduce(**args)


def test_direct_dcp_a2a_rejects_unpacked_head_stride():
    args = _make_args()
    args["partial_output"] = torch.zeros(
        2, 8, 4, dtype=torch.float16, device="cuda"
    ).transpose(1, 2)
    with pytest.raises(RuntimeError, match="packed heads"):
        direct_dcp_a2a_lse_reduce(**args)


def test_direct_dcp_a2a_rejects_capacity_overflow():
    with pytest.raises(RuntimeError, match="exceeds symmetric buffer capacity"):
        direct_dcp_a2a_lse_reduce(**_make_args(num_tokens=2, max_num_tokens=1))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

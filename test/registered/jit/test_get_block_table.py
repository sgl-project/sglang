import pytest
import torch

from sglang.jit_kernel.minicpm_sala.get_block_table import (
    get_block_table_v1,
    get_block_table_v2,
    get_block_table_v3,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_HEAD_GROUP = 2
_SPARSE_BLOCK_SIZE = 64


def _make_inputs(token_num, seqlen_q_max, topk, batch_size=1, device="cuda"):
    """Build the same kind of inputs as 3rdparty/sparse_kernel/ut/test_v2.py."""
    topk_idx = torch.full(
        (_HEAD_GROUP, token_num, topk), -1, dtype=torch.int32, device=device
    )
    # Plant a few valid blocks at fixed positions, like the original UT.
    topk_idx[0, 32, 0:2] = torch.tensor([0, 1], dtype=torch.int32, device=device)
    topk_idx[1, 32, 0:2] = torch.tensor([0, 1], dtype=torch.int32, device=device)
    topk_idx[1, 64, 0:2] = torch.tensor([0, 1], dtype=torch.int32, device=device)
    topk_idx[0, 1000, 0:10] = torch.tensor(
        [0, 1, 5, 11, 14, 16, 17, 25, 26, 27], dtype=torch.int32, device=device
    )

    block_table = torch.arange(
        1, seqlen_q_max * batch_size + 1, dtype=torch.int32, device=device
    ).reshape(batch_size, seqlen_q_max)
    token_to_bs = torch.zeros((token_num,), dtype=torch.int32, device=device)
    token_pos_in_bs = torch.arange(1, token_num + 1, dtype=torch.int32, device=device)
    seqlen_q = torch.tensor([seqlen_q_max], dtype=torch.int32, device=device)
    return topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q


def _make_valid_inputs(token_num, seqlen_q_max, topk, batch_size=1, device="cuda"):
    """Build well-formed inputs with only non-negative block indices.

    The original v3 kernel (unlike v1/v2) has no ``sparse_block_idx < 0`` guard,
    so the three variants only provably agree when every topk entry is valid.
    """
    num_blocks = seqlen_q_max // _SPARSE_BLOCK_SIZE
    torch.manual_seed(0)
    topk_idx = torch.randint(
        0, num_blocks, (_HEAD_GROUP, token_num, topk), dtype=torch.int32, device=device
    )
    block_table = torch.arange(
        1, seqlen_q_max * batch_size + 1, dtype=torch.int32, device=device
    ).reshape(batch_size, seqlen_q_max)
    token_to_bs = torch.zeros((token_num,), dtype=torch.int32, device=device)
    token_pos_in_bs = torch.arange(1, token_num + 1, dtype=torch.int32, device=device)
    seqlen_q = torch.tensor([seqlen_q_max], dtype=torch.int32, device=device)
    return topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q


def _golden_check_v2(out_block_table, block_table, token_num):
    """The assertions ported verbatim from the original test_v2.py."""
    # check token 32
    assert (out_block_table[32, 0] != 0).sum().item() == 33
    assert (out_block_table[32, 1] != 0).sum().item() == 33
    assert torch.equal(out_block_table[32, 0, 0:33], block_table[0][:33] * 2)
    assert torch.equal(out_block_table[32, 1, 0:33], block_table[0][:33] * 2 + 1)

    # check token 64
    assert (out_block_table[64, 1] != 0).sum().item() == 65
    assert torch.equal(out_block_table[64, 1, 0:65], block_table[0][:65] * 2 + 1)

    # check token 1000
    topk_blocks = [0, 1, 5, 11, 14, 16, 17, 25, 26, 27]
    tokens = []
    for b in topk_blocks:
        tokens.extend(range(b * _SPARSE_BLOCK_SIZE, (b + 1) * _SPARSE_BLOCK_SIZE))
    tokens = [t for t in tokens if t < token_num and t < 1001]
    assert (out_block_table[1000, 0] != 0).sum().item() == len(tokens)
    assert torch.equal(
        out_block_table[1000, 0, : len(tokens)], block_table[0][tokens] * 2
    )


@pytest.mark.parametrize("topk", [96, 128])
def test_get_block_table_v2_golden(topk):
    token_num, seqlen_q_max = 8192, 8192
    inputs = _make_inputs(token_num, seqlen_q_max, topk)
    out = get_block_table_v2(*inputs)
    assert out.shape == (token_num, _HEAD_GROUP, topk * _SPARSE_BLOCK_SIZE)
    _golden_check_v2(out, inputs[1], token_num)


@pytest.mark.parametrize("topk", [96, 128])
def test_get_block_table_versions_agree(topk):
    """v1, v2, v3 must produce identical block tables for well-formed input.

    Uses only non-negative block indices because the original v3 kernel has no
    ``sparse_block_idx < 0`` guard (so the variants only agree on valid inputs).
    """
    token_num, seqlen_q_max = 2048, 2048
    inputs = _make_valid_inputs(token_num, seqlen_q_max, topk)
    out1 = get_block_table_v1(*inputs)
    out2 = get_block_table_v2(*inputs)
    out3 = get_block_table_v3(*inputs)
    assert torch.equal(out1, out2)
    assert torch.equal(out1, out3)


@pytest.mark.parametrize("topk", [96, 128])
def test_get_block_table_matches_reference(topk):
    """Cross-check each version against the original sparse_kernel_extension.

    This is the authoritative faithfulness check for the migration. The
    original extension takes ``topk`` as an explicit 6th argument.
    """
    ext = pytest.importorskip("sparse_kernel_extension")
    token_num, seqlen_q_max = 4096, 4096
    inputs = _make_valid_inputs(token_num, seqlen_q_max, topk)

    ref_v1 = ext.get_block_table_v1(*inputs, topk)
    assert torch.equal(ref_v1, get_block_table_v1(*inputs))

    ref_v2 = ext.get_block_table_v2(*inputs, topk)
    assert torch.equal(ref_v2, get_block_table_v2(*inputs))

    ref_v3 = ext.get_block_table_v3(*inputs, topk)
    assert torch.equal(ref_v3, get_block_table_v3(*inputs))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))

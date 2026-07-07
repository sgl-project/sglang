import types

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.triton_ops.vocab_parallel_embedding import (
    vocab_parallel_embedding,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test."
)


def _reference_embedding(input_ids, weight, cfg):
    # The kernel's contract is bit-parity with the eager production path it
    # replaces, so use that exact path as the oracle.
    masked_input, input_mask = get_masked_input_and_mask(input_ids, **cfg)
    output = F.embedding(masked_input.long(), weight)
    output.masked_fill_(input_mask.unsqueeze(-1), 0)
    return output


def _run_case(input_ids, weight, cfg):
    expected = _reference_embedding(input_ids, weight, cfg)
    actual = vocab_parallel_embedding(input_ids, weight, **cfg)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("hidden_dim", [7, 128, 6144])
def test_vocab_parallel_embedding_no_added_vocab(dtype, input_dtype, hidden_dim):
    cfg = dict(
        org_vocab_start_index=16,
        org_vocab_end_index=32,
        num_org_vocab_padding=0,
        added_vocab_start_index=64,
        added_vocab_end_index=64,
    )
    weight = torch.randn((16, hidden_dim), dtype=dtype, device="cuda")
    # Include negative and garbage ids: the mask is the only defense against
    # them when the OOB probe is disabled, so they must produce zero rows.
    token_ids = [-100, -1, 0, 16, 17, 31, 32, 63, 2**30]
    if input_dtype == torch.int64:
        token_ids += [2**62, -(2**62)]
    input_ids = torch.tensor(token_ids, dtype=input_dtype, device="cuda")
    _run_case(input_ids, weight, cfg)


def test_vocab_parallel_embedding_added_vocab_with_padding():
    cfg = dict(
        org_vocab_start_index=10,
        org_vocab_end_index=18,
        num_org_vocab_padding=4,
        added_vocab_start_index=100,
        added_vocab_end_index=103,
    )
    weight = torch.randn((16, 257), dtype=torch.bfloat16, device="cuda")
    input_ids = torch.tensor(
        [[9, 10, 17], [18, 100, 102]], dtype=torch.int64, device="cuda"
    )
    _run_case(input_ids, weight, cfg)


def test_vocab_parallel_embedding_strided_weight():
    cfg = dict(
        org_vocab_start_index=10,
        org_vocab_end_index=18,
        num_org_vocab_padding=4,
        added_vocab_start_index=100,
        added_vocab_end_index=103,
    )
    # Column slice of a wider buffer: stride(0) != hidden_dim, stride(1) == 1.
    weight = torch.randn((16, 300), dtype=torch.bfloat16, device="cuda")[:, :257]
    assert weight.stride(0) == 300 and weight.stride(1) == 1
    input_ids = torch.tensor(
        [[9, 10, 17], [18, 100, 102]], dtype=torch.int64, device="cuda"
    )
    _run_case(input_ids, weight, cfg)


def test_vocab_parallel_embedding_empty_input():
    cfg = dict(
        org_vocab_start_index=0,
        org_vocab_end_index=8,
        num_org_vocab_padding=0,
        added_vocab_start_index=8,
        added_vocab_end_index=8,
    )
    weight = torch.randn((8, 64), dtype=torch.bfloat16, device="cuda")
    input_ids = torch.empty((0,), dtype=torch.int64, device="cuda")
    _run_case(input_ids, weight, cfg)


def _stub_layer(**overrides):
    # The gate reads only tp_size, quant_method, and weight, so a stub avoids
    # needing distributed init for tp_size > 1.
    layer = types.SimpleNamespace(
        tp_size=2,
        quant_method=UnquantizedEmbeddingMethod(),
        weight=torch.empty((16, 32), dtype=torch.bfloat16, device="cuda"),
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def test_use_triton_embedding_gate():
    # The gate's failure direction is silent (the eager fallback is
    # numerically correct), so pin the cases that matter: eligibility, the
    # kill-switch, and the quantized-method exclusion.
    gate = VocabParallelEmbedding._use_triton_embedding
    input_ids = torch.zeros((4,), dtype=torch.int64, device="cuda")
    with envs.SGLANG_OPT_USE_TRITON_VOCAB_PARALLEL_EMBEDDING.override(True):
        assert gate(_stub_layer(), input_ids)
        assert not gate(_stub_layer(quant_method=object()), input_ids)
    with envs.SGLANG_OPT_USE_TRITON_VOCAB_PARALLEL_EMBEDDING.override(False):
        assert not gate(_stub_layer(), input_ids)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

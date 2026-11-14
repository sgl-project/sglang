import itertools
import sys

import pytest
import torch

from sglang.kernels.ops.speculative.verify_target_probs import verify_target_probs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.spec_verify_target_probs import reference_verify_target_probs

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

VOCAB = 128256  # Llama-3.1 vocab, the default EAGLE3 test target
BATCH_SIZES = [1, 4, 16, 64]
ROWS_PER_REQUEST = [1, 4, 8]
# (additive, scaling, logit_bias)
FEATURES = [
    (False, False, False),  # temperature only
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, True),
]


def _inputs(bs, k, features, dtype=torch.float32, seed=0):
    additive, scaling, bias = features
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    kwargs = dict(
        rows_per_request=k,
        temperatures=torch.rand(bs, 1, device=dev, generator=g) * 1.5 + 0.1,
    )
    if additive:
        kwargs["additive_penalties"] = -torch.rand(bs, VOCAB, device=dev, generator=g)
    if scaling:
        kwargs["scaling_penalties"] = 1 + torch.rand(bs, VOCAB, device=dev, generator=g)
    if bias:
        kwargs["logit_bias"] = torch.randn(bs, VOCAB, device=dev, generator=g)
    logits = (torch.randn(bs * k, VOCAB, device=dev, generator=g) * 4).to(dtype)
    return logits, kwargs


def _assert_probs(actual, expected, rows):
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(
        actual.sum(-1), torch.ones(rows, device="cuda"), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize(
    "bs,k,features", list(itertools.product(BATCH_SIZES, ROWS_PER_REQUEST, FEATURES))
)
def test_matches_reference(bs, k, features):
    logits, kwargs = _inputs(bs, k, features)
    expected, _ = reference_verify_target_probs(logits, **kwargs)
    before = logits.clone()
    actual = verify_target_probs(logits, **kwargs)
    _assert_probs(actual, expected, bs * k)
    assert torch.equal(logits, before), "logits must be untouched without write-back"


@pytest.mark.parametrize("features", FEATURES[1:])
def test_write_back_matches_in_place_penalties(features):
    bs, k = 4, 4
    logits, kwargs = _inputs(bs, k, features)
    expected, penalized = reference_verify_target_probs(logits, **kwargs)
    actual = verify_target_probs(logits, write_back_logits=True, **kwargs)
    _assert_probs(actual, expected, bs * k)
    torch.testing.assert_close(logits, penalized, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_logits(dtype):
    bs, k = 4, 4
    logits, kwargs = _inputs(bs, k, (True, True, True), dtype=dtype)
    expected, _ = reference_verify_target_probs(logits.float(), **kwargs)
    actual = verify_target_probs(logits, **kwargs)
    _assert_probs(actual, expected, bs * k)


def test_masked_tokens():
    """-inf logits (e.g. a -inf logit bias) get zero probability, no NaN."""
    bs, k = 2, 4
    logits, kwargs = _inputs(bs, k, (False, False, False))
    logits[:, 1000:70000] = -float("inf")
    expected, _ = reference_verify_target_probs(logits, **kwargs)
    actual = verify_target_probs(logits, **kwargs)
    assert not actual.isnan().any()
    assert (actual[:, 1000:70000] == 0).all()
    _assert_probs(actual, expected, bs * k)


def test_out_buffer():
    bs, k = 2, 4
    logits, kwargs = _inputs(bs, k, (False, False, False))
    out = torch.empty(bs * k, VOCAB, device="cuda")
    assert verify_target_probs(logits, out=out, **kwargs) is out


def test_rejects_bad_shapes():
    logits = torch.randn(6, 32, device="cuda")
    with pytest.raises(ValueError):
        verify_target_probs(
            logits, rows_per_request=4, temperatures=torch.ones(2, device="cuda")
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

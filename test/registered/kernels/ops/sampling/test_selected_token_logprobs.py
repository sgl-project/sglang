import pytest
import torch

from sglang.kernels.ops.sampling.selected_token_logprobs import (
    selected_token_logprobs,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _reference(
    logits: torch.Tensor,
    ids: torch.Tensor,
    temperatures: torch.Tensor | None,
    width: int,
) -> torch.Tensor:
    """The materialized stock computation the operator replaces."""
    if temperatures is None:
        z = logits
    else:
        z = logits / temperatures.repeat_interleave(width, dim=0)
    lp = torch.log_softmax(z, dim=-1)
    lp.clamp_(min=torch.finfo(lp.dtype).min)
    return lp.gather(1, ids.long().view(-1, 1)).reshape(-1, width)


def _case(bs, width, vocab, greedy, seed, dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(bs * width, vocab, device="cuda", dtype=dtype, generator=g)
    ids = torch.randint(0, vocab, (bs * width,), device="cuda", generator=g)
    temperatures = None
    if not greedy:
        temperatures = torch.empty(bs, 1, device="cuda", dtype=torch.float32).uniform_(
            0.6, 1.0, generator=g
        )
    return logits, ids, temperatures


@pytest.mark.parametrize(
    "bs,width,vocab", [(8, 6, 129_280), (64, 6, 157_184), (3, 5, 2_049)]
)
@pytest.mark.parametrize("greedy", [False, True])
def test_selected_token_logprobs_matches_log_softmax(bs, width, vocab, greedy):
    logits, ids, temperatures = _case(bs, width, vocab, greedy, seed=bs * 7 + width)
    got = selected_token_logprobs(logits, ids, temperatures, width)
    want = _reference(logits, ids, temperatures, width)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_id_dtypes(ids_dtype):
    logits, ids, temperatures = _case(4, 3, 1_000, greedy=False, seed=9)
    got = selected_token_logprobs(logits, ids.to(ids_dtype), temperatures, 3)
    want = _reference(logits, ids, temperatures, 3)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_nonfinite_rows_match_log_softmax():
    """Whole masked tiles and nonfinite rows follow the PyTorch reference."""
    g = torch.Generator(device="cuda").manual_seed(11)
    logits = torch.randn(12, 4_100, device="cuda", generator=g)
    logits[0:3] = float("-inf")
    logits[0:2, -2:] = torch.tensor([1.0, 2.0], device="cuda")
    logits[3:6] = float("inf")
    logits[6:9, 10] = float("nan")
    logits[9:12, 4_099] = float("nan")
    ids = torch.randint(0, 4_100, (12,), device="cuda", generator=g)
    ids[0] = 4_099  # Select a finite logit after two fully masked tiles.
    ids[1] = 0  # Select a masked logit in a row with a finite normalizer.
    temperatures = torch.empty(4, 1, device="cuda", dtype=torch.float32).uniform_(
        0.6, 1.0, generator=g
    )
    got = selected_token_logprobs(logits, ids, temperatures, 3)
    want = _reference(logits, ids, temperatures, 3)
    assert got[0, 1].item() == torch.finfo(logits.dtype).min
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6, equal_nan=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))

"""Regression test for the XPU torch.compile guard on
``VocabParallelEmbedding.get_masked_input_and_mask`` and
``apply_rotary_pos_emb_native``.

Both functions are ``@torch.compile``-decorated. torch.compile/Inductor
codegen on Intel XPU under tensor parallelism is unreliable for them,
previously causing either a scheduler watchdog timeout (stuck in Inductor's
compile-worker subprocess) or a hard crash ("RuntimeError: level_zero backend
failed with error: 20/40") on real multi-GPU TP servers. That failure mode
needs a full TP server under sustained load to reproduce, so it isn't
practical to reproduce directly in a unit test; instead this test locks in
the fix itself (torch.compile is actually disabled on XPU, not just
numerically correct) plus functional correctness on an XPU device.
"""

import sys

import pytest
import torch

from sglang.srt.layers.rotary_embedding.utils import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_native,
    apply_rotary_pos_emb_native_eager,
)
from sglang.srt.layers.vocab_parallel_embedding import get_masked_input_and_mask
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=10, suite="nightly-xpu-kernel-main-1-gpu", nightly=True)

pytestmark = pytest.mark.skipif(not is_xpu(), reason="XPU required")

DEVICE = "xpu"


def _assert_not_dynamo_wrapped(fn) -> None:
    # torch.compile(fn, disable=True) returns fn itself, unwrapped -- no
    # _torchdynamo_orig_callable attribute gets attached. If a future edit
    # drops the `or _is_xpu` guard, this function would instead be a dynamo
    # wrapper and this assertion catches it immediately, without needing to
    # reproduce the original multi-GPU TP compile-worker hang/crash.
    assert not hasattr(fn, "_torchdynamo_orig_callable"), (
        f"{fn.__name__} is torch.compile-wrapped on XPU -- the _is_xpu guard "
        "appears to have been removed"
    )


def test_get_masked_input_and_mask_not_compiled_on_xpu():
    _assert_not_dynamo_wrapped(get_masked_input_and_mask)


def test_apply_rotary_pos_emb_native_not_compiled_on_xpu():
    _assert_not_dynamo_wrapped(apply_rotary_pos_emb_native)
    # The platform dispatcher (used directly by e.g. the vision attention
    # path) must resolve to this same guarded function on XPU.
    assert apply_rotary_pos_emb is apply_rotary_pos_emb_native


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
def test_get_masked_input_and_mask_correctness_xpu(dtype, input_dtype):
    cfg = dict(
        org_vocab_start_index=16,
        org_vocab_end_index=32,
        num_org_vocab_padding=0,
        added_vocab_start_index=64,
        added_vocab_end_index=64,
    )
    # Include negative and garbage ids: the mask is the only defense against
    # them when the OOB probe is disabled, so they must produce zero rows.
    token_ids = [-100, -1, 0, 16, 17, 31, 32, 63, 2**30]
    if input_dtype == torch.int64:
        token_ids += [torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max]
    input_ids = torch.tensor(token_ids, dtype=input_dtype, device=DEVICE)

    masked_input, input_mask = get_masked_input_and_mask(input_ids, **cfg)

    weight = torch.randn((16, 8), dtype=dtype, device=DEVICE)
    output = torch.nn.functional.embedding(masked_input.long(), weight)
    output.masked_fill_(input_mask.unsqueeze(-1), 0)

    expected_mask = torch.tensor(
        [True, True, True, False, False, False, True, True, True]
        + ([True, True] if input_dtype == torch.int64 else []),
        device=DEVICE,
    )
    torch.testing.assert_close(input_mask, expected_mask)
    # In-vocab rows (org or added) embed normally; everything else is zeroed.
    assert torch.all(output[input_mask] == 0)
    assert not torch.any(torch.all(output[~input_mask] == 0, dim=-1))


@pytest.mark.parametrize("num_tokens", [1, 257])
@pytest.mark.parametrize("head_size", [64, 128])
def test_apply_rotary_pos_emb_correctness_xpu(num_tokens: int, head_size: int):
    torch.manual_seed(0)
    num_heads, num_kv_heads = 8, 1
    q = torch.randn(num_tokens, num_heads, head_size, device=DEVICE)
    k = torch.randn(num_tokens, num_kv_heads, head_size, device=DEVICE)
    cos = torch.randn(num_tokens, head_size, device=DEVICE)
    sin = torch.randn(num_tokens, head_size, device=DEVICE)

    q_expected, k_expected = apply_rotary_pos_emb_native_eager(
        q.clone(), k.clone(), cos, sin
    )
    # On XPU this is the exact same (uncompiled) code path as the eager
    # reference above -- bit-for-bit parity is expected, not just "close".
    q_actual, k_actual = apply_rotary_pos_emb(q.clone(), k.clone(), cos, sin)

    torch.testing.assert_close(q_actual, q_expected, rtol=0, atol=0)
    torch.testing.assert_close(k_actual, k_expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

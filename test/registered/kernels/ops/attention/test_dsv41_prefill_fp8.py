"""SM90 FP8 prefill scoring and its shared full/publish/consume dispatch."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.prefill_fp8 import prefill_fp8_scores
from sglang.kernels.ops.attention.dsv4.prefill_index_k_fp8 import gather_index_k_fp8
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
    BlockIds,
    DenseBlocksBackend,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.full_topk import FullTopKIndexer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 FP8 Tensor Cores required",
)


def reference(q, k, weights, lens):
    dots = torch.einsum("rhd,nd->rhn", q.bfloat16(), k.bfloat16())
    scores = (dots.relu() * weights.unsqueeze(-1)).sum(1).float()
    return scores.masked_fill(
        torch.arange(k.shape[0], device=q.device)[None, :] >= lens[:, None], -torch.inf
    )


@pytest.mark.parametrize("page_size", [1, 8, 64])
@pytest.mark.parametrize("count", [0, 1, 7, 17])
def test_fp4_readback(page_size, count):
    # Exercise all E2M1 codes, per-32 scales, page strides and repeated slots.
    pages = 4
    n = pages * page_size
    codes = torch.arange(n * 128, device="cuda").reshape(n, 128) % 16
    exps = (torch.arange(n * 4, device="cuda").reshape(n, 4) % 11 + 120).byte()
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).byte()
    storage = torch.zeros(pages, page_size * 68 + 16, dtype=torch.uint8, device="cuda")
    cache = storage[:, : page_size * 68]
    cache[:, : page_size * 64] = packed.reshape(pages, -1)
    cache[:, page_size * 64 :] = exps.reshape(pages, -1)
    slots = ((torch.arange(count * 2, device="cuda") * 7 + 3) % n)[::2]
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    values = lut[codes] * torch.exp2(exps.float() - 127).repeat_interleave(32, 1)
    expected = values[slots].bfloat16().to(torch.float8_e4m3fn)
    actual = gather_index_k_fp8(cache, slots, page_size)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("length", [0, 1, 127, 128, 129, 1025])
def test_scores_exact_and_causal(length):
    torch.manual_seed(17)
    # Small dyadic values make conversion and sums exact, isolating addressing,
    # masking and post-dot rounding from the general FP8 error budget.
    q = (torch.randint(-2, 3, (5, 32, 128), device="cuda") / 8).to(torch.float8_e4m3fn)
    k = (torch.randint(-2, 3, (length, 128), device="cuda") / 8).to(torch.float8_e4m3fn)
    weights = (torch.randint(-2, 3, (5, 32), device="cuda") / 8).bfloat16()
    lens = torch.tensor(
        [0, 1, length // 2, length, length + 3], device="cuda", dtype=torch.int32
    )
    lens_storage = torch.zeros(10, device="cuda", dtype=lens.dtype)
    lens_storage[::2] = lens
    actual = prefill_fp8_scores(q, k, weights, lens_storage[::2])
    torch.testing.assert_close(actual, reference(q, k, weights, lens), rtol=0, atol=0)


def test_empty_queries():
    q = torch.empty(0, 32, 128, device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.empty(10, 128, device="cuda", dtype=q.dtype)
    weights = torch.empty(0, 32, device="cuda", dtype=torch.bfloat16)
    lens = torch.empty(0, device="cuda", dtype=torch.int32)
    assert prefill_fp8_scores(q, k, weights, lens).shape == (0, 10)


@pytest.mark.parametrize("role", ["full", "publish", "consume"])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("budget", [16 * 9, 1 << 30])
def test_shared_switch_across_prefill_roles(role, ratio, budget):
    torch.manual_seed(29)
    rows, length = 4, 41
    q = (torch.randint(-2, 3, (rows, 32, 128), device="cuda") / 8).bfloat16()
    keys = (torch.randint(-2, 3, (length * 2, 128), device="cuda") / 8).bfloat16()
    weights = (torch.randint(-2, 3, (rows, 32), device="cuda") / 8).bfloat16()
    req = torch.tensor([0, 0, 1, 1], device="cuda")
    pos = torch.tensor([10, 20, 9, 16], device="cuda") * ratio - 1
    table = torch.arange(length * 2 * ratio, device="cuda").reshape(2, -1)
    pool = SimpleNamespace(
        get_low_ratio_index_k_dequant=lambda layer, slots: keys[slots],
        get_low_ratio_index_k_fp8=lambda layer, slots: keys[slots].to(
            torch.float8_e4m3fn
        ),
    )
    indexer = SimpleNamespace(
        index_topk=8,
        queries=lambda *args: q,
        head_weights=lambda x: weights,
        scores=lambda q, k, w: (
            (torch.einsum("rhd,nd->rhn", q, k).relu() * w[:, :, None]).sum(1).float()
        ),
    )
    raw = torch.full((rows, 8), -1, device="cuda", dtype=torch.int32)
    pages = torch.full_like(raw, -1)
    inputs = SimpleNamespace(
        layer_id=20,
        compress_ratio=ratio,
        indexer=indexer,
        req_rows=req,
        positions=pos,
        q_lora=None,
        freqs_cis=torch.zeros(length * ratio, device="cuda"),
        x=None,
        rows_per_request=[2, 2],
        out_raw_indices=raw,
        out_page_indices=pages,
        reset_outputs=lambda: (raw.fill_(-1), pages.fill_(-1)),
    )
    backend = SimpleNamespace(
        token_to_kv_pool=pool, req_to_token=table, topk_blocks=2, block_size=8
    )
    published = BlockIds(
        blocks=torch.tensor(
            [[0, -1], [0, 2], [1, 0], [1, -1]], device="cuda", dtype=torch.int32
        ),
        rows_per_request=[2, 2],
    )

    def run():
        if role == "full":
            return FullTopKIndexer._torch_prefill(backend, inputs)
        if role == "publish":
            return DenseBlocksBackend._torch_publish_prefill(backend, inputs)
        return DenseBlocksBackend._torch_consume_prefill(backend, inputs, published)

    for oracle_override in (False, True):
        with (
            envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(oracle_override),
            patch.object(
                scoring, "get_platform", return_value=SimpleNamespace(is_sm90=True)
            ),
            patch.object(scoring, "_TORCH_SCORE_BUDGET_BYTES", budget),
        ):
            with envs.SGLANG_DSV41_FP8_PREFILL_INDEXER.override(False):
                baseline = run()
                expected_raw, expected_pages = raw.clone(), pages.clone()
            with (
                envs.SGLANG_DSV41_FP8_PREFILL_INDEXER.override(True),
                patch.object(
                    pool,
                    "get_low_ratio_index_k_fp8",
                    wraps=pool.get_low_ratio_index_k_fp8,
                ) as fp8_read,
            ):
                actual = run()
                assert fp8_read.call_count == (0 if oracle_override else 2)
            torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)
            torch.testing.assert_close(pages, expected_pages, rtol=0, atol=0)
            if role == "publish":
                torch.testing.assert_close(
                    actual.blocks, baseline.blocks, rtol=0, atol=0
                )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

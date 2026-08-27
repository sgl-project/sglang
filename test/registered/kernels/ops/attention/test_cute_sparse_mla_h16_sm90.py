from __future__ import annotations

import importlib
import math

import pytest
import torch

from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=420, stage="base-b-kernel-unit", runner_config="1-gpu-large")


H_LOGICAL = 16
H_FLASHMLA = 64
D_QK = 512
D_V = 512
NO_SINK = -1.0e30

# The fixed-input H20 comparison against FlashMLA observed <=2.44e-4 maximum
# and roughly 6.5e-7 mean absolute error. Keep the elementwise limit within a
# small number of BF16 output ULPs and independently lock down aggregate drift.
MAX_ABS_ERROR = 5.0e-4
MAX_MEAN_ABS_ERROR = 5.0e-6
STRESS_MAX_ABS_ERROR = 2.0e-3
STRESS_MAX_MEAN_ABS_ERROR = 1.0e-4
TORCH_REF_MAX_ABS_ERROR = 1.0e-3
# The CPU-FP32 oracle intentionally includes the BF16 WGMMA and final-output
# rounding gap. On SM90 that active-row mean is about 5.4e-5 even when the
# CuTe result is bitwise identical to FlashMLA, so retain margin without
# weakening the much tighter implementation-to-implementation parity gates.
TORCH_REF_MAX_MEAN_ABS_ERROR = 1.0e-4


def _sm90_available() -> bool:
    return is_sm90_supported()


requires_sm90 = pytest.mark.skipif(
    not _sm90_available(), reason="native H16 CuTe sparse MLA requires SM90 CUDA"
)


def _load_implementations():
    # Keep the CuTe DSL import inside an SM90-gated test. This lets CPU-only
    # collection succeed while making a missing GPU dependency fail on the
    # runner where the kernel is expected to execute.
    cute_h16 = importlib.import_module(
        "sglang.kernels.ops.attention.dsv4.cute_sparse_mla_h16"
    )
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    return cute_h16, flash_mla_sparse_fwd


def _make_case(
    *,
    tq: int,
    skv: int,
    topk: int,
    lengths: list[int],
    seed: int,
    invalid_in_prefix: bool = True,
    std: float = 0.1,
):
    assert len(lengths) == tq
    assert all(0 <= length <= topk for length in lengths)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = (
        torch.randn(
            (tq, H_LOGICAL, D_QK),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * std
    ).to(torch.bfloat16)
    kv = (
        torch.randn(
            (skv, 1, D_QK),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * std
    ).to(torch.bfloat16)
    indices = torch.randint(
        0,
        skv,
        (tq, 1, topk),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    topk_length = torch.tensor(lengths, dtype=torch.int32, device="cuda")

    # Garbage outside the declared prefix must never be gathered. Alternate
    # negative and high sentinels so both invalid-index predicates execute.
    for row, length in enumerate(lengths):
        # Keep at least one active row ID above the signed-int16 range when the
        # workspace is large enough; rebased production workspaces commonly
        # exceed 32K rows.
        if length > 0 and skv > 32768:
            indices[row, 0, 0] = 32768 + (row * 97) % (skv - 32768)
        if length < topk:
            tail = torch.arange(topk - length, device="cuda")
            indices[row, 0, length:] = torch.where(
                tail % 2 == 0,
                torch.full_like(tail, -1),
                torch.full_like(tail, skv + 17),
            ).to(torch.int32)

        # FlashMLA documents both -1 and values >= SKV as invalid even when
        # they occur inside topk_length. Exercise that contract independently
        # from the trailing-prefix mask.
        if invalid_in_prefix and length >= 4:
            indices[row, 0, 1] = -1
            indices[row, 0, 3] = skv + 31

    attn_sink = torch.linspace(
        0.05, 0.90, H_LOGICAL, dtype=torch.float32, device="cuda"
    )
    return q, kv, indices, topk_length, attn_sink


def _length_boundaries(topk: int) -> list[int]:
    candidates = (0, 1, 63, 64, 65, 127, 128, 129, topk - 1, topk)
    return [
        value
        for position, value in enumerate(candidates)
        if 0 <= value <= topk and value not in candidates[:position]
    ]


def _pad_flashmla_inputs(
    q: torch.Tensor, attn_sink: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_padded = torch.zeros(
        (q.shape[0], H_FLASHMLA, D_QK), dtype=q.dtype, device=q.device
    )
    q_padded[:, :H_LOGICAL].copy_(q)
    sink_padded = torch.full(
        (H_FLASHMLA,), NO_SINK, dtype=torch.float32, device=q.device
    )
    sink_padded[:H_LOGICAL].copy_(attn_sink)
    return q_padded, sink_padded


def _unaligned_contiguous_copy(tensor: torch.Tensor) -> torch.Tensor:
    storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
    result = storage[1:].view(tensor.shape)
    result.copy_(tensor)
    assert result.is_contiguous()
    assert result.data_ptr() % 16 != 0
    return result


def _flashmla_reference(
    flash_mla_sparse_fwd,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    # FlashMLA's SM90 sparse-prefill WGMMA specialization pays for an H64
    # query tile. Explicit padding makes that baseline visible while keeping
    # the first 16 heads numerically identical to the production TP=8 input.
    q_padded, sink_padded = _pad_flashmla_inputs(q, attn_sink)
    out, _, _ = flash_mla_sparse_fwd(
        q=q_padded,
        kv=kv,
        indices=indices,
        sm_scale=D_QK**-0.5,
        d_v=D_V,
        attn_sink=sink_padded,
        topk_length=topk_length,
    )
    return out[:, :H_LOGICAL]


def _torch_fp32_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Small, direct FP32 definition of sparse attention plus a zero-V sink."""

    assert q.ndim == 3 and q.shape[1:] == (H_LOGICAL, D_QK)
    assert kv.ndim == 2 and kv.shape[1] == D_QK
    assert indices.ndim == 2 and indices.shape[0] == q.shape[0]

    # CPU float32 keeps this oracle independent from both SM90 attention
    # implementations and from any CUDA TF32 matmul policy on the test runner.
    q = q.detach().cpu().float()
    kv = kv.detach().cpu().float()
    indices = indices.detach().cpu()
    topk_length = topk_length.detach().cpu()
    attn_sink = attn_sink.detach().cpu().float()

    rows = []
    for row in range(q.shape[0]):
        # Only the declared prefix participates. Invalid IDs inside that prefix
        # are absent from both the softmax denominator and the value reduction.
        length = int(topk_length[row].item())
        prefix = indices[row, :length].long()
        valid_ids = prefix[(prefix >= 0) & (prefix < kv.shape[0])]
        selected_kv = kv[valid_ids]

        q_row = q[row]
        logits = torch.matmul(q_row, selected_kv.T) * sm_scale
        # The virtual sink is appended after all valid key logits. Its value is
        # exactly zero, so its probability contributes only to normalization.
        logits_with_sink = torch.cat((logits, attn_sink.unsqueeze(1)), dim=1)
        key_prob = torch.softmax(logits_with_sink, dim=1)[:, : valid_ids.numel()]
        rows.append(torch.matmul(key_prob, selected_kv))

    return torch.stack(rows)


def _assert_matches_flashmla(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    topk_length: torch.Tensor,
    max_abs_error: float = MAX_ABS_ERROR,
    max_mean_abs_error: float = MAX_MEAN_ABS_ERROR,
) -> None:
    assert actual.shape == expected.shape
    assert actual.shape[1:] == (H_LOGICAL, D_V)
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.isfinite(actual.float()).all()
    assert torch.isfinite(expected.float()).all()

    diff = (actual.float() - expected.float()).abs()
    assert diff.max().item() <= max_abs_error
    assert diff.mean().item() <= max_mean_abs_error
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=max_abs_error, rtol=1.0e-3
    )

    # With no real key, the positive virtual sink has a zero value vector, so
    # both implementations must return exact zero rather than NaN/Inf.
    zero_rows = topk_length == 0
    if bool(zero_rows.any()):
        assert torch.count_nonzero(actual[zero_rows]).item() == 0
        assert torch.count_nonzero(expected[zero_rows]).item() == 0


@requires_sm90
@torch.inference_mode()
def test_cute_sparse_mla_h16_matches_independent_torch_fp32_reference():
    """Check the mathematical contract without using FlashMLA as the oracle."""

    cute_h16 = importlib.import_module(
        "sglang.kernels.ops.attention.dsv4.cute_sparse_mla_h16"
    )
    tq, skv, topk = 4, 17, 128
    q, kv, indices, topk_length, attn_sink = _make_case(
        tq=tq,
        skv=skv,
        topk=topk,
        lengths=[0, 5, 6, 8],
        seed=77,
        invalid_in_prefix=False,
    )

    # Valid-looking data after each prefix catches implementations that ignore
    # topk_length. The active prefixes cover mixed invalid IDs, an all-invalid
    # row, duplicates, and both negative and high out-of-bounds sentinels.
    indices.fill_(10)
    indices[1, 0, :5] = torch.tensor(
        [2, -1, 7, skv + 3, 4], dtype=torch.int32, device="cuda"
    )
    indices[2, 0, :6] = torch.tensor(
        [-1, skv, -7, skv + 1, -1, skv + 99],
        dtype=torch.int32,
        device="cuda",
    )
    indices[3, 0, :8] = torch.tensor(
        [1, 3, 3, 5, -1, skv + 2, 8, 2],
        dtype=torch.int32,
        device="cuda",
    )

    q_padded, sink_padded = _pad_flashmla_inputs(q, attn_sink)
    kv_compact = kv.squeeze(1)
    indices_compact = indices.squeeze(1)
    sm_scale = D_QK**-0.5
    actual = cute_h16.cute_sparse_mla_h16_fwd(
        q=q_padded,
        kv=kv_compact,
        indices=indices_compact,
        topk_length=topk_length,
        attn_sink=sink_padded,
        sm_scale=sm_scale,
    )
    expected_fp32 = _torch_fp32_reference(
        q=q,
        kv=kv_compact,
        indices=indices_compact,
        topk_length=topk_length,
        attn_sink=attn_sink,
        sm_scale=sm_scale,
    )
    torch.cuda.synchronize()

    assert actual.shape == expected_fp32.shape == (tq, H_LOGICAL, D_V)
    assert actual.dtype == torch.bfloat16
    assert expected_fp32.dtype == torch.float32
    assert torch.isfinite(actual.float()).all()
    assert torch.isfinite(expected_fp32).all()
    # Rows with zero valid keys contain only the zero-valued virtual sink.
    assert torch.count_nonzero(actual[[0, 2]]).item() == 0
    assert torch.count_nonzero(expected_fp32[[0, 2]]).item() == 0
    assert torch.count_nonzero(expected_fp32[[1, 3]]).item() > 0

    actual_fp32 = actual.float().cpu()
    diff = (actual_fp32 - expected_fp32).abs()
    # This compares BF16 WGMMA/output rounding directly with an FP32 definition;
    # the max gate is a few BF16 ULPs at this test's ~1e-2 output scale, while
    # the mean gate prevents a broad low-amplitude drift from hiding underneath.
    assert diff.max().item() <= TORCH_REF_MAX_ABS_ERROR
    assert diff[[1, 3]].mean().item() <= TORCH_REF_MAX_MEAN_ABS_ERROR
    torch.testing.assert_close(
        actual_fp32, expected_fp32, atol=TORCH_REF_MAX_ABS_ERROR, rtol=1.0e-2
    )


@requires_sm90
@torch.inference_mode()
def test_cute_sparse_mla_h16_multitile_matches_independent_torch_fp32_reference():
    """Check multi-tile online softmax against an implementation-independent oracle."""

    cute_h16 = importlib.import_module(
        "sglang.kernels.ops.attention.dsv4.cute_sparse_mla_h16"
    )
    tq, skv, topk = 3, 521, 256
    q, kv, indices, topk_length, _ = _make_case(
        tq=tq,
        skv=skv,
        topk=topk,
        lengths=[129, 191, 256],
        seed=88,
        invalid_in_prefix=False,
        std=0.25,
    )
    kv_compact = kv.squeeze(1)
    indices_compact = indices.squeeze(1)
    sm_scale = D_QK**-0.5

    # Make every later 64-row tile raise head 0's running maximum. This forces
    # the online-softmax path to rescale already accumulated PV state.
    generator = torch.Generator(device="cuda").manual_seed(89)
    for row in range(tq):
        selected = torch.randperm(skv, device="cuda", generator=generator)[:topk]
        scores = torch.mv(kv_compact[selected].float(), q[row, 0].float()) * sm_scale
        indices_compact[row].copy_(selected[torch.argsort(scores)].to(torch.int32))

    full_scores = (
        torch.mv(kv_compact[indices_compact[-1].long()].float(), q[-1, 0].float())
        * sm_scale
    )
    tile_max = full_scores.view(-1, 64).amax(dim=1)
    assert bool(torch.all(tile_max[1:] > tile_max[:-1]))

    sink_cases = {
        "disabled": NO_SINK,
        "negative": -2.0,
        "dominant": 2.0,
    }
    for name, sink_score in sink_cases.items():
        attn_sink = torch.full(
            (H_LOGICAL,), sink_score, dtype=torch.float32, device="cuda"
        )
        q_padded, sink_padded = _pad_flashmla_inputs(q, attn_sink)
        actual = cute_h16.cute_sparse_mla_h16_fwd(
            q=q_padded,
            kv=kv_compact,
            indices=indices_compact,
            topk_length=topk_length,
            attn_sink=sink_padded,
            sm_scale=sm_scale,
        )
        expected_fp32 = _torch_fp32_reference(
            q=q,
            kv=kv_compact,
            indices=indices_compact,
            topk_length=topk_length,
            attn_sink=attn_sink,
            sm_scale=sm_scale,
        )
        torch.cuda.synchronize()

        actual_fp32 = actual.float().cpu()
        diff = (actual_fp32 - expected_fp32).abs()
        assert torch.isfinite(actual_fp32).all(), name
        assert diff.max().item() <= TORCH_REF_MAX_ABS_ERROR, name
        assert diff.mean().item() <= TORCH_REF_MAX_MEAN_ABS_ERROR, name
        torch.testing.assert_close(
            actual_fp32,
            expected_fp32,
            atol=TORCH_REF_MAX_ABS_ERROR,
            rtol=1.0e-2,
            msg=lambda message: f"sink case {name}: {message}",
        )


@requires_sm90
@torch.inference_mode()
def test_cute_sparse_mla_h16_fresh_jit_dynamic_shapes_and_repeat_allocation():
    """Compile once, reuse dynamic shapes, and return independent outputs."""

    cute_h16, flash_mla_sparse_fwd = _load_implementations()
    cute_h16._COMPILE_CACHE.clear()

    q1, kv1, indices1, lengths1, sink1 = _make_case(
        tq=2,
        skv=257,
        topk=128,
        lengths=[128, 65],
        seed=101,
    )
    q1_padded, sink1_padded = _pad_flashmla_inputs(q1, sink1)
    actual1 = cute_h16.cute_sparse_mla_h16_fwd(
        q=q1_padded,
        kv=kv1.squeeze(1),
        indices=indices1.squeeze(1),
        topk_length=lengths1,
        attn_sink=sink1_padded,
        sm_scale=D_QK**-0.5,
    )
    torch.cuda.synchronize()

    assert actual1.is_contiguous()
    assert len(cute_h16._COMPILE_CACHE) == 1
    expected1 = _flashmla_reference(
        flash_mla_sparse_fwd, q1, kv1, indices1, lengths1, sink1
    )
    torch.cuda.synchronize()
    _assert_matches_flashmla(actual1, expected1, topk_length=lengths1)

    # Every public call owns its output. A repeated launch must not alias or
    # overwrite the first allocation and must remain deterministic.
    snapshot1 = actual1.clone()
    actual1_repeat = cute_h16.cute_sparse_mla_h16_fwd(
        q=q1_padded,
        kv=kv1.squeeze(1),
        indices=indices1.squeeze(1),
        topk_length=lengths1,
        attn_sink=sink1_padded,
        sm_scale=D_QK**-0.5,
    )
    torch.cuda.synchronize()
    assert actual1_repeat.data_ptr() != actual1.data_ptr()
    assert torch.equal(actual1, snapshot1)
    assert torch.equal(actual1_repeat, snapshot1)

    # TQ and SKV are runtime dimensions. Changing both while H and TOPK stay
    # fixed must reuse the one compiled executor rather than specialize again.
    q2, kv2, indices2, lengths2, sink2 = _make_case(
        tq=5,
        skv=40009,
        topk=128,
        lengths=[0, 1, 64, 127, 128],
        seed=202,
    )
    q2_padded, sink2_padded = _pad_flashmla_inputs(q2, sink2)
    actual2 = cute_h16.cute_sparse_mla_h16_fwd(
        q=q2_padded,
        kv=kv2.squeeze(1),
        indices=indices2.squeeze(1),
        topk_length=lengths2,
        attn_sink=sink2_padded,
        sm_scale=D_QK**-0.5,
    )
    expected2 = _flashmla_reference(
        flash_mla_sparse_fwd, q2, kv2, indices2, lengths2, sink2
    )
    torch.cuda.synchronize()

    assert len(cute_h16._COMPILE_CACHE) == 1
    _assert_matches_flashmla(actual2, expected2, topk_length=lengths2)


@requires_sm90
@pytest.mark.parametrize("topk", [128, 384, 512, 640, 1152, 8192])
@torch.inference_mode()
def test_cute_sparse_mla_h16_matches_flashmla_boundaries(topk: int):
    cute_h16, flash_mla_sparse_fwd = _load_implementations()
    lengths = _length_boundaries(topk)
    q, kv, indices, topk_length, attn_sink = _make_case(
        tq=len(lengths),
        skv=40009 + topk,
        topk=topk,
        lengths=lengths,
        seed=1000 + topk,
    )
    q_padded, sink_padded = _pad_flashmla_inputs(q, attn_sink)

    actual = cute_h16.cute_sparse_mla_h16_fwd(
        q=q_padded,
        kv=kv.squeeze(1),
        indices=indices.squeeze(1),
        topk_length=topk_length,
        attn_sink=sink_padded,
        sm_scale=1.0 / math.sqrt(D_QK),
    )
    expected = _flashmla_reference(
        flash_mla_sparse_fwd, q, kv, indices, topk_length, attn_sink
    )
    torch.cuda.synchronize()

    _assert_matches_flashmla(actual, expected, topk_length=topk_length)


@requires_sm90
@torch.inference_mode()
def test_cute_sparse_mla_h16_rms1_multitile_online_rescale():
    """Stress repeated online-max/PV rescaling across all 18 K tiles."""

    cute_h16, flash_mla_sparse_fwd = _load_implementations()
    tq, skv, topk = 4, 4096, 1152
    q, kv, indices, topk_length, attn_sink = _make_case(
        tq=tq,
        skv=skv,
        topk=topk,
        lengths=[topk] * tq,
        seed=7001,
        invalid_in_prefix=False,
        std=1.0,
    )
    kv_compact = kv.squeeze(1)
    indices_compact = indices.squeeze(1)

    # Sort each query's unique gather rows by head-0 score. Every successive
    # 64-row tile then raises that head's running maximum, forcing the old PV
    # state to be rescaled instead of merely accumulating under a fixed max.
    generator = torch.Generator(device="cuda").manual_seed(7002)
    sm_scale = D_QK**-0.5
    for row in range(tq):
        selected = torch.randperm(skv, device="cuda", generator=generator)[:topk]
        scores = torch.mv(kv_compact[selected].float(), q[row, 0].float()) * sm_scale
        indices_compact[row].copy_(selected[torch.argsort(scores)].to(torch.int32))

    ordered_scores = (
        torch.mv(kv_compact[indices_compact[0].long()].float(), q[0, 0].float())
        * sm_scale
    )
    tile_max = ordered_scores.view(-1, 64).amax(dim=1)
    assert bool(torch.all(tile_max[1:] > tile_max[:-1]))
    assert 0.9 < q.float().square().mean().sqrt().item() < 1.1
    assert 0.9 < kv.float().square().mean().sqrt().item() < 1.1

    q_padded, sink_padded = _pad_flashmla_inputs(q, attn_sink)
    actual = cute_h16.cute_sparse_mla_h16_fwd(
        q=q_padded,
        kv=kv_compact,
        indices=indices_compact,
        topk_length=topk_length,
        attn_sink=sink_padded,
        sm_scale=sm_scale,
    )
    expected = _flashmla_reference(
        flash_mla_sparse_fwd, q, kv, indices, topk_length, attn_sink
    )
    torch.cuda.synchronize()

    stress_diff = (actual.float() - expected.float()).abs()
    print(
        "\nRMS1 K=1152 FlashMLA parity: "
        f"max_abs={stress_diff.max().item():.8e}, "
        f"mean_abs={stress_diff.mean().item():.8e}"
    )
    # A legacy normalized-input H20 run reached 1.953125e-3 max error. Keep
    # this stress-only gate independent from the tight ordinary-input gate;
    # tighten it after measuring this exact test on the rebased upstream main.
    _assert_matches_flashmla(
        actual,
        expected,
        topk_length=topk_length,
        max_abs_error=STRESS_MAX_ABS_ERROR,
        max_mean_abs_error=STRESS_MAX_MEAN_ABS_ERROR,
    )


@requires_sm90
@torch.inference_mode()
def test_cute_sparse_mla_h16_h16_and_h64_inputs_are_bitwise_equal():
    cute_h16, _ = _load_implementations()
    q, kv, indices, topk_length, attn_sink = _make_case(
        tq=3,
        skv=1024,
        topk=128,
        lengths=[1, 65, 128],
        seed=8001,
    )
    q_padded, sink_padded = _pad_flashmla_inputs(q, attn_sink)
    generator = torch.Generator(device="cuda").manual_seed(8002)
    q_padded[:, H_LOGICAL:].copy_(
        torch.randn(
            q_padded[:, H_LOGICAL:].shape,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
    )
    sink_padded[H_LOGICAL:] = torch.linspace(
        -4.0,
        4.0,
        H_FLASHMLA - H_LOGICAL,
        dtype=torch.float32,
        device="cuda",
    )

    kwargs = dict(
        kv=kv.squeeze(1),
        indices=indices.squeeze(1),
        topk_length=topk_length,
        sm_scale=D_QK**-0.5,
    )
    out_h16 = cute_h16.cute_sparse_mla_h16_fwd(q=q, attn_sink=attn_sink, **kwargs)
    out_h64 = cute_h16.cute_sparse_mla_h16_fwd(
        q=q_padded, attn_sink=sink_padded, **kwargs
    )
    torch.cuda.synchronize()

    assert torch.equal(out_h16, out_h64)


@requires_sm90
@torch.inference_mode()
def test_cute_sparse_mla_h16_invalid_redirects_mask_poisoned_kv_zero():
    cute_h16, flash_mla_sparse_fwd = _load_implementations()
    lengths = [1, 65, 128]
    q, kv, indices, topk_length, attn_sink = _make_case(
        tq=len(lengths),
        skv=257,
        topk=128,
        lengths=lengths,
        seed=9001,
    )
    indices[indices == 0] = 1
    kv[0, 0, :3] = torch.tensor(
        [float("nan"), float("inf"), float("-inf")],
        dtype=torch.bfloat16,
        device="cuda",
    )

    for row, length in enumerate(lengths):
        prefix = indices[row, 0, :length]
        valid = (prefix >= 0) & (prefix < kv.shape[0])
        assert not bool(torch.any(prefix[valid] == 0))

    q_padded, sink_padded = _pad_flashmla_inputs(q, attn_sink)
    actual = cute_h16.cute_sparse_mla_h16_fwd(
        q=q_padded,
        kv=kv.squeeze(1),
        indices=indices.squeeze(1),
        topk_length=topk_length,
        attn_sink=sink_padded,
        sm_scale=D_QK**-0.5,
    )
    expected = _flashmla_reference(
        flash_mla_sparse_fwd, q, kv, indices, topk_length, attn_sink
    )
    torch.cuda.synchronize()

    assert torch.isfinite(actual.float()).all()
    assert torch.isfinite(expected.float()).all()
    _assert_matches_flashmla(actual, expected, topk_length=topk_length)


@requires_sm90
@pytest.mark.parametrize("unaligned_input", ["q", "kv"])
@torch.inference_mode()
def test_cute_sparse_mla_h16_rejects_unaligned_contiguous_storage(
    unaligned_input: str,
):
    cute_h16, _ = _load_implementations()
    q, kv, indices, topk_length, attn_sink = _make_case(
        tq=1,
        skv=257,
        topk=128,
        lengths=[128],
        seed=10001,
    )
    kv_compact = kv.squeeze(1)
    if unaligned_input == "q":
        q = _unaligned_contiguous_copy(q)
    else:
        kv_compact = _unaligned_contiguous_copy(kv_compact)

    with pytest.raises(ValueError, match="aligned to 16 bytes"):
        cute_h16.cute_sparse_mla_h16_fwd(
            q=q,
            kv=kv_compact,
            indices=indices.squeeze(1),
            topk_length=topk_length,
            attn_sink=attn_sink,
            sm_scale=D_QK**-0.5,
        )


@requires_sm90
@torch.inference_mode()
def test_cute_sparse_mla_h16_two_nondefault_streams_share_executor():
    cute_h16, flash_mla_sparse_fwd = _load_implementations()
    cute_h16._COMPILE_CACHE.clear()
    topk = 512
    case_a = _make_case(
        tq=3,
        skv=1024,
        topk=topk,
        lengths=[65, 129, topk],
        seed=11001,
    )
    case_b = _make_case(
        tq=5,
        skv=40009,
        topk=topk,
        lengths=[0, 1, 128, 384, topk],
        seed=11002,
    )
    q_a, kv_a, indices_a, lengths_a, sink_a = case_a
    q_b, kv_b, indices_b, lengths_b, sink_b = case_b
    q_a64, sink_a64 = _pad_flashmla_inputs(q_a, sink_a)
    q_b64, sink_b64 = _pad_flashmla_inputs(q_b, sink_b)

    warm_a = cute_h16.cute_sparse_mla_h16_fwd(
        q=q_a64,
        kv=kv_a.squeeze(1),
        indices=indices_a.squeeze(1),
        topk_length=lengths_a,
        attn_sink=sink_a64,
        sm_scale=D_QK**-0.5,
    )
    torch.cuda.synchronize()
    assert len(cute_h16._COMPILE_CACHE) == 1

    default_stream = torch.cuda.current_stream(q_a.device)
    stream_a = torch.cuda.Stream(device=q_a.device)
    stream_b = torch.cuda.Stream(device=q_b.device)
    stream_a.wait_stream(default_stream)
    stream_b.wait_stream(default_stream)
    with torch.cuda.stream(stream_a):
        actual_a = cute_h16.cute_sparse_mla_h16_fwd(
            q=q_a64,
            kv=kv_a.squeeze(1),
            indices=indices_a.squeeze(1),
            topk_length=lengths_a,
            attn_sink=sink_a64,
            sm_scale=D_QK**-0.5,
        )
    with torch.cuda.stream(stream_b):
        actual_b = cute_h16.cute_sparse_mla_h16_fwd(
            q=q_b64,
            kv=kv_b.squeeze(1),
            indices=indices_b.squeeze(1),
            topk_length=lengths_b,
            attn_sink=sink_b64,
            sm_scale=D_QK**-0.5,
        )
    stream_a.synchronize()
    stream_b.synchronize()

    assert len(cute_h16._COMPILE_CACHE) == 1
    assert torch.equal(actual_a, warm_a)
    expected_a = _flashmla_reference(
        flash_mla_sparse_fwd, q_a, kv_a, indices_a, lengths_a, sink_a
    )
    expected_b = _flashmla_reference(
        flash_mla_sparse_fwd, q_b, kv_b, indices_b, lengths_b, sink_b
    )
    torch.cuda.synchronize()
    _assert_matches_flashmla(actual_a, expected_a, topk_length=lengths_a)
    _assert_matches_flashmla(actual_b, expected_b, topk_length=lengths_b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

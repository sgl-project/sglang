"""The 4over6 activation recipe on the per-token NVFP4 path.

flashinfer's recipe, as its CuTe-DSL quantizer evaluates it: per 16-block the
candidates ``sf6 = amax / (6 s)`` and ``sf4 = 1.5 sf6`` cast to E4M3
(saturating at 448, never at ``e4m3_max``), the block quantized with both, the
candidate error summed over the block in lane order (absolute or squared
differences, in the encode domain under ``err_use_fast_math`` or dequantized
otherwise), ``err4 < err6`` picking ``sf4`` and ties keeping ``sf6``; a
candidate whose E4M3 scale rounds to zero quantizes to codes 0; ``e4m3_max``
256 only changes the stored scale's denominator to 1536. Every per-token
quantizer (the row quantizer, both dispatch fills, the decode two-pass
activation) is checked bitwise against the float64 torch reference of that
rule under each recipe, the pre-registered recipe cases of the per-token
design are asserted outright, and the providers run under a recipe eager and
under captured graphs."""

from __future__ import annotations

import itertools

import msgspec
import pytest
import test_moe_lora_blockscaled_provider_numerics as P
import test_moe_lora_cutedsl_blockscaled_numerics as K
import torch
import torch.nn.functional as F

from sglang.kernels.ops.lora.moe.activation import ActivationFn
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.srt.lora.moe.plan import BFamily, resolve_plans
from sglang.srt.lora.moe.quant_info import NvFp4ActivationRecipe
from sglang.srt.lora.moe.runner import MoeLoraBatch, MoeLoraRunner
from sglang.srt.lora.utils import Phase
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci

# Every test here needs SM100 (block-scaled tcgen05 MMA): on the H100 lane the
# file is collected and skipped whole, so it is registered on the SM100 lane
# only. It runs on one GPU: 4-gpu-b200 is the kernel lane's only SM100 pool, so
# its four GPUs are allocation granularity, not multi-GPU coverage.
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_FLT_MAX = torch.finfo(torch.float32).max
_RECIPES = [
    NvFp4ActivationRecipe(e4m3_max=e4m3_max, err_mode=err_mode, err_use_fast_math=fast)
    for e4m3_max, err_mode, fast in itertools.product(
        (448, 256), ("MAE", "MSE"), (False, True)
    )
]
_RECIPE_IDS = [
    f"{r.err_mode}-{r.e4m3_max}-{'fast' if r.err_use_fast_math else 'strict'}"
    for r in _RECIPES
]
# The provider gate covers both denominators and both error paths.
_PROVIDER_RECIPES = [
    _RECIPES[_RECIPE_IDS.index(i)] for i in ("MAE-256-strict", "MSE-448-fast")
]
_E4M3_384, _E4M3_448, _E4M3_MIN = 0x7C, 0x7E, 0x01


def _denominator(recipe: NvFp4ActivationRecipe | None) -> float:
    return 6.0 * (448 if recipe is None else recipe.e4m3_max)


def _rows_buffer(num_rows, rows_per_slab, k, recipe, device, fill=0x55):
    from sglang.kernels.ops.lora.moe.cutedsl.api import scale_atom_bytes
    from sglang.kernels.ops.lora.moe.quant_blockscaled import BlockScaledRows

    return BlockScaledRows(
        data=torch.full((num_rows, k // 2), fill, dtype=torch.uint8, device=device),
        scales=torch.full(
            (scale_atom_bytes(rows_per_slab, k, 16, num_rows // rows_per_slab),),
            fill,
            dtype=torch.uint8,
            device=device,
        ),
        sf_vec_size=16,
        rows_per_slab=rows_per_slab,
        row_scales=torch.full((num_rows,), -1.0, device=device),
        row_denominator=_denominator(recipe),
        candidate_select=recipe is not None,
        err_mode="MAE" if recipe is None else recipe.err_mode,
        err_use_fast_math=recipe is not None and recipe.err_use_fast_math,
    )


def _quant_nvfp4_recipe_ref(x: torch.Tensor, recipe: NvFp4ActivationRecipe | None):
    """Per-token NVFP4 under ``recipe`` as the kernels evaluate it, in fp32
    where they round (tensor divisors: a Python float divisor multiplies by a
    reciprocal on CUDA) and float64 for the dequantized values. The stored
    row scale is fl(row_amax / D); each block's ``sf6 = fl(amax / fl(6 s))``
    and ``sf4 = fl(1.5 sf6)`` saturate at 448 and cast to E4M3; its codes
    against a candidate are e2m1((x / s) / dec(sf)), 0 where the E4M3 scale
    is 0; the candidate error is the sum over the block, in lane order with
    fp32 adds, of |d - o| or (d - o)^2 with ``d = code * dec(sf)`` and ``o =
    x * min(D / row_amax, fp32_max)`` under err_use_fast_math, else ``d =
    ((code * dec(sf)) * row_amax) / D`` and ``o = x``; sf4 wins iff err4 <
    err6. Returns the packed codes, scale bytes [.., k // 16], row scales,
    the float64 dequantized values and the per-block selection detail."""
    *lead, k = x.shape
    device = x.device
    xf = x.float()
    denominator = torch.tensor(_denominator(recipe), device=device)
    row_amax = xf.abs().amax(dim=-1)
    s = row_amax / denominator
    s_safe = torch.where(s > 0, s, torch.ones_like(s))
    groups = xf.reshape(*lead, k // 16, 16)
    amax = groups.abs().amax(dim=-1)
    scaled = groups / s_safe[..., None, None]
    sf6 = amax / (6.0 * s_safe)[..., None]
    table = K._E2M1.to(device)

    def candidate(sf_float):
        sf = sf_float.clamp(max=448.0).to(torch.float8_e4m3fn)
        decoded = sf.float()
        live = decoded > 0
        denom = torch.where(live, decoded, torch.ones_like(decoded))
        q = scaled / denom[..., None]
        q = torch.where(live[..., None], q, torch.zeros_like(q)).clamp(-6.0, 6.0)
        idx = K._e2m1_codes(q.abs())
        sign = torch.signbit(q)
        values = torch.where(sign, -table[idx], table[idx])
        codes = (idx | (sign.long() << 3)).to(torch.uint8)
        return sf.view(torch.uint8), decoded, values, codes

    sf_bytes, decoded, values, codes = candidate(sf6)
    detail = dict(sf6=sf_bytes)
    if recipe is not None:
        sf4_bytes, decoded4, values4, codes4 = candidate(sf6 * 1.5)
        # The kernels read the row amax back from the stored scale; a zero
        # row reads as D (its blocks are dead either way).
        amax_eff = torch.where(row_amax > 0, row_amax, denominator.expand_as(row_amax))
        encode = torch.minimum(
            denominator / amax_eff, torch.full_like(amax_eff, _FLT_MAX)
        )

        def error(values_c, decoded_c):
            cand = values_c * decoded_c[..., None]
            if recipe.err_use_fast_math:
                orig = groups * encode[..., None, None]
            else:
                cand = (cand * amax_eff[..., None, None]) / denominator
                orig = groups
            diff = cand - orig
            term = diff * diff if recipe.err_mode == "MSE" else diff.abs()
            err = term[..., 0]
            for lane in range(1, 16):
                err = err + term[..., lane]
            return err

        err6, err4 = error(values, decoded), error(values4, decoded4)
        use4 = err4 < err6
        sf_bytes = torch.where(use4, sf4_bytes, sf_bytes)
        decoded = torch.where(use4, decoded4, decoded)
        values = torch.where(use4[..., None], values4, values)
        codes = torch.where(use4[..., None], codes4, codes)
        detail.update(sf4=sf4_bytes, err6=err6, err4=err4, use4=use4)
    codes = codes.reshape(*lead, k)
    data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    dequant = (
        values.double() * decoded.double()[..., None] * s.double()[..., None, None]
    ).reshape(*lead, k)
    return data, sf_bytes, s, dequant, detail


def _decode(data, sf_bytes, s, k):
    """Packed codes, scale bytes and row scales -> float64 values."""
    low, high = data & 0x0F, data >> 4
    nibbles = torch.stack([low, high], dim=-1).reshape(*data.shape[:-1], k)
    table = K._E2M1.to(data.device)
    values = table[(nibbles & 7).long()]
    values = torch.where(nibbles & 8 > 0, -values, values).double()
    scale = sf_bytes.view(torch.float8_e4m3fn).double()
    return (values.reshape(*data.shape[:-1], k // 16, 16) * scale[..., None]).reshape(
        *data.shape[:-1], k
    ) * s.double()[..., None]


# The recipe cases of the design (6.4), one row each over K = 128 (8 blocks);
# block 0 holds the row maximum unless stated.
_CASE_K = 128
_CASES = (
    "above_256",  # row amax 1536: block 0 has sf6 = 256 and sf4 = 384 under 256
    "sf384_only_max",  # row amax 2688; block 1 at 2304: sf6 = 384, sf4 saturates to 448
    "sf384_decided",  # the same block with elements on the 448 grid
    "zero_scale_both",  # row amax 1; block 1 at 2^-20: both candidates round to 0
    "zero_scale_rescued",  # row amax 1; block 1 where sf6 rounds to 0 and sf4 does not
    "zero",
    "tiny_2^-133",
    "tiny_2^-126",
    "bf16_max",
    "flat",
)


def _case_rows(recipe: NvFp4ActivationRecipe, generator: torch.Generator):
    rows = torch.zeros((len(_CASES), _CASE_K))
    r = {name: i for i, name in enumerate(_CASES)}
    rows[r["above_256"], :3] = torch.tensor([1536.0, 576.0, 1152.0])
    rows[r["sf384_only_max"], 0] = 2688.0
    rows[r["sf384_only_max"], 16] = 2304.0
    rows[r["sf384_decided"], 0] = 2688.0
    rows[r["sf384_decided"], 16:21] = torch.tensor(
        [2304.0, 224.0, 672.0, 1344.0, 1792.0]
    )
    rows[r["zero_scale_both"], 0] = 1.0
    rows[r["zero_scale_both"], 16:19] = torch.tensor([2.0**-20, -(2.0**-20), 2.0**-21])
    rows[r["zero_scale_rescued"], 0] = 1.0
    # sf6 = amax * e4m3_max lands in [2^-10 / 1.5, 2^-10): E4M3 rounds it to 0
    # and 1.5x it to the smallest subnormal scale 2^-9.
    rescued = 2.0**-19 if recipe.e4m3_max == 448 else 3 * 2.0**-20
    rows[r["zero_scale_rescued"], 16:19] = torch.tensor(
        [rescued, -rescued, rescued / 2]
    )
    for name, amax in (("tiny_2^-133", 2.0**-133), ("tiny_2^-126", 2.0**-126)):
        rows[r[name]] = (torch.rand((_CASE_K,), generator=generator) * 2 - 1) * amax
        rows[r[name], 0] = amax
    rows[r["bf16_max"]] = torch.finfo(torch.bfloat16).max
    rows[r["bf16_max"], 1::2] *= -1
    rows[r["flat"]] = 0.75
    rows = rows.to(torch.bfloat16)
    assert torch.equal(rows.float(), rows.float().to(torch.bfloat16).float())
    return rows, r


def _quantize_rows_under(x, recipe, device):
    from sglang.kernels.ops.lora.moe.quant_blockscaled import quantize_rows

    num_rows, k = x.shape
    target = _rows_buffer(num_rows, num_rows, k, recipe, device)
    quantize_rows(x, target, torch.arange(num_rows, dtype=torch.int32, device=device))
    torch.cuda.synchronize()
    sf = K._unpack_scale_atoms(target.scales, num_rows, k, 16, num_rows)
    return target.data, sf, target.row_scales


@pytest.mark.parametrize("e4m3_max", (448, 256))
def test_row_amax_reads_back_from_the_stored_scale(e4m3_max):
    """The error evaluation needs the row amax; the kernels recover it from
    the stored scale as bf16(fl(s * D)). Every positive finite BF16 value
    round-trips under both denominators, the subnormal stored scales (rows
    below about 2^-115) included, because the stored scale's absolute error
    stays below half a BF16 ulp of the amax."""
    K._skip_unless_sm100()
    device = torch.device("cuda")
    bits = torch.arange(1, 0x7F80, dtype=torch.int32, device=device)  # (0, +inf)
    amax = (bits << 16).view(torch.float32)
    denominator = torch.tensor(6.0 * e4m3_max, device=device)
    s = amax / denominator
    assert bool((s > 0).all())
    assert bool((s[amax < 2.0**-120] < torch.finfo(torch.float32).tiny).all())
    assert torch.equal((s * denominator).to(torch.bfloat16).float(), amax)


@pytest.mark.parametrize("recipe", _RECIPES, ids=_RECIPE_IDS)
def test_row_quantizer_matches_the_recipe_reference(recipe):
    """The row quantizer under each recipe: codes, scale bytes and stored row
    scales bitwise with the reference over the per-token rows (zero, tiny,
    the BF16 maximum, a zero-scale block, a subnormal-scale block, a flat
    row, log-uniform rows) and the recipe cases; every stored value finite;
    the recipe exercised (sf4 selected somewhere); the reconstruction within
    the per-token tolerance on the rows that keep their blocks."""
    K._skip_unless_sm100()
    device = torch.device("cuda")
    g = torch.Generator().manual_seed(0x4076)
    k = 640
    special = K._per_token_rows(k, g)
    cases, _ = _case_rows(recipe, g)
    x = torch.cat([special, F.pad(cases, (0, k - _CASE_K))]).to(device)
    data, sf, s = _quantize_rows_under(x, recipe, device)
    data_ref, sf_ref, s_ref, deq_ref, detail = _quant_nvfp4_recipe_ref(x, recipe)
    assert torch.equal(data, data_ref)
    assert torch.equal(sf, sf_ref)
    assert torch.equal(s, s_ref)
    assert bool(torch.isfinite(s).all()) and bool(torch.isfinite(deq_ref).all())
    assert bool(detail["use4"].any())
    assert bool((detail["sf4"] != detail["sf6"]).any())
    x64 = x.double()
    for row in range(special.shape[0]):
        if row == K._ROW_ZERO:
            continue
        cols = slice(0, k)
        if row in (K._ROW_ZERO_SCALE_BLOCK, K._ROW_SUBNORMAL_SCALE_BLOCK):
            cols = slice(0, 16)
        err = K._rel_l2(deq_ref[row, cols], x64[row, cols])
        assert err < 0.25, (row, float(err))
    # The row-maximum block's candidates are e4m3_max and 1.5x it: under 448
    # both saturate to 448; under 256 either 256 or the representable 384
    # wins. Rows with a subnormal stored scale are left out (their sf6 can
    # land one E4M3 step away).
    rows = special.shape[0]
    normal = (s_ref >= torch.finfo(torch.float32).tiny)[:rows, None]
    magnitudes = x[:rows].float().abs()
    row_max_block = (
        magnitudes.reshape(rows, k // 16, 16).amax(-1) == magnitudes.amax(-1)[:, None]
    )
    top = sf_ref[:rows][row_max_block & normal]
    allowed = (_E4M3_448,) if recipe.e4m3_max == 448 else (0x78, _E4M3_384)
    assert bool(sum(top == v for v in allowed).bool().all())


@pytest.mark.parametrize("recipe", _RECIPES, ids=_RECIPE_IDS)
def test_recipe_cases(recipe):
    """The pre-registered recipe cases, asserted outright and bitwise with
    the kernel: under 256 the row-maximum block with sf6 = 256 selects the
    representable sf4 = 384 and reconstructs its elements exactly; under 448
    the same block's sf4 saturates to 448, coincides with sf6 and ties; a
    block with sf6 = 384 under 448 keeps it when its one element is exact
    there, and the saturated sf4 = 448 wins or loses by the error mode when
    the block sits on the 448 grid (MAE takes it, MSE does not); a block
    whose scale rounds to zero under both candidates keeps scale 0 and codes
    0; a block whose sf6 rounds to zero while sf4 does not keeps its values
    through sf4 (the guard is per candidate); zero rows, tiny rows with
    subnormal stored scales and BF16-maximum rows stay finite."""
    K._skip_unless_sm100()
    device = torch.device("cuda")
    g = torch.Generator().manual_seed(0x6406)
    x, r = _case_rows(recipe, g)
    x = x.to(device)
    data, sf, s = _quantize_rows_under(x, recipe, device)
    data_ref, sf_ref, s_ref, deq_ref, detail = _quant_nvfp4_recipe_ref(x, recipe)
    assert torch.equal(data, data_ref) and torch.equal(sf, sf_ref)
    assert torch.equal(s, s_ref)
    assert bool(torch.isfinite(deq_ref).all()) and bool(torch.isfinite(s).all())
    sf6, sf4, use4 = detail["sf6"], detail["sf4"], detail["use4"]
    deq = _decode(data, sf, s, _CASE_K)
    x64 = x.double()

    row = r["above_256"]
    if recipe.e4m3_max == 256:
        assert s[row] == 1.0  # amax 1536 = D
        assert int(sf6[row, 0]) == 0x78 and int(sf4[row, 0]) == _E4M3_384  # 256, 384
        assert (
            torch.tensor([_E4M3_384], dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .float()
            == 384
        )
        assert bool(use4[row, 0]) and int(sf[row, 0]) == _E4M3_384
        assert torch.equal(deq[row, :3], x64[row, :3])  # 1536, 576, 1152 exact
        assert detail["err4"][row, 0] == 0 < detail["err6"][row, 0]
    else:
        assert int(sf6[row, 0]) == _E4M3_448 and int(sf4[row, 0]) == _E4M3_448
        assert detail["err4"][row, 0] == detail["err6"][row, 0]
        assert not bool(use4[row, 0]) and int(sf[row, 0]) == _E4M3_448

    if recipe.e4m3_max == 448:
        row = r["sf384_only_max"]
        assert s[row] == 1.0
        assert int(sf6[row, 1]) == _E4M3_384 and int(sf4[row, 1]) == _E4M3_448
        assert detail["err6"][row, 1] == 0 < detail["err4"][row, 1]
        assert not bool(use4[row, 1]) and int(sf[row, 1]) == _E4M3_384
        assert deq[row, 16] == 2304  # code 6 at scale 384
        row = r["sf384_decided"]
        assert int(sf6[row, 1]) == _E4M3_384 and int(sf4[row, 1]) == _E4M3_448
        err6, err4 = float(detail["err6"][row, 1]), float(detail["err4"][row, 1])
        if recipe.err_mode == "MAE":
            assert (err6, err4) == (576.0, 384.0)
            assert bool(use4[row, 1]) and int(sf[row, 1]) == _E4M3_448
            assert torch.equal(deq[row, 17:21], x64[row, 17:21])  # on the 448 grid
            assert deq[row, 16] == 2688  # 2304 rounds up to code 6 at 448
        else:
            assert (err6, err4) == (112640.0, 147456.0)
            assert not bool(use4[row, 1]) and int(sf[row, 1]) == _E4M3_384
            assert deq[row, 16] == 2304

    row = r["zero_scale_both"]
    assert int(sf6[row, 1]) == 0 and int(sf4[row, 1]) == 0
    assert not bool(use4[row, 1]) and int(sf[row, 1]) == 0
    assert bool((data[row, 8:16] == 0).all())
    assert int(sf[row, 0]) > 0 and int(data[row, 0]) != 0
    assert detail["err6"][row, 1] == detail["err4"][row, 1] > 0

    row = r["zero_scale_rescued"]
    assert int(sf6[row, 1]) == 0 and int(sf4[row, 1]) == _E4M3_MIN
    assert bool(use4[row, 1]) and int(sf[row, 1]) == _E4M3_MIN
    assert bool((data[row, 8:10] != 0).all())
    assert float(detail["err4"][row, 1]) < float(detail["err6"][row, 1])
    # The rescued block reconstructs within its two-code grid, not as the
    # full loss a dead candidate would leave.
    assert K._rel_l2(deq[row, 16:32], x64[row, 16:32]) < 0.3

    assert s[r["zero"]] == 0 and bool((data[r["zero"]] == 0).all())
    assert bool((sf[r["zero"]] == 0).all())
    tiny = torch.finfo(torch.float32).tiny
    for name in ("tiny_2^-133", "tiny_2^-126"):
        assert 0 < s[r[name]] < tiny, name
        assert bool((data[r[name]] != 0).any()), name
    assert bool(torch.isfinite(deq[r["bf16_max"]]).all())
    assert K._rel_l2(deq[r["bf16_max"]], x64[r["bf16_max"]]) < 1e-6
    # Every block of the flat row is the row maximum: under 448 both
    # candidates are 448; under 256 both reconstruct 0.75 exactly (6 * 256 =
    # 4 * 384, and 0.75 / 1536 is a power of two) and the tie keeps sf6 = 256.
    if recipe.e4m3_max == 448:
        assert bool((sf[r["flat"]] == _E4M3_448).all())
        assert torch.allclose(deq[r["flat"]], x64[r["flat"]], rtol=2.0**-23, atol=0)
    else:
        assert bool((sf[r["flat"]] == 0x78).all())
        assert torch.equal(deq[r["flat"]], x64[r["flat"]])


_OVERFLOW_ROW = 2.0**110  # lifts the ``above_256`` block's values above fp32_max / D


@pytest.mark.parametrize("recipe", _RECIPES, ids=_RECIPE_IDS)
def test_strict_error_overflow_keeps_sf6(recipe):
    """One example of the inherited limit of the strict candidate error,
    pinned on the kernel: ``((code * sf) * row_amax) / D`` in fp32
    (flashinfer's order) overflows in a lane whose ``|code * sf| * row_amax``
    exceeds fp32_max, i.e. for values above about fp32_max / D, inside BF16's
    range. The ``above_256`` block [1536, 576, 1152, 0, ...] times 2^110 puts
    its three nonzero values (7.5e35 to 2.0e36) above that bound under both
    denominators, so under 256 every nonzero lane overflows for both
    candidates, both strict errors are inf and the strict recipes keep sf6 =
    256 (codes 6, 2, 4, reconstruction error 192 * 2^110) although sf4 = 384
    reconstructs the three elements exactly, which the fast recipes
    (comparing in the encode domain) select; under 448 both candidates
    saturate to 448 and the overflow decides nothing. The kernel and the
    reference agree bitwise either way, the stored row scale and the
    reconstruction stay finite. Which lanes overflow, and whether only one
    candidate's error does, depends on the block: this is a specific case,
    not the general rule. A change of this behavior is a change of
    flashinfer's rule and must be deliberate."""
    K._skip_unless_sm100()
    device = torch.device("cuda")
    x = torch.zeros((1, _CASE_K))
    x[0, :3] = torch.tensor([1536.0, 576.0, 1152.0]) * _OVERFLOW_ROW
    x = x.to(torch.bfloat16)
    assert float(x[0, 0]) == 1536.0 * _OVERFLOW_ROW  # BF16 holds the row
    x = x.to(device)
    assert float(x.float().abs().amax()) > _FLT_MAX / _denominator(recipe)
    data, sf, s = _quantize_rows_under(x, recipe, device)
    data_ref, sf_ref, s_ref, deq_ref, detail = _quant_nvfp4_recipe_ref(x, recipe)
    assert torch.equal(data, data_ref) and torch.equal(sf, sf_ref)
    assert torch.equal(s, s_ref)
    assert bool(torch.isfinite(s).all()) and bool(torch.isfinite(deq_ref).all())
    err6, err4 = detail["err6"][0, 0], detail["err4"][0, 0]
    if recipe.err_use_fast_math:
        assert bool(torch.isfinite(err6)) and bool(torch.isfinite(err4))
    else:
        assert bool(torch.isinf(err6)) and bool(torch.isinf(err4))
    deq = _decode(data, sf, s, _CASE_K)[0, :3] / _OVERFLOW_ROW
    x64 = x.double()[0, :3] / _OVERFLOW_ROW
    if recipe.e4m3_max == 448:
        assert int(detail["sf6"][0, 0]) == _E4M3_448 == int(detail["sf4"][0, 0])
        assert int(sf[0, 0]) == _E4M3_448
        return
    assert float(s[0]) == _OVERFLOW_ROW  # amax / 1536
    assert int(detail["sf6"][0, 0]) == 0x78 and int(detail["sf4"][0, 0]) == _E4M3_384
    if recipe.err_use_fast_math:
        assert bool(detail["use4"][0, 0]) and int(sf[0, 0]) == _E4M3_384
        assert torch.equal(deq, x64)
    else:
        # ``err4 < err6`` is false for two infinities: sf6 stays.
        assert not bool(detail["use4"][0, 0]) and int(sf[0, 0]) == 0x78
        expected = torch.tensor([1536.0, 512.0, 1024.0], dtype=torch.float64)
        assert torch.equal(deq, expected.to(device))
        assert float((deq - x64).abs().sum()) == 192.0


@pytest.mark.parametrize("recipe", _RECIPES, ids=_RECIPE_IDS)
def test_fills_match_the_row_quantizer(recipe):
    """Both dispatch fills (masked slabs; contiguous segments with the token
    pre-pass) under each recipe: codes, scale atoms and stored row scales
    bitwise with the row quantizer applied to the rows they wrote."""
    K._skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.dispatch_contiguous import (
        contiguous_m_pad_ceiling,
        dispatch_fill_rows_contiguous_blockscaled,
        dispatch_layout_contiguous,
    )
    from sglang.kernels.ops.lora.moe.dispatch_masked import (
        dispatch_fill_masked_blockscaled,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import quantize_rows

    device = torch.device("cuda")
    num_tokens, top_k, num_experts, k = 24, 4, 8, 640
    g = torch.Generator().manual_seed(0x5B1F)
    scores = torch.rand((num_tokens, num_experts), generator=g)
    topk_ids = torch.topk(scores, top_k, dim=1).indices.to(torch.int32)
    topk_ids[0] = -1
    topk_ids[1, 2] = -1
    topk_ids = topk_ids.to(device)
    x = (torch.randn((num_tokens, k), generator=g) * 0.3).to(torch.bfloat16)
    x[2].zero_()
    x[3, ::128] = 32
    x[4] = K._per_token_rows(k, g)[K._ROWS_TINY[0]]
    x[5] = torch.finfo(torch.bfloat16).max
    cases, _ = _case_rows(recipe, g)
    x[6 : 6 + cases.shape[0]] = F.pad(cases, (0, k - _CASE_K))
    x = x.to(device)
    pair_to_row = torch.empty(num_tokens * top_k, dtype=torch.int32, device=device)
    valid = topk_ids.view(-1) >= 0
    tokens = torch.arange(num_tokens, device=device).repeat_interleave(top_k)[valid]
    for domain in ("masked", "contiguous"):
        if domain == "masked":
            m_max = 32
            masked_m = torch.empty(num_experts, dtype=torch.int32, device=device)
            target = _rows_buffer(num_experts * m_max, m_max, k, recipe, device)
            dispatch_fill_masked_blockscaled(
                x,
                topk_ids,
                top_k,
                masked_m_out=masked_m,
                pair_to_row_out=pair_to_row,
                rows_out=target,
            )
        else:
            counts = torch.empty(num_experts, dtype=torch.int32, device=device)
            offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
            m_pad = contiguous_m_pad_ceiling(pair_to_row.numel(), num_experts, 128)
            dispatch_layout_contiguous(
                x,
                topk_ids,
                num_experts,
                top_k,
                128,
                seg_counts_out=counts,
                seg_offsets_out=offsets,
                pair_to_row_out=pair_to_row,
            )
            target = _rows_buffer(m_pad, m_pad, k, recipe, device)
            dispatch_fill_rows_contiguous_blockscaled(
                x,
                topk_ids,
                pair_to_row,
                rows_out=target,
                token_scales=torch.empty(num_tokens, device=device),
            )
        torch.cuda.synchronize()
        reference = _rows_buffer(
            target.data.shape[0], target.rows_per_slab, k, recipe, device
        )
        source = torch.zeros(
            (target.data.shape[0], k), dtype=torch.bfloat16, device=device
        )
        source[pair_to_row[valid].long()] = x[tokens]
        quantize_rows(source, reference, pair_to_row)
        torch.cuda.synchronize()
        assert torch.equal(target.data, reference.data), domain
        assert torch.equal(target.scales, reference.scales), domain
        assert torch.equal(target.row_scales, reference.row_scales), domain
        live_rows = pair_to_row[valid].long()
        assert bool((target.row_scales[live_rows] >= 0).all()), domain
        # The live rows also follow the reference rule, the recipe exercised.
        rows = target.data.shape[0]
        data_ref, sf_ref, s_ref, _, detail = _quant_nvfp4_recipe_ref(source, recipe)
        sf = K._unpack_scale_atoms(target.scales, rows, k, 16, target.rows_per_slab)
        assert torch.equal(target.data[live_rows], data_ref[live_rows]), domain
        assert torch.equal(sf[live_rows], sf_ref[live_rows]), domain
        assert bool(detail["use4"][live_rows].any()), domain


@pytest.mark.parametrize("recipe", _RECIPES, ids=_RECIPE_IDS)
def test_act_delta_matches_the_row_quantizer(recipe):
    """The decode activation's two-pass per-token quantization under each
    recipe: codes, scale atoms and stored row scales bitwise with the row
    quantizer applied to the BF16 rows it stored; the stored scale follows
    the BF16 maximum, which exceeds the FP32 maximum on the rows whose
    largest activation rounded up."""
    K._skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.activation_delta import act_delta
    from sglang.kernels.ops.lora.moe.quant_blockscaled import quantize_rows

    device = torch.device("cuda")
    g = torch.Generator().manual_seed(0xAC7)
    experts, m_max, inter, num_tokens, top_k = 4, 64, 384, 40, 2
    rows = experts * m_max
    gateup = (torch.randn((rows, 2 * inter), generator=g) * 2).to(torch.bfloat16)
    delta = (torch.randn((num_tokens, top_k, 2 * inter), generator=g) * 0.5).to(
        torch.bfloat16
    )
    topk_ids = torch.stack(
        [torch.randperm(experts, generator=g)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_ids[1, 0] = -1
    topk_ids[2] = -1
    pair_to_row = torch.full((num_tokens * top_k,), -1, dtype=torch.int32)
    slots = [0] * experts
    for pair, expert in enumerate(topk_ids.flatten().tolist()):
        if expert >= 0:
            pair_to_row[pair] = expert * m_max + slots[expert]
            slots[expert] += 1
    gateup, delta = gateup.to(device), delta.to(device)
    topk_ids, pair_to_row = topk_ids.to(device), pair_to_row.to(device)
    act_out = torch.zeros((rows, inter), dtype=torch.bfloat16, device=device)
    lora_in = torch.zeros(
        (num_tokens, top_k, inter), dtype=torch.bfloat16, device=device
    )
    target = _rows_buffer(rows, m_max, inter, recipe, device)
    act_delta(
        gateup,
        delta,
        act_out,
        lora_in,
        pair_to_row,
        topk_ids,
        num_local_experts=experts,
        act_quant=target,
    )
    torch.cuda.synchronize()
    reference = _rows_buffer(rows, m_max, inter, recipe, device)
    quantize_rows(act_out, reference, pair_to_row)
    torch.cuda.synchronize()
    assert torch.equal(target.data, reference.data)
    assert torch.equal(target.scales, reference.scales)
    assert torch.equal(target.row_scales, reference.row_scales)
    live = pair_to_row[pair_to_row >= 0].long()
    assert bool(torch.isfinite(target.row_scales).all())
    pairs = torch.nonzero(pair_to_row >= 0, as_tuple=True)[0]
    g32 = gateup.float()[live] + delta.view(-1, 2 * inter).float()[pairs]
    act32 = F.silu(g32[:, :inter]) * g32[:, inter:]
    bf16_amax = act_out.float()[live].abs().amax(dim=1)
    rounded_up = bf16_amax > act32.abs().amax(dim=1)
    assert bool(rounded_up.any())
    expected = bf16_amax / torch.tensor(_denominator(recipe), device=device)
    assert torch.equal(target.row_scales[live], expected)
    data_ref, sf_ref, _, _, detail = _quant_nvfp4_recipe_ref(act_out, recipe)
    sf = K._unpack_scale_atoms(target.scales, rows, inter, 16, m_max)
    assert torch.equal(target.data[live], data_ref[live])
    assert torch.equal(sf[live], sf_ref[live])
    assert bool(detail["use4"][live].any())


@pytest.mark.parametrize(
    "recipe", _PROVIDER_RECIPES, ids=["MAE-256-strict", "MSE-448-fast"]
)
@pytest.mark.parametrize(
    "phase", [Phase.DECODE, Phase.PREFILL], ids=["decode", "prefill"]
)
def test_provider_under_a_recipe(monkeypatch: pytest.MonkeyPatch, phase: Phase, recipe):
    """The NVFP4 per-token providers under a recipe through the sm100
    per-expert rows, decode (expert slabs, the two-pass activation) and
    prefill (flat segments, the separate row quantizer, the down-B fused
    into the base GEMM epilogue by the packaged table's row): eight resident
    slots with adapters beside base tokens, an EP hole, a zero token and two
    tiny tokens, every pattern against the reference that quantizes both GEMM
    inputs under the recipe; then one captured graph per phase replayed over
    the patterns and over changed routing and inputs, bitwise with eager."""
    K._skip_unless_sm100()
    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", P._standalone_output_allocation
    )
    g = torch.Generator(device="cpu").manual_seed(41)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    tokens, slots, rank = (8 if phase is Phase.DECODE else 40), 8, 16
    quant_info, w13_eff, w2_eff, _ = P._quant_info(
        "nvfp4_pt", rand(P._E, 2 * P._I, P._H), rand(P._E, P._H, P._I)
    )
    quant_info = msgspec.structs.replace(quant_info, recipe=recipe)
    assert quant_info.activation_denominator == _denominator(recipe)
    selected = resolve_plans(
        architecture="sm100",
        is_shared_outer=False,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=P._H,
        num_local_experts=P._E,
        quant_family="nvfp4",
        vendor="cutedsl",
        act_scale="per_token",
    )[phase]
    provider = select_provider_cls(selected.base_gemm_rows, "nvfp4", "cutedsl")(
        quant_info
    )
    if phase is Phase.PREFILL:  # the packaged table's epilogue row
        assert selected.plan.down_b.family is BFamily.EPILOGUE
        provider.enable_epilogue_down_lora(rank)
    runner = MoeLoraRunner(
        providers={selected.base_gemm_rows: provider},
        top_k=P._TOPK,
        routed_scaling_factor=P._ROUTED_SCALING,
        activation=ActivationFn.SILU,
    )
    runner.validate_plan(selected.plan, base_gemm_rows=selected.base_gemm_rows)
    runner.plans = {phase: selected}

    hidden0 = rand(tokens, P._H, scale=0.5)
    hidden0[0].zero_()
    for t, amax in P._TINY_TOKENS.items():
        row = (torch.rand(P._H, generator=g) * 2 - 1) * amax
        row[0] = amax
        hidden0[t] = row.to(torch.bfloat16).to(device)
    topk0 = torch.stack(
        [torch.randperm(P._E, generator=g)[: P._TOPK] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    topk0[3, 1] = -1
    weights = torch.tensor([0.65, 0.35], device=device).repeat(tokens, 1)
    gate_up_a = rand(slots, P._E, 2 * rank, P._H, scale=0.1)
    gate_up_b = rand(slots, P._E, 2 * P._I, rank, scale=0.1)
    down_a = rand(slots, P._E, rank, P._I, scale=0.1)
    down_b = rand(slots, P._E, P._H, rank, scale=0.1)
    for adapters, scale in (
        (gate_up_a, 2.0**40),
        (gate_up_b, 2.0**40),
        (down_a, 2.0**20),
        (down_b, 2.0**20),
    ):
        adapters[P._TINY_SLOT] = (adapters[P._TINY_SLOT].float() * scale).to(
            torch.bfloat16
        )
    hidden, topk_ids = hidden0.clone(), topk0.clone()
    mapping = torch.empty(tokens, dtype=torch.int32, device=device)
    states = [(name, False) for name in P._PER_TOKEN_PATTERNS] + [("mixed", True)]

    def load(state):
        name, flipped = state
        mapping.copy_(P._per_token_mapping(name, tokens))
        topk_ids.copy_(topk0.flip(1) if flipped else topk0)
        hidden.copy_(hidden0 * -0.5 if flipped else hidden0)

    def quant(x):
        return _quant_nvfp4_recipe_ref(x.to(torch.bfloat16), recipe)[3].float()

    def reference(mapping):
        pair_tokens = torch.arange(tokens, device=device).repeat_interleave(P._TOPK)
        experts = topk_ids.flatten().long()
        routed = (experts >= 0).float()[:, None]
        experts = experts.clamp(min=0)
        slot = mapping[pair_tokens].long()
        adapted = (slot >= 0).float()[:, None]
        slot = slot.clamp(min=0)
        inputs = hidden.float()[pair_tokens]
        gate_up = P._bf16(torch.einsum("ph,pnh->pn", quant(inputs), w13_eff[experts]))
        bridge = P._bf16(
            torch.einsum("ph,prh->pr", inputs, gate_up_a.float()[slot, experts])
        )
        inner = gate_up_b.float()[slot, experts]
        delta = torch.cat(
            [
                torch.einsum(
                    "pir,pr->pi",
                    inner[:, s * P._I : (s + 1) * P._I],
                    bridge[:, s * rank : (s + 1) * rank],
                )
                for s in range(2)
            ],
            dim=1,
        )
        gate_up = gate_up + adapted * (
            P._bf16(delta) if phase is Phase.DECODE else delta
        )
        act = P._bf16(F.silu(gate_up[:, : P._I]) * gate_up[:, P._I :])
        pairs = P._bf16(torch.einsum("pi,phi->ph", quant(act), w2_eff[experts]))
        down_rank = P._bf16(
            torch.einsum("pi,pri->pr", act, down_a.float()[slot, experts])
        )
        pairs = pairs + adapted * P._bf16(
            torch.einsum("pr,phr->ph", down_rank, down_b.float()[slot, experts])
        )
        pairs = pairs * routed
        return (pairs.view(tokens, P._TOPK, P._H) * weights[..., None]).sum(
            1
        ) * P._ROUTED_SCALING

    dispatch = StandardDispatchOutput(
        hidden_states=hidden,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=weights,
            topk_ids=topk_ids,
            router_logits=torch.zeros((tokens, P._E), device=device),
        ),
    )
    batch = MoeLoraBatch(
        gate_up_lora_a=gate_up_a,
        gate_up_lora_b=gate_up_b,
        down_lora_a=down_a,
        down_lora_b=down_b,
        token_lora_mapping=mapping,
        is_prefill=phase is Phase.PREFILL,
    )

    def forward():
        runner.workspace.routes.clear()
        with get_context().override_server_args():
            out = runner.run(dispatch, batch).hidden_states
        torch.cuda.synchronize()
        return out

    eager = {}
    for state in states:
        load(state)
        out = eager[state] = forward()
        ref = reference(mapping)
        assert bool(torch.isfinite(out).all()), state
        err = float(P._rel_l2(out.float(), ref))
        assert err < P._RUNNER_TOL, (state, err)
        for t in P._TINY_TOKENS:
            row, ref_row = out[t].double(), ref[t].double()
            if ref_row.norm() == 0:
                assert bool((row == 0).all()), (state, t)
            elif int(mapping[t]) >= 0:
                err = float(P._rel_l2(row, ref_row))
                assert err < 2.5e-2, (state, t, err)
            else:
                assert torch.allclose(row, ref_row, rtol=0.1, atol=2.0**-130), (
                    state,
                    t,
                )

    load(states[0])
    batch.use_cuda_graph = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward()
    torch.cuda.synchronize()
    runner.workspace.routes.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        with get_context().override_server_args():
            captured = runner.run(dispatch, batch).hidden_states
    for state in states + states[:1]:
        load(state)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager[state]), state


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

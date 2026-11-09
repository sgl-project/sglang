"""Bitwise small-M BF16 GEMM routes for SM107 (default off; SGLANG_ENABLE_SM107_BF16_SMALLM_ROUTES=1).

On SM107 the stock unquantized BF16 dispatch serves some small-M decode and verify GEMMs of Qwen3.8-Flash-Next at TP1
as a cuBLAS split-K kernel plus a separate split-K reduce, or on a slower non-split tile. Each key listed in
``ROUTES`` runs instead as one kernel whose output bits equal the stock route's:

* ``"tgv"``: the CuTe DSL TGV kernel, which never splits K, at keys where stock runs a non-split cuBLAS kernel (the
  same sequential K order); tactic ``None`` is TGV's own pick.
* ``"tgvc"``: the TGV kernel at a tile outside its tactic table (a 2-CTA token tile covering every row), same K order.
* ``"cl"``: FlashInfer's cluster split-K kernel at cuBLAS's slicing (equal K slices, FP32 partials summed in rank
  order); tactic ``(mma_m, mma_n, split_k, stages, cta_k)``.

Every key was checked bit-equal to the stock dispatch on SM107 with cuBLAS 13.8 and FlashInfer 0.7.2, eagerly and
under CUDA graph replay. The cluster routes install only against that FlashInfer split-K source, and at load
``precompile`` checks every key once more against the live stock dispatch: a mismatch on either side turns the
routes off, and until that check has run (it runs when a model calls ``precompile_splitk_tactics`` after loading,
as Qwen4-Exp does) the routes are not served. Keys where no bit-equal single kernel beat stock keep the stock
route: gdn_in_proj and attn_qkv at M 1-4, the shared expert's gate_up at M 17-32 and out_proj below M 64. Any
other shape, a bias, an addend or a non-contiguous operand takes the stock dispatch.

``NONBITWISE_ROUTES`` (opt-in, SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_ROUTES=1) are faster kernels whose bits
differ from stock in FP32 summation order; each is checked at load against a relative-L2 bound instead.
``NONBITWISE_LOWM_ROUTES`` (opt-in, SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_LOWM_ROUTES=1) are the same kind of
key for the same shape at smaller M, checked the same way.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Callable, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_GDN_IN_PROJ = (16480, 2560)
_ATTN_QKV = (13312, 2560)
_ROUTER = (512, 2560)
_INDEX_QK = (640, 2560)
_SHARED_GATE_UP = (1280, 2560)
_OUT_PROJ = (2560, 6144)

# (M, N, K) -> (kind, tactic)
ROUTES = {
    **{(m, *_GDN_IN_PROJ): ("tgv", None) for m in (20, 24)},
    **{(m, *_GDN_IN_PROJ): ("tgv", 26) for m in (28, 32)},
    **{(m, *_ATTN_QKV): ("tgv", None) for m in (20, 24, 28, 32, 72, 80, 96)},
    **{(m, *_ATTN_QKV): ("tgvc", (64, 80, 8, True)) for m in (68, 76)},
    **{(m, *_ATTN_QKV): ("tgvc", (64, 96, 7, True)) for m in (84, 88, 92)},
    # cuBLAS 13.8 sums 4 K slices of 640 in rank order at M <= 16 and 8 slices of 320 at M 17-64 for these shapes.
    **{
        (m, *nk): ("cl", (64, 8, 4, 5, 128))
        for nk in (_ROUTER, _INDEX_QK)
        for m in (5, 6, 7, *range(9, 17))
    },
    **{
        (m, *nk): ("cl", (64, 16, 8, 5, 64))
        for nk in (_ROUTER, _INDEX_QK)
        for m in range(17, 33)
    },
    **{(m, *_SHARED_GATE_UP): ("cl", (64, 8, 4, 5, 128)) for m in range(1, 17)},
    # Stock runs TGV tactic 18 here; tactic 21 is the same unsplit K order.
    (64, *_OUT_PROJ): ("tgv", 21),
}
# SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_ROUTES=1 (with the routes above on) adds these keys, whose output is
# the same product with FP32 partial sums added in another order, so not the stock bits. ``precompile`` checks each
# against the stock dispatch by relative L2 instead of bit equality and drops only a key that fails. GDN out_proj and
# attention o_proj share (N 2560, K 6144); at M 20-32 (verify of 5-8 requests) stock runs TGV tactic 18, 80 CTAs on
# 212 SMs, where four K slices of 1536 run 160 CTAs.
NONBITWISE_ROUTES = {
    (m, *_OUT_PROJ): ("cl", (64, 32, 4, 8, 128)) for m in range(20, 33)
}
# SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_LOWM_ROUTES=1 (with the routes above on) adds the same shape at M 4-16
# (verify of 1-4 requests, draft-extend and short extends): stock runs TGV cta64x16 on 40 CTAs there; four K slices
# of 1536 run 160 CTAs (320 for mma_n 16 at M 9-16). Checked and dropped key by key exactly as NONBITWISE_ROUTES.
# Tactics measured on SM107 (mb-mid2-op1): 5.5 against 7.4 us per call at M 4-8, 5.7 against 7.4 at M 9-16.
NONBITWISE_LOWM_ROUTES = {
    **{(m, *_OUT_PROJ): ("cl", (64, 8, 4, 8, 128)) for m in range(4, 9)},
    **{(m, *_OUT_PROJ): ("cl", (64, 16, 4, 10, 128)) for m in range(9, 17)},
}
# Measured 1.2-1.7e-4 at these keys on SM107; one BF16 ulp on every output element is about 4e-3.
_NONBITWISE_REL_L2_MAX = 1e-3
# FlashInfer 0.7.2's dense_bf16_gemm_sm100_splitk.py with FlashInfer #7443's hand-off fix, vendored as
# kernels/ops/gemm/dense_bf16_gemm_sm100_splitk.py (0.7.2's tiles, K slices and reduction order, so 0.7.2's bits): the
# "cl" routes call into its compile path, so they are installed only against that exact source.
_FI_SPLITK_SHA256 = "f00c89b75ade23d868f9fd11fef2bb5b7c8deee72e61fa27e78bd0c390fb75d3"

_routes: Optional[dict] = None
_fi = None  # (FlashInfer split-K module, CUstream constructor)
_compiled: dict = {}
_logged: set = set()
_checked = False  # set by precompile's check; routes serve only after it
_fused_act = None  # sm107_shared_expert_act when SGLANG_ENABLE_SM107_SHARED_EXPERT_FUSED_ACT is on
_nonbitwise: frozenset = frozenset()  # the NONBITWISE_ROUTES keys in _routes
_precompiling = False


def _cluster_smem(mma_m: int, mma_n: int, split_k: int, stages: int, cta_k: int) -> int:
    """FlashInfer 0.7.2 dense_bf16_gemm_sm100_splitk._smem_bytes with the K tile as a parameter."""

    def align(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    cursor = align(mma_m * cta_k * 2 * stages, 1024) + mma_n * cta_k * 2 * stages
    cursor = align(align(cursor, 8) + 2 * stages * 8 + 3 * 8, 4) + 4
    if split_k == 1:
        return cursor
    return align(align(cursor, 128) + (split_k - 1) * mma_m * mma_n * 4, 8) + 8


def _cluster_ok(k: int, tactic: tuple, smem_capacity: int) -> bool:
    mma_m, mma_n, split_k, stages, cta_k = tactic
    return (
        k % (split_k * cta_k) == 0
        and 1 <= stages <= k // (split_k * cta_k)
        and _cluster_smem(*tactic) <= smem_capacity
    )


def _tgvc_ok(cta_m: int, cta_n: int, stages: int, use_2cta: bool) -> bool:
    step = 16 if use_2cta else 8
    smem = stages * (cta_m + cta_n // (2 if use_2cta else 1)) * 128 * 2 + 4096
    return (
        cta_m in (64, 128)
        and step <= cta_n <= 256
        and cta_n % step == 0
        and 2 <= stages
        and smem <= 227 * 1024
    )


def install(device_sm: int, splitk_enabled: bool, cutedsl_available: bool) -> bool:
    """Install the routes; returns whether they are on. Called once by unquant.initialize_bf16_gemm_config."""
    routes_on = _install_routes(device_sm, splitk_enabled, cutedsl_available)
    _install_fused_act(routes_on)
    return routes_on


def _install_fused_act(routes_on: bool) -> None:
    global _fused_act
    _fused_act = None
    if not envs.SGLANG_ENABLE_SM107_SHARED_EXPERT_FUSED_ACT.get():
        return
    if not routes_on:
        logger.warning(
            "SGLANG_ENABLE_SM107_SHARED_EXPERT_FUSED_ACT needs the SM107 small-M routes on; the stock pair stays"
        )
        return
    from sglang.srt.layers.quantization import sm107_shared_expert_act

    sm107_shared_expert_act.install()
    _fused_act = sm107_shared_expert_act


def _install_routes(
    device_sm: int, splitk_enabled: bool, cutedsl_available: bool
) -> bool:
    global _routes, _fi, _checked, _nonbitwise
    _routes = None
    _checked = False
    _nonbitwise = frozenset()
    if not (splitk_enabled and cutedsl_available and device_sm == 107):
        logger.warning(
            "SGLANG_ENABLE_SM107_BF16_SMALLM_ROUTES needs SM107 and the cutedsl BF16 GEMM backend with split-K; "
            "the stock dispatch stays"
        )
        return False
    try:
        import cuda.bindings.driver as cuda_driver
        from sglang.kernels.ops.gemm import dense_bf16_gemm_sm100_splitk as fi_splitk

        with open(fi_splitk.__file__, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
    except (ImportError, OSError) as err:
        logger.warning(
            "SGLANG_ENABLE_SM107_BF16_SMALLM_ROUTES: FlashInfer's split-K module is unavailable (%s); "
            "the stock dispatch stays",
            err,
        )
        return False
    if digest != _FI_SPLITK_SHA256:
        logger.warning(
            "SGLANG_ENABLE_SM107_BF16_SMALLM_ROUTES: FlashInfer's split-K source (sha256 %s) is not the one the routes "
            "were checked against; the stock dispatch stays",
            digest[:16],
        )
        return False
    for (m, n, k), (kind, tactic) in ROUTES.items():
        if (kind == "tgvc" and not _tgvc_ok(*tactic)) or (
            kind == "cl" and not _cluster_ok(k, tactic, fi_splitk._SMEM_CAPACITY_BYTES)
        ):
            logger.warning(
                "SGLANG_ENABLE_SM107_BF16_SMALLM_ROUTES: route %s %s at %s is invalid; the stock dispatch stays",
                kind,
                tactic,
                (m, n, k),
            )
            return False
    _fi = (fi_splitk, cuda_driver.CUstream)
    _routes = ROUTES
    kinds = [route[0] for route in ROUTES.values()]
    logger.info(
        "SM107 bitwise small-M BF16 GEMM routes on: %d keys (cluster split-K %d, TGV %d)",
        len(kinds),
        kinds.count("cl"),
        kinds.count("tgv") + kinds.count("tgvc"),
    )
    if envs.SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_ROUTES.get():
        _install_nonbitwise(fi_splitk._SMEM_CAPACITY_BYTES)
    if envs.SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_LOWM_ROUTES.get():
        _install_nonbitwise_lowm(fi_splitk._SMEM_CAPACITY_BYTES)
    return True


def _install_nonbitwise(smem_capacity: int) -> None:
    """Add NONBITWISE_ROUTES to the installed routes; they serve after precompile's relative-L2 check."""
    global _routes, _nonbitwise
    for (m, n, k), (kind, tactic) in NONBITWISE_ROUTES.items():
        if kind != "cl" or not _cluster_ok(k, tactic, smem_capacity):
            logger.warning(
                "SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_ROUTES: route %s %s at %s is invalid; "
                "only the bitwise routes serve",
                kind,
                tactic,
                (m, n, k),
            )
            return
    _routes = {**ROUTES, **NONBITWISE_ROUTES}
    _nonbitwise = frozenset(NONBITWISE_ROUTES)


def _install_nonbitwise_lowm(smem_capacity: int) -> None:
    """Add NONBITWISE_LOWM_ROUTES to the installed routes (after NONBITWISE_ROUTES, if on); an invalid tactic leaves
    every other installed key alone. They serve after precompile's relative-L2 check, as NONBITWISE_ROUTES."""
    global _routes, _nonbitwise
    for (m, n, k), (kind, tactic) in NONBITWISE_LOWM_ROUTES.items():
        if kind != "cl" or not _cluster_ok(k, tactic, smem_capacity) or (m, n, k) in _routes:
            logger.warning(
                "SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_LOWM_ROUTES: route %s %s at %s is invalid or already "
                "routed; the low-M keys keep the stock dispatch",
                kind,
                tactic,
                (m, n, k),
            )
            return
    _routes = {**_routes, **NONBITWISE_LOWM_ROUTES}
    _nonbitwise = _nonbitwise | frozenset(NONBITWISE_LOWM_ROUTES)


def _cluster_out(
    x_2d: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, tactic
) -> None:
    fi_splitk, stream_ctor = _fi
    mma_m, mma_n, split_k, stages, cta_k = tactic
    tensors = fi_splitk._to_cute_swap(x_2d, weight.T, out, None)
    key = (x_2d.get_device(), x_2d.dtype, tactic, tensors[4])
    compiled = _compiled.get(key)
    if compiled is None:
        fi_tactic = fi_splitk.SplitKTactic(mma_m, mma_n, split_k, stages)
        if cta_k == fi_splitk._CTA_K:
            compiled = fi_splitk._get_compiled_splitk_kernel(
                device_index=x_2d.get_device(),
                dtype=x_2d.dtype,
                tactic=fi_tactic,
                use_pdl=True,
                has_bias=False,
                leading_dims=tensors[4],
            )
        else:
            # The kernel body reads only self.cta_k; FlashInfer's _CTA_K is host-side validation.
            kernel = fi_splitk.SplitKDenseGemmKernel(
                tactic=fi_tactic, use_pdl=True, has_bias=False
            )
            kernel.cta_k = cta_k
            with torch.cuda.device(x_2d.device):
                rep = fi_splitk._make_compile_repr_tensors(
                    x_2d.dtype, False, *tensors[4]
                )
                compiled = fi_splitk.cute_ext.compile(
                    fi_splitk._bmm_no_bias,
                    kernel,
                    *rep[:3],
                    stream_ctor(torch.cuda.current_stream().cuda_stream),
                )
        _compiled[key] = compiled
    compiled(
        *tensors[:3], stream_ctor(torch.cuda.current_stream(x_2d.device).cuda_stream)
    )


def _tgvc_out(
    x_2d: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, tactic
) -> None:
    import cuda.bindings.driver as cuda_driver

    from sglang.kernels.ops.gemm import cutedsl_bf16_gemm as tgv

    cta_m, cta_n, stages, use_2cta = tactic
    a_, b_, c_, _, (a_leading, b_leading, c_leading) = tgv._to_cute_swap(
        x_2d, weight.t(), out, None
    )
    compiled = tgv._get_compiled_cute_ext_kernel(
        dtype=x_2d.dtype,
        c_dtype=out.dtype,
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=tgv._TGV_CUTE_EXT_CTA_K,
        num_ab_stage=stages,
        use_2cta=use_2cta,
        use_pdl=True,
        has_bias=False,
        a_leading=a_leading,
        b_leading=b_leading,
        c_leading=c_leading,
    )
    compiled(
        a_,
        b_,
        c_,
        cuda_driver.CUstream(torch.cuda.current_stream(x_2d.device).cuda_stream),
    )


def _tgv_out(
    x_2d: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, tactic
) -> None:
    from sglang.kernels.ops.gemm import cutedsl_bf16_gemm as tgv

    m, k = x_2d.shape
    tgv._run_tgv(
        x_2d,
        weight.t(),
        None,
        out,
        True,
        tactic if tactic is not None else tgv._pick_tactic(m, weight.shape[0], k),
    )


_RUNNERS = {"cl": _cluster_out, "tgvc": _tgvc_out, "tgv": _tgv_out}


def route_for(x: torch.Tensor, weight: torch.Tensor) -> Optional[tuple]:
    """The (kind, tactic) that serves ``x @ weight.T``, or None when the stock dispatch does."""
    if _routes is None or not _checked:
        return None
    k = x.shape[-1]
    m = x.numel() // k
    route = _routes.get((m, weight.shape[0], weight.shape[1]))
    if route is None or not (x.is_contiguous() and weight.is_contiguous()):
        return None
    if not (
        x.view(-1, k).stride() == (k, 1)
        and weight.stride() == (k, 1)
        and x.data_ptr() % 32 == 0
        and weight.data_ptr() % 32 == 0
    ):
        return None
    return route


def gemm(x: torch.Tensor, weight: torch.Tensor) -> Optional[torch.Tensor]:
    """``x @ weight.T`` from one kernel for a listed (M, N, K), or None: a ROUTES key with the stock route's bits,
    a NONBITWISE_ROUTES key within its load-checked relative-L2 bound."""
    route = route_for(x, weight)
    if route is None:
        return None
    k = x.shape[-1]
    m = x.numel() // k
    kind, tactic = route
    out = torch.empty((m, weight.shape[0]), dtype=x.dtype, device=x.device)
    _RUNNERS[kind](x.view(-1, k), weight, out, tactic)
    if (kind, tuple(weight.shape)) not in _logged:
        _logged.add((kind, tuple(weight.shape)))
        logger.debug(
            "SM107 small-M route %s: m=%d n=%d k=%d tactic=%s",
            kind,
            m,
            weight.shape[0],
            k,
            tactic,
        )
    if _nonbitwise and not _precompiling and "nonbitwise" not in _logged:
        if (m, *weight.shape) in _nonbitwise:
            _logged.add("nonbitwise")
            logger.info(
                "SM107 non-bitwise small-M route serving: m=%d n=%d k=%d %s %s (first use)",
                m,
                weight.shape[0],
                k,
                kind,
                tactic,
            )
    if (
        _nonbitwise
        and not _precompiling
        and "nonbitwise_lowm" not in _logged
        and (m, *weight.shape) in NONBITWISE_LOWM_ROUTES
        and (m, *weight.shape) in _nonbitwise
    ):
        _logged.add("nonbitwise_lowm")
        logger.info(
            "SM107 non-bitwise low-M route serving: m=%d n=%d k=%d %s %s "
            "(first use; SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_LOWM_ROUTES)",
            m,
            weight.shape[0],
            k,
            kind,
            tactic,
        )
    return out.view(*x.shape[:-1], weight.shape[0])


def precompile(
    *,
    stock_gemm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device=None,
) -> bool:
    """JIT-compile every key at load and check it bit for bit against ``stock_gemm`` (the dispatch with the routes
    off) on random operands, so a change on the stock side (cuBLAS's slicing) cannot go unnoticed; any mismatch turns
    the routes off. NONBITWISE_ROUTES keys are checked after, by relative L2, and a failure drops only that key.
    Returns whether the routes stay on."""
    global _routes, _fused_act, _checked
    if _routes is None:
        return False
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(0)
    _checked = True  # gemm() serves the routes during the check
    weights = {}
    for m, n, k in sorted(key for key in _routes if key not in _nonbitwise):
        if (n, k) not in weights:
            weights[(n, k)] = (
                torch.randn(n, k, generator=gen, device=device) * 0.02
            ).to(torch.bfloat16)
        x = torch.randn(m, k, generator=gen, device=device).to(torch.bfloat16)
        got = gemm(x, weights[(n, k)])
        want = stock_gemm(x, weights[(n, k)])
        if got is None or not torch.equal(
            got.view(torch.int16), want.view(torch.int16)
        ):
            logger.warning(
                "SGLANG_ENABLE_SM107_BF16_SMALLM_ROUTES: route %s at %s differs from the stock dispatch on this "
                "build; the stock dispatch stays",
                _routes[(m, n, k)],
                (m, n, k),
            )
            _routes = None
            _checked = False
            return False
    if _nonbitwise:
        _check_nonbitwise(stock_gemm=stock_gemm, device=device)
    if _fused_act is not None and not _fused_act.precompile(
        stock_gemm=stock_gemm, device=device
    ):
        _fused_act = None
    return True


def _rel_l2(got: Optional[torch.Tensor], want: torch.Tensor) -> float:
    if got is None:
        return float("inf")
    return float(
        torch.linalg.vector_norm(got.float() - want.float())
        / torch.linalg.vector_norm(want.float())
    )


def _check_nonbitwise(
    *,
    stock_gemm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device,
) -> None:
    """Drop each NONBITWISE_ROUTES key farther than _NONBITWISE_REL_L2_MAX (relative L2) from ``stock_gemm`` on random
    operands; the ROUTES keys are untouched."""
    global _routes, _nonbitwise, _precompiling
    gen = torch.Generator(device=device).manual_seed(1)
    weights, errors = {}, {}
    _precompiling = True
    try:
        for m, n, k in sorted(_nonbitwise):
            if (n, k) not in weights:
                weights[(n, k)] = (
                    torch.randn(n, k, generator=gen, device=device) * 0.02
                ).to(torch.bfloat16)
            x = torch.randn(m, k, generator=gen, device=device).to(torch.bfloat16)
            errors[(m, n, k)] = _rel_l2(
                gemm(x, weights[(n, k)]), stock_gemm(x, weights[(n, k)])
            )
    finally:
        _precompiling = False
    # ``not <=`` also fails a NaN.
    failed = {key for key, err in errors.items() if not err <= _NONBITWISE_REL_L2_MAX}
    for key in sorted(failed):
        logger.warning(
            "SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_ROUTES: route %s at %s is %.2e from the stock dispatch "
            "(relative L2, bound %.0e); that key keeps the stock dispatch",
            _routes[key],
            key,
            errors[key],
            _NONBITWISE_REL_L2_MAX,
        )
    if failed:
        _routes = {key: route for key, route in _routes.items() if key not in failed}
        _nonbitwise = _nonbitwise - failed
    if _nonbitwise:
        logger.info(
            "SM107 non-bitwise small-M BF16 GEMM routes on: %d of %d keys (%s), relative L2 to the stock dispatch "
            "at most %.2e (bound %.0e)",
            len(_nonbitwise),
            len(errors),
            ", ".join(
                sorted(
                    {f"N {n} K {k}: {_routes[(m, n, k)]}" for m, n, k in _nonbitwise}
                )
            ),
            max(errors[key] for key in _nonbitwise),
            _NONBITWISE_REL_L2_MAX,
        )

"""DeepSeek-V4 MXFP4 expert backend backed by the b12x W4A16 fused MoE.

Same checkpoint as :mod:`mxfp4_flashinfer_cutlass_moe`, different runtime: one
fused kernel per expert layer instead of seven, and bf16 activations instead of
MXFP8.

This uses the standalone ``b12x`` package rather than the copy vendored inside
FlashInfer. Three things depend on that choice:

* ``source_format="fp4_e8m0_k32"`` consumes the checkpoint's E8M0 K/32 block
  scales byte for byte, so there is no scale relayout to get wrong.
* ``swiglu_limit`` is honoured. DSv4-Flash asks for a SwiGLU clamp at 10.0 and
  it binds on real data (gate magnitudes reach 10.1, and half the layers sit
  within 7% of the limit), so dropping it is a correctness change, not a
  rounding one.
* Kernels are warmed before CUDA graph capture. b12x specializes on token
  count only as ``size_m == 1`` vs everything else (``_m_specialization_key``
  in its w4a16 kernel), so this is two compiles, not one per prompt length --
  but it refuses to compile at all while a graph is being captured, which is
  what the warm-up exists for.

Install with ``pip install --no-deps b12x==1.2.2``; without ``--no-deps`` pip
pulls a newer torch and breaks the rest of the image.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.environ import envs
from sglang.srt.utils import log_info_on_rank0
from sglang.srt.utils.common import is_sm120_supported

# Suppress TRT-LLM CUTLASS trace logs without overriding user configuration.
os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

# One scratch buffer per (num_experts, k, n, top_k, device), shared by every
# layer: they run sequentially and it is pure scratch. Allocated at load time so
# it can never land in a CUDA graph's private memory pool -- a buffer first
# allocated during capture is only valid inside that graph, and reusing it from
# an un-captured forward faults.
_B12X_SCRATCH: dict = {}

# Every expert layer in the model has the same shape, so they share one weight
# plan. That is not just a memory saving: b12x's compiled-kernel cache is keyed
# per plan, so sharing means the warm-up below compiles each token count once
# for the whole model instead of once per layer.
_B12X_WEIGHT_PLANS: dict = {}


def _select_b12x_distribution() -> None:
    """Put the pinned b12x on sys.path ahead of whatever pip installed.

    b12x 1.2.2 generalized its W4A16 kernel to also serve trellis rotation,
    extra gating modes, expert maps and activation-amax collection. The
    weights and scales
    went from ``cute.Tensor`` (compile-time-static strides) to raw
    ``cute.Pointer`` with layouts rebuilt at runtime, which costs 14 registers
    and ~19% on this shape -- measured at 875.6 us/call against 707.9 for
    0.15.3, same GPU, same weights, same inputs, bit-identical outputs.
    DSv4-Flash uses none of the features that bought.
    """
    path = envs.SGLANG_B12X_PATH.get()
    if not path:
        return
    if not os.path.isdir(os.path.join(path, "b12x")):
        raise FileNotFoundError(f"SGLANG_B12X_PATH={path!r} has no b12x package.")
    loaded = sys.modules.get("b12x")
    if loaded is not None:
        # Idempotent: this runs once per expert layer. Only a b12x already
        # imported from somewhere else is a problem, and it is fatal -- the two
        # generations share a module name but not an API.
        want = os.path.realpath(path)
        got = os.path.realpath(os.path.dirname(os.path.dirname(loaded.__file__)))
        if want != got:
            raise RuntimeError(
                f"SGLANG_B12X_PATH={path!r} came too late: b12x was already "
                f"imported from {loaded.__file__}."
            )
        return
    sys.path.insert(0, path)


def is_b12x_available() -> bool:
    try:
        _select_b12x_distribution()
        import b12x  # noqa: F401

        return True
    except ImportError:
        return False


def _b12x_is_legacy() -> bool:
    """True for the 0.15.3-era API (``b12x.integration.tp_moe``)."""
    import b12x  # noqa: F401

    try:
        import b12x.moe.fused_moe  # noqa: F401

        return False
    except ImportError:
        return True


def _b12x_max_tokens() -> int:
    """Upper bound on tokens in a single MoE call (prefill is chunked)."""
    return envs.SGLANG_B12X_MAX_TOKENS.get()


def _core_token_counts(max_tokens: int, tokens_per_seq: int) -> tuple[int, ...]:
    """Token counts to plan capacity for, and to warm kernels at.

    Under CUDA graph capture b12x refuses to compile (``_require_cached``) and
    falls back to the plan's pinned tile config, which does not fit every block
    size -- so the launch fails outright rather than running slowly. Every token
    count a captured graph can present must therefore have been run once
    already. Decode graphs are captured at the configured batch sizes times the
    speculative window; prefill is chunked, so a power-of-two ladder covers it.

    Kernel compilation dedupes down to two variants (see ``_warmup``); the rest
    of this ladder is what sizes the scratch arena.
    """
    counts = set()
    batch_sizes = list(range(1, 33))
    try:
        from sglang.srt.runtime_context import get_exec

        cg_config = get_exec().graph.cuda_graph_config
        decode = cg_config.decode if cg_config is not None else None
        batch_sizes = list(getattr(decode, "bs", None) or batch_sizes)
    except Exception:
        # No server context (unit tests, offline use): the dense 1..32 ladder
        # already covers every batch size a decode graph can be captured at.
        pass
    for bs in batch_sizes:
        counts.add(bs * max(tokens_per_seq, 1))
    n = 128
    while n <= max_tokens:
        counts.add(n)
        n *= 2
    counts.add(max_tokens)
    return tuple(sorted(c for c in counts if 0 < c <= max_tokens))


_BLOCKS_PER_SM_RELAXED = False


def _relax_blocks_per_sm() -> None:
    """Undo b12x's one-CTA-per-SM pin for moe_block_size==8.

    b12x pins ``blocks_per_sm = 1`` whenever the routed block size is 8, to keep
    the fused kernel's grid-barrier atomic from serializing across a wide grid.
    Every decode token count lands on block size 8 here -- the routed fill test
    is ``m * top_k / num_experts < 0.9 * 8``, so any m below ~308 takes it -- and
    GB10 has 48 SMs, so the pin runs the machine at grid 48 where the unpinned
    heuristic would pick 144.

    Must run before the kernels are compiled: ``blocks_per_sm`` is part of the
    launch bounds and therefore of b12x's kernel cache key, and b12x refuses to
    compile during CUDA graph capture.
    """
    global _BLOCKS_PER_SM_RELAXED
    if _BLOCKS_PER_SM_RELAXED:
        return
    from b12x.moe._shared.kernels.w4a16 import kernel as b12x_kernel

    original = b12x_kernel._determine_blocks_per_sm

    def _unpinned(*args, **kwargs):
        # Route to the cta_m_blocks==1 branch, which caps at 4 rather than 1.
        # This also picks the uses_m_block_8=False register count (143 vs 120),
        # the more conservative of the two, so occupancy stays achievable.
        kwargs["uses_m_block_8"] = False
        return original(*args, **kwargs)

    b12x_kernel._determine_blocks_per_sm = _unpinned
    _BLOCKS_PER_SM_RELAXED = True
    log_info_on_rank0(
        logger, "b12x: dropped the one-CTA-per-SM pin for moe_block_size==8."
    )


def _warmup(*, plan, scratch, experts, top_k, num_experts, hidden_size, device, counts):
    """Compile one kernel per token count, outside CUDA graph capture.

    b12x refuses to compile while a graph is being captured, and falls back to
    the plan's pinned tile config -- which does not fit every block size, so the
    launch fails outright rather than running slowly. ``core_token_counts``
    only declares the counts; it does not compile them. So every count a
    captured graph can present has to be run once here, eagerly, first.
    """
    from b12x.moe import fused_moe

    for tokens in counts:
        a = torch.zeros(tokens, hidden_size, dtype=torch.bfloat16, device=device)
        topk_ids = torch.arange(tokens * top_k, dtype=torch.int32, device=device)
        topk_ids = (topk_ids % num_experts).view(tokens, top_k)
        topk_weights = torch.full(
            (tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device
        )
        out = torch.empty(tokens, hidden_size, dtype=torch.bfloat16, device=device)
        fused_moe.run(
            binding=fused_moe.bind(
                plan,
                scratch=scratch,
                a=a,
                experts=experts,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                output=out,
            )
        )
    torch.cuda.synchronize()


_B12X_WARMED = False


def _warmup_legacy(*, launch, top_k, num_experts, hidden_size, device, counts):
    """Compile the 0.15.3 kernels before any CUDA graph capture."""
    global _B12X_WARMED
    for tokens in counts:
        a = torch.zeros(tokens, hidden_size, dtype=torch.bfloat16, device=device)
        ids = torch.arange(tokens * top_k, dtype=torch.int32, device=device)
        ids = (ids % num_experts).view(tokens, top_k)
        w = torch.full((tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
        out = torch.empty(tokens, hidden_size, dtype=torch.bfloat16, device=device)
        launch(a, ids, w, out)
    torch.cuda.synchronize()
    _B12X_WARMED = True


def _legacy_prepare(*, w13, s13, w2, s2, ones):
    """0.15.3 weight prep. Repacks in place, so the caller's tensors stay live."""
    from b12x.integration import prepare_b12x_fp4_moe_weights

    return prepare_b12x_fp4_moe_weights(
        source_format="fp4_e8m0_k32",
        w13_layout="up_gate",
        w1_fp4=w13,
        w1_blockscale=s13,
        w1_global_scale=ones,
        a1_gscale=ones,
        w2_fp4=w2,
        w2_blockscale=s2,
        w2_global_scale=ones,
        a2_gscale=ones,
        activation="silu",
        params_dtype=torch.bfloat16,
        prepare_runtime_alphas=False,
        prepare_w4a16=True,
        reuse_input_storage=True,
    )


def _legacy_plan(
    *, num_experts, hidden_size, intermediate, top_k, device, swiglu_limit, counts
):
    """0.15.3 scratch plan. Returns (plan, scratch)."""
    from b12x.integration.tp_moe import TPMoEScratchCaps, plan_tp_moe_scratch

    caps = TPMoEScratchCaps(
        max_tokens=max(counts),
        weight_E=num_experts,
        k=hidden_size,
        n=intermediate,
        num_topk=top_k,
        device=device,
        dtype=torch.bfloat16,
        core_token_counts=tuple(counts),
        route_num_experts=0,
        route_logits_dtype=None,
        quant_mode="w4a16",
        activation="silu",
        apply_router_weight_on_input=False,
        swiglu_limit=swiglu_limit,
        source_format="fp4_e8m0_k32",
        w13_layout="up_gate",
        frozen=True,
    )
    plan = plan_tp_moe_scratch(caps)
    specs = plan.scratch_specs()
    if len(specs) != 1 or specs[0].dtype != torch.uint8:
        raise RuntimeError(f"unexpected b12x scratch specs: {specs}")
    scratch = torch.empty(int(specs[0].shape[0]), dtype=torch.uint8, device=device)
    return plan, scratch


def _legacy_launcher(*, plan, scratch, prepared, w13, s13, w2, s2, ones, swiglu_limit):
    """Bind-and-run closure for 0.15.3, matching vLLM's call sequence."""
    from b12x.integration.tp_moe import b12x_moe_fp4

    packed = prepared.w4a16
    w1_alphas = prepared.w1_runtime_alphas
    w2_alphas = prepared.w2_runtime_alphas
    if w1_alphas is None:
        w1_alphas = ones
    if w2_alphas is None:
        w2_alphas = ones

    def launch(a, topk_ids, topk_weights, out):
        b12x_moe_fp4(
            binding=plan.bind(
                scratch=scratch,
                a=a,
                a1_gscale=ones,
                w1_fp4=w13,
                w1_blockscale=s13,
                w1_alphas=w1_alphas,
                a2_gscale=ones,
                w2_fp4=w2,
                w2_blockscale=s2,
                w2_alphas=w2_alphas,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output=out,
                activation="silu",
                quant_mode="w4a16",
                source_format="fp4_e8m0_k32",
                w13_layout="up_gate",
                prepared_w4a16=packed,
                swiglu_limit=swiglu_limit,
            )
        )

    return launch


def _get_weight_plan(*, num_experts: int, hidden_size: int, intermediate: int):
    """One weight plan per shape, shared by every expert layer."""
    from b12x.moe import fused_moe

    key = (num_experts, hidden_size, intermediate)
    plan = _B12X_WEIGHT_PLANS.get(key)
    if plan is None:
        plan = fused_moe.plan_weights(
            quant_modes="w4a16",
            source_format="fp4_e8m0_k32",
            activation="silu",
            params_dtype=torch.bfloat16,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate,
            w13_layout="up_gate",
        )
        _B12X_WEIGHT_PLANS[key] = plan
    return plan


def _get_scratch(
    *,
    weight_plan,
    top_k: int,
    device,
    swiglu_limit,
    tokens_per_seq,
    experts,
    num_experts: int,
    hidden_size: int,
):
    """Plan and allocate the shared scratch buffer, once per shape."""
    from b12x.moe import fused_moe

    max_tokens = _b12x_max_tokens()
    key = (id(weight_plan), top_k, str(device), swiglu_limit)
    entry = _B12X_SCRATCH.get(key)
    if entry is None:
        if envs.SGLANG_B12X_RELAX_BLOCKS_PER_SM.get():
            _relax_blocks_per_sm()
        counts = _core_token_counts(max_tokens, tokens_per_seq)
        caps = fused_moe.Caps(
            max_tokens=max_tokens,
            num_topk=top_k,
            device=device,
            weight_plan=weight_plan,
            quant_mode="w4a16",
            swiglu_limit=swiglu_limit,
            core_token_counts=counts,
            frozen=True,
        )
        plan = fused_moe.plan(caps)
        scratch = torch.empty(
            fused_moe.required_nbytes(caps), dtype=torch.uint8, device=device
        )
        log_info_on_rank0(
            logger,
            f"Compiling {len(counts)} b12x W4A16 kernels "
            f"(token counts {counts[0]}..{counts[-1]})...",
        )
        _warmup(
            plan=plan,
            scratch=scratch,
            experts=experts,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            device=device,
            counts=counts,
        )
        entry = (plan, scratch)
        _B12X_SCRATCH[key] = entry
    return entry


class Mxfp4B12xMoEMethod:
    """b12x MXFP4 MoE: one fused W4A16 kernel per expert layer."""

    def __init__(self, fp8_method, prefix: str):
        if not is_b12x_available():
            raise RuntimeError(
                "The b12x MoE backend needs the b12x package: "
                "pip install --no-deps b12x==1.2.2"
            )
        if not is_sm120_supported():
            raise RuntimeError("Mxfp4B12xMoEMethod requires SM120 or SM121.")
        self._fp8 = fp8_method
        self.prefix = prefix
        self._experts: Any = None
        self._plan: Any = None
        self._scratch: torch.Tensor | None = None
        self._swiglu_limit: float | None = None
        # Set on the 0.15.3 path, where the whole call is a prepared closure.
        self._launch: Any = None
        self._legacy_keepalive: Any = None
        self._ep_guard = False

    @property
    def load_up_proj_weight_first(self) -> bool:
        """Load W13 as ``[up; gate]`` -- declared to b12x as w13_layout='up_gate'."""
        return True

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype,
        **extra_weight_attrs,
    ):
        if hidden_size % 128 != 0 or intermediate_size_per_partition % 128 != 0:
            raise ValueError(
                "Mxfp4B12xMoEMethod requires hidden_size and "
                "intermediate_size_per_partition to be multiples of 128 "
                f"(got hidden={hidden_size}, "
                f"intermediate={intermediate_size_per_partition})."
            )
        # Keep checkpoint scales in native E8M0: b12x consumes those bytes as-is.
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            fp4_scale_dtype=torch.float8_e8m0fnu,
            **extra_weight_attrs,
        )

    def create_moe_runner(self, layer: Module, moe_runner_config) -> None:
        from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        self.moe_runner_config = moe_runner_config
        self._swiglu_limit = getattr(moe_runner_config, "swiglu_limit", None)
        # -1 expert sentinels exist only under expert parallelism; resolve once
        # so the per-forward path can skip the guard entirely at EP=1.
        from sglang.srt.runtime_context import get_parallel

        self._ep_guard = get_parallel().moe_ep_size > 1

        # Register the fused func at runner construction so the FusedOpPool
        # lookup at `MoeRunner.__init__` finds it.
        import sglang.srt.layers.moe.moe_runner.b12x  # noqa: F401

        self.runner = MoeRunner(MoeRunnerBackend.B12X, moe_runner_config)

    def process_weights_after_loading(self, layer: Module) -> None:
        # Preserve the base FP4 post-load handling.
        self._fp8.process_weights_after_loading(layer)
        if getattr(layer, "_mega_moe_weights_built", False):
            return

        for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
            scale = getattr(layer, name)
            if scale.dtype != torch.float8_e8m0fnu:
                raise TypeError(
                    f"{name} must stay native E8M0: it is a hard input contract "
                    f"of source_format='fp4_e8m0_k32'. Got {scale.dtype}."
                )

        log_info_on_rank0(
            logger,
            f"Preparing DSv4 MXFP4 experts for b12x W4A16 "
            f"(layer: {self.prefix}, swiglu_limit={self._swiglu_limit})...",
        )

        device = layer.w13_weight.device
        num_experts, two_n, k_half = layer.w13_weight.shape
        hidden_size = k_half * 2
        intermediate = two_n // 2
        ones = torch.ones(num_experts, dtype=torch.float32, device=device)

        try:
            from sglang.srt.runtime_context import get_spec

            spec_tokens = int(get_spec().speculative_num_draft_tokens or 1)
        except Exception:
            spec_tokens = 1

        if _b12x_is_legacy():
            # 0.15.3: one call prepares the weights in place, and the scratch
            # plan is built straight from the shapes -- there is no separate
            # weight plan to share between layers.
            w13 = layer.w13_weight.data.view(torch.uint8)
            s13 = layer.w13_weight_scale_inv.data.view(torch.uint8)
            w2 = layer.w2_weight.data.view(torch.uint8)
            s2 = layer.w2_weight_scale_inv.data.view(torch.uint8)
            prepared = _legacy_prepare(w13=w13, s13=s13, w2=w2, s2=s2, ones=ones)
            counts = _core_token_counts(_b12x_max_tokens(), spec_tokens)
            plan, scratch = _legacy_plan(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate=intermediate,
                top_k=int(layer.top_k),
                device=device,
                swiglu_limit=self._swiglu_limit,
                counts=counts,
            )
            self._launch = _legacy_launcher(
                plan=plan,
                scratch=scratch,
                prepared=prepared,
                w13=w13,
                s13=s13,
                w2=w2,
                s2=s2,
                ones=ones,
                swiglu_limit=self._swiglu_limit,
            )
            # reuse_input_storage=True: the prepared package aliases these
            # tensors, so they must stay reachable. Keep the Parameters as they
            # are and hold our own references.
            self._legacy_keepalive = (prepared, w13, s13, w2, s2, ones, plan, scratch)
            # Same reason as the 1.2.2 path: b12x will not compile while a CUDA
            # graph is being captured, so every shape a graph can present has to
            # have run once already.
            if not _B12X_WARMED:
                log_info_on_rank0(
                    logger,
                    f"Compiling b12x 0.15.3 W4A16 kernels for {len(counts)} "
                    f"token counts ({counts[0]}..{counts[-1]})...",
                )
                _warmup_legacy(
                    launch=self._launch,
                    top_k=int(layer.top_k),
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    device=device,
                    counts=counts,
                )
            layer._dsv4_mxfp4_backend = "b12x_sm12x_015"
            return

        from b12x.moe import fused_moe

        weight_plan = _get_weight_plan(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate=intermediate,
        )
        self._experts = fused_moe.prepare_weights(
            plan=weight_plan,
            params_dtype=torch.bfloat16,
            w1_fp4=layer.w13_weight.data.view(torch.uint8),
            w1_blockscale=layer.w13_weight_scale_inv.data.view(torch.uint8),
            w1_global_scale=ones,
            w2_fp4=layer.w2_weight.data.view(torch.uint8),
            w2_blockscale=layer.w2_weight_scale_inv.data.view(torch.uint8),
            w2_global_scale=ones,
            a1_gscale=ones,
            a2_gscale=ones,
        )
        self._plan, self._scratch = _get_scratch(
            weight_plan=weight_plan,
            top_k=int(layer.top_k),
            device=device,
            swiglu_limit=self._swiglu_limit,
            tokens_per_seq=spec_tokens,
            experts=self._experts,
            num_experts=num_experts,
            hidden_size=hidden_size,
        )

        # b12x owns the runtime weights from here on. Whether it aliased the
        # checkpoint storage or copied it is its choice; either way nothing else
        # reads these Parameters again.
        def _release(param: torch.Tensor) -> Parameter:
            return Parameter(
                torch.empty(0, dtype=param.dtype, device=param.device),
                requires_grad=False,
            )

        layer.w13_weight = _release(layer.w13_weight)
        layer.w2_weight = _release(layer.w2_weight)
        layer.w13_weight_scale_inv = _release(layer.w13_weight_scale_inv)
        layer.w2_weight_scale_inv = _release(layer.w2_weight_scale_inv)
        torch.cuda.empty_cache()

        layer._dsv4_mxfp4_backend = "b12x_sm12x"

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.b12x import B12xMoeQuantInfo

        quant_info = B12xMoeQuantInfo(
            experts=self._experts,
            plan=self._plan,
            scratch=self._scratch,
            launch=self._launch,
            ep_guard=self._ep_guard,
        )
        return self.runner.run(dispatch_output, quant_info)

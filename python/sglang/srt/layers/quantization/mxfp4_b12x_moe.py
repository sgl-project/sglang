"""DeepSeek-V4 MXFP4 expert backend backed by the b12x fused MoE.

Same checkpoint as :mod:`mxfp4_flashinfer_cutlass_moe`, different runtime: one
fused kernel per expert layer instead of seven.

The quant mode is ``SGLANG_B12X_MOE_QUANT_MODE``, passed to b12x verbatim. It
defaults to ``w4a8_mx`` (activations quantized to MXFP8-E4M3 in-kernel), the
recipe the official vLLM DGX Spark image runs and where all of b12x's GB10 MoE
tuning lives; ``w4a16`` keeps bf16 activations on a load-time frozen plan.

This targets exactly one generation of the standalone ``b12x`` package: the
op-registry API (``b12x.moe.fused_moe``, ``Caps -> plan() -> bind() -> run()``)
that lives on ``local-inference-lab/b12x`` master. The pin is the tree the
official vLLM DGX Spark image ships (nightly-20260815, ``b4d6c7593``) plus the
two upstream fixes that path needs -- ``5c8318009`` (route-packed W4A16 top-k
scaling) and ``85d3681b`` (W4A16 route histograms preallocated for CUDA graph
capture); vLLM does not stage either path, so its image runs without them.
Three things depend on that choice:

* ``source_format="fp4_e8m0_k32"`` consumes the checkpoint's E8M0 K/32 block
  scales byte for byte, so there is no scale relayout to get wrong.
* ``swiglu_limit`` is honoured. DSv4-Flash asks for a SwiGLU clamp at 10.0 and
  it binds on real data (gate magnitudes reach 10.1, and half the layers sit
  within 7% of the limit), so dropping it is a correctness change, not a
  rounding one.
* Kernels are warmed before CUDA graph capture: an unwarmed token count under
  capture raises ``RuntimeError`` at launch (``_require_cached``), so every
  token count a captured graph can present is compiled eagerly first (see
  :func:`_core_token_counts`).

Install the pinned tree::

    pip install --no-deps "b12x @ https://github.com/local-inference-lab/b12x/archive/85d3681bb2c3749e4e94e121d5d829c7ec6f0b9f.tar.gz"

(``--no-deps``: b12x's metadata pulls a newer torch and breaks the rest of the
image.) Earlier generations -- 0.15.x and the PyPI 1.2.x wheels -- keep the old
``b12x.integration.tp_moe`` API and different W4A16 kernels; they are rejected
at startup rather than silently producing different numerics -- see
:func:`sglang.srt.utils.b12x.require_aligned_generation`. The last 0.15.3-based
revision of this file is the parent of the commit that introduced this
docstring.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module

from sglang.srt.environ import envs
from sglang.srt.utils import log_info_on_rank0
from sglang.srt.utils.b12x import install_compile_cache_dir, require_aligned_generation
from sglang.srt.utils.common import is_sm120_supported

# Suppress TRT-LLM CUTLASS trace logs without overriding user configuration.
os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")

install_compile_cache_dir()

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput


def _b12x_forward_contract():
    """The scheduler-owned, resolved row domain for this backend."""
    from sglang.srt.runtime_context import get_moe_forward_contract

    return get_moe_forward_contract()


def _core_token_counts(max_tokens: int, tokens_per_seq: int) -> tuple[int, ...]:
    """Token counts to plan capacity for, and to warm kernels at.

    Under CUDA graph capture b12x refuses to compile (``_require_cached``) and
    raises at launch for any token count that was not run once already. Decode
    graphs are captured at the configured batch sizes times the speculative
    window; prefill is chunked, so a power-of-two ladder covers it.

    b12x buckets W4A16 token capacities to powers of two internally, so warming
    every count here resolves to far fewer distinct kernels; the same ladder
    sizes the scratch arena.
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


# Module-level on purpose: this latch mirrors b12x's own process-global
# compiled-kernel cache, which reset_context() cannot reset -- re-warming after
# a context reset would be wasted work, not a correctness need.
_B12X_WARMED = False


def _warmup(*, launch, top_k, num_experts, hidden_size, device, counts):
    """Compile the kernels for every count before any CUDA graph capture."""
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


def _prepare_weights(
    *, w13, s13, w2, s2, ones, num_experts, hidden_size, intermediate, quant_mode
):
    """Weight plan + prep. Repacks in place, so the caller's tensors stay live."""
    from b12x.moe import fused_moe

    weight_plan = fused_moe.plan_weights(
        quant_modes=quant_mode,
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate,
        # sglang stores fused W13 as [up; gate] (load_up_proj_weight_first);
        # vLLM stores [gate; up] and declares "w31" -- do not copy that value.
        w13_layout="up_gate",
    )
    return fused_moe.prepare_weights(
        plan=weight_plan,
        params_dtype=torch.bfloat16,
        w1_fp4=w13,
        w1_blockscale=s13,
        w1_global_scale=ones,
        a1_gscale=ones,
        w2_fp4=w2,
        w2_blockscale=s2,
        w2_global_scale=ones,
        a2_gscale=ones,
    )


def _plan_scratch(*, experts, top_k, device, swiglu_limit, counts):
    """One frozen scratch plan + arena shared by every expert layer.

    Every expert layer presents identical caps, so one plan and one arena
    serve them all (per-layer arenas once wasted ~12 GB here; bind is a pure
    view mapping over the arena, so sharing is allocation- and state-free).
    The weight-plan identity check in bind passes across layers because
    MoEWeightPreparationPlan is a tensor-free frozen dataclass with value
    equality.
    """
    from b12x.moe import fused_moe

    from sglang.srt.runtime_context import get_resources

    fingerprint = (
        int(top_k),
        tuple(counts),
        None if swiglu_limit is None else float(swiglu_limit),
        int(experts.num_experts),
        int(experts.hidden_size),
        int(experts.intermediate_size),
    )
    buffers = get_resources().buffers
    key = f"b12x:moe_plan:{device}"
    cached = buffers.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1], cached[2]

    caps = fused_moe.Caps(
        max_tokens=max(counts),
        num_topk=top_k,
        device=device,
        weight_plan=experts.plan,
        core_token_counts=tuple(counts),
        route_num_experts=0,
        route_logits_dtype=None,
        quant_mode="w4a16",
        apply_router_weight_on_input=False,
        swiglu_limit=swiglu_limit,
        frozen=True,
    )
    plan = fused_moe.plan(caps)
    specs = plan.scratch_specs()
    if len(specs) != 1 or specs[0].dtype != torch.uint8:
        raise RuntimeError(f"unexpected b12x scratch specs: {specs}")
    scratch = torch.empty(int(specs[0].shape[0]), dtype=torch.uint8, device=device)
    buffers[key] = (fingerprint, plan, scratch)
    return plan, scratch


def _make_launcher(*, plan, scratch, experts, max_tokens):
    """W4A16 bind-and-run closure over the load-time frozen plan."""
    from b12x.moe import fused_moe

    def launch(a, topk_ids, topk_weights, out):
        live_tokens = max(1, int(a.shape[0]))
        if live_tokens > max_tokens:
            raise RuntimeError(
                "b12x MoE row-count invariant violated before frozen-plan bind: "
                f"live_tokens={live_tokens}, derived_capacity={max_tokens}"
            )
        fused_moe.run(
            binding=plan.bind(
                scratch=scratch,
                a=a,
                experts=experts,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output=out,
                # W4A16 contract, same as vLLM: activations are bf16 and every
                # global scale is the unit filler prepare_weights built.
                input_scales_static=True,
                unit_scale_contract=True,
            )
        )

    return launch


def _plan_a8_arena(*, experts, top_k, device, swiglu_limit, counts, quant_mode):
    """Shared arena for the W4A8 per-forward-planned path.

    W4A8 selects its implementation (micro vs dynamic) and tile regime from
    the live token count, so the plan is rebuilt per call, vLLM-style --
    b12x's planning is host-only, allocation- and compile-free for this
    recipe, and under full-graph capture it runs once per bucket at capture
    time. Only the arena is fixed: sized here over the whole count ladder
    (required_nbytes covers every bucketed count including the micro/dynamic
    cutover), shared by every layer and every per-call plan.
    """
    from b12x.moe import fused_moe

    from sglang.srt.runtime_context import get_resources

    fingerprint = (
        quant_mode,
        int(top_k),
        tuple(counts),
        None if swiglu_limit is None else float(swiglu_limit),
        int(experts.num_experts),
        int(experts.hidden_size),
        int(experts.intermediate_size),
    )
    buffers = get_resources().buffers
    key = f"b12x:moe_a8_arena:{device}"
    cached = buffers.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1], cached[2]
    # W4A8 chooses its workspace plan from the exact live token count.
    # Size across every count admitted by the resolved runtime contract, not
    # just the sparse decode/power-of-two warmup ladder.
    arena_counts = tuple(range(1, max(counts) + 1))
    caps = fused_moe.Caps(
        max_tokens=max(counts),
        num_topk=top_k,
        device=device,
        weight_plan=experts.plan,
        core_token_counts=arena_counts,
        route_num_experts=0,
        route_logits_dtype=None,
        quant_mode=quant_mode,
        apply_router_weight_on_input=False,
        swiglu_limit=swiglu_limit,
        frozen=True,
    )
    required_bytes = int(fused_moe.required_nbytes(caps))
    arena = torch.empty(required_bytes, dtype=torch.uint8, device=device)
    buffers[key] = (fingerprint, arena, required_bytes)
    return arena, required_bytes


def _make_launcher_a8(
    *, arena, arena_bytes, experts, top_k, device, swiglu_limit, quant_mode, max_tokens
):
    """W4A8 bind-and-run closure; plans at the live token count per call."""
    from b12x.moe import fused_moe

    def launch(a, topk_ids, topk_weights, out):
        m = max(1, int(a.shape[0]))
        if m > max_tokens:
            raise RuntimeError(
                "b12x MoE row-count invariant violated before arena bind: "
                f"live_tokens={m}, derived_capacity={max_tokens}. The resolved "
                "scheduler/model contract changed after runner initialization."
            )
        caps = fused_moe.Caps(
            max_tokens=m,
            num_topk=top_k,
            device=device,
            weight_plan=experts.plan,
            core_token_counts=(m,),
            route_num_experts=0,
            route_logits_dtype=None,
            quant_mode=quant_mode,
            apply_router_weight_on_input=False,
            swiglu_limit=swiglu_limit,
            frozen=True,
        )
        live_required_bytes = int(fused_moe.required_nbytes(caps))
        if live_required_bytes > arena_bytes:
            raise RuntimeError(
                "b12x workspace invariant violated before arena bind: "
                f"live_tokens={m}, required_bytes={live_required_bytes}, "
                f"reserved_bytes={arena_bytes}"
            )
        plan = fused_moe.plan(caps)
        fused_moe.run(
            binding=plan.bind(
                scratch=arena,
                a=a,
                experts=experts,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output=out,
                input_scales_static=True,
            )
        )

    return launch


class Mxfp4B12xMoEMethod:
    """b12x MXFP4 MoE: one fused kernel per expert layer."""

    def __init__(self, fp8_method, prefix: str):
        require_aligned_generation()
        if not is_sm120_supported():
            raise RuntimeError("Mxfp4B12xMoEMethod requires SM120 or SM121.")
        self._fp8 = fp8_method
        self.prefix = prefix
        self._swiglu_limit: float | None = None
        # The whole call is a prepared closure; see _make_launcher.
        self._launch: Any = None
        self._keepalive: Any = None
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

        # Passed straight to b12x. "w4a16" is the frozen-plan path; every other
        # mode quantizes activations in-kernel and plans per call.
        quant_mode = envs.SGLANG_B12X_MOE_QUANT_MODE.get()
        log_info_on_rank0(
            logger,
            f"Preparing DSv4 MXFP4 experts for b12x {quant_mode} "
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

        w13 = layer.w13_weight.data.view(torch.uint8)
        s13 = layer.w13_weight_scale_inv.data.view(torch.uint8)
        w2 = layer.w2_weight.data.view(torch.uint8)
        s2 = layer.w2_weight_scale_inv.data.view(torch.uint8)
        experts = _prepare_weights(
            w13=w13,
            s13=s13,
            w2=w2,
            s2=s2,
            ones=ones,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate=intermediate,
            quant_mode=quant_mode,
        )
        contract = _b12x_forward_contract()
        max_tokens = contract.max_rows
        counts = _core_token_counts(max_tokens, spec_tokens)
        if quant_mode != "w4a16":
            arena, workspace_bytes = _plan_a8_arena(
                experts=experts,
                top_k=int(layer.top_k),
                device=device,
                swiglu_limit=self._swiglu_limit,
                counts=counts,
                quant_mode=quant_mode,
            )
            self._launch = _make_launcher_a8(
                arena=arena,
                arena_bytes=workspace_bytes,
                experts=experts,
                top_k=int(layer.top_k),
                device=device,
                swiglu_limit=self._swiglu_limit,
                quant_mode=quant_mode,
                max_tokens=max_tokens,
            )
            plan_or_arena = arena
        else:
            plan, scratch = _plan_scratch(
                experts=experts,
                top_k=int(layer.top_k),
                device=device,
                swiglu_limit=self._swiglu_limit,
                counts=counts,
            )
            self._launch = _make_launcher(
                plan=plan,
                scratch=scratch,
                experts=experts,
                max_tokens=max_tokens,
            )
            plan_or_arena = (plan, scratch)
        # The prepared weights alias the checkpoint storage (REUSE_SOURCE for
        # 128-aligned fp4_e8m0_k32; w4a8_mx repacks likewise discard the
        # source Parameters), so these tensors must stay reachable. Keep the
        # Parameters as they are and hold our own references.
        self._keepalive = (experts, w13, s13, w2, s2, ones, plan_or_arena)
        # b12x will not compile while a CUDA graph is being captured, so every
        # shape a graph can present has to have run once already.
        if not _B12X_WARMED and envs.SGLANG_B12X_ENABLE_WARMUP.get():
            log_info_on_rank0(
                logger,
                f"Compiling b12x {quant_mode} kernels for {len(counts)} "
                f"token counts ({counts[0]}..{counts[-1]})...",
            )
            _warmup(
                launch=self._launch,
                top_k=int(layer.top_k),
                num_experts=num_experts,
                hidden_size=hidden_size,
                device=device,
                counts=counts,
            )
        layer._dsv4_mxfp4_backend = "b12x_sm12x"

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.b12x import B12xMoeQuantInfo

        quant_info = B12xMoeQuantInfo(
            launch=self._launch,
            ep_guard=self._ep_guard,
        )
        return self.runner.run(dispatch_output, quant_info)

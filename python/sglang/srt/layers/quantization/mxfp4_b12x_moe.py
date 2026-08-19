"""DeepSeek-V4 MXFP4 expert backend backed by the b12x 0.15.3 W4A16 fused MoE.

Same checkpoint as :mod:`mxfp4_flashinfer_cutlass_moe`, different runtime: one
fused kernel per expert layer instead of seven, and bf16 activations instead of
MXFP8.

This targets exactly one generation of the standalone ``b12x`` package: the
0.15.3 API (``b12x.integration.tp_moe``). Three things depend on that choice:

* ``source_format="fp4_e8m0_k32"`` consumes the checkpoint's E8M0 K/32 block
  scales byte for byte, so there is no scale relayout to get wrong.
* ``swiglu_limit`` is honoured. DSv4-Flash asks for a SwiGLU clamp at 10.0 and
  it binds on real data (gate magnitudes reach 10.1, and half the layers sit
  within 7% of the limit), so dropping it is a correctness change, not a
  rounding one.
* Kernels are warmed before CUDA graph capture: b12x refuses to compile while
  a graph is being captured and falls back to a pinned tile config that does
  not fit every block size, so every token count a captured graph can present
  is compiled eagerly first (see :func:`_core_token_counts`).

0.15.3 was never published to PyPI. Install it from the source commit whose
MoE surface matches the released wheel::

    pip install --no-deps "b12x @ https://github.com/local-inference-lab/b12x/archive/7dc6fb8fcc6446ea093537d1657df81985fa5f43.tar.gz"

(``--no-deps``: b12x's metadata pulls a newer torch and breaks the rest of the
image.) Later b12x generations changed the W4A16 kernels (0.20.0) and then the
API (1.2.x); anything but 0.15.3 is rejected at startup rather than silently
producing different numerics -- see :func:`_check_b12x_version`.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module

from sglang.srt.environ import envs
from sglang.srt.utils import log_info_on_rank0
from sglang.srt.utils.common import is_sm120_supported

# Suppress TRT-LLM CUTLASS trace logs without overriding user configuration.
os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

_B12X_INSTALL_HINT = (
    "pip install --no-deps 'b12x @ https://github.com/local-inference-lab/"
    "b12x/archive/7dc6fb8fcc6446ea093537d1657df81985fa5f43.tar.gz'"
)


def _select_b12x_distribution() -> None:
    """Put a local b12x tree on sys.path ahead of whatever pip installed.

    For deployments that carry the 0.15.3 tree as a directory (e.g. extracted
    from the dspark reference image) instead of pip-installing the pinned
    source commit. Unset (the default) is a no-op.
    """
    path = envs.SGLANG_B12X_PATH.get()
    if not path:
        return
    if not os.path.isdir(os.path.join(path, "b12x")):
        raise FileNotFoundError(f"SGLANG_B12X_PATH={path!r} has no b12x package.")
    loaded = sys.modules.get("b12x")
    if loaded is not None:
        # Idempotent: this runs once per expert layer. Only a b12x already
        # imported from somewhere else is a problem, and it is fatal -- the
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
        import b12x.integration.tp_moe  # noqa: F401

        return True
    except ImportError:
        return False


def _check_b12x_version() -> None:
    """This backend is validated against b12x 0.15.3, and only 0.15.3.

    Later generations rewrote the W4A16 kernels behind the same
    ``b12x.integration`` API (0.20.0) and then replaced the API (1.2.x), so a
    version mismatch must fail here rather than surface as different numerics.
    A source tree without dist-info cannot be checked and is trusted to be the
    pinned 0.15.3.
    """
    import importlib.metadata

    try:
        version = importlib.metadata.version("b12x")
    except importlib.metadata.PackageNotFoundError:
        return
    if version != "0.15.3":
        import b12x

        raise RuntimeError(
            f"b12x at {b12x.__file__} is version {version}; this backend is "
            "validated against 0.15.3 only. Install the pinned source commit: "
            f"{_B12X_INSTALL_HINT}"
        )


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

    Each count compiles its own kernel (the startup log line counts them), and
    the same ladder sizes the scratch arena.
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


def _prepare_weights(*, w13, s13, w2, s2, ones):
    """Weight prep. Repacks in place, so the caller's tensors stay live."""
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


def _plan_scratch(
    *, num_experts, hidden_size, intermediate, top_k, device, swiglu_limit, counts
):
    """Frozen scratch plan. Returns (plan, scratch)."""
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


def _make_launcher(*, plan, scratch, prepared, w13, s13, w2, s2, ones, swiglu_limit):
    """Bind-and-run closure, matching vLLM's call sequence."""
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


class Mxfp4B12xMoEMethod:
    """b12x MXFP4 MoE: one fused W4A16 kernel per expert layer."""

    def __init__(self, fp8_method, prefix: str):
        if not is_b12x_available():
            raise RuntimeError(
                "The b12x MoE backend needs the 0.15.3-generation b12x "
                f"package: {_B12X_INSTALL_HINT}"
            )
        _check_b12x_version()
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

        # One call prepares the weights in place, and the scratch plan is
        # built straight from the shapes.
        w13 = layer.w13_weight.data.view(torch.uint8)
        s13 = layer.w13_weight_scale_inv.data.view(torch.uint8)
        w2 = layer.w2_weight.data.view(torch.uint8)
        s2 = layer.w2_weight_scale_inv.data.view(torch.uint8)
        prepared = _prepare_weights(w13=w13, s13=s13, w2=w2, s2=s2, ones=ones)
        counts = _core_token_counts(_b12x_max_tokens(), spec_tokens)
        plan, scratch = _plan_scratch(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate=intermediate,
            top_k=int(layer.top_k),
            device=device,
            swiglu_limit=self._swiglu_limit,
            counts=counts,
        )
        self._launch = _make_launcher(
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
        self._keepalive = (prepared, w13, s13, w2, s2, ones, plan, scratch)
        # b12x will not compile while a CUDA graph is being captured, so every
        # shape a graph can present has to have run once already.
        if not _B12X_WARMED:
            log_info_on_rank0(
                logger,
                f"Compiling b12x W4A16 kernels for {len(counts)} "
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

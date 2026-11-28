"""What paged experts needs from a fused-MoE runner backend, one ``RunnerContract`` per backend.

The executor runs a step in waves (one wave when its experts fit the K slots). Two kinds of
runner:

- A runner that can return every routed (token, expert) output already multiplied by its router
  weight (``no_combine``): the contract runs each wave over only the entries whose expert it
  holds, writes them into one merged buffer, and performs the top-k sum the runner would have
  performed itself, so a multi-wave step is exact.
- A runner that sums the top-k outputs itself (``combines``, e.g. FlashInfer's CUTLASS MoE): a
  wave is the runner's local expert range ``[0, K)``; the entries of other experts get id K and
  are dropped before the grouped GEMM, as expert parallelism drops another rank's experts, and
  the executor adds the waves' outputs in FP32. A step in one wave (every decode step the device
  plans) is exact; in a multi-wave step each wave's output is rounded to the runner's output
  dtype before the FP32 sum, so it can differ from the unpaged layer by one rounding per wave.

The contract follows from the runner the base method builds (``contract_for``); supporting a
new backend means adding a contract here, or registering one from the backend's own module
(``register_runner_contract``).
"""

from __future__ import annotations

import contextlib
import weakref
from dataclasses import replace
from typing import Dict

import torch
import torch.nn.functional as F

#: The most routed entries one compacted runner call computes: it bounds the call's working
#: memory (``RunnerContract.prefill_bytes``) whatever a wave routes, e.g. a long prompt whose
#: tokens all go to the experts one wave holds.
COMPACT_ROWS = 4096


def run_on_slots(base_method, layer, dispatch_output, slots, outside: int = 0):
    """Run the base method over the K-slot table: routed entry i goes to GPU slot ``slots[i]``;
    entries at -1 get expert id ``outside`` and router weight 0, so they contribute nothing."""
    masked = slots < 0
    topk_output = dispatch_output.topk_output
    wave_output = dispatch_output._replace(
        topk_output=topk_output._replace(
            topk_ids=slots.masked_fill(masked, outside),
            topk_weights=topk_output.topk_weights.masked_fill(masked, 0),
        ),
    )
    return base_method.apply(layer=layer, dispatch_output=wave_output).hidden_states


def runner_config(moe_runner_config, num_slots: int):
    """The base runner's config: K local experts, router-weighted per-expert outputs."""
    return replace(
        moe_runner_config,
        num_local_experts=num_slots,
        inplace=False,
        no_combine=True,
        no_combine_keep_router_weight=True,
    )


class RunnerContract:
    #: Whether the runner reads a layout the base method's post-load step repacks the
    #: experts into; the step then runs over the host store K experts at a time.
    repacks: bool = False
    #: Whether the runner sums the top-k outputs itself: a wave's output is then the
    #: ``[tokens, hidden]`` sum over the wave's entries, and the executor adds the waves.
    combines: bool = False

    def runner_config(self, moe_runner_config, num_slots: int):
        """The config to build the base runner with for ``num_slots`` slots, when it is not
        the shared one (``runner_config``: per-expert outputs); None keeps the shared one."""
        return None

    def check_layer(self, layer, expert_format) -> None:
        """Raise if this runner is not known to serve the layer (its expert format, its
        tensors) paged."""

    def validate(self, config) -> None:
        """Raise, before the checkpoint loads, on a server configuration (``get_exec()``) this
        contract cannot serve (beyond the shared checks), e.g. CUDA graphs on a backend
        without them."""

    def resize(self, layer, num_slots: int) -> None:
        """Rebuild what the runner keeps per local expert outside the paged tensors, for
        ``num_slots`` slots: the layer had another K when its weights were processed."""

    def run_wave(
        self, base_method, layer, dispatch_output, slots, merged, num_entries=None
    ):
        """Compute the routed entries of one wave (``slots`` >= 0) into ``merged``, the step's
        ``[tokens, top_k, hidden]`` entries (None before its first wave), and return it.
        ``num_entries``, if the host knows it, is how many routed entries the wave has.
        A combining contract returns the wave's own ``[tokens, hidden]`` output instead
        (``merged`` is always None); the executor accumulates."""
        raise NotImplementedError

    def finish(self, merged, hidden_states, routed_scaling_factor: float):
        """The layer output: the top-k sum of the merged entries, trimmed to the input's width
        (a runner may pad the hidden dimension, e.g. MXFP4 Marlin)."""
        out = hidden_states.new_empty(merged.shape[0], merged.shape[-1])
        self._sum_top_k(merged, out, routed_scaling_factor)
        return out[..., : hidden_states.shape[-1]]

    def _sum_top_k(self, entries, out, routed_scaling_factor: float) -> None:
        """Sum ``[tokens, top_k, hidden]`` into ``out`` exactly as the runner would."""
        raise NotImplementedError

    def compacts(self, base_method, layer, dispatch_output) -> bool:
        """Whether ``run_wave`` runs only the wave's entries (in calls of at most
        ``COMPACT_ROWS``), so a step of any size may go through it."""
        return False

    def prefill_bytes(
        self, hidden_size: int, intermediate_size: int, tokens: int, top_k: int, dtype
    ) -> int:
        """GPU memory a step of ``tokens`` tokens takes in a layer of this size beyond the
        runner's own use for the same step unpaged: the merged ``[tokens, top_k, hidden]``
        entries and the largest compacted call (its gathered inputs, outputs and
        intermediates)."""
        hidden = _padded_hidden(hidden_size)
        inter = intermediate_size
        size = torch.empty((), dtype=dtype).element_size()
        merged = tokens * top_k * hidden * size
        rows = COMPACT_ROWS + _pad_rows(COMPACT_ROWS)
        call = rows * ((4 * hidden + 3 * inter) * size + hidden + inter)
        return merged + call


class CombiningContract(RunnerContract):
    """A runner that sums the top-k outputs itself. A wave sends its entries to their slots
    ``[0, K)`` and every other entry to id K, outside the runner's K local experts, which the
    runner drops before its grouped GEMM (as expert parallelism drops the experts of other
    ranks). The routed scaling factor is the runner's, as unpaged: it is linear, so the waves'
    sum carries it once.

    The base method picks the runner backend while it builds the runner, so the contract is
    known only afterwards and the runner is first built with the shared config. A combining
    runner that never reads ``no_combine`` (FlashInfer's CUTLASS MoE) serves with it as built;
    one that reads or rejects it (trtllm-gen asserts it is off) returns its config from
    ``combining_config`` in ``runner_config``, and is built once more with it.

    A runner without a local expert range (``drops_outside = False``) computes every id it is
    given: the entries of other experts then go to slot 0 with weight 0, so they add nothing,
    but every wave computes every entry of the step. Such a runner's tables are as wide as its
    expert count, so its contract's ``runner_config`` builds it with ``num_experts`` = K."""

    combines = True
    #: Whether the runner drops ids outside its local experts ``[0, K)`` before its GEMMs.
    drops_outside: bool = True

    def validate(self, config) -> None:
        """A multi-wave step sums each token's experts in groups that depend on which experts
        the previous steps left resident, so the same request can round differently."""
        if config.deterministic.enable_deterministic_inference:
            raise RuntimeError(
                "Paged experts: --enable-deterministic-inference needs a MoE runner that "
                "returns each expert's output (e.g. --moe-runner-backend triton); this one "
                "sums the top-k outputs itself, in an order that depends on which experts "
                "are resident"
            )

    @staticmethod
    def combining_config(moe_runner_config, num_slots: int):
        """The shared config with the runner's own combine on, for ``runner_config``."""
        return replace(
            moe_runner_config,
            num_local_experts=num_slots,
            inplace=False,
            no_combine=False,
            no_combine_keep_router_weight=False,
        )

    def run_wave(
        self, base_method, layer, dispatch_output, slots, merged, num_entries=None
    ):
        outside = layer.num_local_experts if self.drops_outside else 0
        return run_on_slots(base_method, layer, dispatch_output, slots, outside=outside)

    def finish(self, merged, hidden_states, routed_scaling_factor: float):
        """The layer output: the summed waves (FP32 when there were several) in the input's
        dtype, trimmed to its width."""
        return merged[..., : hidden_states.shape[-1]].to(hidden_states.dtype)

    def prefill_bytes(
        self, hidden_size: int, intermediate_size: int, tokens: int, top_k: int, dtype
    ) -> int:
        """Beyond the runner's own use for the step unpaged (each wave is one call over the
        whole step): the FP32 accumulator and the wave's output before it is added."""
        hidden = _padded_hidden(hidden_size)
        size = torch.empty((), dtype=dtype).element_size()
        return tokens * hidden * (4 + size)


class FlashInferCutlassContract(CombiningContract):
    """FlashInfer's CUTLASS MoE: its experts per rank are the rows of the weight tables (K),
    and ids outside ``[ep_rank * K, (ep_rank + 1) * K)`` are not computed. It always combines
    and reads none of the shared config's per-entry fields, so it is built once. Only unquantized
    experts without biases so far: the quantized formats' post-load steps repack for this
    runner, which has not been checked against the host store's rows, and the unquantized
    method does not pass expert biases to it. Its per-expert SwiGLU parameters (a clamped
    activation) are rebuilt for the final K."""

    def check_layer(self, layer, expert_format) -> None:
        from sglang.srt.layers.moe.paged_experts.formats import UnquantizedFormat

        if not isinstance(expert_format, UnquantizedFormat):
            raise RuntimeError(
                "Paged experts: the flashinfer_cutlass MoE runner is supported for "
                f"unquantized experts only, the layer has {type(expert_format).__name__}"
            )
        biases = [
            name
            for name in ("w13_weight_bias", "w2_weight_bias")
            if getattr(layer, name, None) is not None
        ]
        if biases:
            raise RuntimeError(
                "Paged experts: the flashinfer_cutlass MoE runner is supported for experts "
                f"without biases only, the layer has {', '.join(biases)}; use "
                "--moe-runner-backend triton"
            )

    def resize(self, layer, num_slots: int) -> None:
        if getattr(layer, "_cutlass_swiglu_params", None) is None:
            return
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            materialize_swiglu_params_for_cutlass,
        )

        layer._cutlass_swiglu_params = materialize_swiglu_params_for_cutlass(
            layer.moe_runner_config, num_slots, layer.w13_weight.device
        )


def _padded_hidden(hidden: int) -> int:
    """The widest hidden dimension a runner may pad its outputs to (MXFP4 Marlin: 256)."""
    return -(-hidden // 256) * 256


def _pad_rows(num: int) -> int:
    """Padding rows that bring ``num`` to the next of a few sizes per power of two, so the
    runner sees few distinct batch shapes."""
    step = max(16, 1 << max(0, num.bit_length() - 4))
    return -num % step


def _compact_calls(slots, num_entries):
    """The wave's routed entries (``slots`` >= 0, flattened), in calls of at most
    ``COMPACT_ROWS``; found without a sync when the host knows how many there are."""
    routed = (slots >= 0).flatten()
    entries = (
        routed.nonzero()
        if num_entries is None
        else routed.nonzero_static(size=num_entries)
    ).squeeze(1)
    return [part for part in entries.split(COMPACT_ROWS) if part.numel()]


def _run_compacted(base_method, layer, dispatch_output, slots, entries, merged):
    """Run ``entries`` of the step (flat indices into ``slots``), one per row with top-k 1,
    and write their outputs into ``merged`` (allocated, zeroed, at the runner's width on the
    first call). Padding rows go to slot 0 with weight 0 and are dropped."""
    hidden_states = dispatch_output.hidden_states
    num = entries.numel()
    pad = _pad_rows(num)
    tokens = F.pad(entries // slots.shape[1], (0, pad))
    topk = dispatch_output.topk_output
    compact = dispatch_output._replace(
        hidden_states=hidden_states[tokens],
        topk_output=topk._replace(
            topk_ids=F.pad(slots.flatten()[entries], (0, pad)).unsqueeze(1),
            topk_weights=F.pad(
                topk.topk_weights.flatten()[entries], (0, pad)
            ).unsqueeze(1),
            router_logits=(
                None if topk.router_logits is None else topk.router_logits[tokens]
            ),
        ),
    )
    out = base_method.apply(layer=layer, dispatch_output=compact).hidden_states
    if merged is None:  # at the runner's width, which may pad the hidden dimension
        merged = out.new_zeros(*slots.shape, out.shape[-1])
    merged.view(-1, merged.shape[-1])[entries] = out.view(-1, merged.shape[-1])[:num]
    return merged


def _unquantized_inputs(dispatch_output) -> bool:
    return (
        dispatch_output.hidden_states_scale is None
        and getattr(dispatch_output, "hidden_states_pre_quant", None) is None
    )


def _config_lookup(num_tokens: int, top_k: int):
    """The triton runner's kernel configs, looked up as for the step's full call."""
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils import (
            override_config_lookup,
        )
    except ImportError:  # no triton: no triton configs to keep
        return contextlib.nullcontext()
    return override_config_lookup(num_tokens, top_k)


def _row_local_activations(info) -> bool:
    """Whether the runner quantizes each row of both GEMMs' inputs on its own (no
    quantization, static scales, per-token or per-group scales): only then does a row's
    output not depend on which other rows share its call. A dynamic per-tensor scale (an
    input or intermediate scale left None) is a max over all rows."""
    if info is None or getattr(info, "use_mxfp8", False):
        return False
    if not (info.use_fp8_w8a8 or info.use_int8_w8a8):
        return True
    return (
        info.block_shape is not None
        or info.per_channel_quant
        or (info.a13_scale is not None and info.a2_scale is not None)
    )


def _triton_quant_info(base_method, layer):
    """The triton quant info the base method (or its scheme, or the scheme's kernel) builds
    for ``layer``; None if none of them exposes it."""
    scheme = getattr(layer, "scheme", None)
    for owner in (base_method, scheme, getattr(scheme, "kernel", None)):
        get = getattr(owner, "get_triton_quant_info", None)
        if get is not None:
            return get(layer)
    return None


class TritonContract(RunnerContract):
    def __init__(self):
        # Per layer: whether a compacted call computes exactly what the full call does.
        self._exact = weakref.WeakKeyDictionary()

    def compacts(self, base_method, layer, dispatch_output) -> bool:
        if not _unquantized_inputs(dispatch_output):
            return False
        exact = self._exact.get(layer)
        if exact is None:
            exact = _row_local_activations(_triton_quant_info(base_method, layer))
            self._exact[layer] = exact
        return exact

    def run_wave(
        self, base_method, layer, dispatch_output, slots, merged, num_entries=None
    ):
        if self.compacts(base_method, layer, dispatch_output):
            # Each row's output is what the step's full call computes for it, with the
            # kernel configs that call would use; only the rows it holds are computed.
            with _config_lookup(*slots.shape):
                for entries in _compact_calls(slots, num_entries):
                    merged = _run_compacted(
                        base_method, layer, dispatch_output, slots, entries, merged
                    )
            return merged
        config = runner_of(base_method, layer).config
        skips = (
            config.num_experts is None or config.num_experts != config.num_local_experts
        )
        partial = run_on_slots(
            base_method, layer, dispatch_output, slots, outside=-1 if skips else 0
        )
        if merged is None:
            return partial
        outside = (slots < 0).unsqueeze(-1)
        return torch.where(outside, merged, partial, out=merged)

    def _sum_top_k(self, entries, out, routed_scaling_factor: float) -> None:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            moe_combine_topk,
        )

        moe_combine_topk(
            intermediate_cache3=entries,
            out_hidden_states=out,
            routed_scaling_factor=routed_scaling_factor,
        )


class MarlinContract(RunnerContract):
    repacks = True

    def compacts(self, base_method, layer, dispatch_output) -> bool:
        return True

    def run_wave(
        self, base_method, layer, dispatch_output, slots, merged, num_entries=None
    ):
        # Only the wave's entries, compacted: the runner sees no masked rows.
        assert _unquantized_inputs(dispatch_output), (
            "Paged experts: compacted waves need unquantized MoE inputs"
        )
        for entries in _compact_calls(slots, num_entries):
            merged = _run_compacted(
                base_method, layer, dispatch_output, slots, entries, merged
            )
        return merged

    def _sum_top_k(self, entries, out, routed_scaling_factor: float) -> None:
        # fused_marlin_moe's own reduce, vectorized: 6.5 us per layer with the general
        # moe_sum_reduce vs ~1.5-2 us here, at bs=8 decode ([8, 10, 2560]).
        if (
            entries.dtype == torch.bfloat16
            and entries.is_contiguous()
            and out.is_contiguous()
            and entries.shape[-1] % 8 == 0
            # moe_topk_sum's grid is (ceil(K / 8 / threads), M) and gridDim.y is capped at
            # 65,535: a prefill chunk past that stays on the general reduce.
            and entries.shape[0] <= 65535
        ):
            from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum

            moe_topk_sum(entries, out, routed_scaling_factor, threads=64)
            return
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            moe_sum_reduce,
        )

        moe_sum_reduce(entries, out, routed_scaling_factor)


TRITON = TritonContract()
MARLIN = MarlinContract()
FLASHINFER_CUTLASS = FlashInferCutlassContract()

# Contracts of backends whose runner lives elsewhere (e.g. a hardware backend's own module):
# registered when that module is imported, so the shared code carries none of their imports.
_REGISTERED: Dict[str, RunnerContract] = {}


def register_runner_contract(backend_name: str, contract: RunnerContract) -> None:
    """The paged-experts contract of a runner backend, by its ``MoeRunnerBackend`` value."""
    if backend_name in _REGISTERED:
        raise ValueError(
            f"Paged experts: a runner contract for {backend_name!r} is already registered"
        )
    _REGISTERED[backend_name] = contract


def runner_of(base_method, layer):
    """The runner the base method built: its own, or its scheme's (or the scheme kernel's)."""
    scheme = getattr(layer, "scheme", None)
    for owner in (base_method, scheme, getattr(scheme, "kernel", None)):
        runner = getattr(owner, "runner", None)
        if runner is not None:
            return runner
    return None


def check_runs_its_runner(base_method) -> None:
    """Raise if the base method's ``apply`` would not run its runner: on XPU some methods call
    sgl-kernel's fused MoE instead, which sums the top-k outputs over all experts (paged
    experts cannot split it into waves yet). The other methods always run their runner."""
    from sglang.srt.layers.moe.paged_experts.store import accelerator_type

    if accelerator_type() != "xpu":
        return
    from sglang.srt.layers.moe.utils import get_moe_runner_backend
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    fused = (
        "sgl-kernel's fused MoE, which sums the top-k outputs (paged experts cannot split "
        "it into waves yet)"
    )
    if isinstance(base_method, Mxfp4MoEMethod):
        raise RuntimeError(
            f"Paged experts: on XPU, the MXFP4 MoE method always runs {fused}"
        )
    if (
        isinstance(base_method, (UnquantizedFusedMoEMethod, Fp8MoEMethod))
        and not get_moe_runner_backend().is_triton()
    ):
        raise RuntimeError(
            f"Paged experts: on XPU, {type(base_method).__name__} runs {fused} unless the "
            "triton MoE runner is selected: pass --moe-runner-backend triton"
        )


def contract_for(runner) -> RunnerContract:
    """The contract of the runner the base method built for paged experts."""
    backend = getattr(runner, "runner_backend", None)
    if backend is not None and backend.is_triton():
        return TRITON
    if (
        backend is not None
        and backend.is_marlin()
        and not backend.is_experimental_sgl_marlin()
    ):
        return MARLIN
    if backend is not None and backend.is_flashinfer_cutlass():
        return FLASHINFER_CUTLASS
    contract = _REGISTERED.get(getattr(backend, "value", None))
    if contract is not None:
        return contract
    raise RuntimeError(
        "Paged experts: requires the triton, Marlin or flashinfer_cutlass MoE runner, the "
        f"layer's method uses {getattr(backend, 'value', None)}; select one with "
        "--moe-runner-backend if its quantization method supports it"
    )

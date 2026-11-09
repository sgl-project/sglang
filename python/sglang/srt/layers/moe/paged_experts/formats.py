"""The fused-MoE methods paged experts runs, one ``ExpertFormat`` each, and how their experts
reach the host store.

The expert tensors are found on the layer, as EPLB finds them: every parameter with one row
per local expert. The checkpoint loads them into the host store, and the ones still per-expert
after the base method's post-load step are paged. A format names only what that rule cannot
see: tensors its runner never reads (``unpaged``) and tensors created for other runners
(``freed``).

Whether the post-load step repacks the experts follows from the runner (``repacks`` of its
contract). A repack runs over the loaded experts K at a time (``PostLoad``), so it must treat
every expert on its own, which is checked; any other step must leave them as loaded.

Adding a quantization method means adding a subclass to ``FORMATS``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.moe.paged_experts.store import (
    ExpertLoads,
    HostExpertStore,
    rows_like,
    same_bytes,
)
from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert

# Prefixes of FusedMoE's per-expert parameters.
_EXPERT_PARAM_PREFIXES = ("w13_", "w2_")
# Attributes every module has; the rest of a layer's attributes are its state.
_MODULE_ATTRS = frozenset(vars(torch.nn.Module()))


class ExpertFormat(ABC):
    #: Shown when no format handles a layer's method.
    label: str = ""
    #: Per-expert tensors the runner never reads: loaded with the checkpoint, never paged.
    unpaged: Tuple[str, ...] = ()
    #: Per-expert tensors the base method creates for other runners: freed before loading.
    freed: Tuple[str, ...] = ()

    @classmethod
    @abstractmethod
    def supports(cls, base_method, scheme) -> bool:
        """Whether this format handles ``base_method``, whose per-layer ``scheme`` (None for
        methods without one) holds the layout for methods that delegate to it."""

    def runner_method(self, base_method):
        """What the executors run waves through: ``base_method``, whose ``apply`` must use the
        runner built from ``runners.runner_config``."""
        return base_method


def expert_params(layer, num_slots: int) -> Tuple[str, ...]:
    """The layer's per-expert tensors, as EPLB finds them; tensors sized for all experts
    (``_sglang_require_global_experts``) stay whole on the GPU."""
    return tuple(
        name
        for name, p in layer.named_parameters(recurse=False)
        if name.startswith(_EXPERT_PARAM_PREFIXES)
        and filter_moe_weight_param_global_expert(name, p, num_slots)
    )


class PostLoad:
    """The base method's post-load step on given experts' loaded rows: a repack runs on a
    layer of K slots, each time from the layer's state before the first run; any other step
    must leave the loaded rows as they are (``finish`` checks it) and runs once, on the
    slots."""

    def __init__(self, layer, base_method, loads: ExpertLoads, repacks: bool, unpaged):
        self.layer, self.base_method, self.loads = layer, base_method, loads
        self.num_slots, self.repacks, self.unpaged = loads.num_slots, repacks, unpaged
        self.rows_like = loads.rows_like  # the repack's output layout, once it ran
        self._templates = {name: getattr(layer, name) for name in loads.rows_like}
        # The step records state on the layer (e.g. that it has run); each run starts from
        # the state before the first.
        self._state = {k: v for k, v in vars(layer).items() if k not in _MODULE_ATTRS}

    def run(
        self, experts: Sequence[int], out: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """The paged rows of ``experts`` (at most K), in ``out`` if given."""
        if not self.repacks:
            return self.loads.rows(experts, out)
        layer = self.layer
        self._reset()  # also before the loads replay: a loader may read the layer's state
        # Rows past a group shorter than K repeat its experts, not what the base method
        # created (torch.empty, which a step that validates every row would reject) nor other
        # experts (whose loads may wait).
        rows = self.loads.rows(
            [experts[i % len(experts)] for i in range(self.num_slots)]
        )
        for name, template in self._templates.items():
            # Fresh parameters in the checkpoint layout: the step may replace or rewrite them.
            fresh = torch.nn.Parameter(template.data.clone(), requires_grad=False)
            fresh.__dict__.update(template.__dict__)
            fresh.data.copy_(rows[name])
            layer.register_parameter(name, fresh)
        self.base_method.process_weights_after_loading(layer)
        names = [
            n for n in expert_params(layer, self.num_slots) if n not in self.unpaged
        ]
        if not names:
            raise RuntimeError(
                f"Paged experts: {type(self.base_method).__name__} leaves no per-expert "
                "tensors after loading"
            )
        self.rows_like = rows_like(layer, names)
        given = {name: getattr(layer, name).data[: len(experts)] for name in names}
        if out is None:
            return given
        for name, rows_ in out.items():
            rows_.copy_(given[name])
        return out

    def mixing_experts(self) -> List[int]:
        """The experts ``check_mixing`` runs again: 1..K, for a repack of more than K."""
        if not self.repacks or self.loads.num_experts <= self.num_slots:
            return []
        return list(range(1, self.num_slots + 1))

    def check_mixing(self, store: HostExpertStore) -> None:
        """Experts 1..K again, in a chunk of their own: a step that mixes experts (e.g. a max
        over them) gives them other rows than their first chunk did."""
        experts = self.mixing_experts()
        if not experts:
            return
        for name, rows in self.run(experts).items():
            if not same_bytes(rows.cpu(), store.host[name][experts]):
                raise RuntimeError(
                    f"Paged experts: {type(self.base_method).__name__}'s post-load step "
                    f"does not treat each expert on its own ({name}), so it cannot run K "
                    "at a time"
                )

    def finish(self, store: HostExpertStore) -> None:
        """Fill the slots with experts 0..K-1. A step that does not repack runs now, once:
        paging copies host rows verbatim, so it must neither transform the loaded weights
        (e.g. a kernel-specific shuffle or row interleave) nor create others."""
        layer, base_method = self.layer, self.base_method
        for name, host in store.host.items():
            getattr(layer, name).data.copy_(host[: self.num_slots])
        if self.repacks:
            return
        base_method.process_weights_after_loading(layer)
        created = set(expert_params(layer, self.num_slots)) - set(store.host)
        if created:
            raise RuntimeError(
                f"Paged experts: {type(base_method).__name__} creates per-expert "
                f"{sorted(created)} after loading, which the host store does not hold"
            )
        for name, host in store.host.items():
            gpu = getattr(layer, name).data[0]
            if not torch.equal(gpu, host[0].to(gpu.device)):
                raise RuntimeError(
                    f"Paged experts: {type(base_method).__name__} transforms {name} after "
                    "loading, which the host store does not replicate"
                )

    def _reset(self) -> None:
        """The layer as before the first run: its state and the created parameters."""
        layer = self.layer
        for k in [
            k for k in vars(layer) if k not in _MODULE_ATTRS and k not in self._state
        ]:
            delattr(layer, k)
        for k, v in self._state.items():
            setattr(layer, k, v)
        for name, template in self._templates.items():
            layer.register_parameter(name, template)


def build_store(
    step: PostLoad,
    store: Optional[HostExpertStore] = None,
    first_done: bool = False,
) -> HostExpertStore:
    """Every expert's paged rows, K experts at a time: ``step`` on each chunk of the loads,
    into ``store`` (a new pinned store if None); ``first_done``: chunk 0's rows are in
    ``store`` already (the caller ran them). A repacking step's rows are views of the
    layer's whole K-slot table on the GPU, so no chunk's rows outlive their copy. Each chunk's checkpoint pages are read ahead while the chunk before
    runs and given back once replayed; the mixing check's are kept until it ran, so that it
    does not read them from the drive again."""
    loads = step.loads
    num_experts, num_slots = loads.num_experts, loads.num_slots
    chunks = [
        list(range(start, min(start + num_slots, num_experts)))
        for start in range(0, num_experts, num_slots)
    ]
    mixing = [e for e in step.mixing_experts() if e not in loads.keep]
    loads.keep.update(mixing)
    try:
        if not first_done:
            loads.read_ahead(chunks[0])
        for i, experts in enumerate(chunks):
            if i + 1 < len(chunks):
                loads.read_ahead(chunks[i + 1])
            if i:
                out = {
                    n: h[experts[0] : experts[-1] + 1] for n, h in store.host.items()
                }
                step.run(experts, out)
                continue
            if first_done:  # run already and not replayed again: its pages go now
                loads.page_out(e for e in experts if e not in loads.keep)
                continue
            first = step.run(experts)  # also gives the paged layout
            if store is None:
                store = HostExpertStore(step.rows_like, num_experts)
            for name, rows in first.items():
                store.host[name][: len(experts)].copy_(rows)
            first = rows = None  # the next chunk's run builds another K-slot table
        step.check_mixing(store)
    finally:
        loads.keep.difference_update(mixing)
        loads.page_out(mixing)
    return store


class UnquantizedFormat(ExpertFormat):
    label = "unquantized (bf16/fp16)"

    @classmethod
    def supports(cls, base_method, scheme) -> bool:
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        return isinstance(base_method, UnquantizedFusedMoEMethod)


class Fp8BlockFormat(ExpertFormat):
    """FP8 weights with per-block scales (e.g. 128x128) and dynamic activation scales."""

    label = "block-quantized FP8"

    @classmethod
    def supports(cls, base_method, scheme) -> bool:
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        return (
            isinstance(base_method, Fp8MoEMethod)
            and base_method.quant_config.weight_block_size is not None
            and base_method.quant_config.is_checkpoint_fp8_serialized
            and not base_method.use_mxfp8
            and not base_method.is_fp4_expert
        )


class W8A8Fp8Format(ExpertFormat):
    label = "W8A8 FP8 (per-channel, dynamic activations)"

    @classmethod
    def supports(cls, base_method, scheme) -> bool:
        from sglang.srt.layers.quantization.w8a8_fp8 import W8A8FP8MoEMethod

        return isinstance(base_method, W8A8FP8MoEMethod)


class W8A8Int8Format(ExpertFormat):
    label = "W8A8 INT8 (per-channel, dynamic activations)"

    @classmethod
    def supports(cls, base_method, scheme) -> bool:
        from sglang.srt.layers.quantization import w8a8_int8

        # On AMX and arm64 hosts the method runs a CPU kernel instead of its runner.
        return (
            isinstance(base_method, w8a8_int8.W8A8Int8MoEMethod)
            and not w8a8_int8._is_cpu_amx_available
            and not w8a8_int8._is_cpu_arm64
        )


class BlockInt8Format(ExpertFormat):
    label = "block-quantized INT8"

    @classmethod
    def supports(cls, base_method, scheme) -> bool:
        from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8MoEMethod

        return isinstance(base_method, BlockInt8MoEMethod)


class GptqFormat(ExpertFormat):
    label = "GPTQ"
    # The Marlin MoE kernel never reads GPTQ's zero points.
    unpaged = ("w13_qzeros", "w2_qzeros")

    @classmethod
    def supports(cls, base_method, scheme) -> bool:
        from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinMoEMethod

        return isinstance(base_method, GPTQMarlinMoEMethod)


class AwqFormat(ExpertFormat):
    label = "AWQ"

    @classmethod
    def supports(cls, base_method, scheme) -> bool:
        from sglang.srt.layers.quantization.awq.awq import AWQMoEMethod

        return isinstance(base_method, AWQMoEMethod)


class Mxfp4Format(ExpertFormat):
    label = "MXFP4"

    @classmethod
    def supports(cls, base_method, scheme) -> bool:
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        return isinstance(base_method, Mxfp4MoEMethod)


class ModelOptNvfp4Format(ExpertFormat):
    label = "ModelOpt NVFP4"
    # Swizzled block scales serve FlashInfer's kernels, not Marlin's.
    freed = ("w13_blockscale_swizzled", "w2_blockscale_swizzled")

    @classmethod
    def supports(cls, base_method, scheme) -> bool:
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptNvFp4FusedMoEMethod,
        )

        return (
            type(base_method) is ModelOptNvFp4FusedMoEMethod
            and base_method.quant_config.is_checkpoint_nvfp4_serialized
        )


class CompressedTensorsWNA16Format(ExpertFormat):
    label = "compressed-tensors W4A16/W8A16"
    unpaged = ("w13_weight_shape", "w2_weight_shape")

    @classmethod
    def supports(cls, base_method, scheme) -> bool:
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
            CompressedTensorsWNA16MoE,
        )

        return type(scheme) is CompressedTensorsWNA16MoE

    def runner_method(self, base_method):
        return _ThroughMarlinRunner(base_method)


class CompressedTensorsFp8Format(ExpertFormat):
    # Per-tensor scales are requantized to a max over the experts after loading, which
    # paging K at a time cannot reproduce.
    label = "compressed-tensors FP8 (per-channel or block, dynamic activations)"

    @classmethod
    def supports(cls, base_method, scheme) -> bool:
        from compressed_tensors.quantization import QuantizationStrategy

        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
            CompressedTensorsW8A8Fp8MoE,
        )

        return (
            type(scheme) is CompressedTensorsW8A8Fp8MoE
            and not scheme.static_input_scales
            and scheme.weight_quant.strategy
            in (QuantizationStrategy.CHANNEL, QuantizationStrategy.BLOCK)
        )


class CompressedTensorsNvfp4Format(ExpertFormat):
    label = "compressed-tensors NVFP4A16"

    @classmethod
    def supports(cls, base_method, scheme) -> bool:
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_nvfp4_moe import (
            CompressedTensorsW4A16Nvfp4MoE,
        )

        return type(scheme) is CompressedTensorsW4A16Nvfp4MoE


class _ThroughMarlinRunner:
    """Applies a method through its scheme's Marlin runner, which honors ``runner_config``; the
    scheme's own ``apply`` calls the kernel directly and sums the top-k outputs."""

    def __init__(self, base_method):
        self.base_method = base_method

    def apply(self, layer, dispatch_output):
        quant_info = self.base_method.get_marlin_quant_info(layer)
        return layer.scheme.runner.run(dispatch_output, quant_info)


FORMATS = (
    UnquantizedFormat,
    Fp8BlockFormat,
    W8A8Fp8Format,
    W8A8Int8Format,
    BlockInt8Format,
    CompressedTensorsFp8Format,
    GptqFormat,
    AwqFormat,
    Mxfp4Format,
    ModelOptNvfp4Format,
    CompressedTensorsWNA16Format,
    CompressedTensorsNvfp4Format,
)


def format_for(layer, base_method) -> Optional[ExpertFormat]:
    scheme = getattr(layer, "scheme", None)
    for fmt in FORMATS:
        if fmt.supports(base_method, scheme):
            return fmt()
    return None

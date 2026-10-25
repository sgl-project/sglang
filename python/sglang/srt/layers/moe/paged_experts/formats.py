"""The fused-MoE methods paged experts runs, one ``ExpertFormat`` each, and how their experts
reach the host store.

The expert tensors are found on the layer, as EPLB finds them: every parameter with one row
per local expert. The checkpoint loads them into the host store, and the ones still per-expert
after the base method's post-load step are paged. A format names only what that rule cannot
see: tensors its runner never reads (``unpaged``) and tensors created for other runners
(``freed``).

Whether the post-load step repacks the experts follows from the runner (``repacks`` of its
contract). A repack runs over the host store K experts at a time, so it must treat every
expert on its own, which is checked; any other step must leave the expert tensors as loaded.

Adding a quantization method means adding a subclass to ``FORMATS``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from sglang.srt.layers.moe.paged_experts.store import HostExpertStore
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


def after_loading(layer, base_method, store, num_slots: int, repacks: bool, unpaged):
    """Run the base post-load step; return the store to page from, whose experts 0..K-1 the
    GPU slots then hold."""
    if repacks:
        return _repack(layer, base_method, store, num_slots, unpaged)
    for name, host in store.host.items():
        getattr(layer, name).data.copy_(host[:num_slots])
    base_method.process_weights_after_loading(layer)
    # Paging copies host rows verbatim, so the base method must neither transform the loaded
    # weights (e.g. a kernel-specific shuffle or row interleave) nor create others.
    created = set(expert_params(layer, num_slots)) - set(store.host)
    if created:
        raise RuntimeError(
            f"Paged experts: {type(base_method).__name__} creates per-expert {sorted(created)} "
            "after loading, which the host store does not hold"
        )
    for name, host in store.host.items():
        gpu = getattr(layer, name).data[0]
        if not torch.equal(gpu, host[0].to(gpu.device)):
            raise RuntimeError(
                f"Paged experts: {type(base_method).__name__} transforms {name} after "
                "loading, which the host store does not replicate"
            )
    return store


def _repack(layer, base_method, staged, num_slots: int, unpaged):
    """Run the base post-load step over the staged checkpoint layout, K experts at a time,
    collecting its per-expert output into a new store."""
    num_experts = next(iter(staged.host.values())).shape[0]
    templates = {name: getattr(layer, name) for name in staged.host}
    # The step records state on the layer (e.g. that it has run); each chunk starts from the
    # state before the first.
    state = {k: v for k, v in vars(layer).items() if k not in _MODULE_ATTRS}

    def post_load(start: int, n: int) -> None:
        for k in [k for k in vars(layer) if k not in _MODULE_ATTRS and k not in state]:
            delattr(layer, k)
        for k, v in state.items():
            setattr(layer, k, v)
        # Fresh parameters in the checkpoint layout: the step may replace or rewrite them.
        # Rows past a short last chunk hold experts from the start, not what the base method
        # created (torch.empty): a step that validates every row would reject those.
        for name, template in templates.items():
            fresh = torch.nn.Parameter(template.data.clone(), requires_grad=False)
            fresh.__dict__.update(template.__dict__)
            fresh.data[:n].copy_(staged.host[name][start : start + n])
            fresh.data[n:].copy_(staged.host[name][: num_slots - n])
            layer.register_parameter(name, fresh)
        base_method.process_weights_after_loading(layer)

    packed = None
    for start in range(0, num_experts, num_slots):
        n = min(num_slots, num_experts - start)
        post_load(start, n)
        if packed is None:
            names = [
                name for name in expert_params(layer, num_slots) if name not in unpaged
            ]
            if not names:
                raise RuntimeError(
                    f"Paged experts: {type(base_method).__name__} leaves no per-expert "
                    "tensors after loading"
                )
            packed = HostExpertStore(layer=layer, names=names, num_experts=num_experts)
        for name, host in packed.host.items():
            host[start : start + n].copy_(getattr(layer, name).data[:n])
    if num_experts > num_slots:
        # Experts 1..K again, in a chunk of their own: a step that mixes experts (e.g. a max
        # over them) gives them other rows than their first chunk did.
        post_load(1, num_slots)
        for name, host in packed.host.items():
            if not torch.equal(
                getattr(layer, name).data.cpu(), host[1 : num_slots + 1]
            ):
                raise RuntimeError(
                    f"Paged experts: {type(base_method).__name__}'s post-load step does not "
                    f"treat each expert on its own ({name}), so it cannot run K at a time"
                )
    for name, host in packed.host.items():
        getattr(layer, name).data.copy_(host[:num_slots])
    return packed


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

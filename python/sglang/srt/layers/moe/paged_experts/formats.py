"""Per-quantization-method expert layout, one ``ExpertFormat`` per fused-MoE method.

A format answers three questions for the method it wraps:

* which per-expert tensors the checkpoint loads (``checkpoint_params``) and which are paged at
  run time (``paged_params``);
* what happens after loading, so the host copy holds exactly what the GPU slots would hold
  (for methods that repack their experts, the repack runs over the host copy K at a time);
* which runner contract (``runners.py``) its base method's runner follows.

Adding a quantization method means adding a subclass to ``FORMATS``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import torch

from sglang.srt.layers.moe.paged_experts.runners import (
    MARLIN,
    TRITON,
    RunnerContract,
)

# Prefixes of FusedMoE's per-expert parameters.
_EXPERT_PARAM_PREFIXES = ("w13_", "w2_")


class ExpertFormat(ABC):
    #: Per-expert tensors the base method creates for loading.
    checkpoint_params: Tuple[str, ...] = ()
    #: Per-expert tensors paged at run time.
    paged_params: Tuple[str, ...] = ()
    #: Whether the base post-load step repacks the experts; it then runs over the host copy K
    #: experts at a time.
    repacks_after_loading: bool = False
    contract: RunnerContract = TRITON

    @classmethod
    @abstractmethod
    def supports(cls, base_method) -> bool:
        """Whether this format handles ``base_method``."""

    def check_params(self, layer, num_slots: int) -> None:
        """Raise unless the layer's per-expert tensors are exactly the declared ones."""
        per_expert = {
            name
            for name, p in layer.named_parameters(recurse=False)
            if name.startswith(_EXPERT_PARAM_PREFIXES)
            and p.dim() > 0
            and p.shape[0] == num_slots
        }
        if per_expert != set(self.checkpoint_params):
            raise RuntimeError(
                f"Paged experts: {type(self).__name__} expects per-expert parameters "
                f"{sorted(self.checkpoint_params)}, the layer has {sorted(per_expert)}"
            )

    def after_loading(
        self, layer, base_method, store, num_slots: int, new_store: Callable
    ):
        """Run the base post-load step; return the store to page from, whose experts 0..K-1
        the GPU slots then hold. ``new_store(names)`` builds an empty pinned store shaped
        like the layer's current ``names`` tensors."""
        if self.repacks_after_loading:
            return self._repack(
                layer=layer,
                base_method=base_method,
                staged=store,
                num_slots=num_slots,
                new_store=new_store,
            )
        for name, host in store.host.items():
            getattr(layer, name).data.copy_(host[:num_slots])
        base_method.process_weights_after_loading(layer)
        # Paging copies host rows verbatim, so the base method must not transform the
        # loaded weights (e.g. a kernel-specific shuffle or row interleave).
        for name, host in store.host.items():
            gpu = getattr(layer, name).data[0]
            if not torch.equal(gpu, host[0].to(gpu.device)):
                raise RuntimeError(
                    f"Paged experts: {type(base_method).__name__} transforms {name} "
                    "after loading, which the host store does not replicate"
                )
        return store

    def _repack(self, layer, base_method, staged, num_slots: int, new_store: Callable):
        """Run the base post-load step over the staged checkpoint layout, K experts at a time,
        collecting its per-expert output into a new store. Only valid for steps that treat
        every expert independently."""
        num_experts = next(iter(staged.host.values())).shape[0]
        templates = {name: getattr(layer, name) for name in staged.host}
        packed = None
        for start in range(0, num_experts, num_slots):
            n = min(num_slots, num_experts - start)
            # Fresh parameters in the checkpoint layout: the step may replace or rewrite them.
            for name, template in templates.items():
                fresh = torch.nn.Parameter(
                    torch.empty_like(template.data), requires_grad=False
                )
                fresh.__dict__.update(template.__dict__)
                fresh.data[:n].copy_(staged.host[name][start : start + n])
                layer.register_parameter(name, fresh)
            base_method.process_weights_after_loading(layer)
            if packed is None:
                packed = new_store(self.paged_params)
            for name, host in packed.host.items():
                host[start : start + n].copy_(getattr(layer, name).data[:n])
        for name, host in packed.host.items():
            getattr(layer, name).data.copy_(host[:num_slots])
        return packed


class UnquantizedFormat(ExpertFormat):
    checkpoint_params = paged_params = ("w13_weight", "w2_weight")

    @classmethod
    def supports(cls, base_method) -> bool:
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        return isinstance(base_method, UnquantizedFusedMoEMethod)


class Fp8BlockFormat(ExpertFormat):
    """FP8 weights with per-block scales (e.g. 128x128) and dynamic activation scales. On the
    triton runner the post-load step leaves both untouched, so they page as stored."""

    checkpoint_params = paged_params = (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        "w2_weight_scale_inv",
    )

    @classmethod
    def supports(cls, base_method) -> bool:
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        return (
            isinstance(base_method, Fp8MoEMethod)
            and base_method.quant_config.weight_block_size is not None
            and base_method.quant_config.is_checkpoint_fp8_serialized
            and not base_method.use_mxfp8
            and not base_method.is_fp4_expert
        )


class GptqMarlinFormat(ExpertFormat):
    """GPTQ int4/int8 experts on the Marlin runner. The post-load step sorts the act-order
    indices and repacks each expert's weights and scales into Marlin's layout."""

    checkpoint_params = (
        "w13_qweight",
        "w2_qweight",
        "w13_scales",
        "w2_scales",
        "w13_qzeros",
        "w2_qzeros",
        "w13_g_idx",
        "w2_g_idx",
        "w13_g_idx_sort_indices",
        "w2_g_idx_sort_indices",
    )
    # GPTQ-Marlin's MoE apply never passes the zero points to the kernel.
    paged_params = tuple(n for n in checkpoint_params if not n.endswith("_qzeros"))
    repacks_after_loading = True
    contract = MARLIN

    @classmethod
    def supports(cls, base_method) -> bool:
        from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinMoEMethod

        return isinstance(base_method, GPTQMarlinMoEMethod)


FORMATS = (UnquantizedFormat, Fp8BlockFormat, GptqMarlinFormat)


def format_for(base_method) -> Optional[ExpertFormat]:
    for fmt in FORMATS:
        if fmt.supports(base_method):
            return fmt()
    return None

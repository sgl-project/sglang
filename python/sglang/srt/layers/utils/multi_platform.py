from typing import Callable, Collection, Optional, Tuple

import torch
from torch import nn

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_musa = is_musa()


class CompilableRegionMixin:
    """Mixin for nn.Modules that expose named sub-regions for torch.compile
    under --torch-compile-scope local.

    Subclasses override ``get_compilable_regions`` to return a mapping from
    region names (used in --torch-compile-override-layers) to the method name
    that should be compiled.  The ``_to_torch`` walk in cuda_graph_runner
    discovers these regions alongside ``MultiPlatformOp`` handling.
    """

    def get_compilable_regions(self) -> dict[str, str]:
        """Return {region_name: method_name} for compilable sub-functions."""
        return {}

    def _get_region_compile_method(
        self,
        region_name: str,
        method_name: str,
        compile_options: Optional[dict] = None,
        compile_dynamic: bool = False,
    ) -> Callable:
        """Return the compiled callable for *method_name*.

        Override in subclasses to customise ``torch.compile`` kwargs per
        region (e.g. force ``dynamic=None``).
        """
        return torch.compile(
            getattr(self, method_name),
            options=compile_options,
            dynamic=compile_dynamic,
        )

    def enter_region_compile(
        self,
        region_name: str,
        compile_options: Optional[dict] = None,
        compile_dynamic: bool = False,
    ):
        """Replace the named method with its ``torch.compile``d version."""
        if hasattr(self, "_compiled_region_originals") and (
            region_name in self._compiled_region_originals
        ):
            return  # already compiled

        regions = self.get_compilable_regions()
        method_name = regions[region_name]
        # Track whether the method lived in __dict__ (instance attr) vs on the
        # class.  This decides whether leave_region_compile restores via setattr
        # or delattr.
        was_instance_attr = method_name in self.__dict__
        original = self.__dict__.get(method_name) if was_instance_attr else None
        compiled = self._get_region_compile_method(
            region_name,
            method_name,
            compile_options=compile_options,
            compile_dynamic=compile_dynamic,
        )
        if not hasattr(self, "_compiled_region_originals"):
            self._compiled_region_originals: dict[
                str, tuple[str, object, bool]
            ] = {}
        self._compiled_region_originals[region_name] = (
            method_name,
            original,
            was_instance_attr,
        )
        setattr(self, method_name, compiled)

    def leave_region_compile(self, region_name: str):
        """Restore the original (un-compiled) method."""
        if not hasattr(self, "_compiled_region_originals"):
            return
        entry = self._compiled_region_originals.pop(region_name, None)
        if entry is None:
            return
        method_name, original, was_instance_attr = entry
        if was_instance_attr:
            setattr(self, method_name, original)
        elif method_name in self.__dict__:
            delattr(self, method_name)

    def is_region_compiled(self, region_name: str) -> bool:
        return hasattr(self, "_compiled_region_originals") and (
            region_name in self._compiled_region_originals
        )


class MultiPlatformOp(nn.Module):
    def __init__(self):
        super().__init__()
        self._forward_method: Callable = self.dispatch_forward()

        # States for torch.compile
        self._original_forward_method = None
        self.is_torch_compile = False

    def _get_local_torch_compile_forward_method(
        self,
        method_name: str,
        compile_options: Optional[dict] = None,
        compile_dynamic: bool = False,
    ) -> Callable:
        return torch.compile(
            getattr(self, method_name),
            options=compile_options,
            dynamic=compile_dynamic,
        )

    def _matches_override_layers(
        self, override_layers: Collection[str]
    ) -> bool:
        """Check if any class in the MRO matches the override allowlist.

        This lets `--torch-compile-override-layers RotaryEmbedding` match all
        subclasses (YaRNScalingRotaryEmbedding, Llama3RotaryEmbedding, etc.)."""
        return any(cls.__name__ in override_layers for cls in type(self).__mro__)

    def _get_torch_compile_forward_method(
        self,
        num_tokens: int,
        compile_scope: str = "full",
        override_layers: Optional[Collection[str]] = None,
        compile_options: Optional[dict] = None,
        compile_dynamic: bool = False,
    ) -> Tuple[bool, Optional[Callable]]:
        class_name = self.__class__.__name__

        if compile_scope == "local":
            if override_layers is None or not self._matches_override_layers(
                override_layers
            ):
                return False, None
            return True, self._get_local_torch_compile_forward_method(
                "forward_native",
                compile_options=compile_options,
                compile_dynamic=compile_dynamic,
            )

        if override_layers is not None:
            # Class-name allowlist from `--torch-compile-override-layers`,
            # e.g. `UnquantizedFusedMoEMethod RMSNorm`. Matches any class in
            # the MRO so that base-class names also cover subclasses.
            if not self._matches_override_layers(override_layers):
                return False, None
            return True, self.forward_native

        # Default fallback
        if "FusedMoE" in class_name:
            if num_tokens == 1:
                from sglang.srt.layers.moe.fused_moe_native import (
                    fused_moe_forward_native,
                )

                return True, fused_moe_forward_native
            return True, self._forward_method

        if "TopK" in class_name:
            if num_tokens == 1:
                return True, self.forward_native
            return True, self._forward_method

        return True, self.forward_native

    def enter_torch_compile(
        self,
        num_tokens: int,
        compile_scope: str = "full",
        override_layers: Optional[Collection[str]] = None,
        compile_options: Optional[dict] = None,
        compile_dynamic: bool = False,
    ):
        # Skip if Op is already entered compile mode.
        # NOTE(alcanderian): Some Ops(for example RotaryEmbedding) will be reused
        # among layers and `enter_torch_compile` will be called many times.
        # We should prevent `self._original_forward_method` from being overridden when
        # it is not the first time `enter_torch_compile` called.
        if self.is_torch_compile:
            return

        should_switch, forward_method = self._get_torch_compile_forward_method(
            num_tokens=num_tokens,
            compile_scope=compile_scope,
            override_layers=override_layers,
            compile_options=compile_options,
            compile_dynamic=compile_dynamic,
        )
        if not should_switch:
            return

        self._original_forward_method = self._forward_method
        # NOTE: By default we keep the existing bs=1-only special handling for MoE
        # and TopK, unless a CLI allowlist explicitly opts a class into compile mode.
        self._forward_method = forward_method
        self.is_torch_compile = True

    def leave_torch_compile(self):
        # Skip if Op is already exited compile mode.
        if not self.is_torch_compile:
            return

        self._forward_method = self._original_forward_method
        self._original_forward_method = None
        self.is_torch_compile = False

    # Please do not override this method, because `self._forward_method` can change when in torch compile mode
    @debug_kernel_api
    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_npu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_musa(self, *args, **kwargs):
        # XXX (MUSA): MUSA kernels follow the CUDA path by default.
        # At this stage, sgl-kernel support for MUSA is still under active
        # development, so we fall back to the PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def forward_hpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self):
        if _is_cuda:
            return self.forward_cuda
        elif _is_hip:
            return self.forward_hip
        elif _is_cpu and _is_cpu_amx_available:
            return self.forward_cpu
        elif _is_npu:
            return self.forward_npu
        elif _is_xpu:
            return self.forward_xpu
        elif _is_musa:
            return self.forward_musa
        else:
            return self.forward_native

from typing import Callable, ClassVar, Collection, Optional, Tuple

from torch import nn

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.platforms import current_platform
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
)
from sglang.srt.utils.torch_compile_utils import CompileConfig, compile_callable

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_musa = is_musa()


class MultiPlatformOp(nn.Module):
    compile_config: ClassVar[CompileConfig] = CompileConfig()

    # OOT forward registry: maps dispatch_key -> {op_cls -> forward_fn}
    _oot_forward_registry: ClassVar[dict[str, dict[type, Callable]]] = {}

    @classmethod
    def register_oot_forward(cls, op_cls: type, fn: Callable, platform_key: str):
        """Register an OOT forward implementation for a specific op class and platform."""
        cls._oot_forward_registry.setdefault(platform_key, {})[op_cls] = fn

    def __init__(self):
        super().__init__()
        self._forward_method: Callable = self.dispatch_forward()

        # States for torch.compile
        self._original_forward_method = None
        self.is_torch_compile = False

    def _get_local_torch_compile_forward_method(
        self,
        method_name: str,
        compile_mode: Optional[str] = None,
        compile_options: Optional[dict] = None,
        compile_dynamic: bool = False,
    ) -> Callable:
        return compile_callable(
            getattr(self, method_name),
            compile_mode=compile_mode,
            compile_options=compile_options,
            compile_dynamic=compile_dynamic,
        )

    def _matches_override_layers(self, override_layers: Collection[str]) -> bool:
        """Check if any class in the MRO matches the override allowlist.

        This lets `--torch-compile-override-layers RotaryEmbedding` match all
        subclasses (YaRNScalingRotaryEmbedding, Llama3RotaryEmbedding, etc.)."""
        return any(cls.__name__ in override_layers for cls in type(self).__mro__)

    def _get_torch_compile_forward_method(
        self,
        num_tokens: int,
        compile_scope: str = "full",
        override_layers: Optional[Collection[str]] = None,
        compile_mode: Optional[str] = None,
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
                compile_mode=compile_mode,
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
        compile_mode: Optional[str] = None,
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
            compile_mode=compile_mode,
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
        return self.forward_cuda(*args, **kwargs)

    def forward_hpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self):
        # OOT platform dispatch: check registry then method lookup
        if current_platform.is_out_of_tree():
            key = current_platform.get_dispatch_key_name()
            oot = self._oot_forward_registry.get(key, {})
            if type(self) in oot:
                return oot[type(self)].__get__(self)
            method = getattr(self, f"forward_{key}", None)
            if method is not None:
                return method
            return self.forward_native

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

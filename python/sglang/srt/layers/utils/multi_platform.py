from typing import Callable, Collection, Optional, Tuple

import torch
from torch import nn

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
            if override_layers is None or class_name not in override_layers:
                return False, None
            return True, self._get_local_torch_compile_forward_method(
                "forward_native",
                compile_options=compile_options,
                compile_dynamic=compile_dynamic,
            )

        if override_layers is not None:
            # Exact class-name allowlist from `--torch-compile-override-layers`,
            # e.g. `UnquantizedFusedMoEMethod RMSNorm`. Allowlisted ops switch
            # to their torch-native implementation for torch.compile.
            if class_name not in override_layers:
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

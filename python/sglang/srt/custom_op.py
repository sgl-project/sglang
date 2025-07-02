from torch import nn

from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_cuda, is_hip, is_npu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_npu = is_npu()


class CustomOp(nn.Module):
    def __init__(self):
        super().__init__()
        self._forward_method = self.dispatch_forward()

        # States for torch.compile
        self._original_forward_method = None
        self.is_torch_compile = False

    def enter_torch_compile(self, num_tokens: int):
        # Skip if Op is already entered compile mode.
        # NOTE(alcanderian): Some Ops(for example RotaryEmbedding) will be reused
        # among layers and `enter_torch_compile` will be called many times.
        # We should prevent `self._original_forward_method` from being overridden when
        # it is not the first time `enter_torch_compile` called.
        if self.is_torch_compile:
            return

        self._original_forward_method = self._forward_method
        # NOTE: Temporarily workaround MoE
        if "FusedMoE" in self.__class__.__name__:
            if num_tokens == 1:
                from sglang.srt.layers.moe.fused_moe_native import (
                    fused_moe_forward_native,
                )

                # The performance of torch.compile on this layer is not always good when bs > 1,
                # so we decide to only use torch.compile when bs =1
                self._forward_method = fused_moe_forward_native
        else:
            self._forward_method = self.forward_native
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
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
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
        else:
            return self.forward_native

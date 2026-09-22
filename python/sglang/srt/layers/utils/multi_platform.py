"""Deprecated compatibility shim for the former ``MultiPlatformOp``.

The multi-platform operator abstraction was unified into
:class:`sglang.kernels.fused_op.BaseFusedOp` (RFC #29630): one class now
covers kernel-backend selection (``forward_aot`` / ``forward_jit`` / ...),
platform dispatch (``forward_cuda`` / ``forward_hip`` / ``forward_npu`` /
...), out-of-tree platform overrides (:meth:`BaseFusedOp.register_oot_forward`),
and the torch.compile enter/leave protocol.

In-repo code must subclass ``BaseFusedOp`` directly. This alias exists only so
out-of-tree platform plugins and external users keep importing from the old
path while they migrate; it will be removed in a future release.
"""

import warnings

from sglang.kernels.fused_op import BaseFusedOp


class MultiPlatformOp(BaseFusedOp):
    """Deprecated alias of :class:`sglang.kernels.fused_op.BaseFusedOp`.

    Kept attribute-compatible with the original class: ``forward_native`` is
    concrete here (raising ``NotImplementedError``) so existing plugin
    subclasses that only define platform forwards keep instantiating, and the
    old per-platform default methods (``forward_hip`` -> ``forward_cuda``,
    ``forward_cpu`` -> ``forward_native``, ...) remain callable for plugin
    code that invokes them directly.
    """

    def __init_subclass__(cls, **kwargs):
        warnings.warn(
            "MultiPlatformOp is deprecated; subclass "
            "sglang.kernels.fused_op.BaseFusedOp instead (RFC #29630).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_musa(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_npu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_hpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

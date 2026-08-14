"""Online (config-driven) quantized FusedMoE methods for Ascend NPU.

These are the ``--quantization <scheme>`` entry points: the checkpoint holds
BF16/FP16 expert weights and the per-gmm kernels quantize them at load time.
Offline (msmodelslim) checkpoints go through the ModelSlim schemes instead and
reuse the same kernels.

Kept out of ``moe_methods.py`` because ``unquant.py`` imports that module at
module scope, so subclassing ``UnquantizedFusedMoEMethod`` there would be a
circular import.
"""

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUMXFP8MoEMethod
from sglang.srt.layers.moe.moe_runner import MoeRunner
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_runner_backend
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


class NPUMXFP8OnlineMoEMethod(UnquantizedFusedMoEMethod):
    """Online MXFP8 FusedMoE entry point (``--quantization mxfp8`` on A5).

    Weight creation, weight post-processing and the forward pass are identical
    to the unquantized Ascend path — the only difference is which per-gmm kernel
    the layer gets, so everything but ``create_moe_runner`` is inherited.
    ``NPUMXFP8MoEMethod`` then quantizes the BF16 expert weights to MXFP8 in
    ``process_weights_after_loading``.
    """

    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        super().__init__()
        self.quant_config = quant_config

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        backend = get_moe_runner_backend()
        if not (backend.is_auto() or backend.is_ascend()):
            # Not merely a wrong-runner check. Because this method subclasses
            # UnquantizedFusedMoEMethod it matches FusedMoE's shard-swap list, so
            # a flashinfer backend would make the weight loader exchange the
            # w1/w3 shards ("flashinfer assumes w31 format"). Every expert would
            # then load with gate and up swapped, and gmm1's fused swiglu would
            # compute silu(up) * gate — no error, just degenerate output.
            raise ValueError(
                "MXFP8 MoE on Ascend requires --moe-runner-backend 'auto' or "
                f"'ascend', got {backend.value!r}."
            )

        # The kernels must be attached before the runner is built:
        # AscendRunnerCore.__init__ reads layer.w2_kernel to pick its activation.
        layer.w13_kernel = NPUMXFP8MoEMethod("w13")
        layer.w2_kernel = NPUMXFP8MoEMethod("w2")
        moe_runner_config.layer = layer
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.ASCEND, moe_runner_config)
        # Inherited apply() consults this; aiter is CUDA/ROCm-only.
        self._aiter_runner = None

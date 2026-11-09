"""SAIL's dispatcher wire format, resolved before the weights are loaded.

The shared ``FusedMoE`` only has to describe the numerics of a MoE layer to its
dispatcher once a quantization method has seen the checkpoint, so upstream
leaves ``set_quant_config`` to ``process_weights_after_loading``. SAIL's DeepEP
sizes its buffers and picks its permute kernels from that same wire format, and
needs it at construction time. The mapping from quant method to wire format is
therefore SAIL knowledge and is kept here rather than in the shared layer.

``FusedMoE.__init__`` calls this on SAIL only; every other platform still lets
the quantization method configure the dispatcher itself, once the checkpoint
dtype is final.
"""

from typing import TYPE_CHECKING

from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.hardware_backend.sail.quantization.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8MoE,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsFusedMoEMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8MoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.srt.layers.quantization.w8a8_fp8 import W8A8FP8MoEMethod
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8MoEMethod

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE


def build_early_dispatcher_quant_config(layer: "FusedMoE") -> dict:
    """Describe the layer's numerics to the dispatcher ahead of loading.

    The dispatcher only accepts dictionaries, so this converts the layer's
    ``quant_config`` object (e.g. ``CompressedTensorsConfig``) into the fields
    DeepEP reads, while leaving that object untouched for the weight-loading
    paths. A method this cannot classify dispatches BF16, which is what SAIL's
    DeepGEMM runners take for the unquantized and WNA16 schemes.
    """
    quant_config_dict: dict = {}

    # compressed-tensors reports its scheme on the layer rather than the quant
    # method, so a compressed-tensors W8A8 INT8 MoE needs its own probe. It
    # drives the same wire format as W8A8Int8MoEMethod.
    ct_int8 = isinstance(
        layer.quant_method, CompressedTensorsFusedMoEMethod
    ) and isinstance(layer.scheme, CompressedTensorsW8A8Int8MoE)

    # SAIL DeepEP reads the dispatch dtype from this same
    # dispatcher_output_dtype field, which keeps every rank on one config.
    if (
        isinstance(layer.quant_method, Fp8MoEMethod)
        or isinstance(layer.quant_method, W8A8FP8MoEMethod)
        or (
            isinstance(layer.quant_method, CompressedTensorsFusedMoEMethod)
            and isinstance(layer.scheme, CompressedTensorsW8A8Fp8MoE)
        )
    ):
        quant_config_dict["dispatcher_output_dtype"] = "fp8"
    elif isinstance(layer.quant_method, W8A8Int8MoEMethod) or ct_int8:
        quant_config_dict["dispatcher_output_dtype"] = "int8"
    elif isinstance(layer.quant_method, Mxfp4MoEMethod):
        quant_config_dict["dispatcher_output_dtype"] = (
            "bf16"
            if (
                layer.quant_method.use_deepgemm_mxfp4_w4a16_valu
                or layer.quant_method.use_deepgemm_mxfp4_w4a16_mma
            )
            else "uint8"
        )
    # Unquantized and WNA16 use BF16 dispatch on SAIL.
    else:
        quant_config_dict["dispatcher_output_dtype"] = "bf16"

    is_channel_quant = False
    if isinstance(layer.quant_config, CompressedTensorsConfig):
        # Resolve from the scheme picked for this expert layer rather than the
        # global "Linear" target: a regex-targeted expert group can carry a
        # different strategy, and a mismatch breaks both ways -- DeepEP would
        # quantize activations in 128-groups while the channelwise runner
        # expects one scale per token, or the reverse.
        scheme_weight_quant = getattr(layer.scheme, "weight_quant", None)
        is_channel_quant = (
            scheme_weight_quant is not None
            and scheme_weight_quant.strategy == QuantizationStrategy.CHANNEL
        )

    if isinstance(layer.quant_method, W8A8Int8MoEMethod) or ct_int8:
        is_channel_quant = True

    quant_config_dict["is_channel_quant"] = is_channel_quant
    return quant_config_dict

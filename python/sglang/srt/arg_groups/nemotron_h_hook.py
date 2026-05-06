import logging
from typing import TYPE_CHECKING

from sglang.srt.utils.common import is_sm100_supported

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def apply_nemotron_h_defaults(server_args: "ServerArgs", model_arch: str) -> None:
    """Apply NemotronH model-specific server arg defaults and constraints."""
    model_config = server_args.get_model_config()
    if model_config.quantization in [
        "modelopt",
        "modelopt_fp8",
        "modelopt_fp4",
        "modelopt_mixed",
    ]:
        assert model_config.hf_config.mlp_hidden_act == "relu2"
        if model_config.quantization == "modelopt":
            quant_algo = model_config.hf_config.quantization_config["quant_algo"]
            if quant_algo == "MIXED_PRECISION":
                server_args.quantization = "modelopt_mixed"
            else:
                server_args.quantization = (
                    "modelopt_fp4" if quant_algo == "NVFP4" else "modelopt_fp8"
                )
        else:
            server_args.quantization = model_config.quantization
        if server_args.moe_runner_backend == "auto":
            if is_sm100_supported() and server_args.moe_a2a_backend == "none":
                server_args.moe_runner_backend = "flashinfer_trtllm"
                logger.info(
                    "Use flashinfer_trtllm as MoE runner backend on sm100 for "
                    f"{model_arch}"
                )
            else:
                server_args.moe_runner_backend = "flashinfer_cutlass"

    server_args._handle_mamba_radix_cache(
        model_arch=model_arch,
        support_mamba_cache=True,
        support_mamba_cache_extra_buffer=False,
        sm100_default_attention_backend="flashinfer",
    )
    assert server_args.attention_backend != "triton", (
        "NemotronHForCausalLM does not support triton attention backend,"
        "as the first layer might not be an attention layer"
    )

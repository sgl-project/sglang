import logging
from typing import TYPE_CHECKING

from sglang.srt.utils.common import is_sm100_supported

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def apply_nemotron_h_defaults(server_args: "ServerArgs", model_arch: str) -> None:
    """Apply NemotronH model-specific server arg defaults and constraints."""
    model_config = server_args.get_model_config()
    # NemotronH-specific config fields live on the inner llm_config for the
    # VL/Omni wrappers (NemotronH_Nano_VL_V2, NemotronH_Nano_Omni_Reasoning_V3)
    # and on hf_config directly for the standalone NemotronHForCausalLM.
    nemotron_h_cfg = getattr(
        model_config.hf_config, "llm_config", model_config.hf_config
    )
    if model_config.quantization in [
        "modelopt",
        "modelopt_fp8",
        "modelopt_fp4",
        "modelopt_mixed",
    ]:
        assert nemotron_h_cfg.mlp_hidden_act == "relu2"
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
                # Blackwell consumer (sm_110 / sm_120 / sm_121): native
                # cutlass_moe_fp4 kernel is sm_100-only; flashinfer_cutlass
                # has SM110/120/121 JIT support and non-gated awareness.
                server_args.moe_runner_backend = "flashinfer_cutlass"
                logger.info(
                    "Use flashinfer_cutlass as MoE runner backend for " f"{model_arch}"
                )

    server_args._handle_mamba_radix_cache(
        model_arch=model_arch,
        sm100_default_attention_backend="flashinfer",
        fallback_attention_backend="flashinfer",
    )
    assert server_args.attention_backend != "triton", (
        "NemotronHForCausalLM does not support triton attention backend,"
        "as the first layer might not be an attention layer"
    )

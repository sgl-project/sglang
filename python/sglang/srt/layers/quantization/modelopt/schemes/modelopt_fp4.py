# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py
from __future__ import annotations

import logging
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
from sglang.srt.layers.moe.utils import should_use_flashinfer_cutlass_moe_fp4_allgather
from sglang.srt.layers.parameter import ModelWeightParameter, PerTensorScaleParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
)
from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
from sglang.srt.layers.quantization.fp8_utils import is_blackwell_supported
from sglang.srt.layers.quantization.modelopt.modelopt import ModelOptQuantConfig
from sglang.srt.layers.quantization.modelopt.utils import (
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight,
    round_up_to_multiple,
    slice_nvfp4_output,
)
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.srt.layers.utils import copy_or_rebind_param
from sglang.srt.utils.common import (
    get_bool_env_var,
    is_cuda,
    is_sm120_supported,
    next_power_of_2,
)
from sglang.srt.utils.custom_op import register_custom_op
from sglang.srt.utils.patch_torch import register_fake_if_exists

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

fp4_quantize = None
try:
    if is_sm120_supported():
        try:
            from flashinfer import fp4_quantize
        except ImportError:
            from sglang.jit_kernel.nvfp4 import scaled_fp4_quant as fp4_quantize
    else:
        from sglang.jit_kernel.nvfp4 import scaled_fp4_quant as fp4_quantize
except ImportError:
    fp4_quantize = None

try:
    from flashinfer import mm_fp4 as flashinfer_fp4_gemm
    from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_sf_a

    enable_flashinfer_fp4_gemm = True
except ImportError:
    enable_flashinfer_fp4_gemm = False
    reorder_rows_for_gated_act_gemm = None
    shuffle_matrix_a = None
    shuffle_matrix_sf_a = None

if is_cuda():
    try:
        from sglang.jit_kernel.nvfp4 import cutlass_scaled_fp4_mm as cutlass_fp4_gemm
    except ImportError:
        cutlass_fp4_gemm = None
else:
    cutlass_fp4_gemm = None

try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType
except ImportError:
    flashinfer_cutlass_fused_moe = None

    # Define a minimal ActivationType enum if flashinfer is not available
    class ActivationType(IntEnum):
        Swiglu = 3
        Relu2 = 6


def _sglang_fp4_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> torch.Tensor:
    M = input.shape[-2]
    N = int(out_features)
    return input.new_empty((M, N), dtype=out_dtype)


@register_custom_op(fake_impl=_sglang_fp4_gemm_fake)
def fp4_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> torch.Tensor:
    fp4_backend = get_fp4_gemm_runner_backend()
    if fp4_backend.is_cutlass() and cutlass_fp4_gemm is not None:
        # flashinfer.fp4_quantize returns scale factors as uint8 (e4m3fn bits
        # stored in uint8 memory). The JIT kernel requires float8_e4m3fn dtype.
        if input_sf.dtype != torch.float8_e4m3fn:
            input_sf = input_sf.view(torch.float8_e4m3fn)
        if weight_sf.dtype != torch.float8_e4m3fn:
            weight_sf = weight_sf.view(torch.float8_e4m3fn)
        return cutlass_fp4_gemm(input, weight, input_sf, weight_sf, alpha, out_dtype)
    elif enable_flashinfer_fp4_gemm:
        # Use the remapping logic to convert SGLang backend names to FlashInfer API names
        backend = fp4_backend.get_flashinfer_backend()
        return flashinfer_fp4_gemm(
            input, weight, input_sf, weight_sf, alpha, out_dtype, backend=backend
        )
    else:
        return cutlass_fp4_gemm(input, weight, input_sf, weight_sf, alpha, out_dtype)


if is_cuda() and (not is_sm120_supported()) and (fp4_quantize is not None):

    @register_fake_if_exists("sgl_kernel::scaled_fp4_quant")
    def _sgl_kernel_scaled_fp4_quant_fake(
        output, input, output_scale, input_global_scale
    ):
        return


CUTEDSL_MOE_SCALAR_INPUT_SCALE = get_bool_env_var(
    "SGLANG_CUTEDSL_MOE_SCALAR_INPUT_SCALE", "true"
)

# TODO make it true by default when the DeepEP PR is merged
MOE_NVFP4_DISPATCH = envs.SGLANG_MOE_NVFP4_DISPATCH.get()
# Supported activation schemes for the current configuration
ACTIVATION_SCHEMES = ["static"]

ACT_STR_TO_TYPE_MAP = {
    "silu": ActivationType.Swiglu,  # This is the default
    "relu2": ActivationType.Relu2,
}

logger = logging.getLogger(__name__)


class ModelOptFp4Config(ModelOptQuantConfig):
    """Config class for FP4."""

    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool = False,
        kv_cache_quant_algo: str = None,
        group_size: int = None,
        exclude_modules: List[str] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__(kv_cache_quant_algo, exclude_modules, packed_modules_mapping)
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected nvfp4 checkpoint. Please note that the "
                "format is experimental and subject to change."
            )
        self.group_size = group_size

    @classmethod
    def override_quantization_method(cls, hf_quant_config, user_quant):
        """Override quantization method based on the model's config."""
        return cls._modelopt_override_quantization_method(hf_quant_config, user_quant)

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_fp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @staticmethod
    def common_group_size(cfg: dict) -> int:
        """Return the unique group_size across the config; raise if missing/mismatched."""
        sizes = set()

        # Top-level and 'quantization' block
        v = cfg.get("group_size")
        if isinstance(v, int):
            sizes.add(v)
        q = cfg.get("quantization")
        if isinstance(q, dict):
            v = q.get("group_size")
            if isinstance(v, int):
                sizes.add(v)

        # config_groups: accept group-level or nested dicts (e.g., weights/input_activations)
        for g in (cfg.get("config_groups") or {}).values():
            if isinstance(g, dict):
                v = g.get("group_size")
                if isinstance(v, int):
                    sizes.add(v)
                for sub in g.values():
                    if isinstance(sub, dict):
                        v = sub.get("group_size")
                        if isinstance(v, int):
                            sizes.add(v)

        if not sizes:
            raise ValueError("No group_size found in config.")
        if len(sizes) > 1:
            raise ValueError(f"Inconsistent group_size values: {sorted(sizes)}")
        return next(iter(sizes))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ModelOptFp4Config:
        # Handle two different config formats:
        # 1. hf_quant_config.json format: {"quantization": {"quant_algo": "NVFP4", ...}}
        # 2. config.json quantization_config format: {"quant_algo": "NVFP4", ...}
        # In future modelopt will deprecate hf_quant_config.json, and only keep config.json.
        # For legacy reasons, we keep hf_quant_config.json for now.

        # Initialize variables
        kv_cache_quant_algo = None
        group_size = None
        exclude_modules = []

        # Try flat format first (config.json quantization_config - preferred format)
        quant_method = config.get("quant_algo")
        if quant_method is not None:
            # Flat format (config.json quantization_config)
            # Derive kv_cache_quant_algo from kv_cache_scheme dict
            kv_cache_scheme = config.get("kv_cache_scheme")
            if isinstance(kv_cache_scheme, dict):
                if (
                    kv_cache_scheme.get("type") == "float"
                    and kv_cache_scheme.get("num_bits") == 8
                ):
                    kv_cache_quant_algo = "FP8"
                else:
                    kv_cache_quant_algo = "auto"
            elif isinstance(kv_cache_scheme, str):
                scheme_name = kv_cache_scheme.strip().upper()
                if scheme_name in ("FP8", "FLOAT8"):
                    kv_cache_quant_algo = "FP8"
                elif scheme_name in ("FP4", "FLOAT4", "NVFP4"):
                    kv_cache_quant_algo = "NVFP4"
                else:
                    kv_cache_quant_algo = "auto"
            else:
                kv_cache_quant_algo = "auto"

            group_size = config.get("group_size")
            # If group_size is not at top level, try to extract from config_groups
            if group_size is None:
                config_groups = config.get("config_groups", {})
                if config_groups:
                    # Get group_size from the first group's weights config
                    first_group = next(iter(config_groups.values()), {})
                    weights_config = first_group.get("weights", {})
                    group_size = weights_config.get("group_size")

            exclude_modules = config.get("ignore", [])
        else:
            # Fall back to nested format (hf_quant_config.json - legacy format)
            try:
                quant_config = cls.get_from_keys(config, ["quantization"])
                quant_method = quant_config["quant_algo"]
                kv_cache_quant_algo = quant_config.get("kv_cache_quant_algo")
                if not kv_cache_quant_algo:
                    kv_cache_quant_algo = "auto"
                group_size = ModelOptFp4Config.common_group_size(config)
                exclude_modules = quant_config.get("exclude_modules", [])
            except (ValueError, KeyError):
                raise ValueError(
                    "Cannot find 'quant_algo' in the model's quantization config. "
                    "Expected either flat format (config.json) or nested format (hf_quant_config.json)."
                )

        if not quant_method in ["FP8", "NVFP4"]:
            raise ValueError(
                f"ModelOpt currently only supports: FP8, NVFP4"
                " quantizations in sglang. Please check the "
                "quantization config for your model's configuration."
            )
        is_checkpoint_nvfp4_serialized = "NVFP4" in quant_method

        if group_size is None or exclude_modules is None:
            logger.warning(
                f"group_size: {group_size},"
                f"kv_cache_quant_algo: {kv_cache_quant_algo},"
                f"exclude_modules: {exclude_modules}"
            )
            raise ValueError(
                "NVFP4 quantization requires group_size and exclude_modules "
                "specified in the quantization config"
            )
        return cls(
            is_checkpoint_nvfp4_serialized,
            kv_cache_quant_algo,
            group_size,
            exclude_modules,
            config.get("packed_modules_mapping"),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return self._get_quant_method(
            layer,
            prefix,
            Linear=ModelOptFp4LinearMethod,
            Moe=ModelOptNvFp4FusedMoEMethod,
        )


class ModelOptFp4LinearMethod(LinearMethodBase):
    """Linear method for NVFP4.
    Supports loading NVFP4 checkpoints with the following structure:

    |Tensor Name           | datatype      |  shape      |
    |----------------------------------------------------|
    |input_scale           | torch.float32 | scalar      |
    |weight                | NVFP4(SE2M1)  | [1, X, y/2] |
    |weight_scale          | FP8-E4M3      | [X, Y]      |
    |weight_scale_2        | torch.float32 | scalar      |

    The weights are quantized per block of 16 elements.
    Args: quant_config: The ModelOpt quantization config.
    """

    def __init__(self, quant_config: ModelOptFp4Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        if not self.quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError(
                "NVFP4 quantization was selected, "
                " dynamic quantization is not supported."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        if input_size_per_partition % 16 != 0:
            raise ValueError(
                "Unsupported model when in features size is " "not multiple of 16"
            )

        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_nvfp4_serialized
            else params_dtype
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                # 2 fp4 data is packed in one uint8 in the input dimension
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        input_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )

        layer.register_parameter("input_scale", input_scale)

        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)

        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        input_scale_2 = layer.input_scale.max().to(torch.float32)
        weight_scale_2 = layer.weight_scale_2.max().to(torch.float32)

        # Keep per-shard scales intact for hot reload; derive scalar params below.
        copy_or_rebind_param(
            layer, "alpha", (input_scale_2 * weight_scale_2).to(torch.float32)
        )
        copy_or_rebind_param(
            layer, "input_scale_inv", (1 / input_scale_2).to(torch.float32)
        )

        # Store original output size before any padding
        layer.output_size_per_partition = layer.weight.shape[0]

        if get_fp4_gemm_runner_backend().is_flashinfer_trtllm():
            # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
            # FlashInfer provides nvfp4_quantize to quantize + shuffle the
            # layout but we use our own quantization so we have to call
            # shuffles ourselves.
            #
            # Alignment requirements:
            #   - shuffle_matrix_a: weight.shape[0] (N) % 32 == 0
            #   - shuffle_matrix_sf_a: scale.shape[0] (N) % 128 == 0, scale.shape[1] (K/16) % 4 == 0
            # We pad N to multiple of 128 and K/16 to multiple of 4.
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            # Pad weight N dimension to 128
            weight, _ = pad_nvfp4_weight(
                layer.weight.data, n_alignment=128, k_alignment=0
            )
            # Pad scale N dimension to match weight
            scale = layer.weight_scale
            if scale.shape[0] != weight.shape[0]:
                pad_n = weight.shape[0] - scale.shape[0]
                scale = torch.nn.functional.pad(scale, (0, 0, 0, pad_n))

            # Pad K dimension: scale K/16 must be multiple of 4
            scale_k = scale.shape[1]  # K/16
            weights_padding_cols = 0
            if scale_k % 4 != 0:
                padded_scale_k = round_up_to_multiple(scale_k, 4)
                pad_scale_k = padded_scale_k - scale_k
                # Pad scale K/16 dimension
                scale = torch.nn.functional.pad(scale, (0, pad_scale_k, 0, 0))
                # Pad weight K/2 dimension correspondingly (K/2 = K/16 * 8)
                pad_weight_k = pad_scale_k * 8
                weight = torch.nn.functional.pad(weight, (0, pad_weight_k, 0, 0))
                # Store K padding for activation padding in apply()
                weights_padding_cols = pad_weight_k

            # Shuffle for TRTLLM layout
            epilogue_tile_m = 128
            shuffled_scale_shape = scale.shape
            weight = shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m)
            scale = (
                shuffle_matrix_sf_a(scale.view(torch.uint8), epilogue_tile_m)
                .reshape(shuffled_scale_shape)
                .view(torch.float8_e4m3fn)
            )

            copy_or_rebind_param(layer, "weight_scale_interleaved", scale)
            copy_or_rebind_param(layer, "weight", weight)
            layer.weights_padding_cols = weights_padding_cols
            return

        # Pad weights for CUTLASS/FlashInfer kernel alignment (K and N divisible by 32)
        weight, weights_padding_cols = pad_nvfp4_weight(layer.weight.data)
        layer.weights_padding_cols = weights_padding_cols
        copy_or_rebind_param(layer, "weight", weight)

        # Pad and blockwise interleave weight_scale
        scales = layer.weight_scale
        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        assert scales.ndim == 3
        B, M, K = scales.shape
        M_padded = round_up_to_multiple(M, 128)
        K_padded = round_up_to_multiple(K, 4)
        padded_scales = torch.zeros((B, M_padded, K_padded), dtype=scales.dtype)
        padded_scales[:B, :M, :K] = scales
        batches, rows, cols = padded_scales.shape
        assert rows % 128 == 0
        assert cols % 4 == 0
        padded_scales = padded_scales.reshape(batches, rows // 128, 4, 32, cols // 4, 4)
        padded_scales = padded_scales.permute((0, 1, 4, 3, 2, 5))
        padded_scales = padded_scales.contiguous().cuda()
        padded_scales = (
            padded_scales.reshape(M_padded, K_padded)
            if scale_ndim == 2
            else padded_scales.reshape(B, M_padded, K_padded)
        )
        copy_or_rebind_param(layer, "weight_scale_interleaved", padded_scales)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_dtype = x.dtype
        x_m, _ = x.shape

        # Get original output size (before padding) and padded weight size
        output_size = layer.output_size_per_partition
        w_n, _ = layer.weight.shape
        output_shape = [x_m, output_size]

        # Quantize BF16 or FP16 to (FP4 and interleaved block scale)
        x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)

        assert x_fp4.dtype == torch.uint8
        assert layer.weight.dtype == torch.uint8
        assert layer.weight_scale_interleaved.dtype == torch.float8_e4m3fn
        assert layer.alpha.dtype == torch.float32

        # Pad activations to match weight K-dimension padding
        weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
        x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

        w = layer.weight
        w_scale_interleaved = layer.weight_scale_interleaved
        if (
            enable_flashinfer_fp4_gemm
            and not get_fp4_gemm_runner_backend().is_cutlass()
        ):
            w = layer.weight.T
            w_scale_interleaved = layer.weight_scale_interleaved.T

        out = fp4_gemm(
            x_fp4,
            w,
            x_scale_interleaved,
            w_scale_interleaved,
            layer.alpha,
            output_dtype,
            w_n,
        )

        # Slice output to remove N-dimension padding
        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class ModelOptNvFp4FusedMoEMethod(FusedMoEMethodBase):
    """
       MoE Method for FP4 Quantization with Blockscales and PerTensorScales
    Args:
        quant_config: NVFP4 Quant Config
    """

    def __init__(self, quant_config: ModelOptFp4Config):
        self.quant_config = quant_config
        if not is_blackwell_supported():
            raise ValueError(
                "Current platform does not support NVFP4"
                " quantization. Please use Blackwell and"
                " above."
            )
        self.enable_flashinfer_trtllm_moe = (
            get_moe_runner_backend().is_flashinfer_trtllm()
        )
        self._cache_permute_indices = {}

    @property
    def enable_flashinfer_cutlass_moe(self) -> bool:
        from sglang.srt.layers.moe import get_moe_runner_backend

        """Access the global enable_flashinfer_cutlass_moe setting."""
        return get_moe_runner_backend().is_flashinfer_cutlass()

    @property
    def enable_flashinfer_cutedsl_moe(self) -> bool:
        from sglang.srt.layers.moe import get_moe_runner_backend

        """Access the global enable_flashinfer_cutedsl_moe setting."""
        return get_moe_runner_backend().is_flashinfer_cutedsl()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if not self.quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError(
                "NVFP4 quantization was selected, "
                " dynamic quantization is not supported."
            )

        # TODO(ch-wan): check if this is needed
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.params_dtype = params_dtype
        layer.quant_config = self.quant_config

        weight_dtype = torch.uint8
        weight_scale_dtype = torch.float8_e4m3fn
        weight_loader = extra_weight_attrs.get("weight_loader")
        # GEMM 1
        num_shards = 2 if layer.moe_runner_config.is_gated else 1

        w13_weight = ModelWeightParameter(
            data=torch.empty(
                layer.num_local_experts,
                num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight", w13_weight)

        # GEMM 2
        w2_weight = ModelWeightParameter(
            data=torch.empty(
                layer.num_local_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight", w2_weight)

        w13_weight_scale = ModelWeightParameter(
            data=torch.empty(
                layer.num_local_experts,
                num_shards * intermediate_size_per_partition,
                hidden_size // self.quant_config.group_size,
                dtype=weight_scale_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        # Only use `swizzle_blockscale` for shapes, not for real content
        layer.w13_blockscale_swizzled = Parameter(
            swizzle_blockscale(layer.w13_weight_scale), requires_grad=False
        )

        w2_weight_scale = ModelWeightParameter(
            data=torch.empty(
                layer.num_local_experts,
                hidden_size,
                intermediate_size_per_partition // self.quant_config.group_size,
                dtype=weight_scale_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        layer.w2_blockscale_swizzled = Parameter(
            swizzle_blockscale(layer.w2_weight_scale), requires_grad=False
        )

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )

        w13_weight_scale_shape = (
            (layer.num_local_experts, 2)
            if layer.moe_runner_config.is_gated
            else (layer.num_local_experts,)
        )
        w13_weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(w13_weight_scale_shape, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)

        w2_weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(layer.num_local_experts, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )

        w13_input_scale_shape = (layer.num_experts, num_shards)
        w13_input_scale = PerTensorScaleParameter(
            data=torch.empty(w13_input_scale_shape, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        w13_input_scale._sglang_require_global_experts = True
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = PerTensorScaleParameter(
            data=torch.empty(layer.num_experts, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        w2_input_scale._sglang_require_global_experts = True
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process FP4 MoE weights after loading from serialized checkpoint.

        Only supports pre-quantized checkpoints with FP8 weights and scales.
        """

        # GEMM 1 scale processing
        if layer.moe_runner_config.is_gated:
            if layer.w13_weight_scale_2.dim() == 1:
                # Some checkpoints store a shared scale for w1/w3.
                w13_weight_scale_2 = layer.w13_weight_scale_2
            else:
                if layer.w13_weight_scale_2.shape[1] >= 2 and not torch.allclose(
                    layer.w13_weight_scale_2[:, 0],
                    layer.w13_weight_scale_2[:, 1],
                ):
                    logger.warning_once(
                        "w1_weight_scale_2 must match w3_weight_scale_2. "
                        "Accuracy may be affected."
                    )

                w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0]
        else:
            w13_weight_scale_2 = layer.w13_weight_scale_2[:]

        # Calculate input scales based on strategy
        if self.enable_flashinfer_cutlass_moe or self.enable_flashinfer_trtllm_moe:
            w13_input_scale = layer.w13_input_scale.max().to(torch.float32)
            w2_input_scale = layer.w2_input_scale.max().to(torch.float32)
        elif self.enable_flashinfer_cutedsl_moe:
            # All-expert-one-input-scale is mathematically different from default per-expert-input-scale
            # Thus we allow users to switch the flag to do thorough testing
            if CUTEDSL_MOE_SCALAR_INPUT_SCALE:
                w13_input_scale = (
                    layer.w13_input_scale.max()
                    .to(torch.float32)
                    .repeat(layer.w13_input_scale.shape[0])
                )
            else:
                w13_input_scale = layer.w13_input_scale.max(dim=1).values.to(
                    torch.float32
                )

            w2_input_scale = layer.w2_input_scale

            def _slice_scale(w):
                assert w.shape == (layer.num_experts,)
                assert layer.moe_ep_size * layer.num_local_experts == layer.num_experts
                return w[
                    layer.moe_ep_rank
                    * layer.num_local_experts : (layer.moe_ep_rank + 1)
                    * layer.num_local_experts
                ]

            w13_input_scale = _slice_scale(w13_input_scale)
            w2_input_scale = _slice_scale(w2_input_scale)

            if MOE_NVFP4_DISPATCH:
                assert torch.all(w13_input_scale == w13_input_scale[0])
                w13_input_scale = w13_input_scale[0]
        else:
            w13_input_scale = layer.w13_input_scale.max(dim=-1).values.to(torch.float32)
            w2_input_scale = layer.w2_input_scale

        # Create shared parameters
        copy_or_rebind_param(
            layer,
            "g1_alphas",
            (w13_input_scale * w13_weight_scale_2).to(torch.float32),
        )
        copy_or_rebind_param(
            layer,
            "g2_alphas",
            (w2_input_scale * layer.w2_weight_scale_2).to(torch.float32),
        )
        copy_or_rebind_param(
            layer,
            "w13_input_scale_quant",
            (1 / w13_input_scale).to(torch.float32),
        )
        copy_or_rebind_param(
            layer,
            "w2_input_scale_quant",
            (1 / w2_input_scale).to(torch.float32),
        )

        # TODO: for flashinfer always do MOE_NVFP4_DISPATCH
        layer.dispatcher.set_quant_config(
            {
                "input_global_scale": (
                    layer.w13_input_scale_quant
                    if MOE_NVFP4_DISPATCH
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                    else None
                )
            }
        )
        block_size = 16
        # Validate weight scales
        assert_dim = 2 if layer.moe_runner_config.is_gated else 1
        for name, weight_scale in [
            ("w13", layer.w13_weight_scale),
            ("w2", layer.w2_weight_scale),
        ]:
            # For NVFP4 TRTLLM we require one scale per 16 inputs (last dim == expected_blocks[name]).
            if get_moe_runner_backend().is_flashinfer_trtllm():
                expected_blocks = {
                    "w13": layer.w13_weight.shape[2] * 2 // block_size,
                    "w2": layer.w2_weight.shape[2] * 2 // block_size,
                }
                assert (
                    weight_scale.shape[-1] == expected_blocks[name]
                ), f"Expected {name}_weight_scale.dim(2) == {expected_blocks[name]}, got {weight_scale.shape[-1]}"
            else:
                if weight_scale.shape[assert_dim] % 4 != 0:
                    logger.warning(
                        "NVFP4 %s_weight_scale K' not multiple of 4: shape=%s, group_size=%s",
                        name,
                        tuple(weight_scale.shape),
                        getattr(self.quant_config, "group_size", None),
                    )
            assert (
                weight_scale.dtype == torch.float8_e4m3fn
            ), f"{name} Weight Blockscale must be represented as FP8-E4M3"

        # Weight processing based on strategy
        if (
            self.enable_flashinfer_trtllm_moe
            and reorder_rows_for_gated_act_gemm is not None
            and shuffle_matrix_sf_a is not None
        ):
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                align_fp4_moe_weights_for_flashinfer_trtllm,
            )

            # FlashInfer TRTLLM processing - handles both w13 and w2
            align_fp4_moe_weights_for_flashinfer_trtllm(layer)

        else:
            # CUTLASS processing - handle w13 and w2 separately

            # Process w13 weights
            w13_blockscale_swizzled = swizzle_blockscale(layer.w13_weight_scale)
            copy_or_rebind_param(
                layer, "w13_blockscale_swizzled", w13_blockscale_swizzled
            )

            w13_weight = layer.w13_weight
            intermediate_size_pad = w13_blockscale_swizzled.size(1) - w13_weight.size(1)
            if intermediate_size_pad:
                # padding gated activations will require to split w1 and w3
                # and pad them individually
                assert not layer.moe_runner_config.is_gated, (
                    "The intermediate size required padding, "
                    "but padding is also implemented for gated activations"
                )

                copy_or_rebind_param(
                    layer,
                    "w13_weight",
                    torch.nn.functional.pad(
                        w13_weight, (0, 0, 0, intermediate_size_pad)
                    ),
                )
                copy_or_rebind_param(
                    layer,
                    "w2_weight",
                    torch.nn.functional.pad(
                        layer.w2_weight, (0, intermediate_size_pad // 2, 0, 0)
                    ),
                )
                copy_or_rebind_param(
                    layer,
                    "w2_weight_scale",
                    torch.nn.functional.pad(
                        layer.w2_weight_scale, (0, intermediate_size_pad // 16)
                    ),
                )

            # Process w2 weights
            w2_blockscale_swizzled = swizzle_blockscale(layer.w2_weight_scale)
            copy_or_rebind_param(
                layer, "w2_blockscale_swizzled", w2_blockscale_swizzled
            )

            # Both flashinfer cutlass and regular cutlass use same processing for w2

            # Set up CUTLASS MoE parameters (reuse to keep CUDA graph stable)
            device = layer.w13_weight.device
            inter_size = layer.w2_weight.shape[2] * 2
            hidden_size = layer.w13_weight.shape[2] * 2
            existing_params = getattr(layer, "cutlass_moe_params", None)
            if (
                existing_params is None
                or existing_params.cutlass_moe_type != CutlassMoEType.BlockscaledFP4
                or existing_params.num_experts != layer.num_experts
                or existing_params.intermediate_size_per_partition != inter_size
                or existing_params.hidden_size != hidden_size
                or existing_params.device != device
            ):
                layer.cutlass_moe_params = CutlassMoEParams(
                    CutlassMoEType.BlockscaledFP4,
                    device,
                    num_experts=layer.num_experts,  # global num experts
                    intermediate_size_per_partition=inter_size,  # n
                    hidden_size=hidden_size,
                )  # k

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13
        return self.enable_flashinfer_cutlass_moe and self.moe_runner_config.is_gated

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if get_moe_runner_backend().is_flashinfer_trtllm():
            self.runner = MoeRunner(
                MoeRunnerBackend.FLASHINFER_TRTLLM, moe_runner_config
            )

    def apply(
        self,
        layer: FusedMoE,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        x = dispatch_output.hidden_states
        x_sf = dispatch_output.hidden_states_scale
        topk_output = dispatch_output.topk_output

        activation = self.moe_runner_config.activation

        assert (
            activation in ACT_STR_TO_TYPE_MAP
        ), f"{activation=} missing from {ACT_STR_TO_TYPE_MAP.keys()=}"
        moe_runner_config = self.moe_runner_config

        # FlashInfer TRTLLM FP4 path - layer has shuffled weights only when
        # backend is flashinfer_trtllm
        if hasattr(layer, "gemm1_weights_fp4_shuffled"):
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                FlashInferTrtllmFp4MoeQuantInfo,
            )
            from sglang.srt.layers.moe.utils import RoutingMethodType

            # Determine routing method type based on layer configuration
            routing_method_type = getattr(
                layer, "routing_method_type", RoutingMethodType.Default
            )

            quant_info = FlashInferTrtllmFp4MoeQuantInfo(
                gemm1_weights_fp4_shuffled=layer.gemm1_weights_fp4_shuffled.data,
                gemm2_weights_fp4_shuffled=layer.gemm2_weights_fp4_shuffled.data,
                gemm1_scales_fp4_shuffled=layer.gemm1_scales_fp4_shuffled.data,
                gemm2_scales_fp4_shuffled=layer.gemm2_scales_fp4_shuffled.data,
                g1_scale_c=layer.g1_scale_c.data,
                g1_alphas=layer.g1_alphas.data,
                g2_alphas=layer.g2_alphas.data,
                w13_input_scale_quant=layer.w13_input_scale_quant,
                global_num_experts=layer.num_experts,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                local_num_experts=layer.num_local_experts,
                intermediate_size_per_partition=layer.intermediate_size_per_partition,
                routing_method_type=routing_method_type,
            )

            return self.runner.run(dispatch_output, quant_info)

        if self.enable_flashinfer_cutlass_moe:
            from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

            assert (
                not moe_runner_config.apply_router_weight_on_input
            ), "apply_router_weight_on_input is not supported for Flashinfer"
            # TRTLLM Cutlass moe takes in activations in BF16/Half/nvfp4 precision
            # and fp4 quantized weights loaded from the checkpoint
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

            output_dtype = torch.bfloat16

            if DispatchOutputChecker.format_is_flashinfer(dispatch_output):
                symm_output = dispatch_output.moe_output
            else:
                # If x_sf is not None, x is FP4 packed (half size), so we need * 2
                # If x_sf is None, x is not packed, so output_col = x.shape[1]
                output_col = x.shape[1]
                if x_sf is not None and layer.moe_runner_config.is_gated:
                    output_col *= 2
                with use_symmetric_memory(
                    get_tp_group(), disabled=not is_allocation_symmetric()
                ):
                    symm_output = torch.empty(
                        x.shape[0],
                        output_col,
                        dtype=output_dtype,
                        device=x.device,
                    )

            output = flashinfer_cutlass_fused_moe(
                output=symm_output,
                input=x,
                token_selected_experts=topk_ids.to(torch.int),
                token_final_scales=topk_weights,
                fc1_expert_weights=layer.w13_weight.view(torch.long),
                fc2_expert_weights=layer.w2_weight.view(torch.long),
                output_dtype=output_dtype,
                input_sf=x_sf,
                # swizzled_input_sf=not get_moe_a2a_backend().is_flashinfer(),
                quant_scales=[
                    layer.w13_input_scale_quant,
                    layer.w13_blockscale_swizzled.view(torch.int32),
                    layer.g1_alphas,
                    layer.w2_input_scale_quant,
                    layer.w2_blockscale_swizzled.view(torch.int32),
                    layer.g2_alphas,
                ],
                ep_size=layer.moe_ep_size,
                ep_rank=layer.moe_ep_rank,
                tp_size=layer.moe_tp_size,
                tp_rank=layer.moe_tp_rank,
                tune_max_num_tokens=next_power_of_2(x.shape[0]),
                activation_type=ACT_STR_TO_TYPE_MAP[activation],
                enable_alltoall=get_moe_a2a_backend().is_flashinfer(),
            )[0]

            from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

            return StandardCombineInput(hidden_states=output)

        from sglang.srt.layers.moe.cutlass_moe import cutlass_moe_fp4

        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        output = cutlass_moe_fp4(
            a=x,
            a1_gscale=layer.w13_input_scale_quant,
            w1_fp4=layer.w13_weight,
            w1_blockscale=layer.w13_blockscale_swizzled,
            w1_alphas=layer.g1_alphas,
            a2_gscale=layer.w2_input_scale_quant,
            w2_fp4=layer.w2_weight,
            w2_blockscale=layer.w2_blockscale_swizzled,
            w2_alphas=layer.g2_alphas,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            params=layer.cutlass_moe_params,
            apply_router_weight_on_input=moe_runner_config.apply_router_weight_on_input,
        ).to(x.dtype)
        # Scale by routed_scaling_factor is fused into select_experts.
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        return StandardCombineInput(hidden_states=output)

    def apply_without_routing_weights(
        self,
        layer: FusedMoE,
        x: tuple[torch.Tensor, Optional[torch.Tensor]],
        masked_m: torch.Tensor,
        moe_runner_config: MoeRunnerConfig,
    ) -> torch.Tensor:
        assert (
            moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        assert self.enable_flashinfer_cutedsl_moe, "only support flashinfer cutedsl moe"
        assert (
            not moe_runner_config.apply_router_weight_on_input
        ), "apply_router_weight_on_input is not supported for Flashinfer"

        from sglang.srt.layers.moe.flashinfer_cutedsl_moe import (
            flashinfer_cutedsl_moe_masked,
        )

        down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = getattr(
            layer, "down_gemm_overlap_args", None
        )

        out = flashinfer_cutedsl_moe_masked(
            hidden_states=x,
            input_global_scale=(
                None if MOE_NVFP4_DISPATCH else layer.w13_input_scale_quant
            ),
            w1=layer.w13_weight,
            w1_blockscale=layer.w13_blockscale_swizzled,
            w1_alpha=layer.g1_alphas,
            w2=layer.w2_weight,
            a2_global_scale=layer.w2_input_scale_quant,
            w2_blockscale=layer.w2_blockscale_swizzled,
            w2_alpha=layer.g2_alphas,
            masked_m=masked_m,
            **(
                dict(
                    down_sm_count=down_gemm_overlap_args.num_sms,
                    down_signals=down_gemm_overlap_args.signal,
                    down_start_event=down_gemm_overlap_args.start_event,
                )
                if down_gemm_overlap_args is not None
                else {}
            ),
        )
        return out

import logging
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from sglang.srt.layers.linear import LinearBase, set_weight_attrs
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.utils import replace_parameter
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()

try:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
    from vllm.model_executor.layers.quantization.gptq_marlin import (
        FusedMoE,
        FusedMoEMethodBase,
        FusedMoeWeightScaleSupported,
        GPTQMarlinLinearMethod,
        marlin_moe_permute_scales,
    )
    from vllm.model_executor.layers.quantization.marlin import MarlinLinearMethod
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        check_marlin_supported,
    )
    from vllm.scalar_type import scalar_types

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    GPTQLinearMethod = MarlinLinearMethod = Any

    FusedMoEMethodBase = QuantizeMethodBase

    class scalar_types:
        uint4b8 = "uint4b8"
        uint8b128 = "uint8b128"


logger = logging.getLogger(__name__)


def check_marlin_format(hf_quant_cfg: Dict[str, Any]) -> bool:
    # compat: gptqmodel and autogptq (eol) main use checkpoint_format: str
    # compat: autogptq <=0.7.1 is_marlin_format: bool
    return hf_quant_cfg.get("checkpoint_format") == "marlin" or hf_quant_cfg.get(
        "is_marlin_format", False
    )


class GPTQConfig(QuantizationConfig):
    """Config class for GPTQ.

    Reference: https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        lm_head_quantized: bool,
        dynamic: Dict[str, Dict[str, Union[int, bool]]],
    ) -> None:
        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # Format is Dict[str, Dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # Negative matching will skip quantization init for this module
        # entirely:
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # Example:
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix
        #  # positive match layer 10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers
        # }
        super().__init__()
        self.dynamic = dynamic

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.pack_factor = Fraction(32, self.weight_bits)
        if self.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ, but got {self.weight_bits} bits."
            )

    def __repr__(self) -> str:
        return (
            f"GPTQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}),"
            f"lm_head_quantized={self.lm_head_quantized}), "
            f"dynamic={self.dynamic}"
        )

    def get_scaled_act_names(self) -> List[str]:
        """Returns the activation function names that should be post-scaled.

        For now, this is only used by AWQ.
        """
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQConfig":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic

        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        return cls(weight_bits, group_size, desc_act, lm_head_quantized, dynamic)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[GPTQLinearMethod]:
        # Delay the import to avoid circular dependency
        from sglang.srt.layers.quantization import get_linear_quant_method

        return get_linear_quant_method(self, layer, prefix, GPTQLinearMethod)


class GPTQMarlinConfig(QuantizationConfig):
    """Config class for GPTQ Marlin"""

    # (num_bits, is_sym) -> quant_type
    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        is_sym: bool,
        lm_head_quantized: bool,
        dynamic: Dict[str, Dict[str, Union[int, bool]]],
        full_config: Dict[str, Any],
    ) -> None:
        super().__init__()
        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # Format is Dict[str, Dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # Negative matching will skip quantization init for this module
        # entirely:
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # Example:
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix
        #  # positive match layer 10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers
        # }
        self.dynamic = dynamic

        self.weight_bits = weight_bits
        self.is_sym = is_sym

        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.full_config = full_config

        if (weight_bits, is_sym) not in self.TYPE_MAP:
            raise ValueError(
                "Unsupported quantization config: " f"bits={weight_bits}, sym={is_sym}"
            )

        # (num_bits, is_sym) -> quant_type
        self.quant_type = self.TYPE_MAP[(weight_bits, is_sym)]

    def __repr__(self) -> str:
        return (
            f"GPTQMarlinConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}, "
            f"lm_head_quantized={self.lm_head_quantized}), "
            f"dynamic={self.dynamic}"
        )

    def get_scaled_act_names(self) -> List[str]:
        """Returns the activation function names that should be post-scaled.

        For now, this is only used by AWQ.
        """
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        return "gptq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQMarlinConfig":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic

        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        is_sym = cls.get_from_keys(config, ["sym"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        return cls(
            weight_bits,
            group_size,
            desc_act,
            is_sym,
            lm_head_quantized,
            dynamic,
            config,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        is_marlin_format = check_marlin_format(hf_quant_cfg)

        can_convert = cls.is_gptq_marlin_compatible(hf_quant_cfg)

        is_valid_user_quant = (
            user_quant is None or user_quant == "marlin" or user_quant == "gptq_marlin"
        )

        if not is_marlin_format and can_convert and is_valid_user_quant:
            msg = (
                "The model is convertible to {} during runtime."
                " Using {} kernel.".format(cls.get_name(), cls.get_name())
            )
            logger.info(msg)
            return cls.get_name()

        if not is_marlin_format and can_convert and user_quant == "gptq":
            logger.info(
                "Detected that the model can run with gptq_marlin"
                ", however you specified quantization=gptq explicitly,"
                " so forcing gptq. Use quantization=gptq_marlin for"
                " faster inference"
            )
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        # Delay the import to avoid circular dependency
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization import get_linear_quant_method

        if isinstance(layer, FusedMoE):
            return GPTQMarlinMoEMethod(self)
            # TODO: re-enable after SGLang syncs with vllm >= 0.7.3
            # if layer.num_experts > 32:
            #     # For MoEs with many experts the moe_wna16 kernel is faster
            #     return MoeWNA16Config.from_config(self.full_config).get_quant_method(
            #         layer, prefix
            #     )
            # else:
            #     return GPTQMarlinMoEMethod(self)
        return get_linear_quant_method(self, layer, prefix, GPTQMarlinLinearMethod)

    @classmethod
    def is_gptq_marlin_compatible(cls, quant_config: Dict[str, Any]):
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        sym = quant_config.get("sym")
        desc_act = quant_config.get("desc_act")

        if not _is_cuda:
            return False

        if quant_method != "gptq":
            return False

        # Marlin conversion is only valid if required properties are found
        if num_bits is None or group_size is None or sym is None or desc_act is None:
            return False

        if (num_bits, sym) not in cls.TYPE_MAP:
            return False

        return check_marlin_supported(
            quant_type=cls.TYPE_MAP[(num_bits, sym)], group_size=group_size
        )


class MarlinConfig(QuantizationConfig):
    """Config class for Marlin.

    Reference: https://github.com/IST-DASLab/marlin/tree/master
    """

    def __init__(
        self,
        group_size: int,
        lm_head_quantized: bool,
    ) -> None:
        # Group size for the quantization.
        self.group_size = group_size
        self.lm_head_quantized = lm_head_quantized
        if self.group_size != 128 and self.group_size != -1:
            raise ValueError(
                "Currently, only group size 128 and -1 (channelwise) "
                "is supported for Marlin, but got group_size of "
                f"{self.group_size}"
            )

        # 4 Bits packed into 32 bit datatype.
        self.pack_factor = 32 // 4

        # Tile size used by marlin kernels.
        self.tile_size = 16

        # Min out_features dim
        self.min_n_threads = 64

        # Min in_features dim
        self.min_k_threads = 128

        # Max parallel problems to solve at once (improves large
        # batch performance)
        self.max_parallel = 16

        # Permutation length used by the marlin kernels.
        self.perm_len = 1024

    def __repr__(self) -> str:
        return (
            f"MarlinConfig(group_size={self.group_size}, "
            f"lm_head_quantized={self.lm_head_quantized})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MarlinConfig":
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        return cls(group_size, lm_head_quantized)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        is_marlin_format = check_marlin_format(hf_quant_cfg)

        is_valid_user_quant = (
            user_quant is None or user_quant == "gptq" or user_quant == "marlin"
        )

        if is_marlin_format and is_valid_user_quant:
            msg = "The model is serialized in {} format. Using {} kernel.".format(
                cls.get_name(), cls.get_name()
            )
            logger.info(msg)
            return cls.get_name()

        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[MarlinLinearMethod]:
        # Delay the import to avoid circular dependency
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            return MarlinLinearMethod(self)
        return None


class GPTQMarlinMoEMethod(FusedMoEMethodBase):
    """MoE Marlin method with quantization."""

    def __init__(self, quant_config: GPTQMarlinConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        intermediate_size = extra_weight_attrs.pop("intermediate_size")

        self.is_k_full = (not self.quant_config.desc_act) or (
            intermediate_size_per_partition == intermediate_size
        )

        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            w2_scales_size = (
                intermediate_size
                if self.quant_config.desc_act
                else intermediate_size_per_partition
            )
            scales_size2 = w2_scales_size // self.quant_config.group_size
            strategy = FusedMoeWeightScaleSupported.GROUP.value
        else:
            scales_size13 = 1
            scales_size2 = 1
            strategy = FusedMoeWeightScaleSupported.CHANNEL.value

        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})
        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.quant_config.pack_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.quant_config.pack_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        # up_proj scales
        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition,
                dtype=torch.half,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)
        # down_proj scales
        w2_scales = torch.nn.Parameter(
            torch.empty(num_experts, scales_size2, hidden_size, dtype=torch.half),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        # dont shard the w2 scales when running act order
        set_weight_attrs(w2_scales, {"load_full_w2": self.quant_config.desc_act})
        # up_proj scales
        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)
        # down_proj scales
        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size2,
                hidden_size // self.quant_config.pack_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)
        # dont shard the w2 scales when running act order
        set_weight_attrs(w2_qzeros, {"load_full_w2": self.quant_config.desc_act})
        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)
        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)
        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)
        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        # Process act_order
        if self.quant_config.desc_act:
            # Get sorting based on g_idx
            num_experts = layer.w13_g_idx.shape[0]
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_g_idx)
            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_g_idx[e]).to(
                    torch.int32
                )
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_g_idx[e]).to(
                    torch.int32
                )
                w13_sorted_g_idx[e] = layer.w13_g_idx[e][w13_g_idx_sort_indices[e]]
                w2_sorted_g_idx[e] = layer.w2_g_idx[e][w2_g_idx_sort_indices[e]]
            replace_parameter(layer, "w13_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        else:
            # Reset g_idx related tensors
            num_experts = layer.w13_g_idx.shape[0]
            device = layer.w13_g_idx.device
            layer.w13_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
        # Repack weights
        marlin_w13_qweight = ops.gptq_marlin_moe_repack(
            layer.w13_qweight,
            layer.w13_g_idx_sort_indices,
            layer.w13_qweight.shape[1] * self.quant_config.pack_factor,
            layer.w13_qweight.shape[2],
            self.quant_config.quant_type.size_bits,
        )
        replace_parameter(layer, "w13_qweight", marlin_w13_qweight)
        marlin_w2_qweight = ops.gptq_marlin_moe_repack(
            layer.w2_qweight,
            layer.w2_g_idx_sort_indices,
            layer.w2_qweight.shape[1] * self.quant_config.pack_factor,
            layer.w2_qweight.shape[2],
            self.quant_config.quant_type.size_bits,
        )
        replace_parameter(layer, "w2_qweight", marlin_w2_qweight)
        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_scales,
            size_k=layer.intermediate_size_per_partition,
            size_n=layer.w13_scales.shape[2],
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "w13_scales", marlin_w13_scales)
        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_scales,
            size_k=layer.w2_scales.shape[1]
            * (
                self.quant_config.group_size
                if self.quant_config.group_size != -1
                else self.quant_config.pack_factor
            ),
            size_n=layer.w2_scales.shape[2],
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "w2_scales", marlin_w2_scales)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert activation == "silu", "Only SiLU activation is supported."

        # The input must currently be float16
        orig_dtype = x.dtype
        x = x.half()

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        return torch.ops.vllm.fused_marlin_moe(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            layer.w13_scales,
            layer.w2_scales,
            router_logits,
            topk_weights,
            topk_ids,
            g_idx1=layer.w13_g_idx,
            g_idx2=layer.w2_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=self.quant_config.quant_type.size_bits,
            is_k_full=self.is_k_full,
        ).to(orig_dtype)

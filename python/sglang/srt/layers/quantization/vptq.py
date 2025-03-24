# Supports VPTQ compression, see https://arxiv.org/abs/2409.17066

import math
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.linear import (
    LinearBase,
    LinearMethodBase,
)
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
)
from sglang.srt.utils import set_weight_attrs
import vptq.libvptq as vptq_ops

_is_cuda = torch.cuda.is_available() and torch.version.cuda
if _is_cuda:
    import sgl_kernel

class MetaData:
    def __init__(self):
        self.num_codebooks = 1
        self.num_centroids = 0
        self.num_res_centroids = 0
        self.vector_len = 0
        self.group_size = 0
        self.output_size = 0

class VPTQConfig(QuantizationConfig):
    """Config class for VPTQ.

    Reference: https://github.com/microsoft/VPTQ
    """

    def __init__(
        self,
        config_for_layers: Dict[str, Dict[str, Any]],
        shared_layer_config: Dict[str, Dict[str, Any]],
    ) -> None:
        self.config_for_layers = config_for_layers
        self.shared_layer_config = shared_layer_config
        self.use_block_quant = False
        self.use_fp8_w8a8 = False
        self.activation_scheme = "static"

    def __repr__(self) -> str:
        return (
            f"VPTQConfig(config_for_layers={self.config_for_layers}, "
            f"shared_layer_config={self.shared_layer_config})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "vptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VPTQConfig":
        config_for_layers: Dict[str, Any] = {}
        shared_layer_config: Dict[str, Any] = {}
        if "config_for_layers" in config:
            config_for_layers = cls.get_from_keys(config, ["config_for_layers"])
        if "shared_layer_config" in config:
            shared_layer_config = cls.get_from_keys(config, ["shared_layer_config"])
        assert len(config_for_layers) > 0 or len(shared_layer_config) > 0, (
            "VPTQConfig must have at least one of 'config_for_layers'\
             or 'shared_layer_config'"
        )

        return cls(config_for_layers, shared_layer_config)

    def get_config_for_key(self, prefix, key):
        merged_name = ".".join([prefix, key])
        if merged_name in self.config_for_layers:
            return self.config_for_layers[merged_name]
        elif key in self.shared_layer_config:
            return self.shared_layer_config[key]
        else:
            raise ValueError(f"Cannot find config for ({prefix}, {key})")

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["VPTQLinearMethod"]:
        from sglang.srt.layers.moe.ep_moe.layer import EPMoE
        print(f'get_quant_method: {prefix}, type: {type(layer)}')
        if isinstance(layer, LinearBase):
            print(f'match linear base layer: {prefix}, type: {type(layer)}')
            linear_name = prefix.split(".")[-1]
            base_name = prefix[: prefix.rfind(".")]
            if linear_name == "qkv_proj":
                quant_config = {
                    "q_proj": self.get_config_for_key(base_name, "q_proj"),
                    "k_proj": self.get_config_for_key(base_name, "k_proj"),
                    "v_proj": self.get_config_for_key(base_name, "v_proj"),
                }
            elif linear_name == "gate_up_proj":
                quant_config = {
                    "gate_proj": self.get_config_for_key(base_name, "gate_proj"),
                    "up_proj": self.get_config_for_key(base_name, "up_proj"),
                }
            else:
                quant_config = self.get_config_for_key(base_name, linear_name)
            return VPTQLinearMethod(quant_config)
        elif isinstance(layer, EPMoE):
            print(f'match ep moe layer: {prefix}')
            return VPTQMoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

class VPTQLinearMethod(LinearMethodBase):
    """Linear method for VPTQ.

    Args:
        quant_config: The VPTQ quantization config.
    """

    def __init__(self, quant_config: Dict[str, Any]):
        self.quant_config = quant_config

    @staticmethod
    def quantized_weight_loader(
        indice_sizes, narrow_dim=1
    ):  # specific for layer.indices/weight_scale&bias
        def wrap_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: Optional[Union[str, int]] = None,
        ):
            if isinstance(loaded_shard_id, str):
                _loaded_shard_id = ["q", "k", "v"].index(loaded_shard_id)
            else:
                _loaded_shard_id = loaded_shard_id or 0

            shard_sizes = [i[1] - i[0] for i in indice_sizes]
            offset, end = indice_sizes[_loaded_shard_id]
            param_data = param.data
            if loaded_shard_id is not None:
                param_data = param_data.narrow(
                    narrow_dim,
                    sum(shard_sizes[:_loaded_shard_id]),
                    shard_sizes[_loaded_shard_id],
                )

            # split for TP
            loaded_weight = loaded_weight.narrow(narrow_dim, offset, end - offset)
            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)

        return wrap_weight_loader

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
        row_parallel_tp_size = input_size // input_size_per_partition
        col_parallel_tp_size = output_size // sum(output_partition_sizes)

        if params_dtype != torch.half and params_dtype != torch.bfloat16:
            raise ValueError("Only half and bfloat16 are currently supported by vptq")
        quant_config = self.quant_config.get("q_proj", self.quant_config)
        quant_config = quant_config.get("gate_proj", quant_config)

        num_codebooks = quant_config["group_num"]
        num_centroids = quant_config["num_centroids"][1]
        group_size = quant_config["group_size"]
        vector_len = quant_config["vector_lens"][1]
        num_res_centroids = quant_config["num_res_centroids"][1]
        enable_residual = num_res_centroids > 0
        enable_norm = quant_config["enable_norm"]
        enable_perm = quant_config["enable_perm"]
        assert not enable_perm, (
            "perm is not absorbed in this model, please process it \
by `pip install vptq && python -m vptq.tools.pre_process \
--input_path xx --output_path xx`"
        )
        assert input_size == group_size
        group_size = input_size_per_partition
        metadata = MetaData()
        metadata.num_centroids = num_centroids
        metadata.num_res_centroids = num_res_centroids
        metadata.vector_len = vector_len
        metadata.group_size = group_size
        layer.metadata = metadata

        num_linears = len(output_partition_sizes)
        orig_weight_loader = extra_weight_attrs["weight_loader"]

        if enable_norm:
            wrapped_weight_loader = VPTQLinearMethod.quantized_weight_loader(
                [
                    [
                        (
                            input_size_per_partition * tp_ind,
                            input_size_per_partition * (tp_ind + 1),
                        )
                        for num in output_partition_sizes
                    ]
                    for tp_ind in range(row_parallel_tp_size)
                ][get_tensor_model_parallel_rank() % row_parallel_tp_size],
                0,
            )
            extra_weight_attrs["weight_loader"] = wrapped_weight_loader

            extra_weight_attrs["output_dim"] = 0
            weight_scale = Parameter(
                torch.empty(input_size_per_partition * num_linears, dtype=params_dtype),
                requires_grad=False,
            )
            weight_bias = Parameter(
                torch.empty(input_size_per_partition * num_linears, dtype=params_dtype),
                requires_grad=False,
            )
            set_weight_attrs(weight_scale, extra_weight_attrs)
            set_weight_attrs(weight_bias, extra_weight_attrs)
            layer.register_parameter("weight_scale", weight_scale)
            layer.register_parameter("weight_bias", weight_bias)
            extra_weight_attrs["weight_loader"] = orig_weight_loader

        index_bits = int(math.log2(num_centroids))
        res_index_bits = int(math.log2(num_res_centroids)) if enable_residual else 0
        total_index_bits = index_bits + res_index_bits
        packed_groupsize = math.ceil(group_size * total_index_bits / 32)

        indice_sizes = [
            [
                (
                    math.floor(num * tp_ind / vector_len),
                    math.ceil(num * (tp_ind + 1) / vector_len),
                )
                for num in output_partition_sizes
            ]
            for tp_ind in range(col_parallel_tp_size)
        ]
        tp_output_offset = [
            [(num * tp_ind) % vector_len for num in output_partition_sizes]
            for tp_ind in range(col_parallel_tp_size)
        ]
        if col_parallel_tp_size > 1:
            this_rank_indice_sizes = indice_sizes[get_tensor_model_parallel_rank()]
        else:
            this_rank_indice_sizes = indice_sizes[0]
        shard_sizes = [i[1] - i[0] for i in this_rank_indice_sizes]
        num_indices = sum(shard_sizes)
        indices = Parameter(
            torch.empty(
                (num_codebooks, num_indices, packed_groupsize), dtype=torch.int32
            ),
            requires_grad=False,
        )
        if row_parallel_tp_size == 1:
            wrapped_weight_loader = VPTQLinearMethod.quantized_weight_loader(
                this_rank_indice_sizes
            )
            extra_weight_attrs["weight_loader"] = wrapped_weight_loader

        set_weight_attrs(
            indices,
            {
                # metadata indicates fixed size concatenated along dim 0
                "output_partition_sizes": output_partition_sizes,
                "output_offset": tp_output_offset,
                "shard_sizes": shard_sizes,
                "input_dim": -1,
            },
        )

        extra_weight_attrs["output_dim"] = 1
        set_weight_attrs(indices, extra_weight_attrs)
        layer.register_parameter("indices", indices)
        extra_weight_attrs["weight_loader"] = orig_weight_loader

        extra_weight_attrs.pop("output_dim")
        extra_weight_attrs["is_metadata"] = True
        centroids = torch.nn.Embedding(
            num_codebooks * num_linears, num_centroids * vector_len, dtype=params_dtype
        )
        set_weight_attrs(centroids.weight, extra_weight_attrs)
        set_weight_attrs(
            centroids.weight,
            {
                # metadata indicates fixed size concatenated along dim 0
                "codebook_sizes": [
                    num_centroids * vector_len for _ in output_partition_sizes
                ],
            },
        )
        layer.centroids = centroids
        # layer.register_parameter("centroids", centroids)
        if enable_residual:
            res_centroids = torch.nn.Embedding(
                num_codebooks * num_linears,
                num_res_centroids * vector_len,
                dtype=params_dtype,
            )
            set_weight_attrs(res_centroids.weight, extra_weight_attrs)
            # layer.register_parameter("res_centroids", res_centroids)
            layer.res_centroids = res_centroids
            set_weight_attrs(
                res_centroids.weight,
                {
                    # metadata indicates fixed size concatenated along dim 1
                    "codebook_sizes": [
                        num_res_centroids * vector_len for _ in output_partition_sizes
                    ],
                },
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight_scale = layer.weight_scale if hasattr(layer, "weight_scale") else None
        weight_bias = layer.weight_bias if hasattr(layer, "weight_bias") else None
        perm = layer.perm if hasattr(layer, "perm") else None
        indices = layer.indices
        output_partition_sizes = getattr(indices, "output_partition_sizes", [])
        centroids = layer.centroids.weight
        res_centroids = (
            layer.res_centroids.weight if hasattr(layer, "res_centroids") else None
        )

        # fall back all unoptimized formats
        return merged_dequantize_gemm(
            x,
            indices,
            centroids,
            res_centroids,
            weight_scale,
            weight_bias,
            perm,
            output_partition_sizes,
            bias,
            layer.metadata,
        )

# Handle QKV projection and gate-up projection
# we will do Q K V separately
def merged_dequantize_gemm(
    input: torch.Tensor,  #  [..., in_features]
    indices: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    res_codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    weight_scale: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    weight_bias: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    perm: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    output_partition_sizes: List[int],
    bias: Optional[torch.Tensor],
    metadata: MetaData,
) -> torch.Tensor:
    output_shape = input.shape[:-1] + (sum(output_partition_sizes),)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    indice_sizes = getattr(indices, "shard_sizes", [])
    output_extra_offsets = getattr(indices, "output_offset", [])
    num_codebooks = indices.shape[0]

    tp_rank = get_tensor_model_parallel_rank()
    input_size = input.shape[-1]
    input_offset = 0
    indice_offset = 0
    output_offset = 0
    codebooks_offset = 0

    num_linears = len(output_partition_sizes)
    for linear_idx, output_size, indice_size in zip(
        range(num_linears), output_partition_sizes, indice_sizes
    ):
        metadata.output_size = output_size
        if len(output_extra_offsets) > 1:
            metadata.output_size = (
                output_size + output_extra_offsets[tp_rank][linear_idx]
            )
        shard_output = optimized_dequantize_gemm(
            input,
            indices.narrow(1, indice_offset, indice_size),
            codebooks.narrow(0, codebooks_offset, num_codebooks),
            res_codebooks.narrow(0, codebooks_offset, num_codebooks) if res_codebooks is not None else None,
            weight_scale.narrow(0, input_offset, input_size),
            weight_bias.narrow(0, input_offset, input_size),
            perm.narrow(0, input_offset, input_size) if perm is not None else None,
            bias if bias is None else bias.narrow(0, output_offset, output_size),
            metadata,
        )

        output_slice = output.narrow(-1, output_offset, output_size)
        if tp_rank > 0 and len(output_extra_offsets) > tp_rank:
            shard_output = shard_output.narrow(
                -1, output_extra_offsets[tp_rank][linear_idx], output_size
            )
        assert output_slice.shape == shard_output.shape
        output_slice.copy_(shard_output)
        output_offset += output_size
        indice_offset += indice_size
        codebooks_offset += num_codebooks
        input_offset += input_size
    return output

# call the optimized version of the dequantized matmul
def optimized_dequantize_gemm(
    input: torch.Tensor,  #  [..., in_features]
    indices: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    res_codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    weight_scale: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    weight_bias: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    perm: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    bias: Optional[torch.Tensor],
    metadata: MetaData,
) -> torch.Tensor:
    
    codebooks = codebooks.view(
        metadata.num_codebooks, metadata.num_centroids, metadata.vector_len
    )
     
    enable_residual = False
    res_codebooks_ = None
    if res_codebooks is not None:
        enable_residual = True
        shape = (metadata.num_codebooks, metadata.num_res_centroids, metadata.vector_len)
        res_codebooks_ = res_codebooks.view(shape)
    
    residual_indices = None
    outlier_indices = None
    outlier_centroids_ = None
    enable_outlier = False
    
    enable_perm = perm is not None
    enable_norm = weight_scale is not None and weight_bias is not None

    invert_perm = None
    if enable_perm:
        invert_perm = torch.argsort(perm.view(torch.uint16).to(torch.int64))
        invert_perm = invert_perm.to(torch.uint16).view(torch.int16)
    
    in_features = input.shape[-1]
    out_features = metadata.output_size
    
    if (input.numel() // input.shape[-1] < 3):
        out = vptq_ops.quant_gemv(
            input,
            indices,
            codebooks,
            residual_indices,
            res_codebooks_,
            outlier_indices,
            outlier_centroids_,
            perm,
            weight_scale,
            weight_bias,
            bias,
            in_features,
            out_features,
        )
        return out
    else: 
        weight = vptq_ops.dequant(
            indices,
            codebooks,
            residual_indices,
            res_codebooks_,
            outlier_indices,
            outlier_centroids_,
            invert_perm,
            weight_scale,
            weight_bias,
            metadata.vector_len,
            in_features,
            out_features) 
    
        return F.linear(input, weight, bias)


class VPTQMoEMethod:
    """MoE method for VPTQ.
    Supports loading VPTQ checkpoints with static weight scale and
    dynamic activation scale.

    Limitations:
    Only support VPTQ quantization

    Args:
        quant_config: The quantization config.
    """

    def __new__(cls, *args, **kwargs):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoEMethodBase

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def moe_weight_loader(self):
        def _moe_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
        ):
            # print(f'param: {param.shape}, loaded_weight: {loaded_weight.shape}, '
            #       f'weight_name: {weight_name}, shard_id: {shard_id}, expert_id: {expert_id}')
            pass
        return _moe_weight_loader

    def create_weights(
        self,
        layer: Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        print(f'num_experts: {num_experts_per_partition}')

        # indices
        up_gate_proj_weight_indices = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition * 2, 1, 256, 3584, dtype=torch.int32
            ),
            requires_grad=False,
        )
        down_proj_weight_indices = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition, 1, 896, 1024, dtype=torch.int32
            ),
            requires_grad=False,
        )
        # centroids
        up_gate_proj_centroids = torch.nn.Embedding(
            num_experts_per_partition * 2, 524288, dtype=params_dtype
        )
        up_gate_proj_centroids.weight.requires_grad = False
        down_proj_centroids = torch.nn.Embedding(
            num_experts_per_partition, 524288, dtype=params_dtype
        )
        down_proj_centroids.weight.requires_grad = False
        
        # weight scale
        up_gate_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition * 2, 7168, dtype=params_dtype
            ),
            requires_grad=False,
        )
        down_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition, 2048, dtype=params_dtype
            ),
            requires_grad=False,
        )
        # weight bias
        up_gate_proj_weight_bias = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition * 2, 7168, dtype=params_dtype
            ),
            requires_grad=False,
        )
        down_proj_weight_bias = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition, 2048, dtype=params_dtype
            ),
            requires_grad=False,
        )
         
        extra_weight_attrs["weight_loader"] = self.moe_weight_loader()
        # indices
        layer.register_parameter("w13_indices", up_gate_proj_weight_indices)
        layer.register_parameter("w2_indices", down_proj_weight_indices)
        set_weight_attrs(up_gate_proj_weight_indices, extra_weight_attrs)
        set_weight_attrs(down_proj_weight_indices, extra_weight_attrs)
        
        # centroids
        layer.w13_centroids = up_gate_proj_centroids
        layer.w2_centroids = down_proj_centroids
        set_weight_attrs(up_gate_proj_centroids.weight, extra_weight_attrs)
        set_weight_attrs(down_proj_centroids.weight, extra_weight_attrs)
        
        # weight scale
        layer.register_parameter("w13_weight_scale", up_gate_proj_weight_scale)
        layer.register_parameter("w2_weight_scale", down_proj_weight_scale)
        set_weight_attrs(up_gate_proj_weight_scale, extra_weight_attrs)
        set_weight_attrs(down_proj_weight_scale, extra_weight_attrs)
        
        # weight bias
        layer.register_parameter("w13_weight_bias", up_gate_proj_weight_bias)
        layer.register_parameter("w2_weight_bias", down_proj_weight_bias)
        set_weight_attrs(up_gate_proj_weight_bias, extra_weight_attrs)
        set_weight_attrs(down_proj_weight_bias, extra_weight_attrs)
         
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        inplace: bool = True,
        no_combine: bool = False,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
        from sglang.srt.layers.moe.topk import select_experts

        # Expert selection
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
        )
        pass


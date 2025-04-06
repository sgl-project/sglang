# TODO: add Aapted from vllm/mllama4.py
import logging
from collections.abc import Iterable
from itertools import tee
from typing import Optional, Set, Tuple

import torch
from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, EPMoE
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.managers.schedule_batch import MultimodalInputs, global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    compute_shared_experts_fusion_weights,
    default_weight_loader,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix
from sglang.srt.utils import add_prefix
from torch import nn
from transformers import Llama4Config
from transformers import Llama4Config, Llama4VisionConfig
from transformers.image_utils import SizeDict

logger = logging.getLogger(__name__)


class Llama4ForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(
        self,
        config: Llama4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.tp_size = get_tensor_model_parallel_world_size()

        # TODO refactor and probably move this
        if global_server_args_dict.get("disable_shared_experts_fusion", False):
            self.n_share_experts_fusion = global_server_args_dict[
                "n_share_experts_fusion"
            ] = None
            logger.info("Shared experts fusion optimization is disabled.")
        elif self.n_share_experts_fusion is None:
            self.n_share_experts_fusion = global_server_args_dict[
                "n_share_experts_fusion"
            ] = self.tp_size
            logger.info(
                f"Shared experts fusion optimization is by default enabled, and n_share_experts_fusion is set to {self.tp_size}. You can tune it by setting --n_share_experts_fusion or disable it by setting --disable_shared_experts_fusion."
            )

        # Initialize the language model
        from sglang.srt.models.llama4 import Llama4ForCausalLM

        self.language_model = Llama4ForCausalLM(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        self.logits_processor = LogitsProcessor(config.text_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ) -> torch.Tensor:

        return self.language_model(input_ids, positions, forward_batch)

    def _separate_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        prefix: str,
    ) -> Tuple[Iterable[Tuple[str, torch.Tensor]], Iterable[Tuple[str, torch.Tensor]]]:
        weights1, weights2 = tee(weights, 2)

        def get_prefix_weights() -> Iterable[Tuple[str, torch.Tensor]]:
            for name, data in weights1:
                if name.startswith(prefix):
                    yield (name, data)

        def get_other_weights() -> Iterable[Tuple[str, torch.Tensor]]:
            for name, data in weights2:
                if not name.startswith(prefix):
                    yield (name, data)

        return get_prefix_weights(), get_other_weights()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        weights = compute_shared_experts_fusion_weights(
            weights,
            n_share_experts_fusion=self.n_share_experts_fusion,
            n_routed_experts=self.config.num_local_experts,
            moe_layer_ids=range(
                1,
                self.config.num_hidden_layers,
                self.config.interleave_moe_layer_step,
            ),
            suffix_list=[
                "down_proj",
                "gate_proj",
                "up_proj",
            ],
            shared_expert_name_template="language_model.model.layers.{moe_layer_id}.feed_forward.shared_expert.{suffix}.weight",
            routed_expert_name_template="language_model.model.layers.{moe_layer_id}.feed_forward.experts.{expert_index}.{suffix}",
        )

        # TODO extract `MoEImpl` computation which occurs multiple times
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        MoEImpl = (
            DeepEPMoE
            if global_server_args_dict["enable_deepep_moe"]
            else (EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE)
        )
        expert_params_mapping = MoEImpl.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
                        + (
                            self.n_share_experts_fusion
                            if self.n_share_experts_fusion is not None
                            else 0
                        ),
        )

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".shared_expert.gate_up_proj", ".shared_expert.gate_proj", 0),
            (".shared_expert.gate_up_proj", ".shared_expert.up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        num_experts = self.config.text_config.num_local_experts

        for name, loaded_weight in weights:

            if name.startswith("vision_model") or name.startswith(
                "multi_modal_projector"
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("feed_forward.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if ".experts" in name:
                    assert len(loaded_weight.shape) in {2, 3}
                    if len(loaded_weight.shape) == 2:
                        for mapping in expert_params_mapping:
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            name = name.replace(weight_name, param_name)
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            weight_loader(
                                param,
                                loaded_weight,
                                name,
                                shard_id=shard_id,
                                expert_id=expert_id,
                            )
                            break
                        else:
                            raise Exception("expert not found")
                    else:
                        if ".gate_up_proj" in name:
                            name_list = [
                                            name.replace(".experts.gate_up_proj", ".experts.w13_weight")
                                        ] * 2
                            loaded_weight_list = loaded_weight.chunk(2, dim=-1)
                            shard_id_list = ["w1", "w3"]
                        else:
                            name_list = [
                                name.replace(".experts.down_proj", ".experts.w2_weight")
                            ]
                            shard_id_list = ["w2"]
                            loaded_weight_list = [loaded_weight]
                        for name, loaded_weight, shard_id in zip(
                            name_list, loaded_weight_list, shard_id_list
                        ):
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            for expert_id in range(num_experts):
                                weight_loader(
                                    param,
                                    loaded_weight[expert_id].T,
                                    name,
                                    shard_id=shard_id,
                                    expert_id=expert_id,
                                )
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


EntryClass = Llama4ForConditionalGeneration

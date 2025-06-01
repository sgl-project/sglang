import torch
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
from tqdm import tqdm


def hack_requant_moe_weight(that, weights):
    print('hi hack_requant_moe_weight')

    weights_dict = dict(list(weights))
    del weights

    moe_layers = range(
        that.config.first_k_dense_replace,
        that.config.num_hidden_layers,
        that.config.moe_layer_freq,
    )

    module_names = [
        "down_proj",
        "gate_proj",
        "up_proj",
    ]

    for moe_layer in tqdm(moe_layers):
        for expert_index in range(that.config.n_routed_experts):
            for module_name in module_names:
                partial_name = f"model.layers.{moe_layer}.mlp.experts.{expert_index}.{module_name}"
                name_weight = partial_name + ".weight"
                name_weight_scale_inv = partial_name + ".weight_scale_inv"

                weight_new, weight_scale_inv_new = \
                    _requant_moe_weight(that, weights_dict[name_weight], weights_dict[name_weight_scale_inv])

                weights_dict[name_weight] = weight_new
                weights_dict[name_weight_scale_inv] = weight_scale_inv_new

    return list(weights_dict.items())


def _requant_moe_weight(that, weight: torch.Tensor, weight_scale_inv: torch.Tensor):
    model_dtype = torch.get_default_dtype()
    weight_block_size = that.quant_config.weight_block_size

    assert model_dtype == torch.bfloat16
    assert weight_block_size == [128, 128]

    weight_dequant = block_quant_dequant(
        weight,
        # TODO does "inv" have trouble?
        weight_scale_inv,
        weight_block_size,
        model_dtype,
    )

    return TODO, TODO

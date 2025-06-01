import torch
from tqdm import tqdm


def hack_model_load_weights(that, weights):
    weights_list = list(weights)
    weights_dict = dict(weights_list)
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
        for module_name in module_names:
            partial_name = f"model.layers.{moe_layer}.mlp.shared_experts.{module_name}"
            name_weight = partial_name + ".weight"
            name_scale_inv = partial_name + ".weight_scale_inv"
            weight = weights_dict[name_weight]
            scale_inv = weights_dict[name_scale_inv]
            weight_new, scale_inv_new = _transform_moe_weight(weight, scale_inv)
            weight[...] = weight_new
            scale_inv[...] = scale_inv_new

    return weights_list


def _transform_moe_weight(weight: torch.Tensor, scale_inv: torch.Tensor):
    return TODO, TODO

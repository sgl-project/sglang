import torch
from tqdm import tqdm


def hack_model_load_weights(that, weights):
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
                    _transform_moe_weight(weights_dict[name_weight], weights_dict[name_weight_scale_inv])

                weights_dict[name_weight] = weight_new
                weights_dict[name_weight_scale_inv] = weight_scale_inv_new

    return list(weights_dict.items())


def _transform_moe_weight(weight: torch.Tensor, weight_scale_inv: torch.Tensor):
    return TODO, TODO

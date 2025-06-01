from tqdm import tqdm


def hack_model_load_weights(that, weights):
    weights_list = list(weights)
    weights_dict = dict(weights_list)
    del weights_dict

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
            TODO

    return TODO

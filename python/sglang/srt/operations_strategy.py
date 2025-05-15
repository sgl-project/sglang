import torch


def compute_layer_operations(
    layer: torch.nn.Module,
):
    if not layer.is_layer_sparse:
        return [
            TODO,
        ]

    # Will add TBO operation orders here
    return [
        TODO,
    ]

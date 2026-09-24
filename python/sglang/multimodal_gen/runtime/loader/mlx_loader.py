# SPDX-License-Identifier: Apache-2.0

import inspect

import mlx.core as mx
import mlx.nn as nn


def create_model(model_class, config):
    fields = inspect.signature(model_class).parameters
    return model_class(**{key: value for key, value in config.items() if key in fields})


def load_quantized_weights(model, weights, quantization):
    if quantization["method"] != "affine":
        raise ValueError("MLX weight loading requires affine quantization")
    nn.quantize(
        model,
        bits=quantization["bits"],
        group_size=quantization["group_size"],
        class_predicate=lambda path, module: (
            isinstance(module, (nn.Linear, nn.Embedding))
            and f"{path}.scales" in weights
        ),
    )
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model

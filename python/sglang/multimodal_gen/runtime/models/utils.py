# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/utils.py
"""Utils for model executor."""
from typing import Any

import torch


# TODO(PY): move it elsewhere
def auto_attributes(init_func):
    """
    Decorator that automatically adds all initialization arguments as object attributes.

    Example:
        @auto_attributes
        def __init__(self, a=1, b=2):
            pass

        # This will automatically set:
        # - self.a = 1 and self.b = 2
        # - self.config.a = 1 and self.config.b = 2
    """

    def wrapper(self, *args, **kwargs):
        # Get the function signature
        import inspect

        signature = inspect.signature(init_func)
        parameters = signature.parameters

        # Get parameter names (excluding 'self')
        param_names = list(parameters.keys())[1:]

        # Bind arguments to parameters
        bound_args = signature.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        # Create config object if it doesn't exist
        if not hasattr(self, "config"):
            self.config = type("Config", (), {})()

        # Set attributes on self and self.config
        for name in param_names:
            if name in bound_args.arguments:
                value = bound_args.arguments[name]
                setattr(self, name, value)
                setattr(self.config, name, value)

        # Call the original __init__ function
        return init_func(self, *args, **kwargs)

    return wrapper


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict[str, Any] | None,
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"

        # NOTE(woosuk): During weight loading, we often do something like:
        # narrowed_tensor = param.data.narrow(0, offset, len)
        # narrowed_tensor.copy_(real_weight)
        # expecting narrowed_tensor and param.data to share the same storage.
        # However, on TPUs, narrowed_tensor will lazily propagate to the base
        # tensor, which is param.data, leading to the redundant memory usage.
        # This sometimes causes OOM errors during model loading. To avoid this,
        # we sync the param tensor after its weight loader is called.
        # TODO(woosuk): Remove this hack once we have a better solution.
        from sglang.multimodal_gen.runtime.platforms import current_platform

        if current_platform.is_tpu() and key == "weight_loader":
            value = _make_synced_weight_loader(value)
        setattr(weight, key, value)


def _make_synced_weight_loader(original_weight_loader) -> Any:

    def _synced_weight_loader(param, *args, **kwargs):
        original_weight_loader(param, *args, **kwargs)
        torch._sync(param)

    return _synced_weight_loader


def extract_layer_index(layer_name: str) -> int:
    """
    Extract the layer index from the module name.
    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError
    """
    subnames = layer_name.split(".")
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    assert len(int_vals) == 1, (
        f"layer name {layer_name} should" " only contain one integer"
    )
    return int_vals[0]


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """modulate by shift and scale

    Args:
        x (torch.Tensor): input tensor.
        shift (torch.Tensor, optional): shift tensor. Defaults to None.
        scale (torch.Tensor, optional): scale tensor. Defaults to None.

    Returns:
        torch.Tensor: the output tensor after modulate.
    """
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))  # type: ignore[union-attr]
    elif scale is None:
        return x + shift.unsqueeze(1)  # type: ignore[union-attr]
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(
            1
        )  # type: ignore[union-attr]


def pred_noise_to_pred_video(
    pred_noise: torch.Tensor,
    noise_input_latent: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: Any,
) -> torch.Tensor:
    """
    Convert predicted noise to clean latent.

    Args:
    pred_noise: the predicted noise with shape [B, C, H, W]
        where B is batch_size or batch_size * num_frames
    noise_input_latent: the noisy latent with shape [B, C, H, W],
    timestep: the timestep with shape [1] or [bs * num_frames] or [bs, num_frames]
    scheduler: the scheduler

    Returns:
        the predicted video with shape [B, C, H, W]
    """
    # If timestep is [bs, num_frames]
    if timestep.ndim == 2:
        timestep = timestep.flatten(0, 1)
        assert timestep.numel() == noise_input_latent.shape[0]
    elif timestep.ndim == 1:
        # If timestep is [1]
        if timestep.shape[0] == 1:
            timestep = timestep.expand(noise_input_latent.shape[0])
        else:
            assert timestep.numel() == noise_input_latent.shape[0]
    else:
        raise ValueError(
            f"[pred_noise_to_pred_video] Invalid timestep shape: {timestep.shape}"
        )
    # timestep shape should be [B]
    dtype = pred_noise.dtype
    device = pred_noise.device
    pred_noise = pred_noise.double().to(device)
    noise_input_latent = noise_input_latent.double().to(device)
    sigmas = scheduler.sigmas.double().to(device)
    timesteps = scheduler.timesteps.double().to(device)
    timestep_id = torch.argmin(
        (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
    )
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
    pred_video = noise_input_latent - sigma_t * pred_noise
    return pred_video.to(dtype)

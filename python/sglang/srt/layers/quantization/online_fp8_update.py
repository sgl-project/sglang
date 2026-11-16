"""Native online-FP8 update sessions with stable serving parameter storage.

A session stages complete local layers in checkpoint precision. This intentionally
uses additional BF16 memory until finalization; it does not dequantize old FP8
weights or require gate/up tensors to arrive in the same transport bucket.
"""

import torch
from torch import nn


class OnlineFP8Update:
    def __init__(self, layer: nn.Module, weight_names: tuple[str, ...]):
        self.weight_names = tuple(
            dict.fromkeys(
                (
                    *weight_names,
                    *(
                        name
                        for name, _ in layer.named_parameters(recurse=False)
                        if name == "bias" or name.endswith("_bias")
                    ),
                )
            )
        )
        # Record metadata only: do not retain an initial BF16 model copy.
        self.specs = {
            name: (
                type(param),
                param.shape,
                param.stride(),
                param.dtype,
                dict(param.__dict__),
            )
            for name, param in layer.named_parameters(recurse=False)
        }
        self.serving = None

    @property
    def active(self) -> bool:
        return self.serving is not None

    def begin(self, layer: nn.Module) -> None:
        if self.active:
            raise RuntimeError("Online FP8 weight update is already open")
        serving = dict(layer.named_parameters(recurse=False))
        staging = {}
        for name, (param_type, shape, stride, dtype, attrs) in self.specs.items():
            tensor = torch.empty_strided(
                shape, stride, dtype=dtype, device=serving[name].device
            )
            # Missing shards must never be quantized as uninitialized memory.
            tensor.fill_(float("nan"))
            param = param_type.__new__(param_type, data=tensor, requires_grad=False)
            param.__dict__.update(attrs)
            staging[name] = param
        self.serving = serving
        for name, param in staging.items():
            setattr(layer, name, param)

    def finish(self, layer: nn.Module, quantize) -> None:
        if not self.active:
            raise RuntimeError("Online FP8 weight update is not open")
        serving = self.serving
        try:
            weights = [getattr(layer, name) for name in self.weight_names]
            # All selected layers require complete finite source weights.
            # Reject empty/incomplete sessions rather than quantizing old values.
            if not all(torch.isfinite(weight).all().item() for weight in weights):
                raise ValueError(
                    "Online FP8 refresh requires complete finite layer weights"
                )
            quantize(layer)
            updated = dict(layer.named_parameters(recurse=False))
            if updated.keys() != serving.keys():
                raise RuntimeError("Online FP8 refresh changed parameter inventory")
            # Validate the entire layer before changing graph-visible storage.
            for name, previous in serving.items():
                current = updated[name]
                if (
                    previous.shape,
                    previous.stride(),
                    previous.dtype,
                    previous.device,
                ) != (
                    current.shape,
                    current.stride(),
                    current.dtype,
                    current.device,
                ):
                    raise RuntimeError(
                        f"Online FP8 refresh changed serving layout: {name}"
                    )
            with torch.no_grad():
                for name, previous in serving.items():
                    previous.copy_(updated[name])
        finally:
            for name in list(layer._parameters):
                if name not in serving:
                    delattr(layer, name)
            for name, param in serving.items():
                setattr(layer, name, param)
            self.serving = None

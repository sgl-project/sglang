# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.vaes.sana import SanaVAEConfig
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class AutoencoderDC(nn.Module):
    """Deep Compression Autoencoder wrapper with 32x spatial compression."""

    def __init__(self, config: SanaVAEConfig = None, **kwargs):
        super().__init__()
        self._config = config
        self._inner_model = None
        self._loaded_state_dict: dict[str, torch.Tensor] = {}

    def _ensure_inner_model(self, state_dict: dict[str, torch.Tensor] | None = None):
        if self._inner_model is not None:
            return

        from diffusers import AutoencoderDC as DiffusersAutoencoderDC

        device = "cpu"
        state_to_load = (
            state_dict if state_dict is not None else self._loaded_state_dict
        )
        if state_to_load:
            first_tensor = next(iter(state_to_load.values()))
            device = first_tensor.device
        hf_config = {}
        if self._config is not None:
            arch = self._config.arch_config
            for key, value in vars(arch).items():
                if key == "extra_attrs" and isinstance(value, dict):
                    for ek, ev in value.items():
                        hf_config[ek] = ev
                elif not key.startswith("_") and not callable(value):
                    hf_config[key] = value

        self._inner_model = DiffusersAutoencoderDC.from_config(hf_config)

        if state_to_load:
            missing, unexpected = self._inner_model.load_state_dict(
                state_to_load, strict=False
            )
            if missing:
                logger.warning(
                    "AutoencoderDC missing keys when loading: %d keys", len(missing)
                )
                if len(missing) > 10:
                    logger.debug("First 10 missing keys: %s", list(missing)[:10])
                else:
                    logger.debug("Missing keys: %s", list(missing))
            if unexpected:
                logger.debug(
                    "AutoencoderDC unexpected keys when loading: %d keys",
                    len(unexpected),
                )
            if state_dict is None:
                self._loaded_state_dict.clear()

        self._inner_model = self._inner_model.to(device)

    @property
    def config(self):
        if self._inner_model is not None:
            return self._inner_model.config
        return self._config

    @property
    def dtype(self):
        if self._inner_model is not None:
            return next(self._inner_model.parameters()).dtype
        return torch.float32

    @property
    def device(self):
        if self._inner_model is not None:
            return next(self._inner_model.parameters()).device
        return torch.device("cpu")

    def encode(self, x: torch.Tensor, **kwargs):
        self._ensure_inner_model()
        return self._inner_model.encode(x, **kwargs)

    def decode(self, z: torch.Tensor, **kwargs):
        self._ensure_inner_model()
        z = z.to(dtype=self.dtype)
        return self._inner_model.decode(z, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs):
        self._ensure_inner_model()
        return self._inner_model(x, **kwargs)

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Intercept load_state_dict to route weights into the inner diffusers model."""
        self._ensure_inner_model(state_dict=state_dict)

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        self._ensure_inner_model()
        return self._inner_model.state_dict(*args, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Buffer weights for deferred loading. The inner model is built lazily."""
        loaded_params: set[str] = set()
        for name, weight in weights:
            self._loaded_state_dict[name] = weight
            loaded_params.add(name)
        return loaded_params

    def to(self, *args, **kwargs):
        if self._inner_model is not None:
            self._inner_model = self._inner_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)


EntryClass = AutoencoderDC

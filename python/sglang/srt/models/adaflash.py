"""Official AdaFlash checkpoints on the upstream DFlash backbone."""

from collections.abc import Iterable

import torch
from sglang.srt.models.dflash import DFlashDraftModel
from torch import nn


class ThreshHeadTwoModel(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_dim: int = 256):
        super().__init__()
        self.down = nn.Linear(hidden_size * 2, bottleneck_dim)
        self.res = nn.Sequential(nn.Linear(bottleneck_dim, bottleneck_dim), nn.SiLU())
        self.out = nn.Linear(bottleneck_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        features = torch.cat([hidden.mean(dim=-2), hidden[:, -1]], dim=-1)
        projected = nn.functional.silu(self.down(features))
        projected = projected + self.res(projected)
        return torch.sigmoid(self.out(projected))


class DFlashAdaFlashDraftModel(DFlashDraftModel):
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        head_config = getattr(config, "dflash_config", {}) or {}
        if not head_config.get("use_thresh_head_two_model"):
            raise ValueError("AdaFlash requires use_thresh_head_two_model=true.")
        if not head_config.get("thresh_head_direct_len"):
            raise ValueError("AdaFlash requires thresh_head_direct_len=true.")
        self.thresh_head_two_model = ThreshHeadTwoModel(
            int(config.hidden_size),
            int(head_config.get("thresh_head_bottleneck_dim", 256)),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loaded_head = set()

        def checked_weights():
            for name, weight in weights:
                unprefixed = name.removeprefix("model.")
                if unprefixed.startswith("thresh_head_two_model."):
                    loaded_head.add(unprefixed)
                yield name, weight

        result = super().load_weights(checked_weights())
        required = {
            f"thresh_head_two_model.{name}"
            for name, _ in self.thresh_head_two_model.named_parameters()
        }
        if missing := required - loaded_head:
            raise ValueError(
                f"AdaFlash checkpoint is missing length-head weights: {sorted(missing)}"
            )
        return result


EntryClass = [DFlashAdaFlashDraftModel]

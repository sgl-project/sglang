from __future__ import annotations

from pathlib import Path

import msgspec
import torch


class DSparkStsCalibration(msgspec.Struct, frozen=True, omit_defaults=True):
    temperatures: list[float]
    dataset: str = ""
    num_samples: int = 0
    ece_before: list[float] = []
    ece_after: list[float] = []

    def __post_init__(self) -> None:
        if not self.temperatures:
            raise ValueError("DSparkStsCalibration requires at least one temperature.")
        for temperature in self.temperatures:
            if temperature <= 0:
                raise ValueError(
                    "DSparkStsCalibration temperatures must all be > 0, got "
                    f"{self.temperatures}."
                )

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode("utf-8")

    @classmethod
    def from_json(cls, data: str) -> DSparkStsCalibration:
        return msgspec.json.decode(data.encode("utf-8"), type=cls)


def load_sts_calibration_from_path(path: str) -> DSparkStsCalibration:
    with open(path, "r", encoding="utf-8") as f:
        return DSparkStsCalibration.from_json(f.read())


class StsDataRecorder:
    def __init__(self, *, path_stem: str, gamma: int, flush_every: int) -> None:
        self.path_stem = path_stem
        self.gamma = int(gamma)
        self.flush_every = int(flush_every)
        self._logits_buffer: list[torch.Tensor] = []
        self._prefix_mask_buffer: list[torch.Tensor] = []
        self._shard_ct = 0

    def record(
        self, *, confidence_raw: torch.Tensor, num_correct_drafts: torch.Tensor
    ) -> None:
        logits = confidence_raw.detach().to(device="cpu", dtype=torch.float32)
        positions = torch.arange(self.gamma).view(1, -1)
        counts = (
            num_correct_drafts.detach().to(device="cpu", dtype=torch.int64).view(-1, 1)
        )
        prefix_mask = (positions < counts).to(torch.float32)
        self._logits_buffer.append(logits)
        self._prefix_mask_buffer.append(prefix_mask)
        if len(self._logits_buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._logits_buffer:
            return
        shard_path = Path(f"{self.path_stem}.{self._shard_ct}.pt")
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "logits": torch.cat(self._logits_buffer, dim=0),
                "prefix_mask": torch.cat(self._prefix_mask_buffer, dim=0),
            },
            shard_path,
        )
        self._logits_buffer.clear()
        self._prefix_mask_buffer.clear()
        self._shard_ct += 1

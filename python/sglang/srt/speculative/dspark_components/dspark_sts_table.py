from __future__ import annotations

import logging

import msgspec

logger = logging.getLogger(__name__)


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

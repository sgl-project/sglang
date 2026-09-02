import json
import math
from pathlib import Path
from typing import Any, Optional

import msgspec

from sglang.srt.sampling.watermark.request import (
    WatermarkRequestConfig,
    WatermarkRequestError,
)

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


class WatermarkConfigError(ValueError):
    pass


class TextSealConfig(msgspec.Struct, frozen=True, kw_only=True):
    key_a: int
    key_b: int
    ngram: int = 1
    mixing_probability: float = 0.5

    def __repr__(self) -> str:
        return (
            "TextSealConfig(key_a=<redacted>, key_b=<redacted>, "
            f"ngram={self.ngram!r}, "
            f"mixing_probability={self.mixing_probability!r})"
        )


class WatermarkRegistry(msgspec.Struct, frozen=True, kw_only=True):
    textseal: Optional[TextSealConfig] = None

    @property
    def textseal_enabled(self) -> bool:
        return self.textseal is not None

    def resolve_request(self, request: WatermarkRequestConfig) -> TextSealConfig:
        if request.provider != "textseal":
            raise WatermarkRequestError(
                "watermark_provider_unknown", "unknown watermark provider"
            )
        if not self.textseal_enabled:
            raise WatermarkRequestError(
                "watermark_disabled", "requested watermark provider is disabled"
            )
        assert self.textseal is not None
        return self.textseal

    def __repr__(self) -> str:
        return f"WatermarkRegistry(textseal_enabled={self.textseal_enabled!r})"


def load_watermark_config(path: Optional[str]) -> WatermarkRegistry:
    if path is None:
        return WatermarkRegistry()

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise WatermarkConfigError("watermark config must be a readable regular file")

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WatermarkConfigError("failed to read watermark config JSON") from error

    return _parse_watermark_config(raw)


def _parse_watermark_config(raw: Any) -> WatermarkRegistry:
    root = _require_object(raw, "watermark config")
    _reject_unknown_fields(root, {"providers"}, "watermark config")
    providers = _require_object(root.get("providers"), "providers")
    _reject_unknown_fields(providers, {"textseal"}, "providers")

    textseal = providers.get("textseal")
    if textseal is None:
        return WatermarkRegistry()
    textseal = _require_object(textseal, "TextSeal provider")
    _reject_unknown_fields(
        textseal,
        {"enabled", "key_a", "key_b", "ngram", "mixing_probability"},
        "TextSeal provider",
    )

    enabled = textseal.get("enabled")
    if not isinstance(enabled, bool):
        raise WatermarkConfigError("TextSeal enabled must be a boolean")
    if not enabled:
        return WatermarkRegistry()

    return WatermarkRegistry(textseal=_parse_textseal_config(textseal))


def _parse_textseal_config(raw: dict[str, Any]) -> TextSealConfig:
    key_a = _parse_secret_integer(raw.get("key_a"), "key_a")
    key_b = _parse_secret_integer(raw.get("key_b"), "key_b")
    if key_a == key_b:
        raise WatermarkConfigError("TextSeal keys must be distinct")

    ngram = raw.get("ngram", 1)
    if isinstance(ngram, bool) or not isinstance(ngram, int) or not 1 <= ngram <= 10:
        raise WatermarkConfigError("TextSeal ngram must be an integer in [1, 10]")

    mixing_probability = raw.get("mixing_probability", 0.5)
    if isinstance(mixing_probability, bool) or not isinstance(
        mixing_probability, (int, float)
    ):
        raise WatermarkConfigError(
            "TextSeal mixing_probability must be a finite number in [0, 1]"
        )
    mixing_probability = float(mixing_probability)
    if not math.isfinite(mixing_probability) or not 0 <= mixing_probability <= 1:
        raise WatermarkConfigError(
            "TextSeal mixing_probability must be a finite number in [0, 1]"
        )

    return TextSealConfig(
        key_a=key_a,
        key_b=key_b,
        ngram=ngram,
        mixing_probability=mixing_probability,
    )


def _parse_secret_integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise WatermarkConfigError(f"TextSeal {field} must be a signed 64-bit integer")
    if isinstance(value, str):
        try:
            value = int(value, 10)
        except ValueError as error:
            raise WatermarkConfigError(
                f"TextSeal {field} must be a signed 64-bit integer"
            ) from error
    if not isinstance(value, int) or not _INT64_MIN <= value <= _INT64_MAX:
        raise WatermarkConfigError(f"TextSeal {field} must be a signed 64-bit integer")
    return value


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WatermarkConfigError(f"{field} must be a JSON object")
    return value


def _reject_unknown_fields(
    value: dict[str, Any], allowed: set[str], field: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise WatermarkConfigError(f"{field} contains unknown fields: {unknown!r}")

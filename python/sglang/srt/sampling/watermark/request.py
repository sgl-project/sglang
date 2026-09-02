from typing import Optional

import msgspec


class WatermarkRequestConfig(msgspec.Struct, frozen=True, kw_only=True):
    provider: str


class WatermarkRequestError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def parse_watermark_header(value: Optional[str]) -> Optional[WatermarkRequestConfig]:
    if value is None:
        return None

    value = value.strip()
    if value == "off":
        return None

    if value != "textseal":
        if value.split(";", 1)[0] not in {"off", "textseal"}:
            raise WatermarkRequestError(
                "watermark_provider_unknown", "unknown watermark provider"
            )
        raise WatermarkRequestError(
            "watermark_invalid_request", "invalid watermark request header"
        )

    return WatermarkRequestConfig(provider="textseal")

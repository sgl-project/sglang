# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from sglang.omni.protocol import OmniResponse


def serialize_response(response: OmniResponse) -> dict[str, Any]:
    payload = response.to_dict()
    for segment in payload["segments"]:
        if segment["type"] == "image":
            segment["image"] = serialize_image(segment["image"])
    return payload


def serialize_image(image: Any) -> Any:
    if image is None:
        return None
    if isinstance(image, dict):
        return image
    if isinstance(image, bytes):
        return {
            "b64_json": base64.b64encode(image).decode("ascii"),
            "mime_type": "application/octet-stream",
        }
    save = getattr(image, "save", None)
    if callable(save):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return {
            "b64_json": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "mime_type": "image/png",
        }
    return image

# SPDX-License-Identifier: Apache-2.0
import base64
import os
import re


def save_base64_image_to_path(base64_data: str, target_path: str) -> str:
    b64_format_hint = (
        "Failed to decode base64 image. "
        "Expected format: `data:[<media-type>];base64,<data>`"
    )

    match = re.match(r"data:(.*?)(;base64)?,(.*)", base64_data)
    if not match:
        raise ValueError(b64_format_hint)
    media_type = match.group(1)
    is_base64 = match.group(2)
    if not is_base64:
        raise ValueError(f"{b64_format_hint} (missing ;base64 marker)")
    data = match.group(3)
    if not data:
        raise ValueError(f"{b64_format_hint} (empty data payload)")

    if media_type.startswith("image/"):
        ext = media_type.split("/")[-1].lower()
        if ext == "jpeg":
            ext = "jpg"
    else:
        ext = "jpg"
    target_path = f"{target_path}.{ext}"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    try:
        image_data = base64.b64decode(data)
    except Exception as exc:
        raise Exception(f"Failed to decode base64 image: {str(exc)}") from exc

    with open(target_path, "wb") as f:
        f.write(image_data)

    return target_path

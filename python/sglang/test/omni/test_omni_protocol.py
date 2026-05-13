# SPDX-License-Identifier: Apache-2.0

import dataclasses
import unittest
from typing import Any

from sglang.omni.core.protocol import (
    GeneratedSegment,
    OmniContextRef,
    OmniInputSegment,
    OmniOutputSegment,
    OmniRequest,
    OmniResponse,
)

FORBIDDEN_KV_WORDS = ("allocator", "page", "slot")


class TestOmniProtocol(unittest.TestCase):
    def test_request_response_shapes_are_serializable_and_map_task_alias(self):
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="draw a cup"),),
            model="sensenova-u1",
        )
        response = OmniResponse(
            segments=(
                OmniOutputSegment.from_generated(
                    GeneratedSegment(type="image", image={"b64_json": "abc"})
                ),
            ),
            context=OmniContextRef(context_id="ctx", token_count=8),
        )

        self.assertEqual("draw a cup", request.messages[0].text)
        self.assertEqual("image", response.to_dict()["segments"][0]["type"])
        self.assertEqual(
            "edit",
            OmniRequest.from_payload(
                {
                    "task": "edit",
                    "messages": [{"type": "text", "text": "edit this image"}],
                }
            ).mode,
        )

    def test_public_shapes_do_not_expose_raw_kv_names(self):
        objects = [
            OmniRequest(messages=(OmniInputSegment(type="text", text="x"),)),
            OmniContextRef(context_id="ctx"),
            GeneratedSegment(type="text", text="ok"),
        ]

        for obj in objects:
            self.assertEqual([], _find_forbidden(dataclasses.asdict(obj)))
            self.assertEqual([], _find_forbidden(_field_names(obj)))


def _field_names(obj: Any) -> dict[str, None]:
    return {field.name: None for field in dataclasses.fields(obj)}


def _find_forbidden(value: Any) -> list[str]:
    found: list[str] = []
    _collect_forbidden(value, found=found, path="$")
    return found


def _collect_forbidden(value: Any, *, found: list[str], path: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            for word in FORBIDDEN_KV_WORDS:
                if word in lowered:
                    found.append(f"{path}.{key}")
            _collect_forbidden(nested, found=found, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _collect_forbidden(nested, found=found, path=f"{path}[{index}]")


if __name__ == "__main__":
    unittest.main()

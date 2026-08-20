"""Unit tests for msgpack_encode_spliced.

Mental model (what we exercise):

    payload = {meta..., "data": <big bytes>, nested: [...]}
                              |
        msgpack_encode_spliced(payload)
                              |
        [small-parts bytes][big bytes BY REFERENCE][small-parts bytes]
                              |
        b"".join(parts)  ==  msgspec.msgpack.encode(payload)   (byte-identical)

Covered:
1. Joined parts are byte-identical to a plain msgspec encode, across nested
   dicts/lists, >15-element containers (map16/array16 headers), and all
   scalar leaves the rollout payload uses.
2. Large bytes land in the parts list by reference (zero copy), small bytes
   stay inline.
"""

import unittest

import msgspec

from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    msgpack_encode_spliced,
)


def _sample_payload(big: bytes) -> list:
    wide_map = {f"k{i}": i for i in range(20)}
    return [
        {
            "request_id": "req-0",
            "seed": 7,
            "scale": 1.5,
            "flag": True,
            "missing": None,
            "dit_trajectory": {
                "timesteps": {"__tensor__": True, "data": b"tiny", "shape": [4]},
                "latents": {"__tensor__": True, "data": big, "shape": [2, 3]},
            },
            "wide": wide_map,
            "long_list": list(range(30)),
        }
    ]


class TestMsgpackEncodeSpliced(unittest.TestCase):
    def test_byte_identical_to_msgspec(self):
        big = bytes(range(256)) * 8192
        payload = _sample_payload(big)
        parts = msgpack_encode_spliced(payload, threshold=1 << 10)
        self.assertEqual(b"".join(parts), msgspec.msgpack.encode(payload))

    def test_large_bytes_spliced_by_reference(self):
        big = b"\x00" * (1 << 20)
        payload = _sample_payload(big)
        parts = msgpack_encode_spliced(payload, threshold=1 << 10)
        self.assertTrue(any(part is big for part in parts))
        decoded = msgspec.msgpack.decode(b"".join(parts))
        self.assertEqual(decoded[0]["dit_trajectory"]["latents"]["data"], big)
        self.assertEqual(decoded[0]["dit_trajectory"]["timesteps"]["data"], b"tiny")


if __name__ == "__main__":
    unittest.main()

"""Wire-schema conformance for the KV hint the dynamo router sends.

The dynamo router and SGLang name the same KV block two different ways, and the
seam between them is unforgiving: a mismatch does not raise, it silently makes
every hint cover zero pages, so a P2P fetch degrades to a full recompute with no
error anywhere. These tests pin the conversion.

SGLang publishes KV events carrying ``hash_str_to_int64(page_hash)`` -- the
leading 16 hex chars of the SHA256 digest, reinterpreted as a signed int64. The
router indexes that as ``ExternalSequenceBlockHash(u64)`` and echoes it back in
the hint as a bare JSON number, so ``page hash -> event int64 -> u64 -> hint ->
page key`` must land back on the same 16 hex chars the store compares against.

The hint travels inside the v0.1 KV-hint envelope (dynamo #13134, SGLang RFC
#36224) as a ``kv.source_locations@1.0`` action; ``EnvelopeTest`` covers that
outer layer, everything else covers the payload. Needs no ``kvcr`` wheel.

    python -m pytest test/registered/mem_cache/test_kvcr_router_hint_schema.py -v
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.storage.kvcr.router_hint import (
    ROUTER_HINT_KEY,
    SOURCE_LOCATIONS_ACTION_TYPE,
    SOURCE_LOCATIONS_ACTION_VERSION,
    RouterHint,
    normalize_block_hash,
    page_hash_key,
)
from sglang.srt.mem_cache.utils import hash_str_to_int64
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# A realistic SGLang page key: a full SHA256 hex digest. The leading 16 chars
# are >= 2**63 as a u64, so it exercises the signed/unsigned wrap.
_PAGE_HASH = "f" * 16 + "0123456789abcdef" * 3
# One whose leading 16 chars stay below 2**63 (no wrap), for the other branch.
_SMALL_PAGE_HASH = "0123456789abcdef" * 4


def _envelope(payload, *, action_type=None, action_version=None):
    """Wrap a kv.source_locations payload in the v0.1 KV-hint envelope."""
    return {
        "protocol_version": "0.1",
        "message_id": "2f82414c-0ab8-4b9e-a806-168d3ad8a1fd",
        "actions": [
            {
                "action_id": "src-0",
                "action_type": action_type or SOURCE_LOCATIONS_ACTION_TYPE,
                "action_version": action_version or SOURCE_LOCATIONS_ACTION_VERSION,
                "payload": payload,
            }
        ],
    }


def _extra_info(payload):
    """extra_info carrying `payload` as the envelope's one source-locations action."""
    return SimpleNamespace(extra_info={ROUTER_HINT_KEY: _envelope(payload)})


def _extra_info_raw(value):
    """extra_info carrying `value` verbatim, with no envelope wrapping."""
    return SimpleNamespace(extra_info={ROUTER_HINT_KEY: value})


class RoundTripTest(unittest.TestCase):
    """page hash -> KV event int64 -> router u64 -> back to the page key."""

    def _round_trip(self, page_hash: str) -> str:
        event_value = hash_str_to_int64(page_hash)
        # The router's wire deserializer is BlockHashValue::Signed(i64)
        # .cast_unsigned(); JSON then carries the u64. Model both steps.
        wire_value = event_value & ((1 << 64) - 1)
        return normalize_block_hash(wire_value)

    def test_negative_event_value_round_trips(self):
        """A leading-bit-set digest becomes a negative i64 and must survive."""
        self.assertLess(hash_str_to_int64(_PAGE_HASH), 0)
        self.assertEqual(self._round_trip(_PAGE_HASH), page_hash_key(_PAGE_HASH))

    def test_positive_event_value_round_trips(self):
        self.assertGreater(hash_str_to_int64(_SMALL_PAGE_HASH), 0)
        self.assertEqual(
            self._round_trip(_SMALL_PAGE_HASH), page_hash_key(_SMALL_PAGE_HASH)
        )

    def test_normalized_form_is_the_digest_prefix(self):
        """The canonical key is exactly what the event schema kept: 16 hex chars."""
        self.assertEqual(page_hash_key(_PAGE_HASH), _PAGE_HASH[:16])
        self.assertEqual(len(page_hash_key(_PAGE_HASH)), 16)


class NormalizeBlockHashTest(unittest.TestCase):
    def test_int_is_zero_padded_to_full_width(self):
        """A small u64 must not shorten the key, or it won't match a digest."""
        self.assertEqual(normalize_block_hash(1), "0000000000000001")

    def test_signed_int_wraps_into_u64(self):
        self.assertEqual(normalize_block_hash(-1), "f" * 16)

    def test_hex_string_is_truncated_and_lowercased(self):
        self.assertEqual(normalize_block_hash(_PAGE_HASH.upper()), _PAGE_HASH[:16])

    def test_uninterpretable_values_are_rejected(self):
        for value in (None, 1.5, b"abc", "", [], True):
            with self.subTest(value=value):
                self.assertIsNone(normalize_block_hash(value))


class ParseTest(unittest.TestCase):
    def test_router_number_payload_covers_the_page(self):
        """The end-to-end shape: what the router actually puts on the wire."""
        wire_value = hash_str_to_int64(_PAGE_HASH) & ((1 << 64) - 1)
        hint = RouterHint.maybe_from_extra_info(
            _extra_info(
                {
                    "source_control_endpoint": "tcp://peer:25000",
                    "block_hashes": [wire_value],
                }
            )
        )
        self.assertIsNotNone(hint)
        self.assertTrue(hint.covers(_PAGE_HASH))

    def test_hex_payload_covers_the_page(self):
        """A direct caller (tests, /generate) may send digests instead."""
        hint = RouterHint.maybe_from_extra_info(
            _extra_info(
                {
                    "source_control_endpoint": "tcp://peer:25000",
                    "block_hashes": [_PAGE_HASH],
                }
            )
        )
        self.assertTrue(hint.covers(_PAGE_HASH))

    def test_segment_key_is_covered_by_its_page(self):
        """The core matches on segment keys; the hint only names whole pages."""
        hint = RouterHint(
            source_control_endpoint="tcp://peer:25000",
            block_hashes=(page_hash_key(_PAGE_HASH),),
        )
        self.assertTrue(hint.covers(f"{_PAGE_HASH}#3"))
        self.assertFalse(hint.covers(f"{_SMALL_PAGE_HASH}#3"))

    def test_uncovered_page_is_rejected(self):
        hint = RouterHint.maybe_from_extra_info(
            _extra_info(
                {
                    "source_control_endpoint": "tcp://peer:25000",
                    "block_hashes": [_PAGE_HASH],
                }
            )
        )
        self.assertFalse(hint.covers(_SMALL_PAGE_HASH))

    def test_bad_hash_truncates_rather_than_shifting(self):
        """Hints are root-aligned, so an unreadable entry ends the prefix.

        Dropping it instead would silently renumber every block after it, and
        the store would fetch the wrong KV for a position it believes matched.
        """
        hint = RouterHint.maybe_from_extra_info(
            _extra_info(
                {
                    "source_control_endpoint": "tcp://peer:25000",
                    "block_hashes": [_PAGE_HASH, None, _SMALL_PAGE_HASH],
                }
            )
        )
        self.assertEqual(hint.block_hashes, (page_hash_key(_PAGE_HASH),))
        self.assertFalse(hint.covers(_SMALL_PAGE_HASH))

    def test_malformed_payloads_yield_no_hint(self):
        """Fail-closed: a bad hint degrades to local-only, never raises."""
        for payload in (
            None,
            "not-a-dict",
            {},
            {"block_hashes": [1]},  # no endpoint
            {"source_control_endpoint": "", "block_hashes": [1]},
            {"source_control_endpoint": "tcp://peer:1"},  # no hashes
            {"source_control_endpoint": "tcp://peer:1", "block_hashes": "abc"},
        ):
            with self.subTest(payload=payload):
                self.assertIsNone(
                    RouterHint.maybe_from_extra_info(_extra_info(payload))
                )

    def test_absent_extra_info_yields_no_hint(self):
        self.assertIsNone(RouterHint.maybe_from_extra_info(None))
        self.assertIsNone(
            RouterHint.maybe_from_extra_info(SimpleNamespace(extra_info=None))
        )
        self.assertIsNone(
            RouterHint.maybe_from_extra_info(SimpleNamespace(extra_info={}))
        )


class EnvelopeTest(unittest.TestCase):
    """The v0.1 envelope layer: which actions are read, which are stepped over."""

    _PAYLOAD = {
        "source_control_endpoint": "tcp://peer:25000",
        "block_hashes": [_PAGE_HASH],
    }

    def test_a_bare_payload_is_still_accepted(self):
        """Pre-envelope shape (dynamo #11695) must keep working until it retires."""
        hint = RouterHint.maybe_from_extra_info(_extra_info_raw(self._PAYLOAD))
        self.assertIsNotNone(hint)
        self.assertTrue(hint.covers(_PAGE_HASH))

    def test_an_unimplemented_action_does_not_suppress_ours(self):
        """Actions are independent: one we ignore must not hide one we implement.

        An envelope is a list, and a router is free to add actions for other
        consumers. Scanning only the first entry would make our fetch depend on
        the router's action ordering.
        """
        envelope = _envelope(self._PAYLOAD)
        envelope["actions"].insert(
            0,
            {
                "action_id": "demote-0",
                "action_type": "kv.demote",
                "action_version": "1.0",
                "payload": {"session_id": "agent-42"},
            },
        )
        hint = RouterHint.maybe_from_extra_info(_extra_info_raw(envelope))
        self.assertIsNotNone(hint)
        self.assertTrue(hint.covers(_PAGE_HASH))

    def test_a_newer_action_version_is_skipped_not_misparsed(self):
        """A future payload shape read against this schema would misread it."""
        self.assertIsNone(
            RouterHint.maybe_from_extra_info(
                _extra_info_raw(_envelope(self._PAYLOAD, action_version="2.0"))
            )
        )

    def test_an_unknown_envelope_version_still_yields_the_action(self):
        """Actions carry their own version, so the envelope's does not gate them."""
        envelope = _envelope(self._PAYLOAD)
        envelope["protocol_version"] = "0.2"
        hint = RouterHint.maybe_from_extra_info(_extra_info_raw(envelope))
        self.assertIsNotNone(hint)
        self.assertTrue(hint.covers(_PAGE_HASH))

    def test_malformed_envelopes_yield_no_hint(self):
        """Fail-closed at the envelope layer too."""
        for envelope in (
            {"protocol_version": "0.1", "actions": "not-a-list"},
            {"protocol_version": "0.1", "actions": ["not-a-dict"]},
            {"protocol_version": "0.1", "actions": [{}]},
            _envelope(self._PAYLOAD, action_type="kv.deref"),
            _envelope(None),
            _envelope("not-a-dict"),
        ):
            with self.subTest(envelope=envelope):
                self.assertIsNone(
                    RouterHint.maybe_from_extra_info(_extra_info_raw(envelope))
                )


class ExtraInfoKeyAgreementTest(unittest.TestCase):
    def test_the_controller_and_the_backend_name_the_same_key(self):
        """The producer and consumer of the hint hold separate literals.

        ``cache_controller`` writes the hint under its own constant so the
        generic controller does not import a backend, and the KVCR backend
        reads it under its own. That layering is deliberate, but it means a
        rename on one side is invisible to the other: every hint would simply
        stop being found, no fetch would be issued, and the only symptom is
        that P2P quietly stops working. Nothing else pins the two together.
        """
        from sglang.srt.managers.cache_controller import _ROUTER_HINT_EXTRA_INFO_KEY

        self.assertEqual(_ROUTER_HINT_EXTRA_INFO_KEY, ROUTER_HINT_KEY)


if __name__ == "__main__":
    unittest.main()

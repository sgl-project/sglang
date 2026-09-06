"""Wire-schema conformance for the KV hint the dynamo router sends.

The dynamo router and SGLang name the same KV block two different ways, and a
mismatch does not raise -- it silently makes every hint cover zero pages, so a
P2P fetch degrades to a full recompute with no error anywhere.

SGLang publishes KV events carrying ``hash_str_to_int64(page_hash)``: the
leading 16 hex chars of the SHA256 digest as a signed int64. The router indexes
it as ``ExternalSequenceBlockHash(u64)`` and echoes it back as a bare JSON
number, so ``page hash -> event int64 -> u64 -> hint -> page key`` must land
back on the same 16 hex chars the store compares against.

The hint travels inside the v0.1 KV-hint envelope (dynamo #13134, SGLang RFC
#36224) as a ``kv.source_locations@1.0`` action; ``EnvelopeTest`` covers that
outer layer. Needs no ``kvcr`` wheel.

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
    StrKeyAdapter,
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
_U64_MASK = (1 << 64) - 1


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


class CoreHandoffTest(unittest.TestCase):
    """The last leg: what crosses submit_hint, and what decode() answers.

    KVCR parses the hint itself and tests ``KeyAdapter.decode(key) in hashes``,
    having validated every wire hash into ``0 <= h < 1<<64``. Both sides of that
    comparison are produced here, so a signed value on either one makes every
    hinted block miss with nothing logged.
    """

    def test_submitted_hashes_are_unsigned(self):
        hint = RouterHint.maybe_from_extra_info(
            _extra_info(
                {
                    "source_control_endpoint": "tcp://peer:25000",
                    "block_hashes": [hash_str_to_int64(_PAGE_HASH) & _U64_MASK],
                }
            )
        )
        submitted = hint.to_kvcr_hint()["block_hashes"]
        self.assertTrue(all(0 <= h < 1 << 64 for h in submitted))
        self.assertEqual(submitted, [int(page_hash_key(_PAGE_HASH), 16)])

    def test_decode_matches_a_submitted_hash(self):
        """A segment key must decode onto the hash its own page was hinted with."""
        adapter = StrKeyAdapter()
        hint = RouterHint.maybe_from_extra_info(
            _extra_info(
                {
                    "source_control_endpoint": "tcp://peer:25000",
                    "block_hashes": [hash_str_to_int64(_PAGE_HASH) & _U64_MASK],
                }
            )
        )
        hashes = frozenset(hint.to_kvcr_hint()["block_hashes"])
        segment_key = adapter.encode(f"{_PAGE_HASH}#3")
        self.assertIn(adapter.decode(segment_key), hashes)

    def test_decode_rejects_an_unhinted_page(self):
        adapter = StrKeyAdapter()
        hint = RouterHint.maybe_from_extra_info(
            _extra_info(
                {
                    "source_control_endpoint": "tcp://peer:25000",
                    "block_hashes": [hash_str_to_int64(_PAGE_HASH) & _U64_MASK],
                }
            )
        )
        hashes = frozenset(hint.to_kvcr_hint()["block_hashes"])
        other = adapter.encode(f"{_SMALL_PAGE_HASH}#0")
        self.assertNotIn(adapter.decode(other), hashes)


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

    def test_bad_hash_truncates_rather_than_shifting(self):
        """Hints are root-aligned, so an unreadable entry ends the prefix.

        Dropping it instead would renumber every block after it, and the store would
        fetch the wrong KV for a position it believes matched.
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
        """Actions are independent, and a router is free to add ones for other
        consumers. Scanning only the first entry would make our fetch depend on the
        router's action ordering.
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

        ``cache_controller`` writes the hint under its own constant so the generic
        controller does not import a backend, and the KVCR backend reads it under its
        own. A rename on one side is invisible to the other: every hint stops being
        found, no fetch is issued, and P2P quietly stops working.
        """
        from sglang.srt.managers.cache_controller import _ROUTER_HINT_EXTRA_INFO_KEY

        self.assertEqual(_ROUTER_HINT_EXTRA_INFO_KEY, ROUTER_HINT_KEY)


if __name__ == "__main__":
    unittest.main()

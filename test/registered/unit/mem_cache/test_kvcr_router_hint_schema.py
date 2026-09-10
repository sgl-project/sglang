"""Wire-schema conformance for the KV hint the dynamo router sends.

The dynamo router and SGLang name the same KV block two different ways, and a
mismatch does not raise -- it silently makes every hint cover zero pages, so a
P2P fetch degrades to a full recompute with no error anywhere.

SGLang publishes KV events carrying ``hash_str_to_int64(page_hash)``: the
leading 16 hex chars of the SHA256 digest as a signed int64. The router indexes
it as ``ExternalSequenceBlockHash(u64)`` and echoes it back as a bare JSON
number, so ``page hash -> event int64 -> u64 -> hint -> page key`` must land
back on the same 16 hex chars the store compares against.

The hint travels inside the KV-hint envelope the request carries
(``sglang.srt.managers.kv_hints``) as a ``kv.fetch@1.0`` action; ``EnvelopeTest``
covers that outer layer and ``CoreHandoffTest`` covers the envelope handed on to
KVCR. Needs no ``kvcr`` wheel.

    python -m pytest test/registered/unit/mem_cache/test_kvcr_router_hint_schema.py -v
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from sglang.srt.managers.kv_hints import KV_HINTS_PROTOCOL_VERSION
from sglang.srt.mem_cache.storage.kvcr.router_hint import (
    KV_FETCH_ACTION_TYPE,
    KV_FETCH_ACTION_VERSION,
    ROUTER_HINT_KEY,
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
_MESSAGE_ID = "2f82414c-0ab8-4b9e-a806-168d3ad8a1fd"


def _envelope(payload, *, action_type=None, action_version=None):
    """Wrap a kv.fetch payload in the KV-hint envelope, as a plain dict."""
    return {
        "protocol_version": KV_HINTS_PROTOCOL_VERSION,
        "message_id": _MESSAGE_ID,
        "actions": [
            {
                "action_id": "fetch-0",
                "action_type": action_type or KV_FETCH_ACTION_TYPE,
                "action_version": action_version or KV_FETCH_ACTION_VERSION,
                "payload": payload,
            }
        ],
    }


def _extra_info(payload):
    """extra_info carrying `payload` as the envelope's one kv.fetch action."""
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
        wire_value = event_value & _U64_MASK
        return normalize_block_hash(wire_value)

    def test_negative_event_value_round_trips(self):
        """A leading-bit-set digest becomes a negative i64 and must survive."""
        self.assertLess(hash_str_to_int64(_PAGE_HASH), 0)
        self.assertEqual(self._round_trip(_PAGE_HASH), page_hash_key(_PAGE_HASH))


class CoreHandoffTest(unittest.TestCase):
    """The last leg: what crosses submit_hint, and what decode() answers.

    KVCR re-parses the envelope itself and tests ``KeyAdapter.decode(key) in
    hashes``, having validated every wire hash into ``0 <= h < 1<<64``. Both
    sides of that comparison are produced here, so a signed value on either one
    makes every hinted block miss with nothing logged.
    """

    def _hint(self):
        return RouterHint.maybe_from_extra_info(
            _extra_info(
                {
                    "source_control_endpoint": "tcp://peer:25000",
                    "block_hashes": [hash_str_to_int64(_PAGE_HASH) & _U64_MASK],
                }
            )
        )

    def _fetch_payload(self, hint):
        envelope = hint.to_kvcr_hint()
        return envelope["actions"][0]["payload"]

    def test_submit_hint_receives_a_whole_envelope(self):
        """KVCR's parser walks `actions`; a bare payload raises on every request.

        It raises inside the core, which the store reports as a fault and then
        returns no hint -- so the wrong shape here costs every remote fetch
        while looking like an ordinary cache miss from outside.
        """
        envelope = self._hint().to_kvcr_hint()
        self.assertEqual(envelope["protocol_version"], KV_HINTS_PROTOCOL_VERSION)
        self.assertEqual(envelope["message_id"], _MESSAGE_ID)
        (action,) = envelope["actions"]
        self.assertEqual(action["action_type"], KV_FETCH_ACTION_TYPE)
        self.assertEqual(action["action_version"], KV_FETCH_ACTION_VERSION)
        self.assertEqual(action["payload"]["mode"], "copy")
        self.assertEqual(
            action["payload"]["block_hashes"], [int(page_hash_key(_PAGE_HASH), 16)]
        )

    def test_a_segment_key_decodes_onto_its_own_hinted_hash(self):
        """Both sides of KVCR's ``decode(key) in hashes`` test are produced here,
        so a signed value on either one makes every hinted block miss silently.
        """
        adapter = StrKeyAdapter()
        hashes = frozenset(self._fetch_payload(self._hint())["block_hashes"])

        self.assertIn(adapter.decode(adapter.encode(f"{_PAGE_HASH}#3")), hashes)
        self.assertNotIn(
            adapter.decode(adapter.encode(f"{_SMALL_PAGE_HASH}#0")), hashes
        )


class ParseTest(unittest.TestCase):
    def test_router_number_payload_covers_the_page(self):
        """The end-to-end shape: what the router actually puts on the wire."""
        wire_value = hash_str_to_int64(_PAGE_HASH) & _U64_MASK
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
            # KVCR rejects an empty hash set outright, so it must not get there.
            {"source_control_endpoint": "tcp://peer:1", "block_hashes": []},
        ):
            with self.subTest(payload=payload):
                self.assertIsNone(
                    RouterHint.maybe_from_extra_info(_extra_info(payload))
                )


class EnvelopeTest(unittest.TestCase):
    """The envelope layer: which actions are read, which are stepped over."""

    _PAYLOAD = {
        "source_control_endpoint": "tcp://peer:25000",
        "block_hashes": [_PAGE_HASH],
    }

    def test_the_typed_request_envelope_parses(self):
        """The request path hands down a KvHintsEnvelope, not a dict.

        The controller stashes ``Req.kv_hints`` in extra_info as-is, so this is
        the shape that actually arrives in production; the dict branch only
        serves callers that build a hint themselves.
        """
        from sglang.srt.managers.kv_hints import decode_kv_hints_envelope

        typed = decode_kv_hints_envelope(_envelope(self._PAYLOAD))
        hint = RouterHint.maybe_from_extra_info(_extra_info_raw(typed))
        self.assertIsNotNone(hint)
        self.assertTrue(hint.covers(_PAGE_HASH))
        self.assertEqual(hint.message_id, _MESSAGE_ID)

    def test_the_legacy_action_type_is_still_read(self):
        """A router that has not picked up the kv.fetch rename still works."""
        envelope = _envelope(self._PAYLOAD, action_type="kv.source_locations")
        hint = RouterHint.maybe_from_extra_info(_extra_info_raw(envelope))
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
            None,
            "not-an-envelope",
            # A bare payload: rejected at ingress by decode_kv_hints_envelope, so
            # reaching here at all means someone bypassed the request path.
            self._PAYLOAD,
            {"protocol_version": "0.1", "message_id": "m", "actions": "not-a-list"},
            {"protocol_version": "0.1", "message_id": "m", "actions": ["not-a-dict"]},
            {"protocol_version": "0.1", "message_id": "m", "actions": [{}]},
            _envelope(self._PAYLOAD, action_type="kv.deref"),
            _envelope(self._PAYLOAD, action_version="2.0"),
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
        from sglang.srt.managers.cache_controller import _KV_HINTS_EXTRA_INFO_KEY

        self.assertEqual(_KV_HINTS_EXTRA_INFO_KEY, ROUTER_HINT_KEY)


if __name__ == "__main__":
    unittest.main()

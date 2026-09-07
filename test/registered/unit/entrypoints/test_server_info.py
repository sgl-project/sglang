"""Endpoint-level tests for `/server_info`.

`/server_info` is the introspection surface that external consumers
(SGLang's own deprecated `/get_server_info` alias, monitoring tools,
KV-aware routers) scrape to learn about the running server's
configuration. New `/server_info` behaviours should add their test
classes to this file as the surface grows.

Current coverage:

* `TestServerInfoKvEventsField` — the `kv_events` publisher descriptor
  surfaced by `_build_kv_events_block`. Covers the full input matrix
  end-to-end (happy path / disabled / malformed JSON / inproc endpoint /
  port edge cases / missing-or-non-positive page_size) because the
  helper has no separate test target; the handler is its only caller.

* `TestServerInfoExistingFieldsPreserved` — regression guard that no
  field existing consumers depend on is silently dropped: every
  `ServerArgs` dataclass field, `internal_states`, `version`, and the
  pre-existing flat `kv_events_config` string all remain visible.
"""

import asyncio
import dataclasses
import json
import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.validation_hook import check_load_publish_args
from sglang.srt.entrypoints import http_server
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


def _stub_tokenizer_manager(
    server_args: ServerArgs, get_internal_state=None
) -> TokenizerManager:
    """A manager carrying the state `/server_info` and its writers read.

    `__new__` skips `__init__`, which would open the ZMQ sockets and start
    the handle loop.
    """
    tokenizer_manager = TokenizerManager.__new__(TokenizerManager)
    tokenizer_manager.server_args = server_args
    tokenizer_manager.model_path = server_args.model_path
    tokenizer_manager.served_model_name = server_args.served_model_name
    tokenizer_manager.startup_time = None
    tokenizer_manager.get_internal_state = get_internal_state
    return tokenizer_manager


def _call_server_info_with(
    server_args: ServerArgs,
    internal_states: list[dict] | None = None,
    config_updates: dict | None = None,
) -> dict:
    """Invoke `http_server.server_info()` against a stub global state.

    Bypasses the FastAPI HTTP layer (no TestClient): the handler is an
    `async def` that reads module-level `_global_state`, so wiring a
    `SimpleNamespace` stub via `set_global_state` and awaiting the
    coroutine directly is enough to exercise the handler logic without
    booting a model server.

    `config_updates` are applied the way production applies them -- through
    `record_config_updates`, in a process that has published -- rather than
    planted on the stub. The endpoint answers the record and does not read that
    log, so what the real writer buys is that an assertion on the record is not
    satisfied for free by an update that never landed.
    """

    async def _fake_internal_state():
        return internal_states or [{"max_req_input_len": 1024}]

    tokenizer_manager = _stub_tokenizer_manager(server_args, _fake_internal_state)
    stub_state = SimpleNamespace(
        tokenizer_manager=tokenizer_manager,
        scheduler_info={"max_req_input_len": 1024},
    )
    prior_state = http_server.get_global_state()
    http_server.set_global_state(stub_state)
    published = False
    if config_updates:
        # The writer runs in a published process, as it does in production.
        publish(server_args, role="tokenizer")
        published = True
        tokenizer_manager.record_config_updates("test", **config_updates)
    try:
        return asyncio.run(http_server.server_info())
    finally:
        # Restore so a later test in the same process isn't surprised.
        http_server._global_state = prior_state
        if published:
            reset_context()


class TestServerInfoKvEventsField(CustomTestCase):
    """The new `kv_events` field is wired correctly across the full
    `_build_kv_events_block` input matrix.
    """

    # ----- happy path --------------------------------------------------

    def test_kv_events_key_present_when_publishing_enabled(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=(
                '{"publisher": "zmq", "endpoint": "tcp://*:5557", "topic": "kv"}'
            ),
            load_publish_endpoint="auto",
            page_size=64,
            dp_size=2,
        )

        info = _call_server_info_with(args)

        self.assertIn("kv_events", info)
        self.assertEqual(
            info["kv_events"],
            {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "kv",
                "block_size": 64,
                "dp_size": 2,
                # Load range packed immediately after the KV range: 5557 + dp_size.
                "load_endpoint_port_base": 5559,
                "load_topic": "load",
            },
        )

    def test_load_port_skips_an_overlapping_replay_range(self):
        # Conventional replay = kv + 1: the load range must be advertised
        # past the replay ROUTER range (5558 + dp_size), matching where
        # SchedulerLoadPublisher actually binds — both sides resolve it via
        # resolve_load_pub_range.
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=(
                '{"publisher": "zmq", "endpoint": "tcp://*:5557", '
                '"replay_endpoint": "tcp://*:5558"}'
            ),
            load_publish_endpoint="auto",
            page_size=64,
            dp_size=2,
        )

        info = _call_server_info_with(args)

        self.assertEqual(info["kv_events"]["load_endpoint_port_base"], 5560)
        self.assertEqual(info["kv_events"]["load_topic"], "load")

    def test_explicit_load_publish_endpoint_moves_the_advertised_base(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config='{"publisher": "zmq", "endpoint": "tcp://*:5557"}',
            load_publish_endpoint="tcp://*:7000",
            page_size=64,
            dp_size=2,
        )

        info = _call_server_info_with(args)

        self.assertEqual(info["kv_events"]["load_endpoint_port_base"], 7000)
        self.assertEqual(info["kv_events"]["load_topic"], "load")

    def test_load_keys_omitted_for_connect_style_kv_endpoint(self):
        # A concrete host is connected to rather than bound: the KV-events
        # descriptor is still valid (KV events work connect-style), but no
        # load range can be bound there, so the load keys must be omitted
        # rather than advertising a port nothing listens on.
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=(
                '{"publisher": "zmq", "endpoint": "tcp://10.0.0.5:5557"}'
            ),
            load_publish_endpoint="auto",
            page_size=64,
            dp_size=1,
        )

        info = _call_server_info_with(args)

        self.assertIsNotNone(info["kv_events"])
        self.assertEqual(info["kv_events"]["endpoint_host"], "10.0.0.5")
        self.assertNotIn("load_endpoint_port_base", info["kv_events"])
        self.assertNotIn("load_topic", info["kv_events"])

    def test_ipv6_wildcard_endpoint_advertises_bracketed_host_and_load_keys(self):
        # "[::]" is a bind-all wildcard: the descriptor must keep the
        # brackets (consumers splice tcp://{host}:{port}) and the load
        # range resolves right after the KV range.
        args = ServerArgs(
            model_path="dummy",
            kv_events_config='{"publisher": "zmq", "endpoint": "tcp://[::]:5557"}',
            load_publish_endpoint="auto",
            page_size=64,
            dp_size=1,
        )

        info = _call_server_info_with(args)

        self.assertEqual(info["kv_events"]["endpoint_host"], "[::]")
        self.assertEqual(info["kv_events"]["endpoint_port_base"], 5557)
        self.assertEqual(info["kv_events"]["load_endpoint_port_base"], 5558)

    def test_concrete_ipv6_endpoint_advertises_kv_but_not_load(self):
        # A concrete IPv6 host works connect-style for KV events (advertised,
        # brackets kept) but is not bindable for the load range — and "::"
        # appearing inside the address must not be mistaken for a wildcard.
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=(
                '{"publisher": "zmq", "endpoint": "tcp://[2001:db8::5]:5557"}'
            ),
            load_publish_endpoint="auto",
            page_size=64,
            dp_size=1,
        )

        info = _call_server_info_with(args)

        self.assertEqual(info["kv_events"]["endpoint_host"], "[2001:db8::5]")
        self.assertNotIn("load_endpoint_port_base", info["kv_events"])

    def test_load_keys_omitted_when_explicitly_off(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config='{"publisher": "zmq", "endpoint": "tcp://*:5557"}',
            load_publish_endpoint="off",
            page_size=64,
            dp_size=1,
        )

        info = _call_server_info_with(args)

        self.assertIsNotNone(info["kv_events"])
        self.assertNotIn("load_endpoint_port_base", info["kv_events"])
        self.assertNotIn("load_topic", info["kv_events"])

    def test_load_keys_omitted_when_no_load_range_fits(self):
        # kv base 65535 leaves no u16 room for a load range: the kv_events
        # descriptor must still be served, with only the load keys omitted,
        # so routers fall back to their in-flight counter for load.
        args = ServerArgs(
            model_path="dummy",
            kv_events_config='{"publisher": "zmq", "endpoint": "tcp://*:65535"}',
            load_publish_endpoint="auto",
            page_size=64,
            dp_size=1,
        )

        info = _call_server_info_with(args)

        self.assertIsNotNone(info["kv_events"])
        self.assertEqual(info["kv_events"]["endpoint_port_base"], 65535)
        self.assertNotIn("load_endpoint_port_base", info["kv_events"])
        self.assertNotIn("load_topic", info["kv_events"])

    def test_kv_events_descriptor_carries_specific_host_and_topic(self):
        # No --load-publish-endpoint: KV descriptor served, load keys absent.
        # Upgrade safety rests on this default silence, so pin it here.
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=(
                '{"publisher": "zmq", "endpoint": "tcp://0.0.0.0:7777", "topic": "kv"}'
            ),
            page_size=128,
            dp_size=1,
        )

        info = _call_server_info_with(args)

        self.assertIsNotNone(info["kv_events"])
        self.assertEqual(info["kv_events"]["endpoint_host"], "0.0.0.0")
        self.assertEqual(info["kv_events"]["endpoint_port_base"], 7777)
        self.assertEqual(info["kv_events"]["topic"], "kv")
        self.assertEqual(info["kv_events"]["block_size"], 128)
        self.assertEqual(info["kv_events"]["dp_size"], 1)
        self.assertNotIn("load_endpoint_port_base", info["kv_events"])
        self.assertNotIn("load_topic", info["kv_events"])

    # ----- disabled / unconfigured -------------------------------------

    def test_kv_events_is_null_when_no_publisher_configured(self):
        args = ServerArgs(model_path="dummy")  # no --kv-events-config

        info = _call_server_info_with(args)

        # The key must still be present so consumers can detect
        # "publishing disabled" via a single shape check.
        self.assertIn("kv_events", info)
        self.assertIsNone(info["kv_events"])

    def test_kv_events_is_null_when_publisher_explicitly_null(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config='{"publisher": "null"}',
            page_size=64,
        )

        info = _call_server_info_with(args)

        self.assertIsNone(info["kv_events"])

    # ----- malformed config --------------------------------------------

    def test_kv_events_is_null_for_malformed_json(self):
        # Not JSON — the publisher would have failed at server startup,
        # but /server_info must keep working.
        args = ServerArgs(
            model_path="dummy",
            kv_events_config="not-json",
            page_size=64,
        )

        info = _call_server_info_with(args)

        self.assertIsNone(info["kv_events"])

    # ----- unreachable endpoints ---------------------------------------

    def test_kv_events_is_null_for_inproc_endpoint(self):
        # `inproc://` is not reachable across process boundaries, so the
        # descriptor must hide it from external routers.
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=(
                '{"publisher": "zmq", "endpoint": "inproc://cache", "topic": ""}'
            ),
            page_size=64,
        )

        info = _call_server_info_with(args)

        self.assertIsNone(info["kv_events"])

    def test_kv_events_is_null_when_endpoint_missing_port(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=(
                '{"publisher": "zmq", "endpoint": "tcp://0.0.0.0", "topic": ""}'
            ),
            page_size=64,
        )

        info = _call_server_info_with(args)

        self.assertIsNone(info["kv_events"])

    def test_kv_events_is_null_when_port_not_integer(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=(
                '{"publisher": "zmq", "endpoint": "tcp://0.0.0.0:abc", "topic": ""}'
            ),
            page_size=64,
        )

        info = _call_server_info_with(args)

        self.assertIsNone(info["kv_events"])

    def test_kv_events_is_null_for_port_out_of_range(self):
        # TCP ports are 1..65535; values outside the range can't bind, so
        # the descriptor refuses to advertise them rather than handing
        # subscribers a non-dialable address.
        for bad_port in (0, -1, 65536, 1_000_000):
            with self.subTest(port=bad_port):
                args = ServerArgs(
                    model_path="dummy",
                    kv_events_config=(
                        f'{{"publisher": "zmq", "endpoint": "tcp://0.0.0.0:{bad_port}", "topic": ""}}'
                    ),
                    page_size=64,
                )
                info = _call_server_info_with(args)
                self.assertIsNone(info["kv_events"])

    # ----- bad scheduler context ---------------------------------------

    def test_kv_events_is_null_when_page_size_missing_or_non_positive(self):
        # Without a real positive `page_size` the descriptor's
        # `block_size` would be a misleading placeholder; subscribers
        # would hash prompts at the wrong granularity and miss every
        # cache entry. Refuse to advertise instead.
        good_cfg = '{"publisher": "zmq", "endpoint": "tcp://*:5557", "topic": ""}'
        for bad_page_size in (None, 0, -1):
            with self.subTest(page_size=bad_page_size):
                args = ServerArgs(
                    model_path="dummy",
                    kv_events_config=good_cfg,
                    page_size=bad_page_size,
                )
                info = _call_server_info_with(args)
                self.assertIsNone(info["kv_events"])


class TestServerInfoControlPlaneUpdates(CustomTestCase):
    """/server_info answers what was asked for, not what is in effect."""

    def test_the_readback_reports_the_record_not_the_control_plane(self):
        # A runtime weight-version change is reported by /model_info and its
        # gRPC / Engine twins; this endpoint reports what the operator supplied.
        server_args = ServerArgs(model_path="dummy", weight_version="v1")
        payload = _call_server_info_with(
            server_args, config_updates={"weight_version": "v2"}
        )
        self.assertEqual(payload["weight_version"], "v1")
        self.assertEqual(server_args.weight_version, "v1")

    def test_the_recorded_update_reaches_the_readback_overlay(self):
        """The update the case above records is one `/server_info` could see.

        `resolved_config_dict` is the overlay the endpoint dropped; an update
        that never reached it would satisfy the assertion above for free.
        """
        server_args = ServerArgs(model_path="dummy", weight_version="v1")
        tokenizer_manager = _stub_tokenizer_manager(server_args)
        publish(server_args, role="tokenizer")
        try:
            tokenizer_manager.record_config_updates("test", weight_version="v2")
            self.assertEqual(tokenizer_manager.config_value("weight_version"), "v2")
            overlaid = tokenizer_manager.resolved_config_dict(
                server_args.resolved_dict()
            )
            self.assertEqual(overlaid["weight_version"], "v2")
        finally:
            reset_context()


class TestServerInfoExistingFieldsPreserved(CustomTestCase):
    """Regression guard: the new `kv_events` field is additive — none of
    the fields existing consumers depend on may be silently dropped.

    Existing `/server_info` consumers in the wild include:
      * SGLang's own deprecated `/get_server_info` (forwards to the
        same handler).
      * External monitoring tools that scrape the full ServerArgs.
      * KV-aware routers reading `kv_events_config`, `page_size`,
        `dp_size` directly to derive subscription info (this is the
        path the new `kv_events` block enriches but does not replace).
    """

    def test_every_server_args_field_appears_in_response(self):
        # `server_args.resolved_dict()` is spread into the response; asserting
        # every dataclass field surfaces is the strongest backward-compat
        # guarantee that's still implementation-agnostic.
        args = ServerArgs(model_path="dummy")

        info = _call_server_info_with(args)

        for field in dataclasses.fields(ServerArgs):
            self.assertIn(
                field.name,
                info,
                f"existing ServerArgs field '{field.name}' missing from "
                f"/server_info response — kv_events patch must not "
                f"shadow or drop ServerArgs fields",
            )

    def test_internal_states_and_version_keys_preserved(self):
        # These two top-level keys predate the kv_events patch and are
        # named individually (not spread from a dataclass), so a stray
        # edit could remove them without breaking syntax. Lock them down.
        args = ServerArgs(model_path="dummy")

        info = _call_server_info_with(args)

        self.assertIn("internal_states", info)
        self.assertIn("version", info)

    def test_kv_events_config_raw_field_still_surfaced(self):
        # The new structured `kv_events` block sits alongside the
        # pre-existing flat `kv_events_config` field (the raw CLI string
        # already on ServerArgs). Both must remain visible so
        # consumers that hand-parse the raw config keep working.
        raw_cfg = '{"publisher": "zmq", "endpoint": "tcp://*:5557", "topic": ""}'
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=raw_cfg,
            page_size=64,
            dp_size=1,
        )

        info = _call_server_info_with(args)

        self.assertIn("kv_events_config", info)
        self.assertEqual(info["kv_events_config"], raw_cfg)
        # And the new structured block is separately present:
        self.assertIn("kv_events", info)
        self.assertIsNotNone(info["kv_events"])

    def test_lora_refs_are_json_serializable_dicts(self):
        lora_ref = LoRARef(
            lora_id="lora-id",
            lora_name="adapter",
            lora_path="/tmp/adapter",
            pinned=True,
        )
        args = ServerArgs(model_path="dummy")
        args.lora_paths = [lora_ref]

        info = _call_server_info_with(
            args,
            internal_states=[{"lora_paths": [lora_ref]}],
        )

        expected = {
            "lora_id": "lora-id",
            "lora_name": "adapter",
            "lora_path": "/tmp/adapter",
            "pinned": True,
        }
        self.assertEqual(info["lora_paths"], [expected])
        self.assertEqual(info["internal_states"][0]["lora_paths"], [expected])
        json.dumps(info)


class TestLoadPublishEndpointValidation(CustomTestCase):
    """--load-publish-endpoint fails fast at the entrypoint, not silently in a
    scheduler subprocess log."""

    def test_requires_kv_events_config(self):
        args = ServerArgs(model_path="dummy", load_publish_endpoint="tcp://*:6000")
        with self.assertRaisesRegex(ValueError, "kv-events"):
            check_load_publish_args(args)

    def test_rejects_non_bindable_endpoint(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config='{"publisher": "zmq", "endpoint": "tcp://*:5557"}',
            load_publish_endpoint="tcp://10.0.0.5:6000",
        )
        with self.assertRaisesRegex(ValueError, "bindable"):
            check_load_publish_args(args)

    def test_rejects_endpoint_overlapping_the_kv_range(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config='{"publisher": "zmq", "endpoint": "tcp://*:5557"}',
            dp_size=4,
            load_publish_endpoint="tcp://*:5558",
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            check_load_publish_args(args)

    def test_rejects_null_publisher(self):
        # publisher='null' disables KV events, so there is nothing to advertise
        # through — accepting the opt-in would silently do nothing.
        args = ServerArgs(
            model_path="dummy",
            kv_events_config='{"publisher": "null"}',
            load_publish_endpoint="auto",
        )
        with self.assertRaisesRegex(ValueError, "null"):
            check_load_publish_args(args)

    def test_rejects_unparseable_kv_events_config(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config="{not json",
            load_publish_endpoint="auto",
        )
        with self.assertRaisesRegex(ValueError, "not parseable"):
            check_load_publish_args(args)

    def test_off_and_valid_endpoint_pass(self):
        for endpoint in (None, "off", "OFF", "auto", "tcp://*:6000"):
            with self.subTest(endpoint=endpoint):
                args = ServerArgs(
                    model_path="dummy",
                    kv_events_config=(
                        '{"publisher": "zmq", "endpoint": "tcp://*:5557"}'
                    ),
                    load_publish_endpoint=endpoint,
                )
                check_load_publish_args(args)  # must not raise


if __name__ == "__main__":
    unittest.main()

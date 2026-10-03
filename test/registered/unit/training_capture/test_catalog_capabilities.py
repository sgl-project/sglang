"""Catalog startup agreement must precede payload ownership and lease creation."""

import copy
import json
import tempfile
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import jsonschema
from sglang.srt.training_capture.catalog import (
    MAX_CATALOG_REQUEST_BYTES,
    MAX_CATALOG_RESPONSE_BYTES,
    CatalogConflict,
    CatalogError,
    CatalogUnavailable,
    HTTPCaptureCatalog,
)
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.resources import CaptureResources
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_utils import (
    BufferStore,
    FakeReplicateConfig,
    make_kv_spec,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestCatalogCapabilities(CustomTestCase):
    def setUp(self):
        self.catalog = TestCaptureCatalog()
        self.addCleanup(self.catalog.close)
        self.client = HTTPCaptureCatalog(self.catalog.endpoint, attempts=2)
        self.kv = make_kv_spec()

    def check(self):
        return self.client.check_compatibility(
            contract_id="maas-target-kv-top128-v1",
            kv_codec=self.kv.codec,
            store_protocol="tcp",
        )

    def config(self, root):
        return CaptureConfig(
            dataset_id="handshake-test",
            model_id="fixture",
            producer_revision="test",
            selected_layer_ids=self.kv.selected_layer_ids,
            catalog_endpoint=self.catalog.endpoint,
            journal_directory=root,
            store=StoreSetup(
                local_hostname="localhost", master_server_addr="localhost:1"
            ),
        )

    def response(self, data):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = data
        return response

    def test_selects_one_complete_contract_without_creating_capture(self):
        contract_dir = (
            Path(__file__).resolve().parents[4]
            / "mooncake-study/training-data-contract"
        )
        example = json.loads(
            (contract_dir / "catalog-capabilities.example.json").read_bytes()
        )
        schema = json.loads(
            (contract_dir / "catalog-capabilities.schema.json").read_bytes()
        )
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(example, schema)
        self.assertEqual(example, self.catalog.capabilities)
        future = copy.deepcopy(self.catalog.capabilities["contracts"][0])
        future["schema_version"] = 2
        self.catalog.capabilities["contracts"].insert(0, future)
        capabilities = self.check()
        self.assertEqual(capabilities.protocol_version, 1)
        self.assertEqual(self.catalog.capability_requests, 1)
        self.assertFalse(self.catalog.captures)
        self.assertFalse(self.catalog.publications)

    def test_incompatible_declarations_and_cross_record_matches_are_rejected(self):
        original = copy.deepcopy(self.catalog.capabilities)
        cases = [
            ("protocol_version", 2),
            ("store_protocols", ["rdma"]),
            ("hard_pin", False),
            ("retention_policy", "consume_once"),
            ("max_request_bytes", MAX_CATALOG_REQUEST_BYTES - 1),
            ("contracts", []),
        ]
        for field, value in (
            ("contract_id", "another-contract"),
            ("schema_version", 2),
            ("payload_format", "hidden_state_v1"),
            ("kv_codecs", []),
        ):
            contract = copy.deepcopy(original["contracts"][0])
            contract[field] = value
            cases.append(("contracts", [contract]))
        wrong_id, wrong_codec = [
            copy.deepcopy(original["contracts"][0]) for _ in range(2)
        ]
        wrong_id["contract_id"] = "other"
        wrong_codec["kv_codecs"] = ["unknown"]
        cases.append(("contracts", [wrong_id, wrong_codec]))
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self.catalog.capabilities = {**original, field: value}
                with self.assertRaises(CatalogConflict):
                    self.check()
        self.assertEqual(self.catalog.capability_requests, len(cases))
        self.assertFalse(self.catalog.captures)

    def test_malformed_or_incomplete_capabilities_cannot_use_defaults(self):
        original = copy.deepcopy(self.catalog.capabilities)
        cases = [
            {key: value for key, value in original.items() if key != missing}
            for missing in original
        ]
        cases += [
            {**original, "protocol_version": True},
            {**original, "hard_pin": "true"},
            {**original, "contracts": [None]},
            {**original, "max_request_bytes": -1},
            {**original, "unknown": True},
        ]
        for value in cases:
            with self.subTest(value=value):
                self.catalog.capabilities = value
                with self.assertRaises(CatalogError):
                    self.check()
        self.assertFalse(self.catalog.captures)

    def test_get_retries_with_auth_and_has_no_body(self):
        response = self.response(json.dumps(self.catalog.capabilities).encode())
        self.client.bearer_token = "unit-test-credential"
        with (
            patch.object(
                self.client.opener, "open", side_effect=[TimeoutError(), response]
            ) as opened,
            patch("sglang.srt.training_capture.catalog.time.sleep"),
        ):
            self.check()
        self.assertEqual(opened.call_count, 2)
        for call in opened.call_args_list:
            request = call.args[0]
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(request.full_url, self.catalog.endpoint + "/capabilities")
            self.assertIsNone(request.data)
            self.assertEqual(
                request.get_header("Authorization"), "Bearer unit-test-credential"
            )
            self.assertEqual(request.get_header("X-training-capture-protocol"), "1")
        response.read.assert_called_once_with(MAX_CATALOG_RESPONSE_BYTES + 1)

    def test_get_budget_decode_http_errors_and_bounded_retry(self):
        for data in (b"!", b"[]", b"x" * (MAX_CATALOG_RESPONSE_BYTES + 1)):
            with (
                self.subTest(length=len(data)),
                patch.object(
                    self.client.opener, "open", return_value=self.response(data)
                ) as opened,
                self.assertRaises(CatalogError),
            ):
                self.check()
            self.assertEqual(opened.call_count, 1)
        for status in (401, 403, 404, 422, 503):
            error = urllib.error.HTTPError(
                self.catalog.endpoint, status, "fixture", {}, None
            )
            with (
                self.subTest(status=status),
                patch.object(self.client.opener, "open", side_effect=error) as opened,
                patch("sglang.srt.training_capture.catalog.time.sleep"),
                self.assertRaises(
                    CatalogUnavailable if status == 503 else CatalogError
                ),
            ):
                self.check()
            self.assertEqual(opened.call_count, 2 if status == 503 else 1)

    def test_redirect_is_not_followed(self):
        location = self.catalog.endpoint + "/capabilities"

        class Redirect(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.client = HTTPCaptureCatalog(
                f"http://127.0.0.1:{server.server_port}", bearer_token="test-token"
            )
            with self.assertRaisesRegex(CatalogError, "302"):
                self.check()
            self.assertEqual(self.catalog.capability_requests, 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())

    def test_incompatible_catalog_prevents_allocation_and_lease_creation(self):
        self.catalog.capabilities["hard_pin"] = False
        partition = plan_capture_layout(
            self.kv, tp_size=1, pp_layer_ranges=[(0, 4)]
        ).partitions[0]
        with (
            tempfile.TemporaryDirectory() as root,
            patch(
                "sglang.srt.training_capture.resources.MooncakeSnapshotStore.connect"
            ) as connect,
            patch(
                "sglang.srt.training_capture.resources.SelectedLayerKVExporter.from_pool"
            ) as exporter,
            patch("sglang.srt.training_capture.resources.HostBufferPool") as pool,
            patch(
                "sglang.srt.training_capture.resources.PublicationJournal"
            ) as journal,
            self.assertRaises(CatalogConflict),
        ):
            CaptureResources.prepare(
                config=self.config(root),
                kv=self.kv,
                partition=partition,
                source_pool=None,
                pin_memory=False,
            )
        for operation in (connect, exporter, pool, journal):
            operation.assert_not_called()
        self.assertEqual(self.catalog.capability_requests, 1)
        self.assertFalse(self.catalog.captures)

    def test_connected_rejection_closes_transport_before_any_host_allocation(self):
        self.catalog.capabilities["retention_policy"] = "consume_once"
        store = MooncakeSnapshotStore(BufferStore(), FakeReplicateConfig())
        self.addCleanup(store.close)
        with (
            tempfile.TemporaryDirectory() as root,
            patch("sglang.srt.training_capture.resources.HostBufferPool") as pool,
            self.assertRaises(CatalogConflict),
        ):
            CaptureResources.from_connected(
                config=self.config(root),
                kv=self.kv,
                exporter=None,
                store=store,
                catalog=self.client,
                pin_memory=False,
            )
        pool.assert_not_called()
        self.assertTrue(store.closed)
        self.assertFalse(store.registered)
        self.assertFalse(self.catalog.captures)


if __name__ == "__main__":
    unittest.main()

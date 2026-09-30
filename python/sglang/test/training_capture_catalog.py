"""In-memory HTTP Catalog test double, never a production retention service."""

import base64
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import msgspec

from sglang.srt.training_capture.protocol import decode_manifest, digest_bytes


class TestCaptureCatalog:
    __test__ = False

    def __init__(self):
        self.condition = threading.Condition()
        self.captures = {}
        self.publications = {}
        self.errors = []
        catalog = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    payload = json.loads(
                        self.rfile.read(int(self.headers["Content-Length"]))
                    )
                    with catalog.condition:
                        result = catalog.handle(self.path, payload)
                        catalog.condition.notify_all()
                    status = 200
                except Exception as error:
                    with catalog.condition:
                        catalog.errors.append(repr(error))
                    status, result = 409, {"error": str(error)}
                data = json.dumps(result).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def handle(self, route, payload):
        if route == "/captures:begin":
            key = payload["idempotency_key"]
            for record in self.captures.values():
                if record["begin"]["idempotency_key"] == key:
                    assert record["begin"] == payload
                    return record["lease"]
            lease = {
                name: payload[name]
                for name in ("dataset_id", "sample_id", "generation_id")
            }
            lease.update(
                capture_id=uuid.uuid4().hex,
                fencing_token=1,
                expires_in_seconds=120.0,
                renew_after_seconds=20.0,
            )
            self.captures[lease["capture_id"]] = dict(
                begin=payload, lease=lease, registered={}, written={}, state="CAPTURING"
            )
            return lease
        record = self.captures[payload["capture_id"]]
        assert record["lease"]["fencing_token"] == payload["fencing_token"]
        if route.endswith("/heartbeat"):
            return record["lease"]
        if route.endswith("/objects"):
            for obj in payload["objects"]:
                key = obj["object_id"]
                if payload["phase"] == "REGISTERED":
                    assert (
                        key not in record["registered"]
                        or record["registered"][key] == obj
                    )
                    record["registered"][key] = obj
                else:
                    assert record["registered"][key] == obj
                    record["written"][key] = obj
            return {
                "phase": payload["phase"],
                "accepted_object_ids": sorted(
                    o["object_id"] for o in payload["objects"]
                ),
            }
        if route.endswith("/seal"):
            data = base64.b64decode(payload["manifest_base64"], validate=True)
            assert digest_bytes(data) == payload["manifest"]["sha256"]
            manifest = decode_manifest(data)
            assert (
                manifest.dataset_id,
                manifest.sample_id,
                manifest.generation_id,
            ) == tuple(
                record["lease"][k] for k in ("dataset_id", "sample_id", "generation_id")
            )
            assert all(
                record["written"][o.object_id]
                == msgspec.json.decode(msgspec.json.encode(o))
                for o in manifest.objects
            ), "written descriptors must match the manifest wire representation"
            record["manifest"] = payload["manifest"]
            record["state"] = "PREPARED"
            return {"state": "PREPARED", "manifest_sha256": digest_bytes(data)}
        if route == "/samples:publish":
            descriptor = record["manifest"]
            assert record["written"]["manifest"] == descriptor
            assert payload["manifest_sha256"] == descriptor["sha256"]
            assert payload["manifest_nbytes"] == descriptor["nbytes"]
            assert payload["manifest_key"] == descriptor["key"]
            self.publications[payload["capture_id"]] = payload
            record["state"] = "AVAILABLE"
            return {
                "state": "AVAILABLE",
                "publication_id": payload["capture_id"],
                "catalog_cursor": str(len(self.publications)),
            }
        if route.endswith("/fail"):
            assert record["state"] != "AVAILABLE"
            record["state"] = "FAILED"
            record["reason"] = payload["reason"]
            return {"state": "FAILED"}
        raise ValueError("unknown test Catalog route")

    def wait_publications(self, count, timeout=30):
        deadline = time.monotonic() + timeout
        with self.condition:
            while len(self.publications) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    states = [
                        (r["state"], r.get("reason")) for r in self.captures.values()
                    ]
                    raise TimeoutError(
                        f"test Catalog publications={len(self.publications)} states={states} errors={self.errors}"
                    )
                self.condition.wait(remaining)
            return list(self.publications.values())

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

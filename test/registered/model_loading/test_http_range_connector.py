import json
import os
import struct
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from sglang.srt.connector import create_remote_connector
from sglang.srt.connector.http_range import (
    HttpRangeConnector,
    iter_safetensors,
    parse_endpoint,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


SHARDS = {
    "model-00001-of-00002.safetensors": {
        "model.embed_tokens.weight": torch.randn(128, 64).to(torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 16),
    },
    "model-00002-of-00002.safetensors": {
        "model.layers.0.self_attn.k_proj.weight": torch.randn(16, 64).to(
            torch.bfloat16
        ),
        "model.layers.0.input_layernorm.weight": torch.randn(64).to(torch.float16),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.tensor(0.5),
    },
}


class _RangeHandler(BaseHTTPRequestHandler):
    """Minimal HTTP/1.1 object server with byte ranges and keep-alive."""

    protocol_version = "HTTP/1.1"
    root = ""
    ignore_range = False
    close_every_response = False

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        path = os.path.join(self.root, unquote(self.path).lstrip("/"))
        if not os.path.isfile(path):
            self._respond(404, b"not found\n")
            return
        with open(path, "rb") as f:
            body = f.read()
        header_range = self.headers.get("Range")
        if header_range and not self.ignore_range:
            start, _, end = header_range.split("=", 1)[1].partition("-")
            first = int(start)
            last = int(end) if end else len(body) - 1
            self._respond(
                206,
                body[first : last + 1],
                {"Content-Range": f"bytes {first}-{last}/{len(body)}"},
            )
        else:
            self._respond(200, body)

    def _respond(self, status, body, extra=None):
        self.send_response(status)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        if self.close_every_response:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _Server:
    """Serves ``root`` on an ephemeral localhost port for the test duration."""

    def __init__(self, root, **handler_attrs):
        handler = type("_Handler", (_RangeHandler,), {"root": root, **handler_attrs})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.host, self.port = self.server.server_address[:2]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def url(self, prefix="/models/tiny", query=""):
        base = f"http-range://{self.host}:{self.port}{prefix}"
        return f"{base}?{query}" if query else base


def _write_model(root, with_index=True):
    model_dir = os.path.join(root, "models", "tiny")
    os.makedirs(model_dir)
    for name, tensors in SHARDS.items():
        save_file(tensors, os.path.join(model_dir, name), metadata={"format": "pt"})
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({"model_type": "qwen3"}, f)
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"model_max_length": 8}, f)
    if with_index:
        weight_map = {
            key: shard for shard, tensors in SHARDS.items() for key in tensors
        }
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f)
    return model_dir


class TestHttpRangeUrl(CustomTestCase):
    def test_parse_endpoint(self):
        self.assertEqual(parse_endpoint("host", 80), ("host", 80))
        self.assertEqual(parse_endpoint("host:9000", 80), ("host", 9000))
        self.assertEqual(parse_endpoint("[fd00::1]:9000", 80), ("fd00::1", 9000))
        self.assertEqual(parse_endpoint("[fd00::1]", 80), ("fd00::1", 80))
        # An unbracketed IPv6 literal has no port to split off.
        self.assertEqual(parse_endpoint("fd00::1", 80), ("fd00::1", 80))
        with self.assertRaises(ValueError):
            parse_endpoint("[fd00::1:9000", 80)

    def test_endpoint_per_rank(self):
        client = HttpRangeConnector(
            "http-range://h0:9000/bucket/model"
            "?endpoints=[fd00::1]:9100,[fd00::2]:9101&nics=bond0,,bond2"
        )
        try:
            self.assertEqual(client.endpoint_for_rank(0), ("fd00::1", 9100, "bond0"))
            self.assertEqual(client.endpoint_for_rank(1), ("fd00::2", 9101, None))
            # Lists wrap independently: endpoints has 2 entries, nics has 3.
            self.assertEqual(client.endpoint_for_rank(2), ("fd00::1", 9100, "bond2"))
            self.assertEqual(client.endpoint_for_rank(3), ("fd00::2", 9101, "bond0"))
        finally:
            client.close()

    def test_defaults_and_validation(self):
        client = HttpRangeConnector("http-range://h0:9000/bucket/model")
        try:
            self.assertEqual(client.endpoint_for_rank(7), ("h0", 9000, None))
            self.assertEqual(client.connections, 16)
        finally:
            client.close()
        with self.assertRaisesRegex(ValueError, "unknown http-range options"):
            HttpRangeConnector("http-range://h0:9000/m?nic=bond0")
        with self.assertRaisesRegex(ValueError, "expected a http-range"):
            HttpRangeConnector("s3://bucket/model")

    def test_created_through_factory(self):
        client = create_remote_connector("http-range://h0:9000/bucket/model")
        try:
            self.assertIsInstance(client, HttpRangeConnector)
        finally:
            client.close()


class TestHttpRangeConnector(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_dir = _write_model(cls.tmp.name)
        cls.expected = {
            name: load_file(os.path.join(cls.model_dir, name)) for name in SHARDS
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_glob_lists_shards_from_index(self):
        with _Server(self.tmp.name) as server:
            with create_remote_connector(server.url()) as client:
                self.assertEqual(
                    client.glob(),
                    [f"{server.url()}/{name}" for name in sorted(SHARDS)],
                )
                self.assertEqual(
                    client.glob(allow_pattern=["*-00002-*"]),
                    [f"{server.url()}/model-00002-of-00002.safetensors"],
                )

    def test_single_shard_without_index(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = os.path.join(root, "models", "tiny")
            os.makedirs(model_dir)
            save_file(
                SHARDS["model-00001-of-00002.safetensors"],
                os.path.join(model_dir, "model.safetensors"),
            )
            with _Server(root) as server:
                with create_remote_connector(server.url()) as client:
                    self.assertEqual(client.shard_names(), ["model.safetensors"])

    def test_no_safetensors_at_all(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "models", "tiny"))
            with _Server(root) as server:
                with create_remote_connector(server.url()) as client:
                    with self.assertRaisesRegex(FileNotFoundError, "only safetensors"):
                        client.shard_names()

    def test_pull_files_fetches_only_metadata(self):
        with _Server(self.tmp.name) as server:
            with create_remote_connector(server.url()) as client:
                client.pull_files(ignore_pattern=["*.safetensors"])
                pulled = sorted(os.listdir(client.get_local_dir()))
            self.assertEqual(
                pulled,
                [
                    "config.json",
                    "model.safetensors.index.json",
                    "tokenizer_config.json",
                ],
            )

    def test_pull_files_honours_allow_pattern(self):
        # This is the pattern ModelConfig uses to bootstrap a remote model.
        with _Server(self.tmp.name) as server:
            with create_remote_connector(server.url()) as client:
                client.pull_files(allow_pattern=["*config.json"])
                pulled = sorted(os.listdir(client.get_local_dir()))
            self.assertEqual(pulled, ["config.json", "tokenizer_config.json"])

    def test_weight_iterator_matches_local_read(self):
        expected = {
            name: tensor
            for tensors in self.expected.values()
            for name, tensor in tensors.items()
        }
        # A chunk smaller than a tensor, one straddling shard boundaries and one
        # larger than the whole object all have to produce identical bytes.
        for connections, chunk_size in ((1, 1 << 20), (4, 1024), (8, 97)):
            with self.subTest(connections=connections, chunk_size=chunk_size):
                with _Server(self.tmp.name) as server:
                    url = server.url(
                        query=f"connections={connections}&chunk_size={chunk_size}"
                    )
                    with create_remote_connector(url) as client:
                        streamed = dict(client.weight_iterator())
                self.assertEqual(sorted(streamed), sorted(expected))
                for name, tensor in expected.items():
                    self.assertEqual(streamed[name].dtype, tensor.dtype)
                    self.assertEqual(streamed[name].shape, tensor.shape)
                    self.assertTrue(
                        torch.equal(
                            streamed[name].reshape(-1).view(torch.uint8),
                            tensor.reshape(-1).view(torch.uint8),
                        ),
                        f"{name} differs",
                    )

    def test_rank_selects_its_own_endpoint(self):
        with _Server(self.tmp.name) as server:
            # Rank 1 points at a dead port: reaching it proves the rank is used
            # to pick the endpoint rather than being ignored.
            query = f"endpoints={server.host}:{server.port},{server.host}:1"
            with create_remote_connector(server.url(query=query)) as client:
                self.assertEqual(len(list(client.weight_iterator(0))), 5)
                with self.assertRaises(OSError):
                    list(client.weight_iterator(1))

    def test_missing_shard_reports_the_object(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = os.path.join(root, "models", "tiny")
            os.makedirs(model_dir)
            with open(
                os.path.join(model_dir, "model.safetensors.index.json"), "w"
            ) as f:
                json.dump({"weight_map": {"a": "gone.safetensors"}}, f)
            with _Server(root) as server:
                with create_remote_connector(server.url()) as client:
                    with self.assertRaisesRegex(FileNotFoundError, "gone.safetensors"):
                        list(client.weight_iterator())

    def test_server_ignoring_range_is_rejected(self):
        with _Server(self.tmp.name, ignore_range=True) as server:
            with create_remote_connector(server.url()) as client:
                with self.assertRaisesRegex(ConnectionError, "ignored the Range"):
                    list(client.weight_iterator())

    def test_server_without_keepalive_is_rejected(self):
        with _Server(self.tmp.name, close_every_response=True) as server:
            with create_remote_connector(server.url()) as client:
                with self.assertRaisesRegex(ConnectionError, "keep-alive"):
                    client.shard_names()

    @unittest.skipIf(sys.platform.startswith("linux"), "SO_BINDTODEVICE exists")
    def test_nics_need_linux(self):
        with _Server(self.tmp.name) as server:
            with create_remote_connector(server.url(query="nics=bond0")) as client:
                with self.assertRaisesRegex(RuntimeError, "SO_BINDTODEVICE"):
                    list(client.weight_iterator())


class TestIterSafetensors(CustomTestCase):
    @staticmethod
    def _image(header, blob=b""):
        raw = json.dumps(header).encode()
        return struct.pack("<Q", len(raw)) + raw + blob

    def _as_buffer(self, data):
        return np.frombuffer(bytearray(data), dtype=np.uint8)

    def test_metadata_entry_is_skipped(self):
        image = self._image(
            {
                "__metadata__": {"format": "pt"},
                "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            },
            struct.pack("<2f", 1.0, 2.0),
        )
        tensors = dict(iter_safetensors(self._as_buffer(image)))
        self.assertEqual(list(tensors), ["w"])
        self.assertTrue(torch.equal(tensors["w"], torch.tensor([1.0, 2.0])))

    def test_empty_tensor(self):
        image = self._image(
            {"w": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}}
        )
        tensors = dict(iter_safetensors(self._as_buffer(image)))
        self.assertEqual(tensors["w"].numel(), 0)

    def test_unsupported_dtype(self):
        image = self._image(
            {"w": {"dtype": "F4", "shape": [2], "data_offsets": [0, 1]}}, b"\x00"
        )
        with self.assertRaisesRegex(ValueError, "F4"):
            dict(iter_safetensors(self._as_buffer(image)))

    def test_truncated_image(self):
        image = self._image(
            {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}, b"\x00" * 4
        )
        with self.assertRaisesRegex(ValueError, "past the"):
            dict(iter_safetensors(self._as_buffer(image)))

    def test_truncated_header(self):
        with self.assertRaisesRegex(ValueError, "header claims"):
            dict(iter_safetensors(self._as_buffer(struct.pack("<Q", 4096))))


if __name__ == "__main__":
    unittest.main()

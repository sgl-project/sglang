# SPDX-License-Identifier: Apache-2.0
"""Image-to-3D (mesh) client and node for the ComfyUI plugin, against a fake server."""

import email
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.server_api import (
    SGLDiffusionServerAPI,
)

GLB = b"glTF-fake-mesh-bytes"


class _FakeMeshServer:
    """Speaks the /v1/meshes job protocol; records what the client sent."""

    def __init__(self, *, file_path=None, url=None, final="completed"):
        self.uploaded: dict[str, bytes] = {}
        self.auth_seen: list[tuple[str, str | None]] = []
        self.polls = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body=b"", content_type="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                message = email.message_from_bytes(
                    b"Content-Type: "
                    + self.headers["Content-Type"].encode()
                    + b"\r\n\r\n"
                    + raw
                )
                for part in message.get_payload():
                    server.uploaded[
                        part.get_param("name", header="content-disposition")
                    ] = part.get_payload(decode=True)
                self._send(200, json.dumps({"id": "m1", "status": "queued"}).encode())

            def do_GET(self):
                server.auth_seen.append((self.path, self.headers.get("Authorization")))
                if self.path == "/v1/meshes/m1":
                    server.polls += 1
                    if server.polls < 3:
                        job = {"id": "m1", "status": "queued"}
                    elif final == "failed":
                        job = {
                            "id": "m1",
                            "status": "failed",
                            "error": {"message": "no GPU"},
                        }
                    else:
                        job = {
                            "id": "m1",
                            "status": "completed",
                            "format": "glb",
                            "file_path": file_path,
                            "url": url,
                        }
                    self._send(200, json.dumps(job).encode())
                elif self.path in ("/v1/meshes/m1/content", "/cloud/m1.glb"):
                    self._send(200, GLB, "model/gltf-binary")
                else:
                    self._send(404)

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "in.png"
    path.write_bytes(b"\x89PNG-fake")
    return str(path)


def _client(server):
    return SGLDiffusionServerAPI(base_url=server.base)


def test_generate_mesh_uploads_the_image_and_waits_for_the_job(image_file):
    server = _FakeMeshServer()
    try:
        job = _client(server).generate_mesh(
            image_path=image_file, output_format="obj", seed=7, poll_interval=0
        )
    finally:
        server.close()
    assert job["status"] == "completed" and server.polls == 3
    assert server.uploaded["image"] == b"\x89PNG-fake"
    assert server.uploaded["output_format"] == b"obj"
    assert server.uploaded["seed"] == b"7"


def test_unset_options_are_left_to_the_server_default(image_file):
    """-1 / 0 are the node's 'unset' values and must not reach the server."""
    server = _FakeMeshServer()
    try:
        _client(server).generate_mesh(
            image_path=image_file,
            seed=-1,
            num_inference_steps=0,
            guidance_scale=-1.0,
            poll_interval=0,
        )
    finally:
        server.close()
    assert set(server.uploaded) == {"image", "output_format"}


def test_failed_job_raises_the_server_message(image_file):
    server = _FakeMeshServer(final="failed")
    try:
        with pytest.raises(RuntimeError, match="no GPU"):
            _client(server).generate_mesh(image_path=image_file, poll_interval=0)
    finally:
        server.close()


def test_fetch_mesh_downloads_when_the_server_files_are_not_local(tmp_path):
    """A remote server's file_path does not exist here; /content must be used."""
    server = _FakeMeshServer(file_path="/nonexistent/on/this/host.glb")
    try:
        client = _client(server)
        job = client.generate_mesh(image_path=_png(tmp_path), poll_interval=0)
        dest = client.fetch_mesh(job, str(tmp_path / "out" / "m.glb"))
    finally:
        server.close()
    assert open(dest, "rb").read() == GLB
    assert ("/v1/meshes/m1/content", "Bearer sk-proj-1234567890") in server.auth_seen


def test_fetch_mesh_copies_a_local_file_without_a_request(tmp_path):
    local = tmp_path / "server_side.glb"
    local.write_bytes(b"local-bytes")
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1")
    dest = client.fetch_mesh(
        {"id": "m1", "file_path": str(local)}, str(tmp_path / "out" / "m.glb")
    )
    assert open(dest, "rb").read() == b"local-bytes"


def test_fetch_mesh_uses_the_cloud_url_without_the_api_key(tmp_path):
    """/content refuses cloud-stored meshes, and the key must not go to the cloud host."""
    server = _FakeMeshServer()
    try:
        job = {"id": "m1", "file_path": None, "url": f"{server.base}/cloud/m1.glb"}
        dest = _client(server).fetch_mesh(job, str(tmp_path / "m.glb"))
    finally:
        server.close()
    assert open(dest, "rb").read() == GLB
    assert server.auth_seen == [("/cloud/m1.glb", None)]


def _png(tmp_path) -> str:
    path = tmp_path / "in.png"
    path.write_bytes(b"\x89PNG-fake")
    return str(path)


@pytest.mark.skipif(
    "COMFYUI_PATH" not in os.environ, reason="needs a ComfyUI checkout (COMFYUI_PATH)"
)
def test_node_returns_a_file_3d_and_saves_into_the_output_dir(tmp_path):
    """The node's result must be what ComfyUI's Preview 3D accepts."""
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
    import folder_paths
    from comfy_api.latest import Types

    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.nodes import (
        NODE_CLASS_MAPPINGS,
        SGLDiffusionGenerateMesh,
    )

    assert NODE_CLASS_MAPPINGS["SGLDiffusionGenerateMesh"] is SGLDiffusionGenerateMesh
    folder_paths.set_output_directory(str(tmp_path / "output"))
    server = _FakeMeshServer(file_path="/nonexistent/on/this/host.glb")
    try:
        model, path = SGLDiffusionGenerateMesh().generate_mesh(
            _client(server), torch.rand(1, 16, 16, 3), seed=3
        )
    finally:
        server.close()
    assert isinstance(model, Types.File3D) and model.format == "glb"
    assert path == str(tmp_path / "output" / "mesh" / "sgld_m1.glb")
    assert open(path, "rb").read() == GLB

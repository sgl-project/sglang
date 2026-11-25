"""Request admission and process inheritance using ordinary, locally owned media."""

import asyncio
import base64
import http.server
import io
import ipaddress
import multiprocessing
import ssl
import subprocess
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from sglang.srt.multimodal import media_io
from sglang.srt.multimodal.media_budget import (
    MediaBudgetExceeded,
    MediaThreadPool,
    media_budget,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _png():
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "navy").save(image, format="PNG")
    return image.getvalue()


def _worker_policy(path):
    allowed_inline = media_io.get_image_bytes(
        "data:image/png;base64," + base64.b64encode(_png()).decode()
    )
    with pytest.raises(ValueError, match="not allowed"):
        media_io.get_image_bytes(Path(path).as_uri())
    return media_io.media_url_security_config(), allowed_inline


def _encoder_boot_policy(config, path):
    from sglang.srt.disaggregation.encoder import server

    # Stop immediately after the real encoder policy setup, before model/device initialization.
    with (
        patch.object(server, "assert_published"),
        patch.object(server, "get_parallel", return_value=SimpleNamespace(tp_size=1)),
        patch.object(server, "get_mm", return_value=config),
        patch.object(server, "get_disagg", side_effect=StopIteration),
        pytest.raises(StopIteration),
    ):
        server.MMEncoder(SimpleNamespace())
    return _worker_policy(path)


def test_encoder_and_spawned_cpu_worker_inherit_monolithic_policy(tmp_path):
    from sglang.srt.multimodal.processors.base_processor import _init_cpu_worker

    path = tmp_path / "allowed-fixture.png"
    path.write_bytes(_png())
    previous = media_io.media_url_security_config()
    config = SimpleNamespace(
        allowed_media_domains=["Media.TEST"],
        media_url_max_file_size_mb=7,
        media_url_timeout=2.5,
        allow_local_media_files=False,
        allow_private_media_hosts=False,
    )
    try:
        media_io.configure_worker_media_policy(config)
        reference = _worker_policy(path)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(1, mp_context=ctx) as pool:
            assert (
                pool.submit(_encoder_boot_policy, config, str(path)).result(90)
                == reference
            )
        with ProcessPoolExecutor(
            1, mp_context=ctx, initializer=_init_cpu_worker, initargs=(reference[0],)
        ) as pool:
            assert pool.submit(_worker_policy, str(path)).result(90) == reference
    finally:
        media_io.configure_media_url_security(*previous)


def test_request_budget_spans_threads_and_source_forms(tmp_path):
    data = _png()
    path = tmp_path / "image with spaces.png"
    path.write_bytes(data)
    inline = "data:image/png;base64," + base64.b64encode(data).decode()
    with MediaThreadPool(2) as pool:
        with media_budget(2 * len(data)):
            futures = [
                pool.submit(media_io.get_image_bytes, src)
                for src in (inline, path.as_uri(), data)
            ]
            successes = [f.result() for f in futures if f.exception() is None]
            assert successes == [data, data]
            assert (
                sum(isinstance(f.exception(), MediaBudgetExceeded) for f in futures)
                == 1
            )
        with media_budget(len(data)):
            assert pool.submit(media_io.get_image_bytes, data).result() == data


def test_pinned_https_preserves_original_sni_host_and_redirect_policy(
    tmp_path, monkeypatch
):
    from sglang.multimodal_gen.runtime.entrypoints.openai import utils

    crt = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-x509",
            "-days",
            "1",
            "-subj",
            "/CN=one.media.test",
            "-addext",
            "subjectAltName=DNS:one.media.test,DNS:two.media.test",
            "-keyout",
            str(key),
            "-out",
            str(crt),
        ],
        check=True,
        capture_output=True,
    )
    seen, names = [], []
    body = _png()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers["Host"])
            if self.path == "/first":
                self.send_response(302)
                self.send_header(
                    "Location",
                    f"https://two.media.test:{self.server.server_port}/image.png",
                )
            else:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
            self.end_headers()
            if self.path != "/first":
                self.wfile.write(body)

        def log_message(self, *_):
            pass

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    ctx.set_servername_callback(lambda sock, name, context: names.append(name))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    previous = media_io.media_url_security_config()
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(crt))
    monkeypatch.setenv("SSL_CERT_FILE", str(crt))
    # The operating system cannot resolve these fixture hostnames. Any unpinned dial fails.
    monkeypatch.setattr(
        media_io, "_resolve_host_ips", lambda _: [ipaddress.ip_address("127.0.0.1")]
    )
    media_io.configure_media_url_security(
        ["one.media.test", "two.media.test", "other.media.test"]
    )
    port = server.server_port
    url = f"https://one.media.test:{port}/first"
    try:
        assert media_io.get_image_bytes(url) == body
        saved = asyncio.run(utils._save_url_image_to_path(url, str(tmp_path / "saved")))
        assert Path(saved).read_bytes() == body
        assert seen == [f"one.media.test:{port}", f"two.media.test:{port}"] * 2
        assert names == ["one.media.test", "two.media.test"] * 2
        import requests

        with pytest.raises(requests.exceptions.SSLError):
            media_io.get_image_bytes(f"https://other.media.test:{port}/image.png")
        monkeypatch.setenv("no_proxy", "")
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("https_proxy", "http://proxy.media.test:3000")
        with pytest.raises(ValueError, match="direct route"):
            media_io.get_image_bytes(url)
        assert len(seen) == 4
    finally:
        media_io.configure_media_url_security(*previous)
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("backend", ["soundfile", "torchcodec"])
def test_ordinary_pcm_has_a_decoded_sample_budget(monkeypatch, backend):
    import numpy as np
    import soundfile as sf

    from sglang.srt.multimodal import media_budget as budget_module

    stream = io.BytesIO()
    sf.write(stream, np.zeros((64, 2), dtype="float32"), 16000, format="WAV")
    payload = stream.getvalue()
    monkeypatch.setattr(media_io, "_BACKEND", backend)

    def limit(n):
        monkeypatch.setattr(media_io, "MAX_DECODED_BYTES", n)
        monkeypatch.setattr(budget_module, "MAX_DECODED_BYTES", n)

    limit(4096)
    assert media_io.load_audio(payload, sr=16000, mono=False).shape == (64, 2)
    limit(128)
    with pytest.raises(MediaBudgetExceeded):
        media_io.load_audio(payload, sr=16000, mono=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

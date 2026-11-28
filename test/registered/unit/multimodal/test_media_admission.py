"""Request admission and process inheritance using ordinary, locally owned media."""

import base64
import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from sglang.srt.multimodal import media_io
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

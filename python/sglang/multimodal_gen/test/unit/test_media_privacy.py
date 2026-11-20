import asyncio
import logging
import secrets
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import Depends, FastAPI

from sglang.multimodal_gen.runtime.entrypoints.openai.media_privacy import (
    gateway_tenant_dependency,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import (
    VideoJobStore,
    media_tenant,
)


@pytest.fixture
def gateway_key(tmp_path):
    key = secrets.token_hex(32)
    path = tmp_path / "gateway-key"
    path.write_text(key + "\n")
    return str(path), key


def test_gateway_key_required_at_startup(tmp_path):
    path = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        gateway_tenant_dependency(str(path))
    path.write_text("invalid")
    with pytest.raises(ValueError, match="64-digit"):
        gateway_tenant_dependency(str(path))


def test_gateway_scope_and_background_task_lifetime(gateway_key):
    store = VideoJobStore()
    path, key = gateway_key
    header = ("x-sglang-gateway-key", key)
    app = FastAPI(dependencies=[Depends(gateway_tenant_dependency(path))])

    async def run():
        release = asyncio.Event()
        tasks = []

        @app.post("/{key}")
        async def create(key: str):
            async def finish():
                await release.wait()
                await store.update_fields(key, {"status": "completed"})

            await store.upsert(key, {"id": key, "status": "queued", "progress": 0})
            tasks.append(asyncio.create_task(finish()))

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for headers in (
                [],
                [("x-sglang-tenant", "a")],
                [("x-sglang-gateway-key", "invalid"), ("x-sglang-tenant", "a")],
                [header],
                [header, ("x-sglang-tenant", "a,b")],
                [header] + [("x-sglang-tenant", "a")] * 2,
                [header, header, ("x-sglang-tenant", "a")],
            ):
                assert (
                    await client.post("/invalid", headers=headers)
                ).status_code == 403
            assert not tasks
            responses = await asyncio.gather(
                *(
                    client.post(f"/{t}", headers=[header, ("x-sglang-tenant", t)])
                    for t in ("a", "b")
                )
            )
            assert all(r.status_code == 200 for r in responses)
        assert media_tenant.get() is None
        await store.record_scheduler_progress("a", 50)
        assert await store.get("a") is None
        release.set()
        await asyncio.gather(*tasks)
        token = media_tenant.set("a")
        try:
            assert await store.get("a") == {
                "id": "a",
                "status": "completed",
                "progress": 50,
            }
            assert await store.update_fields("b", {"progress": 1}) is None
            with pytest.raises(ValueError):
                await store.upsert("b", {})
            assert await store.pop("b") is None
            assert (await store.list_page())["last_id"] == "a"
            assert not (await store.list_page(after="b"))["data"]
        finally:
            media_tenant.reset(token)

    asyncio.run(run())


@pytest.mark.parametrize("scoped", [False, True])
def test_media_http_ownership_and_private_cache(tmp_path, scoped, gateway_key):
    from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app
    from sglang.multimodal_gen.runtime.entrypoints.openai.stores import (
        IMAGE_STORE,
        MESH_STORE,
        VIDEO_STORE,
    )

    args = SimpleNamespace(
        enable_metrics=False,
        media_gateway_key_file=gateway_key[0] if scoped else None,
        pipeline_config=SimpleNamespace(
            supports_action_endpoint=lambda: False,
            supports_openpi_endpoint=lambda: False,
        ),
    )
    app = create_app(args)
    owner = {"X-SGLang-Tenant": "blue", "X-SGLang-Gateway-Key": gateway_key[1]}
    other = {**owner, "X-SGLang-Tenant": "green"}
    path = tmp_path / "artifact.png"
    path.write_bytes(b"synthetic media")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for route, store in (
                ("images", IMAGE_STORE),
                ("videos", VIDEO_STORE),
                ("meshes", MESH_STORE),
            ):
                key = f"fixture-{route}"
                token = media_tenant.set("blue" if scoped else None)
                try:
                    await store.upsert(
                        key, {"id": key, "status": "completed", "file_path": str(path)}
                    )
                finally:
                    media_tenant.reset(token)
                root = f"/v1/{route}"
                try:
                    if scoped:
                        assert (
                            await client.get(f"{root}/{key}/content")
                        ).status_code == 403
                        denied = await client.get(
                            f"{root}/{key}/content", headers=other
                        )
                        unknown = await client.get(
                            f"{root}/absent/content", headers=other
                        )
                        assert denied.status_code == unknown.status_code == 404
                        assert denied.json() == unknown.json()
                        assert denied.headers["cache-control"] == "private, no-store"
                        if route != "images":
                            assert not (await client.get(root, headers=other)).json()[
                                "data"
                            ]
                            assert (
                                await client.delete(f"{root}/{key}", headers=other)
                            ).status_code == 404
                            assert (
                                await client.get(f"{root}/{key}", headers=owner)
                            ).status_code == 200
                    response = await client.get(
                        f"{root}/{key}/content", headers=owner if scoped else other
                    )
                    assert response.content == b"synthetic media"
                    assert response.headers["cache-control"] == "private, no-store"
                    if route == "videos":
                        token = media_tenant.set("blue" if scoped else None)
                        try:
                            await store.update_fields(
                                key, {"status": "failed", "url": "private-url"}
                            )
                        finally:
                            media_tenant.reset(token)
                        response = await client.get(
                            f"{root}/{key}/content", headers=owner
                        )
                        assert (
                            response.status_code == 404
                            and "private-url" not in response.text
                        )
                finally:
                    token = media_tenant.set("blue" if scoped else None)
                    try:
                        await store.pop(key)
                    finally:
                        media_tenant.reset(token)

    asyncio.run(run())


@pytest.mark.parametrize("action", [False, True])
@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_worker_does_not_log_request_content(caplog, action, level):
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.configs.task_type import DataType
    from sglang.multimodal_gen.runtime.pipelines_core import schedule_batch as sb

    params = SamplingParams(
        prompt="private-prompt", negative_prompt="private-neg", seed=7251382
    )
    if action:
        params.data_type = DataType.ACTION
    req = sb.Req(sampling_params=params)
    req.system_prompt = "private-system"
    req.image_path = "private-image"
    with caplog.at_level(level, logger=sb.logger.name):
        req.log(
            SimpleNamespace(pipeline_config=SimpleNamespace(embedded_cfg_scale=None))
        )
    text = caplog.text + repr(req) + str(req)
    assert "private-" not in text and "7251382" not in text
    if not action or level == logging.DEBUG:
        assert "steps" in caplog.text


def test_object_upload_metadata_and_logs(caplog):
    from sglang.multimodal_gen.runtime.entrypoints.openai.storage import CloudStorage

    storage = CloudStorage.__new__(CloudStorage)
    storage.enabled = True
    storage.bucket_name, storage.region_name, storage.endpoint_url = (
        "private-bucket",
        None,
        None,
    )
    storage.client = Mock()
    with caplog.at_level(logging.INFO):
        url = asyncio.run(storage.upload_file("private-input.png", "private-key.png"))
        assert url.endswith("/private-key.png")
        assert storage.client.upload_file.call_args.kwargs["ExtraArgs"] == {
            "ContentType": "image/png",
            "CacheControl": "private, no-store",
        }
        storage.client.upload_file.side_effect = OSError("private-error")
        assert (
            asyncio.run(storage.upload_file("private-input.png", "private-key.png"))
            is None
        )
    assert "private-" not in caplog.text


def test_request_logger_requires_opt_in_for_reproduction_inputs():
    from sglang.multimodal_gen.runtime.server_args import ServerArgs
    from sglang.multimodal_gen.runtime.utils.request_logger import (
        DiffusionRequestLogger,
    )

    assert ServerArgs.log_requests_level == 0
    req = SimpleNamespace(
        request_id="test",
        sampling_params=SimpleNamespace(prompt="secret-text", seed=12345),
    )
    result = SimpleNamespace(error="secret-text")
    for level in range(4):
        logger = DiffusionRequestLogger(True, level, "json", None)
        _, record = logger._batch_record([req])
        assert ("seed" in record.get("sampling_params", {})) == (level >= 1)
        assert ("prompt" in record) == (level >= 2)
        assert "secret-text" not in str(logger._result_view(result))

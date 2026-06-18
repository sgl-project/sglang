import dataclasses
import unittest
from typing import Annotated, List

import msgspec
from fastapi import Body, FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    BaseReq,
    UpdateWeightFromDiskReqInput,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ToyReqInput(BaseReq, kw_only=True):
    required: str
    count: int = 3
    values: List[str] = msgspec.field(default_factory=list)
    mode: str = "ok"

    def __post_init__(self):
        if self.mode != "ok":
            raise ValueError(f"Invalid mode: {self.mode!r}")


class TestHttpMsgspecReqInput(CustomTestCase):
    def test_pydantic_type_adapter_constructs_msgspec_struct(self):
        adapter = TypeAdapter(ToyReqInput)

        obj = adapter.validate_python({"required": "x"})
        self.assertFalse(dataclasses.is_dataclass(obj))
        self.assertIsInstance(obj, ToyReqInput)
        self.assertEqual(obj.required, "x")
        self.assertEqual(obj.count, 3)
        self.assertEqual(obj.values, [])
        self.assertIsNone(obj.rid)

        first = adapter.validate_python({"required": "x"})
        second = adapter.validate_python({"required": "y"})
        first.values.append("mutated")
        self.assertEqual(second.values, [])

        with self.assertRaisesRegex(ValueError, "Invalid mode"):
            adapter.validate_python({"required": "x", "mode": "bad"})

    def test_msgspec_req_input_is_fastapi_body_param(self):
        app = FastAPI()

        @app.post("/toy")
        def toy(obj: Annotated[ToyReqInput, Body()]):
            return {
                "required": obj.required,
                "count": obj.count,
                "values": obj.values,
                "rid": obj.rid,
            }

        openapi = app.openapi()
        operation = openapi["paths"]["/toy"]["post"]
        self.assertIn("requestBody", operation)
        self.assertNotIn("parameters", operation)
        self.assertNotIn("rid", openapi["components"]["schemas"]["ToyReqInput"])

        client = TestClient(app)
        response = client.post("/toy", json={"required": "x", "rid": "ignored"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"required": "x", "count": 3, "values": [], "rid": None},
        )

        response = client.post("/toy", json={"count": 4})
        self.assertEqual(response.status_code, 422)

    def test_update_weight_from_disk_req_input_fastapi_body_validation(self):
        app = FastAPI()

        @app.post("/update_weights_from_disk")
        def update_weights_from_disk(
            obj: Annotated[UpdateWeightFromDiskReqInput, Body()],
        ):
            return {
                "model_path": obj.model_path,
                "load_format": obj.load_format,
                "abort_all_requests": obj.abort_all_requests,
                "weight_version": obj.weight_version,
                "is_async": obj.is_async,
                "torch_empty_cache": obj.torch_empty_cache,
                "keep_pause": obj.keep_pause,
                "recapture_cuda_graph": obj.recapture_cuda_graph,
                "token_step": obj.token_step,
                "flush_cache": obj.flush_cache,
                "manifest": obj.manifest,
                "rid": obj.rid,
            }

        openapi_schema = app.openapi()["components"]["schemas"][
            "UpdateWeightFromDiskReqInput"
        ]
        self.assertIn("model_path", openapi_schema["properties"])
        self.assertNotIn("rid", openapi_schema["properties"])

        client = TestClient(app)
        response = client.post(
            "/update_weights_from_disk",
            json={"model_path": "/tmp/model", "rid": "ignored"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "model_path": "/tmp/model",
                "load_format": None,
                "abort_all_requests": False,
                "weight_version": None,
                "is_async": False,
                "torch_empty_cache": False,
                "keep_pause": False,
                "recapture_cuda_graph": False,
                "token_step": 0,
                "flush_cache": True,
                "manifest": None,
                "rid": None,
            },
        )

        response = client.post(
            "/update_weights_from_disk",
            json={"load_format": "auto"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()

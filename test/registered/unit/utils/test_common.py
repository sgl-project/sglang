import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.utils.common import (
    _get_fastapi_request_path,
    flatten_arrays_to_int64_tensor,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFlattenArraysToInt64Tensor(CustomTestCase):
    """`flatten_arrays_to_int64_tensor` is invoked by `prepare_for_extend`
    to build the per-batch input_ids tensor (pinned, async H2D) from a
    list of array.array('q') per-req get_fill_ids() slices. Tests the
    full matrix of (device, pin) the production code paths through.
    """

    DEVICES = ("cpu", "cuda")
    PIN_OPTIONS = (False, True)

    def _check(self, parts: list, expected: list[int]) -> None:
        for device in self.DEVICES:
            for pin in self.PIN_OPTIONS:
                with self.subTest(device=device, pin=pin):
                    out = flatten_arrays_to_int64_tensor(parts, device, pin)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    self.assertEqual(out.dtype, torch.int64)
                    self.assertEqual(out.device.type, device)
                    self.assertEqual(out.shape, (len(expected),))
                    self.assertEqual(out.cpu().tolist(), expected)

    def test_single_part(self):
        parts = [array("q", [1, 2, 3, 4, 5])]
        self._check(parts, [1, 2, 3, 4, 5])

    def test_multiple_parts(self):
        parts = [
            array("q", [10, 20, 30]),
            array("q", [100, 200]),
            array("q", [1000]),
        ]
        self._check(parts, [10, 20, 30, 100, 200, 1000])


class TestGetFastapiRequestPath(CustomTestCase):
    """`_get_fastapi_request_path` resolves the templated route path for the
    Prometheus HTTP middleware. FastAPI >=0.137 no longer flattens included
    routers into ``app.routes`` -- they become wrapper objects without a
    ``.path`` -- which used to raise AttributeError -> HTTP 500 on every
    request (#28887).
    """

    @staticmethod
    def _request(app, path):
        scope = {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": [],
            "query_string": b"",
            "root_path": "",
        }
        return SimpleNamespace(app=app, scope=scope, url=SimpleNamespace(path=path))

    def test_included_router_path_is_resolved(self):
        from fastapi import APIRouter, FastAPI

        app = FastAPI()
        router = APIRouter()

        @router.get("/v1/chat/completions")
        def _endpoint():
            return {}

        app.include_router(router)
        path, handled = _get_fastapi_request_path(
            self._request(app, "/v1/chat/completions")
        )
        self.assertEqual(path, "/v1/chat/completions")
        self.assertTrue(handled)

    def test_unmatched_path_falls_back(self):
        from fastapi import FastAPI

        app = FastAPI()
        path, handled = _get_fastapi_request_path(self._request(app, "/nope"))
        self.assertEqual(path, "/nope")
        self.assertFalse(handled)


if __name__ == "__main__":
    unittest.main()

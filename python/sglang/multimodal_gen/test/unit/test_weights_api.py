# SPDX-License-Identifier: Apache-2.0
"""A weight endpoint must report a failed scheduler handler, not crash or answer 200.

The scheduler replies `OutputBatch(error=...)` with `output=None` when a handler
raises; every weight route has to turn that reply into an error response that
carries the message.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints.post_training import weights_api
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


@pytest.mark.parametrize(
    "endpoint,body",
    [
        ("/update_weights_from_tensor", {"serialized_named_tensors": ["payload"]}),
        (
            "/update_weights_from_tensor_checker",
            {
                "target_module": "transformer",
                "expected_named_tensors_sha256": {"weight": "0" * 64},
            },
        ),
        ("/get_weights_checksum", {"module_names": ["transformer"]}),
    ],
)
def test_failed_handler_reply_is_reported_with_its_message(endpoint, body):
    async def forward(req):
        return OutputBatch(error="handler raised: out of memory")

    app = FastAPI()
    app.include_router(weights_api.router)
    with patch.object(
        weights_api, "async_scheduler_client", SimpleNamespace(forward=forward)
    ):
        response = TestClient(app).post(endpoint, json=body)

    assert response.status_code == 500
    assert "handler raised: out of memory" in response.text

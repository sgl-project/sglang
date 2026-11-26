import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import (
    PullWeightsReqInput,
    UpdateWeightFromDiskReqInput,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPullWeightsResponse(CustomTestCase):
    def test_reload_path_only_after_a_successful_snapshot_pull(self):
        local = Path.cwd() / "local"
        for snapshot, success in ((False, True), (True, False), (True, True)):
            with self.subTest(snapshot=snapshot, success=success):
                message = "ok" if success else "one host failed"
                manager = SimpleNamespace(
                    pull_weights=AsyncMock(return_value=(success, message)),
                    update_weights_from_disk=AsyncMock(return_value=(True, "ok", 0)),
                )
                req = PullWeightsReqInput(
                    local_checkpoint_dir=str(local),
                    source_dir="published",
                    target_version=9,
                    snapshot=snapshot,
                )
                with patch.object(
                    http_server,
                    "_global_state",
                    SimpleNamespace(tokenizer_manager=manager),
                ):
                    response = asyncio.run(http_server.pull_weights(req, None))
                    self.assertEqual(response.status_code, 200 if success else 400)
                    body = json.loads(response.body)
                    self.assertEqual(body["success"], success)
                    if success and snapshot:
                        path = str(local / ".weight_sync" / "snapshots" / "v000009")
                        self.assertEqual(body["model_path"], path)
                        reload_req = UpdateWeightFromDiskReqInput(
                            model_path=body["model_path"]
                        )
                        asyncio.run(
                            http_server.update_weights_from_disk(reload_req, None)
                        )
                        manager.update_weights_from_disk.assert_awaited_once_with(
                            reload_req, None
                        )
                    else:
                        self.assertNotIn("model_path", body)


if __name__ == "__main__":
    unittest.main()

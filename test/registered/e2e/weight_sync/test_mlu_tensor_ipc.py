"""MLU weight tensors share storage across processes with different device indices."""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=90, suite="pr-test-2-mlu")


def _worker(role, path, device_index):
    import torch
    import torch_mlu  # noqa: F401

    import sglang.srt.plugins as plugins
    from sglang.srt.platforms import current_platform
    from sglang.srt.platforms.mlu import MluSRTPlatform
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
    from sglang.srt.utils.serialization import MultiprocessingSerializer

    device_index = int(device_index)
    assert type(current_platform) is MluSRTPlatform
    assert not plugins._plugins_loaded
    expected = torch.arange(48, dtype=torch.float32).reshape(6, 8)[1:, 1::2]
    if role == "producer":
        monkey_patch_torch_reductions()
        monkey_patch_torch_reductions()
        base = torch.arange(48, dtype=torch.float32, device=f"mlu:{device_index}")
        tensor = base.reshape(6, 8)[1:, 1::2]
        parameter = torch.nn.Parameter(torch.ones(3, device=f"mlu:{device_index}"))
        torch.mlu.synchronize(device_index)
        payload = MultiprocessingSerializer.serialize(
            [("weight", tensor), ("parameter", parameter)]
        )
        staging = Path(path).with_suffix(".tmp")
        staging.write_bytes(payload)
        staging.replace(path)
        # Keep both exporting allocations alive until the consumer releases them.
        assert sys.stdin.readline().strip() == "done"
        torch.mlu.synchronize(device_index)
        assert torch.equal(tensor.cpu(), expected + 100)
        assert torch.equal(parameter.detach().cpu(), torch.full((3,), 2.0))
    else:
        # A receiver must resolve rebuild functions without initializing reducers.
        named_tensors = dict(
            MultiprocessingSerializer.deserialize(Path(path).read_bytes())
        )
        tensor = named_tensors.pop("weight")
        parameter = named_tensors.pop("parameter")
        assert tensor.device.index == parameter.device.index == device_index
        assert tensor.stride() == expected.stride()
        assert tensor.storage_offset() == expected.storage_offset()
        assert torch.equal(tensor.cpu(), expected)
        assert isinstance(parameter, torch.nn.Parameter) and parameter.requires_grad
        assert torch.equal(parameter.detach().cpu(), torch.ones(3))
        with torch.no_grad():
            tensor.add_(100)
            parameter.add_(1)
        torch.mlu.synchronize(device_index)
        del tensor, parameter
    assert not plugins._plugins_loaded


class TestMLUTensorIPC(CustomTestCase):
    def test_reordered_visible_devices(self):
        visible = (
            os.environ.get("CN_VISIBLE_DEVICES")
            or os.environ.get("MLU_VISIBLE_DEVICES")
            or "0,1"
        ).split(",")[:2]
        self.assertEqual(len(visible), 2, "This test requires two visible MLU devices")
        # Select the in-tree platform even when an out-of-tree plugin is installed.
        env = dict(os.environ, SGLANG_PLUGINS=",", SGLANG_USE_CPU_ENGINE="0")
        env.pop("SGLANG_PLATFORM", None)
        command = [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])['_worker'](*sys.argv[2:])",
            str(Path(__file__).resolve()),
        ]
        for producer_index in (0, 1):
            with self.subTest(producer_index=producer_index):
                with tempfile.TemporaryDirectory() as directory:
                    payload = Path(directory) / "weights.bin"
                    with (Path(directory) / "producer.log").open("w+") as output:
                        producer = subprocess.Popen(
                            command + ["producer", str(payload), str(producer_index)],
                            stdin=subprocess.PIPE,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            text=True,
                            env=dict(env, CN_VISIBLE_DEVICES=",".join(visible)),
                        )
                        try:
                            deadline = time.monotonic() + 90
                            while (
                                not payload.exists()
                                and producer.poll() is None
                                and time.monotonic() < deadline
                            ):
                                time.sleep(0.1)
                            if not payload.exists():
                                output.seek(0)
                                self.fail(output.read() or "Producer timed out")
                            consumer = subprocess.run(
                                command
                                + ["consumer", str(payload), str(1 - producer_index)],
                                env=dict(
                                    env,
                                    CN_VISIBLE_DEVICES=",".join(reversed(visible)),
                                ),
                                capture_output=True,
                                text=True,
                                timeout=90,
                            )
                            self.assertEqual(
                                consumer.returncode,
                                0,
                                consumer.stdout + consumer.stderr,
                            )
                            producer.communicate("done\n", timeout=30)
                            output.seek(0)
                            self.assertEqual(producer.returncode, 0, output.read())
                        finally:
                            if producer.poll() is None:
                                producer.kill()
                                producer.wait()
                            if producer.stdin is not None:
                                producer.stdin.close()


if __name__ == "__main__":
    unittest.main()

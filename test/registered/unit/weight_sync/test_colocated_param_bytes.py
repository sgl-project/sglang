"""An engine rank pulls param bytes from a trainer process's CUDA-IPC buffers on the same GPU into its published
weights, and reports a bad or incomplete sync as an error after finishing the exchange."""

import tempfile
import unittest

import torch
import torch.multiprocessing as mp
import zmq

from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.colocated_param_bytes import (
    Ack,
    Closed,
    End,
    Hello,
    Shared,
    Write,
    pull_param_bytes,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_BUFFER_NBYTES = 1 << 20


def _new_bytes(name: str, nbytes: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(len(name) * 7919 + nbytes)
    return torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=generator)


def _run_trainer(socket_path, params, expected_names, ack_errors):
    """params: [(name, nbytes sent)], packed into buffer 1 at 256-byte aligned offsets."""
    torch.cuda.set_device(0)
    monkey_patch_torch_reductions()
    sock = zmq.Context.instance().socket(zmq.PAIR)
    sock.bind(socket_path)
    hello = sock.recv_pyobj()
    assert isinstance(hello, Hello)
    buffers = [
        torch.zeros(_BUFFER_NBYTES, dtype=torch.uint8, device="cuda") for _ in range(2)
    ]
    sock.send_pyobj(
        Shared(
            serialized_buffers=MultiprocessingSerializer.serialize(buffers),
            expected_names_by_role={"target": frozenset(expected_names)},
        )
    )
    write_params, offset = [], 0
    for name, nbytes in params:
        buffers[1][offset : offset + nbytes].copy_(_new_bytes(name, nbytes))
        write_params.append((name, offset, nbytes))
        offset += (nbytes + 255) // 256 * 256
    torch.cuda.synchronize()
    sock.send_pyobj(Write(buffer_index=1, role="target", params=write_params))
    ack = sock.recv_pyobj()
    assert isinstance(ack, Ack)
    ack_errors.put(ack.error)
    sock.send_pyobj(End())
    assert isinstance(sock.recv_pyobj(), Closed)
    sock.close()


class TestPullParamBytes(CustomTestCase):
    def setUp(self):
        torch.cuda.set_device(0)
        self.params = {
            "a.weight": torch.zeros(1000, dtype=torch.float32, device="cuda"),
            "b.weight": torch.zeros(3, 5, dtype=torch.bfloat16, device="cuda"),
        }
        self.published_nbytes = {
            name: param.numel() * param.element_size()
            for name, param in self.params.items()
        }
        self.socket_path = f"ipc://{tempfile.mkdtemp()}/pull.sock"

    def _pull(self, params, expected_names):
        context = mp.get_context("spawn")
        ack_errors = context.Queue()
        trainer = context.Process(
            target=_run_trainer,
            args=(self.socket_path, params, expected_names, ack_errors),
        )
        trainer.start()
        sock = zmq.Context.instance().socket(zmq.PAIR)
        sock.setsockopt(zmq.RCVTIMEO, 60_000)
        sock.connect(self.socket_path)
        try:
            error = pull_param_bytes(
                sock=sock,
                device_uuid="gpu-0",
                params_by_role={"target": self.params},
                published_nbytes_by_role={"target": self.published_nbytes},
            )
        finally:
            sock.close()
        trainer.join(timeout=60)
        self.assertEqual(trainer.exitcode, 0)
        return error, ack_errors.get(timeout=5)

    def _assert_holds(self, name, expected: torch.Tensor):
        param = self.params[name]
        self.assertTrue(
            torch.equal(param.reshape(-1).view(torch.uint8).cpu(), expected),
            f"{name} does not hold the bytes the trainer sent",
        )

    def test_every_published_byte_lands_in_place(self):
        """The engine's post-load pass reads exactly the bytes a P2P write would have placed."""
        error, ack_error = self._pull(
            [(name, nbytes) for name, nbytes in self.published_nbytes.items()],
            expected_names=list(self.params),
        )

        self.assertIsNone(error)
        self.assertIsNone(ack_error)
        for name, nbytes in self.published_nbytes.items():
            self._assert_holds(name, _new_bytes(name, nbytes))

    def test_a_size_mismatch_is_acknowledged_and_writes_nothing(self):
        """Copying past a param's published bytes would corrupt its neighbor; the trainer must learn it failed."""
        error, ack_error = self._pull(
            [("a.weight", 4000), ("b.weight", 32)], expected_names=list(self.params)
        )

        self.assertIn("b.weight: 32 bytes sent, 30 published", error)
        self.assertEqual(ack_error, error)
        self._assert_holds("a.weight", torch.zeros(4000, dtype=torch.uint8))

    def test_a_param_the_loader_writes_but_no_bytes_reached_fails_the_sync(self):
        """A loader-written param left at the previous version would serve stale weights after the update."""
        error, ack_error = self._pull(
            [("a.weight", 4000)], expected_names=list(self.params)
        )

        self.assertIsNone(ack_error)
        self.assertIn("b.weight", error)


if __name__ == "__main__":
    unittest.main()

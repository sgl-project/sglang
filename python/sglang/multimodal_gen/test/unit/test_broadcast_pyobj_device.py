"""broadcast_pyobj(local_device=...) unpickles accelerator tensors onto the receiver's card.

Without it a receiving rank that sees every device restores them on the sender's
device index: its copies then run on the sender's card and, on XPU, reset that card
when they overlap the sender's device barrier (Wan2.2-S2V's second request).
"""

import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.runtime.distributed.utils import broadcast_pyobj
from sglang.test.test_utils import CustomTestCase

_SEED = 123


def _accelerator_count() -> int:
    if not torch.accelerator.is_available():
        return 0
    return torch.accelerator.device_count()


def _check_receiver_device(rank: int, directory: str):
    device_type = torch.accelerator.current_accelerator().type
    device_module = torch.get_device_module(device_type)
    device_module.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=Path(directory, "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        local = torch.device(device_type, rank)
        if rank == 0:
            generator = torch.Generator(device=local)
            generator.manual_seed(_SEED)
            # Advance it, as a pipeline stage does before sending its batch.
            torch.randn(4, generator=generator, device=local)
            payload = {
                "tensor": torch.arange(8.0, device=local),
                "cpu_tensor": torch.ones(3),
                "generator": generator,
            }
            broadcast_pyobj([payload], rank=0, dist_group=dist.group.WORLD, src=0)
            return
        sender_allocated = device_module.memory_allocated(0)
        (received,) = broadcast_pyobj(
            [], rank=1, dist_group=dist.group.WORLD, src=0, local_device=local
        )
        assert received["tensor"].device == local, received["tensor"].device
        assert torch.equal(received["tensor"].cpu(), torch.arange(8.0))
        # A generator's state is a CPU tensor: it must load as-is for set_state.
        assert received["cpu_tensor"].device.type == "cpu"
        assert received["generator"].device == local, received["generator"].device
        expected = torch.Generator(device=local)
        expected.manual_seed(_SEED)
        torch.randn(4, generator=expected, device=local)
        assert torch.equal(received["generator"].get_state(), expected.get_state())
        assert device_module.memory_allocated(0) == sender_allocated
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(_accelerator_count() >= 2, "needs 2 accelerator devices")
class TestBroadcastPyobjDevice(CustomTestCase):
    def test_receiver_loads_onto_its_own_device(self):
        with TemporaryDirectory() as directory:
            mp.spawn(_check_receiver_device, args=(directory,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()

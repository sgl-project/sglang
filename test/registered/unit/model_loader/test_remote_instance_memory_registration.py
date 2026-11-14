"""Registration ranges for remote weights allocated inside shared GPU slabs."""

import unittest
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    register_memory_region,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_model_and_snapshot():
    # CPU tensors supply real parameter pointers; only the GPU allocator's
    # snapshot is replaced. Two weights occupy separate blocks of one slab.
    pool = torch.zeros(1024, dtype=torch.float32)
    model = torch.nn.ParameterDict(
        {
            "first": torch.nn.Parameter(pool[128:256]),
            "second": torch.nn.Parameter(pool[384:512]),
        }
    )
    base = pool.data_ptr()
    blocks = [
        (base, 512, "inactive"),
        (base + 512, 512, "active_allocated"),
        (base + 1024, 512, "active_allocated"),
        (base + 1536, 512, "active_allocated"),
        (base + 2048, 2048, "inactive"),
    ]
    snapshot = [
        {
            "device": 0,
            "address": base,
            "total_size": 4096,
            "allocated_size": 1536,
            "active_size": 1536,
            "requested_size": 1536,
            "stream": 0,
            "segment_type": "small",
            "is_expandable": False,
            "blocks": [
                {
                    "address": address,
                    "size": size,
                    "requested_size": size if state == "active_allocated" else 0,
                    "state": state,
                    "frames": [],
                }
                for address, size, state in blocks
            ],
        }
    ]
    return model, snapshot, base


def register(model, snapshot, hip, engine=None):
    if engine is None:
        engine = Mock(register_memory=Mock(return_value=0))
    with (
        patch.object(torch.version, "hip", hip),
        patch.object(torch.cuda.memory, "memory_snapshot", return_value=snapshot),
    ):
        info = register_memory_region(model, engine)
    return info, engine


class TestRemoteInstanceMemoryRegistration(unittest.TestCase):
    def test_hip_registers_allocation_base_once_for_suballocated_weights(self):
        model, snapshot, base = make_model_and_snapshot()

        _, engine = register(model, snapshot, hip="7.2.0")

        # HIP IPC opens the allocation base. Publishing an interior address
        # as that base silently shifts reads in the pinned Mooncake build.
        engine.register_memory.assert_called_once_with(base, 4096)

    def test_cuda_keeps_separate_weight_blocks(self):
        model, snapshot, base = make_model_and_snapshot()

        _, engine = register(model, snapshot, hip=None)

        self.assertEqual(
            engine.register_memory.call_args_list,
            [call(base + 512, 512), call(base + 1536, 512)],
        )

    def test_expandable_hip_segment_keeps_virtual_memory_registration(self):
        model, snapshot, base = make_model_and_snapshot()
        snapshot[0]["is_expandable"] = True

        _, engine = register(model, snapshot, hip="7.2.0")

        # A VMM segment can span multiple physical allocations; it must not
        # be treated as a single hipMalloc allocation for legacy IPC.
        self.assertEqual(
            engine.register_memory.call_args_list,
            [call(base + 512, 512), call(base + 1536, 512)],
        )

    def test_published_weight_metadata_keeps_tensor_addresses_and_sizes(self):
        for hip in (None, "7.2.0"):
            with self.subTest(hip=hip):
                model, snapshot, base = make_model_and_snapshot()

                info, _ = register(model, snapshot, hip=hip)

                self.assertEqual(
                    info,
                    {"first": (base + 512, 128, 4), "second": (base + 1536, 128, 4)},
                )

    def test_segments_without_weights_are_not_registered(self):
        model, snapshot, _ = make_model_and_snapshot()
        model = torch.nn.ParameterDict()

        _, engine = register(model, snapshot, hip="7.2.0")

        engine.register_memory.assert_not_called()

    def test_each_hip_allocation_with_weights_is_registered(self):
        first_model, first_snapshot, first_base = make_model_and_snapshot()
        second_model, second_snapshot, second_base = make_model_and_snapshot()
        model = torch.nn.ModuleDict({"a": first_model, "b": second_model})

        _, engine = register(model, first_snapshot + second_snapshot, hip="7.2.0")

        self.assertEqual(
            engine.register_memory.call_args_list,
            [call(first_base, 4096), call(second_base, 4096)],
        )

    def test_registration_failure_is_reported(self):
        model, snapshot, _ = make_model_and_snapshot()
        engine = Mock(register_memory=Mock(return_value=-1))

        with self.assertRaisesRegex(RuntimeError, "register memory failed"):
            register(model, snapshot, hip="7.2.0", engine=engine)


if __name__ == "__main__":
    unittest.main()

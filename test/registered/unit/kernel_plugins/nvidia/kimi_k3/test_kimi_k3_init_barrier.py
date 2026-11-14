# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU-only guards against initialization barriers on a wrong backend/rank group.

Execute the actual host bodies without importing CuTe/CUDA. This checks early
rejection and call ordering; it does not qualify cross-rank GPU visibility.
"""

import ast
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.kimi_k3_source_paths import source_definition, source_path
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def definition(filename, *names):
    return source_definition("cutedsl/" + filename, *names)


def function(filename, names, namespace):
    node = definition(filename, *names)
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(
        compile(module, str(source_path("cutedsl/" + filename, *names)), "exec"),
        namespace,
    )
    return namespace[node.name]


class TestInitBarrier(CustomTestCase):
    def setUp(self):
        self.device_group = SimpleNamespace(
            backend="nccl", ranks=tuple(range(8)), rank=0
        )
        self.cpu_group = SimpleNamespace(backend="gloo", ranks=tuple(range(8)), rank=0)
        self.dist = SimpleNamespace(
            is_initialized=lambda: True,
            get_backend=lambda group: group.backend,
            get_process_group_ranks=lambda group: group.ranks,
            get_rank=lambda group: group.rank,
            get_world_size=lambda group: len(group.ranks),
        )
        self.torch = SimpleNamespace(distributed=self.dist)
        self.validate = function(
            "common/host.py", ("validate_init_barrier_group",), {"torch": self.torch}
        )

    def test_same_ordered_gloo_group_and_legacy_api(self):
        self.assertIs(self.validate(self.device_group, self.cpu_group), self.cpu_group)
        self.dist.is_initialized = lambda: False
        self.assertIs(self.validate(self.device_group), self.device_group)
        with self.assertRaisesRegex(RuntimeError, "initialized groups"):
            self.validate(self.device_group, self.cpu_group)

    def test_wrong_backend_order_membership_and_nonmember_fail_closed(self):
        for backend, ranks, rank in (
            ("nccl", tuple(range(8)), 0),
            ("gloo", tuple(reversed(range(8))), 7),
            ("gloo", tuple(range(7)), 0),
            ("gloo", tuple(range(8)), -1),
        ):
            with self.subTest(backend=backend, ranks=ranks, rank=rank):
                group = SimpleNamespace(backend=backend, ranks=ranks, rank=rank)
                with self.assertRaises(ValueError):
                    self.validate(self.device_group, group)

    def test_all_box_constructors_reject_before_allocation_or_collective(self):
        bad_group = SimpleNamespace(backend="nccl", ranks=tuple(range(8)), rank=0)
        for filename, cls, args, kwargs in (
            ("oproj/prepared.py", "OprojMailbox", (8, 0), {}),
            ("common/twoshot.py", "TwoShot", (8, 0), {}),
            ("moe/small_batch/chain.py", "SmallBatchBox", (8, 0), {}),
            ("moe/tail_runtime.py", "MoeTailBox", (8, 0), {}),
            ("mla/dcp.py", "DcpBox", (8, 0, 12), {"deadline_ns": 1}),
        ):
            with self.subTest(cls=cls):
                constructor = function(
                    filename,
                    (cls, "__init__"),
                    {
                        "torch": self.torch,
                        "validate_init_barrier_group": self.validate,
                        "_require_deadline_ns": lambda value: value,
                    },
                )
                # The namespace has no GPU allocator, rendezvous or barrier. A
                # constructor reaching any of them before rejection fails here.
                with self.assertRaisesRegex(ValueError, "must use Gloo"):
                    constructor(
                        SimpleNamespace(),
                        self.device_group,
                        object(),
                        *args,
                        init_barrier_group=bad_group,
                        **kwargs,
                    )

    def test_cpu_barriers_remain_after_local_gpu_synchronization(self):
        for filename, cls in (
            ("oproj/prepared.py", "OprojMailbox"),
            ("common/twoshot.py", "TwoShot"),
            ("mla/dcp.py", "DcpBox"),
        ):
            body = definition(filename, cls, "__init__")
            calls = {
                ast.unparse(node.func): node
                for node in ast.walk(body)
                if isinstance(node, ast.Call)
            }
            sync = calls["torch.accelerator.synchronize"]
            barrier = calls["torch.distributed.barrier"]
            self.assertLess(sync.lineno, barrier.lineno)
            self.assertEqual(ast.unparse(barrier.keywords[0].value), "init_group")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))

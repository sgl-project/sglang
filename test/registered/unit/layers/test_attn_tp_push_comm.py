"""CPU unit tests for attention-TP2 symmetric-memory push comm."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import attn_tp_push_comm
from sglang.srt.runtime_context import get_resources
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_fake_comm(rank: int = 0, max_rows: int = 8, hidden: int = 16):
    comm = object.__new__(attn_tp_push_comm._AttnTpPushComm)
    comm.rank = rank
    comm.peer = 1 - rank
    comm.max_rows = max_rows
    comm.hidden = hidden
    comm.device = torch.device("cpu")
    return comm


class TestAttnTpPushComm(CustomTestCase):
    def tearDown(self):
        get_resources().attn_tp_push_comm = None
        super().tearDown()

    def test_takes_checks_shape_dtype_device_and_stride(self):
        comm = _make_fake_comm(max_rows=8, hidden=16)
        full = torch.zeros(4, 16, dtype=torch.bfloat16)
        shard = torch.zeros(2, 16, dtype=torch.bfloat16)
        self.assertTrue(comm.takes(full, shard))
        # Boundary: max_rows accepted, max_rows + 2 and 0 rejected.
        self.assertTrue(
            comm.takes(
                torch.zeros(8, 16, dtype=torch.bfloat16),
                torch.zeros(4, 16, dtype=torch.bfloat16),
            )
        )
        self.assertFalse(
            comm.takes(
                torch.zeros(10, 16, dtype=torch.bfloat16),
                torch.zeros(5, 16, dtype=torch.bfloat16),
            )
        )
        self.assertFalse(
            comm.takes(
                torch.zeros(0, 16, dtype=torch.bfloat16),
                torch.zeros(0, 16, dtype=torch.bfloat16),
            )
        )
        # Wrong shard dim / hidden width / odd rows / dtype / non-unit last stride.
        self.assertFalse(comm.takes(full, torch.zeros(2, 16, 1, dtype=torch.bfloat16)))
        self.assertFalse(comm.takes(full, torch.zeros(2, 8, dtype=torch.bfloat16)))
        self.assertFalse(
            comm.takes(
                torch.zeros(3, 16, dtype=torch.bfloat16),
                torch.zeros(2, 16, dtype=torch.bfloat16),
            )
        )
        self.assertFalse(comm.takes(full.float(), shard))
        self.assertFalse(comm.takes(full, shard.float()))
        strided = torch.zeros(2, 32, dtype=torch.bfloat16)[:, ::2]
        self.assertFalse(comm.takes(full, strided))

    def test_initialize_clears_stale_comm_and_falls_back_on_error(self):
        get_resources().attn_tp_push_comm = object()
        fake_parallel = SimpleNamespace(
            attn_tp_group=SimpleNamespace(world_size=2, cpu_group=object()),
            attn_tp_size=2,
            tp_size=8,
        )
        with patch.object(
            attn_tp_push_comm, "get_parallel", return_value=fake_parallel
        ):
            attn_tp_push_comm.initialize_attn_tp_push_comm(
                hidden_size=16, uses_megamoe=False
            )
        self.assertIsNone(get_resources().attn_tp_push_comm)

        with (
            patch.object(attn_tp_push_comm, "get_parallel", return_value=fake_parallel),
            patch.object(attn_tp_push_comm, "is_cuda", return_value=True),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch(
                "sglang.srt.distributed.device_communicators.custom_all_reduce_utils.in_one_nvlink_domain",
                return_value=True,
            ),
            patch.object(
                attn_tp_push_comm,
                "_AttnTpPushComm",
                side_effect=RuntimeError("symm_mem unsupported"),
            ),
        ):
            attn_tp_push_comm.initialize_attn_tp_push_comm(
                hidden_size=16, uses_megamoe=True
            )
        self.assertIsNone(get_resources().attn_tp_push_comm)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""Q/K/V must survive reuse of the Ulysses collective receive buffer."""

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers import usp
from sglang.test.test_utils import CustomTestCase


class TestUlyssesInputCache(CustomTestCase):
    def test_single_head_outputs_survive_the_next_receive(self):
        # TP2 followed by Ulysses2 can leave one head per rank. For batch1,
        # the receive relayout is already contiguous: returning a view lets
        # later K/V receives overwrite Q. Emulate only the collective boundary
        # and its reusable storage; execute the real production relayout.
        full = [
            torch.arange(1 * 6 * 2 * 4, dtype=torch.float32).reshape(1, 6, 2, 4)
            + 100 * stream
            for stream in range(3)
        ]
        receive = torch.empty(2, 1, 3, 4)

        for head_dim in (1, 2):
            with self.subTest(head_dim=head_dim):
                next_stream = 0

                def receive_collective(x, role=None):
                    nonlocal next_stream
                    tensor = full[next_stream]
                    next_stream += 1
                    # Destination rank0 receives head0 from both sequence
                    # ranks, in source-rank-major all_to_all_single order.
                    incoming = torch.cat(
                        [
                            tensor[:, start : start + 3, :1].permute(2, 0, 1, 3)
                            for start in (0, 3)
                        ],
                        dim=0,
                    )
                    receive.copy_(incoming)
                    return receive.reshape(x.shape)

                with (
                    patch.object(
                        usp, "get_ulysses_parallel_world_size", return_value=2
                    ),
                    patch.object(usp, "_ipc_varlen_fast", return_value=None),
                    patch.object(usp, "_usp_all_to_all_single", receive_collective),
                ):
                    if head_dim == 2:
                        outputs = usp._usp_input_all_to_all_qkv(
                            *(tensor[:, :3].contiguous() for tensor in full)
                        )
                    else:
                        outputs = tuple(
                            usp._usp_input_all_to_all(
                                tensor[:, :3].transpose(1, 2), head_dim=1
                            )
                            for tensor in full
                        )

                # A subsequent attention exchange must not invalidate any
                # previously returned tensor, including the final V result.
                receive.fill_(-999)
                for actual, tensor in zip(outputs, full):
                    expected = tensor[:, :, :1]
                    if head_dim == 1:
                        expected = expected.transpose(1, 2)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

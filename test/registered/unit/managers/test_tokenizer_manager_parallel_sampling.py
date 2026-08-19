import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import (
    _parallel_sample_input_index,
    _parallel_sample_prefix_bootstrap_room,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestParallelSamplingBootstrapRoom(CustomTestCase):
    def test_expanded_index_matches_normalized_batch_order(self):
        req = GenerateReqInput(
            text=["prompt_a", "prompt_b"],
            sampling_params=[{"n": 3}, {"n": 3}],
            bootstrap_room=[100, 200],
        )
        req.normalize_batch_and_arguments()

        # GenerateReqInput expands batch-major fields as [a0, b0, a1, b1, ...].
        # Each original prompt should pick its own element for every parallel
        # sample instead of reusing objs[i].
        self.assertEqual(req.bootstrap_room, [100, 200, 102, 202, 104, 204])
        self.assertEqual(
            [
                req.bootstrap_room[
                    _parallel_sample_input_index(
                        batch_index=0, sample_index=i, batch_size=req.batch_size
                    )
                ]
                for i in range(req.parallel_sample_num)
            ],
            [100, 102, 104],
        )
        self.assertEqual(
            [
                req.bootstrap_room[
                    _parallel_sample_input_index(
                        batch_index=1, sample_index=i, batch_size=req.batch_size
                    )
                ]
                for i in range(req.parallel_sample_num)
            ],
            [200, 202, 204],
        )

    def test_list_bootstrap_room_skips_collisions_during_expansion(self):
        req = GenerateReqInput(
            text=["prompt_a", "prompt_b"],
            sampling_params=[{"n": 3}, {"n": 3}],
            bootstrap_room=[100, 102],
        )
        req.normalize_batch_and_arguments()

        self.assertEqual(req.bootstrap_room, [100, 102, 104, 106, 108, 110])

    def test_prefix_bootstrap_room_does_not_collide_with_scalar_samples(self):
        used_rooms = [7, 8, 9]

        self.assertEqual(
            _parallel_sample_prefix_bootstrap_room(7, used_rooms),
            10,
        )
        self.assertIsNone(
            _parallel_sample_prefix_bootstrap_room(None, used_rooms),
        )

    def test_prefix_bootstrap_room_skips_all_expanded_sample_rooms(self):
        used_rooms = [100, 200, 102, 202, 104, 204]

        self.assertEqual(
            _parallel_sample_prefix_bootstrap_room(100, used_rooms),
            106,
        )
        self.assertEqual(
            _parallel_sample_prefix_bootstrap_room(200, used_rooms),
            206,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""CPU-only regression tests for packed GGUF MoE loading contracts."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

from sglang.srt.layers.quantization.gguf_moe_loader import (
    plan_k3_a_log_tp_shard,
    plan_gguf_moe_stream_destination,
    plan_gguf_moe_tp_shard,
    record_gguf_moe_qtype,
)


class TestK3ALogTPShardingContract(unittest.TestCase):
    def test_tp4_ranks_receive_exact_32_element_ranges(self):
        for rank in range(4):
            with self.subTest(rank=rank):
                self.assertEqual(
                    plan_k3_a_log_tp_shard(
                        checkpoint_shape=(128,),
                        parameter_shape=(1, 1, 32, 1),
                        tp_rank=rank,
                        tp_size=4,
                    ),
                    (rank * 32, 32),
                )

    def test_legacy_4d_checkpoint_uses_same_range(self):
        self.assertEqual(
            plan_k3_a_log_tp_shard(
                checkpoint_shape=(1, 1, 128, 1),
                parameter_shape=(1, 1, 32, 1),
                tp_rank=2,
                tp_size=4,
            ),
            (64, 32),
        )

    def test_invalid_shapes_and_rank_fail_closed(self):
        cases = (
            ((2, 2), (1, 1, 1, 1), 0, 2, "1-D or legacy 4-D"),
            ((129,), (1, 1, 32, 1), 0, 4, "extent differs"),
            ((160,), (1, 1, 32, 1), 0, 4, "extent differs"),
            ((128,), (1, 1, 32, 1), -1, 4, "topology is invalid"),
            ((128,), (1, 1, 32, 1), 4, 4, "topology is invalid"),
            ((32,), (1, 32), 0, 1, "parameter shape is invalid"),
        )
        for checkpoint_shape, parameter_shape, rank, size, message in cases:
            with self.subTest(
                checkpoint_shape=checkpoint_shape,
                parameter_shape=parameter_shape,
                rank=rank,
                size=size,
            ), self.assertRaisesRegex(ValueError, message):
                plan_k3_a_log_tp_shard(
                    checkpoint_shape=checkpoint_shape,
                    parameter_shape=parameter_shape,
                    tp_rank=rank,
                    tp_size=size,
                )


class TestGGUFMoEQTypeContract(unittest.TestCase):
    def test_matching_w1_w3_types_are_accepted(self):
        types = {}
        record_gguf_moe_qtype(types, "w1", 16)
        record_gguf_moe_qtype(types, "w3", 16)
        record_gguf_moe_qtype(types, "w1", 16)
        self.assertEqual(types, {"w1": 16, "w3": 16})

    def test_mixed_w1_w3_types_fail_closed(self):
        types = {}
        record_gguf_moe_qtype(types, "w1", 16)
        with self.assertRaisesRegex(ValueError, "different qtypes"):
            record_gguf_moe_qtype(types, "w3", 17)
        self.assertEqual(types, {"w1": 16})

    def test_one_shard_repeating_with_a_new_type_fails_closed(self):
        types = {}
        record_gguf_moe_qtype(types, "w2", 16)
        with self.assertRaisesRegex(ValueError, "repeats with different qtypes"):
            record_gguf_moe_qtype(types, "w2", 17)
        self.assertEqual(types, {"w2": 16})


class TestGGUFMoETPShardingContract(unittest.TestCase):
    def test_w1_and_w3_split_output_rows(self):
        for shard_id in ("w1", "w3"):
            self.assertEqual(
                plan_gguf_moe_tp_shard(
                    shard_id=shard_id,
                    shape=(3072, 1036),
                    tp_size=4,
                    tp_rank=2,
                ),
                (0, 1536, 768),
            )

    def test_iq2_xxs_w2_tp4_splits_three_whole_blocks_per_rank(self):
        # 3072 logical input weights = 12 IQ2_XXS blocks * 66 bytes.
        self.assertEqual(
            plan_gguf_moe_tp_shard(
                shard_id="w2",
                shape=(3584, 12 * 66),
                tp_size=4,
                tp_rank=3,
                packed_type_size=66,
            ),
            (1, 9 * 66, 3 * 66),
        )

    def test_iq2_xs_w2_tp4_splits_three_whole_blocks_per_rank(self):
        # 3072 logical input weights = 12 IQ2_XS blocks * 74 bytes.
        self.assertEqual(
            plan_gguf_moe_tp_shard(
                shard_id="w2",
                shape=(3584, 12 * 74),
                tp_size=4,
                tp_rank=1,
                packed_type_size=74,
            ),
            (1, 3 * 74, 3 * 74),
        )

    def test_iq2_w2_tp8_fails_instead_of_cutting_blocks(self):
        for type_size in (66, 74):
            with self.subTest(type_size=type_size):
                with self.assertRaisesRegex(ValueError, "cuts a packed"):
                    plan_gguf_moe_tp_shard(
                        shard_id="w2",
                        shape=(3584, 12 * type_size),
                        tp_size=8,
                        tp_rank=0,
                        packed_type_size=type_size,
                    )

    def test_tp1_owns_the_entire_w2_input(self):
        self.assertEqual(
            plan_gguf_moe_tp_shard(
                shard_id="w2",
                shape=(3584, 12 * 66),
                tp_size=1,
                tp_rank=0,
                packed_type_size=66,
            ),
            (1, 0, 12 * 66),
        )


class TestGGUFMoEStreamingDestinationContract(unittest.TestCase):
    def test_w1_streams_into_lower_half_of_final_w13(self):
        self.assertEqual(
            plan_gguf_moe_stream_destination(
                shard_id="w1",
                expert_id=7,
                num_experts=896,
                local_shape=(768, 1036),
            ),
            ((896, 1536, 1036), (7, 0, 768)),
        )

    def test_w3_streams_into_upper_half_of_same_final_w13(self):
        self.assertEqual(
            plan_gguf_moe_stream_destination(
                shard_id="w3",
                expert_id=7,
                num_experts=896,
                local_shape=(768, 1036),
            ),
            ((896, 1536, 1036), (7, 768, 768)),
        )

    def test_w2_streams_into_final_down_parameter(self):
        self.assertEqual(
            plan_gguf_moe_stream_destination(
                shard_id="w2",
                expert_id=895,
                num_experts=896,
                local_shape=(3584, 222),
            ),
            ((896, 3584, 222), (895, 0, 3584)),
        )

    def test_stream_plan_rejects_invalid_expert(self):
        with self.assertRaisesRegex(ValueError, "invalid GGUF MoE expert"):
            plan_gguf_moe_stream_destination(
                shard_id="w2",
                expert_id=896,
                num_experts=896,
                local_shape=(3584, 222),
            )


if __name__ == "__main__":
    unittest.main()

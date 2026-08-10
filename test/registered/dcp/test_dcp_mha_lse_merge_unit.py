"""CPU tests for the allocation-free DCP MHA LSE merge path."""

import unittest

import torch

from sglang.srt.layers.dcp import comm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _reference_contributions(partial_outputs, partial_lses):
    global_lse = torch.logsumexp(partial_lses, dim=0)
    scales = torch.exp(partial_lses - global_lse.unsqueeze(0)).unsqueeze(-1)
    scales = torch.nan_to_num(scales, nan=0.0, posinf=0.0, neginf=0.0)
    sanitized_outputs = torch.nan_to_num(
        partial_outputs, nan=0.0, posinf=0.0, neginf=0.0
    )
    contributions = sanitized_outputs * scales
    return contributions, contributions.sum(dim=0), global_lse


class _FakeDCPGroup:
    def __init__(self, rank, partial_lses, expected_contributions, expected_global):
        self.world_size = partial_lses.shape[0]
        self.rank_in_group = rank
        self._partial_lses = partial_lses
        self._expected_contributions = expected_contributions
        self._expected_global = expected_global
        self.all_reduce_input = None
        self.all_reduce_input_ptr = None

    def all_gather(self, tensor, dim):
        if dim != 0:
            raise AssertionError(f"expected LSE all-gather on dim 0, got {dim}")
        torch.testing.assert_close(
            tensor,
            self._partial_lses[self.rank_in_group],
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        return torch.cat(tuple(self._partial_lses.unbind(0)), dim=0)

    def all_reduce(self, tensor):
        self.all_reduce_input = tensor.clone()
        self.all_reduce_input_ptr = tensor.data_ptr()
        torch.testing.assert_close(
            tensor,
            self._expected_contributions[self.rank_in_group],
            rtol=0,
            atol=0,
        )
        return self._expected_global.clone()


class TestDCPMHALSEMerge(unittest.TestCase):
    @staticmethod
    def _inputs():
        torch.manual_seed(1234)
        world_size, tokens, heads, head_dim = 4, 5, 8, 7
        partial_outputs = torch.randn(
            world_size, tokens, heads, head_dim, dtype=torch.float32
        )
        partial_lses = torch.randn(world_size, tokens, heads, dtype=torch.float32)

        # Exercise sanitization of non-finite partial attention output.
        partial_outputs[1, 0, 0, 0] = float("nan")
        partial_outputs[2, 0, 1, 0] = float("inf")
        partial_outputs[3, 0, 2, 0] = -float("inf")

        # Token 1 has no valid KV on any rank. Token 2 has one empty shard.
        partial_lses[:, 1, :] = -float("inf")
        partial_lses[2, 2, :] = -float("inf")
        return partial_outputs, partial_lses

    def _run_rank(self, rank, partial_outputs, partial_lses):
        contributions, expected_global, expected_lse = _reference_contributions(
            partial_outputs, partial_lses
        )
        group = _FakeDCPGroup(rank, partial_lses, contributions, expected_global)
        local_output = partial_outputs[rank].clone()
        input_ptr = local_output.data_ptr()

        output, output_lse = comm.cp_lse_ag_out_rs_mha(
            local_output,
            partial_lses[rank].clone(),
            group,
            return_lse=True,
        )

        heads_per_rank = partial_outputs.shape[2] // partial_outputs.shape[0]
        head_start = rank * heads_per_rank
        head_end = head_start + heads_per_rank
        torch.testing.assert_close(
            output,
            expected_global[:, head_start:head_end, :],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output_lse,
            expected_lse[:, head_start:head_end],
            rtol=0,
            atol=0,
        )
        return local_output, input_ptr, group, output, output_lse

    def test_matches_reference_for_every_rank(self):
        partial_outputs, partial_lses = self._inputs()

        for rank in range(partial_outputs.shape[0]):
            local_output, _, group, _, _ = self._run_rank(
                rank, partial_outputs, partial_lses
            )
            torch.testing.assert_close(
                local_output, group.all_reduce_input, rtol=0, atol=0
            )

    def test_reuses_partial_output_buffer_for_every_rank(self):
        partial_outputs, partial_lses = self._inputs()

        for rank in range(partial_outputs.shape[0]):
            _, input_ptr, group, _, _ = self._run_rank(
                rank, partial_outputs, partial_lses
            )
            self.assertEqual(group.all_reduce_input_ptr, input_ptr)

    def test_all_empty_kv_shards_contribute_zero_instead_of_nan(self):
        partial_outputs, partial_lses = self._inputs()

        for rank in range(partial_outputs.shape[0]):
            _, _, group, output, output_lse = self._run_rank(
                rank, partial_outputs, partial_lses
            )
            self.assertTrue(
                torch.equal(
                    group.all_reduce_input[1],
                    torch.zeros_like(group.all_reduce_input[1]),
                )
            )
            self.assertTrue(torch.equal(output[1], torch.zeros_like(output[1])))
            self.assertTrue(torch.isneginf(output_lse[1]).all())

    def test_one_empty_rank_does_not_poison_global_result(self):
        partial_outputs, partial_lses = self._inputs()
        _, _, group, output, output_lse = self._run_rank(
            2, partial_outputs, partial_lses
        )

        self.assertTrue(
            torch.equal(
                group.all_reduce_input[2],
                torch.zeros_like(group.all_reduce_input[2]),
            )
        )
        self.assertTrue(torch.isfinite(output[2]).all())
        self.assertTrue(torch.isfinite(output_lse[2]).all())

    def test_nonfinite_partial_outputs_are_sanitized_before_collective(self):
        partial_outputs, partial_lses = self._inputs()

        for rank in range(partial_outputs.shape[0]):
            _, _, group, _, _ = self._run_rank(rank, partial_outputs, partial_lses)
            self.assertTrue(torch.isfinite(group.all_reduce_input[0]).all())

    def test_world_size_one_preserves_input_contract(self):
        partial_output = torch.randn(2, 4, 8)
        partial_lse = torch.randn(2, 4)
        original = partial_output.clone()

        class _SingleRankGroup:
            world_size = 1

        output, output_lse = comm.cp_lse_ag_out_rs_mha(
            partial_output,
            partial_lse,
            _SingleRankGroup(),
            return_lse=True,
        )

        self.assertIs(output, partial_output)
        self.assertIs(output_lse, partial_lse)
        torch.testing.assert_close(partial_output, original, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

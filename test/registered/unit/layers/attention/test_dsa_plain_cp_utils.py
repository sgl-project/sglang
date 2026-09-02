import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.utils import (
    cp_plain_all_gather,
    cp_plain_reduce_scatter,
    cp_plain_split,
    cp_plain_to_scattered,
    cp_scattered_to_plain,
)
from sglang.srt.layers.utils.cp_utils import prepare_context_parallel_metadata
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPlainContextParallelUtils(unittest.TestCase):
    @staticmethod
    def _forward_batch(metadata):
        return SimpleNamespace(attn_cp_metadata=metadata)

    @staticmethod
    def _all_gather(rank_tensors):
        def all_gather(output, input_tensor):
            del input_tensor
            torch.cat(rank_tensors, dim=0, out=output)

        return all_gather

    def test_non_divisible_plain_split_and_gather_round_trip(self):
        cp_size = 2
        total_tokens = 4097
        x = torch.arange(total_tokens * 2).view(total_tokens, 2)
        metadata = SimpleNamespace(total_seq_lens=total_tokens)
        forward_batch = self._forward_batch(metadata)

        shards = []
        for rank in range(cp_size):
            with get_parallel().override(attn_cp_size=cp_size, attn_cp_rank=rank):
                shards.append(cp_plain_split(x, forward_batch))

        self.assertEqual([shard.shape[0] for shard in shards], [2049, 2049])
        self.assertTrue(torch.equal(shards[0], x[:2049]))
        self.assertTrue(torch.equal(shards[1][:-1], x[2049:]))
        self.assertTrue(torch.equal(shards[1][-1], torch.zeros(2, dtype=x.dtype)))

        for rank in range(cp_size):
            with (
                get_parallel().override(
                    attn_cp_size=cp_size,
                    attn_cp_rank=rank,
                    attn_cp_group=object(),
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.attn_cp_all_gather_into_tensor",
                    side_effect=self._all_gather(shards),
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.is_allocation_symmetric",
                    return_value=False,
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.use_symmetric_memory",
                    return_value=torch.no_grad(),
                ),
            ):
                gathered = cp_plain_all_gather(shards[rank], cp_size, forward_batch)

            self.assertTrue(torch.equal(gathered, x))

    def test_non_divisible_plain_reduce_scatter_pads_final_rank(self):
        cp_size = 2
        total_tokens = 4097
        x = torch.arange(total_tokens * 2).view(total_tokens, 2)
        metadata = SimpleNamespace(total_seq_lens=total_tokens)
        forward_batch = self._forward_batch(metadata)
        physical_rank_len = 2049

        for rank in range(cp_size):

            def reduce_scatter(output, input_tensor, rank=rank):
                chunks = input_tensor.view(cp_size, physical_rank_len, 2)
                output.copy_(chunks[rank])

            with (
                get_parallel().override(
                    attn_cp_size=cp_size,
                    attn_cp_rank=rank,
                    attn_cp_group=object(),
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.attn_cp_reduce_scatter_tensor",
                    side_effect=reduce_scatter,
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.is_allocation_symmetric",
                    return_value=False,
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.use_symmetric_memory",
                    return_value=torch.no_grad(),
                ),
            ):
                reduced = cp_plain_reduce_scatter(x, cp_size, forward_batch)
                expected = cp_plain_split(x, forward_batch)

            self.assertTrue(torch.equal(reduced, expected))

    def test_non_divisible_plain_scattered_conversion_round_trip(self):
        cp_size = 2
        total_tokens = 4097
        x = torch.arange(total_tokens * 2).view(total_tokens, 2)
        metadata = []
        plain_shards = []

        for rank in range(cp_size):
            with (
                get_parallel().override(attn_cp_size=cp_size, attn_cp_rank=rank),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.is_dsa_prefill_cp_round_robin_split",
                    return_value=False,
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.is_dsa_enable_prefill_cp",
                    return_value=False,
                ),
            ):
                metadata.append(
                    prepare_context_parallel_metadata(
                        total_tokens,
                        rank,
                        cp_size,
                        [total_tokens],
                        extend_seqs_len=[total_tokens],
                        device="cpu",
                    )
                )
                plain_shards.append(
                    cp_plain_split(x, self._forward_batch(metadata[rank]))
                )

        scattered_shards = []
        for rank in range(cp_size):
            forward_batch = self._forward_batch(metadata[rank])
            with (
                get_parallel().override(
                    attn_cp_size=cp_size,
                    attn_cp_rank=rank,
                    attn_cp_group=object(),
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.attn_cp_all_gather_into_tensor",
                    side_effect=self._all_gather(plain_shards),
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.is_allocation_symmetric",
                    return_value=False,
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.use_symmetric_memory",
                    return_value=torch.no_grad(),
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.is_dsa_prefill_cp_round_robin_split",
                    return_value=False,
                ),
            ):
                scattered_shards.append(
                    cp_plain_to_scattered(plain_shards[rank], forward_batch, cp_size)
                )

        max_scattered_len = max(shard.shape[0] for shard in scattered_shards)
        padded_scattered = [
            torch.cat(
                [
                    shard,
                    shard.new_zeros(
                        max_scattered_len - shard.shape[0], *shard.shape[1:]
                    ),
                ],
                dim=0,
            )
            for shard in scattered_shards
        ]

        for rank in range(cp_size):
            forward_batch = self._forward_batch(metadata[rank])
            with (
                get_parallel().override(
                    attn_cp_size=cp_size,
                    attn_cp_rank=rank,
                    attn_cp_group=object(),
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.attn_cp_all_gather_into_tensor",
                    side_effect=self._all_gather(padded_scattered),
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.is_allocation_symmetric",
                    return_value=False,
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.use_symmetric_memory",
                    return_value=torch.no_grad(),
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.is_dsa_prefill_cp_round_robin_split",
                    return_value=False,
                ),
                patch("torch.cuda.current_stream", return_value=None),
            ):
                plain = cp_scattered_to_plain(
                    scattered_shards[rank], forward_batch, cp_size
                )

            self.assertTrue(torch.equal(plain, plain_shards[rank]))


if __name__ == "__main__":
    unittest.main()

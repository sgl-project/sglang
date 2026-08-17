import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kda_cp import (
    all_gather_cp_heads,
    head_to_sequence_a2a,
    kda_use_prefill_cp,
    sequence_to_head_a2a,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ForwardMode:
    def is_context_parallel_extend(self):
        return True

    def is_mixed(self):
        return False

    def is_target_verify(self):
        return False

    def is_draft_extend_v2(self):
        return False


def _metadata():
    # Ten logical rows, split into four zigzag blocks:
    # rank0 = [0,1,2,8,9], rank1 = [3,4,5,6,7].  CP-v2 pads both ranks to 8.
    return SimpleNamespace(
        split_list=[3, 3, 2, 2],
        cp_reverse_index=[0, 2, 3, 1],
        reverse_split_len=[3, 2, 3, 2],
        per_rank_logical_token=[5, 5],
        per_rank_actual_token=[8, 8],
        max_rank_len=[8, 8],
    )


def _local_inputs():
    natural = torch.arange(10 * 4, dtype=torch.float32).view(10, 4, 1)
    rank_tokens = ([0, 1, 2, 8, 9], [3, 4, 5, 6, 7])
    local = []
    for indices in rank_tokens:
        shard = natural[list(indices)]
        local.append(torch.cat([shard, shard.new_zeros(3, 4, 1)], dim=0))
    return natural, local


class _SequenceToHeadGroup:
    def __init__(self, all_rank_inputs, destination_rank):
        self.all_rank_inputs = all_rank_inputs
        self.destination_rank = destination_rank

    def all_to_all_single(self, output, _input):
        cp_size = len(self.all_rank_inputs)
        for source_rank, source in enumerate(self.all_rank_inputs):
            send = source.view(source.shape[0], cp_size, -1, *source.shape[2:])
            send = send.transpose(0, 1).contiguous()
            output[source_rank].copy_(send[self.destination_rank])


class _HeadToSequenceGroup:
    def __init__(self, all_head_shards, destination_rank):
        self.all_head_shards = all_head_shards
        self.destination_rank = destination_rank

    def all_to_all_single(self, output, _input):
        # Natural -> rank order [block0, block3, block1, block2].
        rank_order = [0, 1, 2, 8, 9, 3, 4, 5, 6, 7]
        for head_rank, natural in enumerate(self.all_head_shards):
            ordered = natural[rank_order]
            chunks = torch.split(ordered, [5, 5], dim=0)
            send = natural.new_zeros((2, 8, *natural.shape[1:]))
            for token_rank, chunk in enumerate(chunks):
                send[token_rank, : chunk.shape[0]].copy_(chunk)
            output[head_rank].copy_(send[self.destination_rank])


class _AllGatherGroup:
    def __init__(self, all_rank_inputs):
        self.all_rank_inputs = all_rank_inputs

    def all_gather_into_tensor(self, output, _input):
        torch.cat(self.all_rank_inputs, dim=0, out=output)


class TestKDAPrefillCP(unittest.TestCase):
    def test_sequence_head_a2a_round_trip_with_cp_v2_padding(self):
        metadata = _metadata()
        forward_batch = SimpleNamespace(attn_cp_metadata=metadata)
        natural, local_inputs = _local_inputs()

        head_shards = []
        for cp_rank in range(2):
            group = _SequenceToHeadGroup(local_inputs, cp_rank)
            with get_parallel().override(attn_cp_rank=cp_rank, attn_cp_size=2):
                shard = sequence_to_head_a2a(
                    local_inputs[cp_rank], forward_batch, group=group
                )
            torch.testing.assert_close(
                shard, natural[:, cp_rank * 2 : (cp_rank + 1) * 2]
            )
            head_shards.append(shard)

        for cp_rank in range(2):
            group = _HeadToSequenceGroup(head_shards, cp_rank)
            with get_parallel().override(attn_cp_rank=cp_rank, attn_cp_size=2):
                restored = head_to_sequence_a2a(
                    head_shards[cp_rank], forward_batch, group=group
                )
            torch.testing.assert_close(restored, local_inputs[cp_rank])

    def test_all_gather_cp_heads_restores_rank_order(self):
        rank0 = torch.tensor([[[0.0], [1.0]]])
        rank1 = torch.tensor([[[2.0], [3.0]]])
        group = _AllGatherGroup([rank0.movedim(1, 0), rank1.movedim(1, 0)])
        with get_parallel().override(attn_cp_size=2):
            gathered = all_gather_cp_heads(rank0, head_dim=1, group=group)
        torch.testing.assert_close(
            gathered, torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
        )

    def test_kda_cp_is_prefill_only(self):
        forward_batch = SimpleNamespace(
            attn_cp_metadata=_metadata(), forward_mode=_ForwardMode()
        )
        with (
            patch(
                "sglang.srt.layers.attention.linear.kda_cp.get_parallel",
                return_value=SimpleNamespace(
                    enable_prefill_context_parallel=True,
                    attn_cp_size=2,
                ),
            ),
            patch(
                "sglang.srt.layers.attention.linear.kda_cp.is_npu",
                return_value=True,
            ),
        ):
            self.assertTrue(kda_use_prefill_cp(forward_batch))

        with patch(
            "sglang.srt.layers.attention.linear.kda_cp.get_parallel",
            return_value=SimpleNamespace(
                enable_prefill_context_parallel=False,
                attn_cp_size=2,
            ),
        ):
            self.assertFalse(kda_use_prefill_cp(forward_batch))


if __name__ == "__main__":
    unittest.main()

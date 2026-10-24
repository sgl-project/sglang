"""Focused regression tests for DFlash with DP attention."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative import dflash_worker_v2 as dflash
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestDFlashDPAttention(CustomTestCase):
    def test_target_graphs_keep_global_dp_metadata(self):
        for is_draft in (False, True):
            runner = SimpleNamespace(
                is_draft_worker=is_draft,
                spec_algorithm=SimpleNamespace(is_dflash=lambda: True),
            )
            self.assertEqual(
                DecodeCudaGraphRunner._forward_is_dp_local(runner), is_draft
            )
        other = SimpleNamespace(
            is_draft_worker=True,
            spec_algorithm=SimpleNamespace(
                is_dflash=lambda: False, is_dspark=lambda: False
            ),
        )
        self.assertFalse(DecodeCudaGraphRunner._forward_is_dp_local(other))

    def test_embedding_cache_respects_sharding_and_padding(self):
        full = torch.arange(14, dtype=torch.float32).reshape(7, 2)
        for layout in ("replicated", "replicated-padded", "attention-tp", "global-tp"):
            for rank in (0, 1):
                with self.subTest(layout=layout, rank=rank):
                    global_group = SimpleNamespace(
                        world_size=2 if layout == "global-tp" else 4,
                        device_group=object(),
                    )
                    attn_group = SimpleNamespace(world_size=2, device_group=object())
                    shard = None
                    parts = [full]
                    if layout != "replicated":
                        sharded = layout != "replicated-padded"
                        num_org_padded = 3 if sharded else 6
                        num_added_padded = 1 if sharded else 2
                        parts = []
                        for shard_rank in range(2 if sharded else 1):
                            part = torch.full(
                                (num_org_padded + num_added_padded, 2), -100.0
                            )
                            start = shard_rank * num_org_padded
                            rows = min(num_org_padded, 5 - start)
                            part[:rows] = full[start : start + rows]
                            if sharded:
                                part[num_org_padded] = full[5 + shard_rank]
                            else:
                                part[num_org_padded:] = full[5:]
                            parts.append(part)
                        shard = SimpleNamespace(
                            num_org_elements_padded=num_org_padded,
                            num_added_elements_padded=num_added_padded,
                        )
                    tp_size = 2 if layout in ("attention-tp", "global-tp") else 1
                    embedding = SimpleNamespace(
                        weight=parts[rank if tp_size > 1 else 0],
                        tp_size=tp_size,
                        use_attn_tp_group=layout == "attention-tp",
                        shard_indices=shard,
                        org_vocab_size=5,
                        num_added_embeddings=2,
                    )
                    worker = SimpleNamespace(
                        _target_worker=SimpleNamespace(
                            model_runner=SimpleNamespace(
                                model=SimpleNamespace(
                                    get_input_embeddings=lambda: embedding
                                ),
                                model_config=SimpleNamespace(vocab_size=7),
                            )
                        ),
                        _target_tp_rank=rank,
                    )

                    def gather(outputs, local, *, group):
                        expected_group = (
                            attn_group if layout == "attention-tp" else global_group
                        )
                        self.assertIs(group, expected_group.device_group)
                        self.assertEqual(tuple(local.shape), (4, 2))
                        for out, part in zip(outputs, parts):
                            out.copy_(part)

                    with (
                        patch.object(
                            dflash,
                            "get_parallel",
                            return_value=SimpleNamespace(
                                attn_dp_enabled=True,
                                attn_tp_group=attn_group,
                                tp_group=global_group,
                            ),
                        ),
                        patch.object(
                            dflash.dist, "all_gather", side_effect=gather
                        ) as all_gather,
                    ):
                        dflash.DFlashWorkerV2._cache_full_embed_weight(worker)
                        self.assertEqual(all_gather.call_count, int(tp_size > 1))
                    torch.testing.assert_close(worker._full_embed_gpu, full)


if __name__ == "__main__":
    unittest.main()

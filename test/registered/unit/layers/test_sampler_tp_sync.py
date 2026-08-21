import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.srt.layers.sampler import Sampler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSamplerTpSync(CustomTestCase):
    def _sync(self, world_size: int):
        group = object()
        sampler = types.SimpleNamespace(
            tp_sync_group=group,
            tp_sync_group_world_size=world_size,
        )
        token_ids = torch.tensor([7], dtype=torch.int64)
        sampling_info = types.SimpleNamespace(grammars=[object()])

        with (
            patch.object(sampler_module, "SYNC_TOKEN_IDS_ACROSS_TP", False),
            patch.object(torch.distributed, "all_reduce") as all_reduce,
        ):
            Sampler._sync_token_ids_across_tp(sampler, token_ids, sampling_info)

        return all_reduce, token_ids, group

    def test_skips_single_rank_group(self):
        all_reduce, token_ids, _ = self._sync(world_size=1)

        all_reduce.assert_not_called()
        torch.testing.assert_close(token_ids, torch.tensor([7]))

    def test_syncs_multi_rank_group(self):
        all_reduce, token_ids, group = self._sync(world_size=2)

        all_reduce.assert_called_once_with(
            token_ids,
            op=torch.distributed.ReduceOp.MIN,
            group=group,
        )


if __name__ == "__main__":
    unittest.main()

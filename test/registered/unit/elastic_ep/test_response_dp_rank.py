import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.elastic_ep.topology import response_dp_rank
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestResponseDpRank(unittest.TestCase):
    def test_public_rank_matches_controller_slot(self):
        # local DP, physical join offset, attention TP, CP, public DP
        cases = [
            (None, 0, 1, 1, None),
            (7, 0, 1, 1, 7),
            (0, 32, 1, 1, 32),
            (7, 32, 1, 1, 39),
            (0, 40, 1, 1, 40),
            (3, 32, 2, 1, 19),
            (1, 32, 2, 2, 9),
        ]
        for local, offset, tp, cp, expected in cases:
            with self.subTest(local=local, offset=offset, tp=tp, cp=cp):
                parallel = SimpleNamespace(
                    dp_rank=local,
                    ep_join_rank_offset=offset,
                    attn_tp_size=tp,
                    attn_cp_size=cp,
                )
                with patch(
                    "sglang.srt.elastic_ep.topology.get_parallel",
                    return_value=parallel,
                ):
                    self.assertEqual(response_dp_rank(), expected)
                self.assertEqual(parallel.dp_rank, local)


if __name__ == "__main__":
    unittest.main()

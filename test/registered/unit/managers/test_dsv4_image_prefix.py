"""Cache matches must not split bidirectional image blocks."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


class TestImagePrefix(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_device_and_host_matches_are_capped_before_load_back(self):
        for device, host in [(256, 0), (64, 192), (0, 256), (128, 128)]:
            with self.subTest(device=device, host=host):
                req = Req("image", "", array("q", range(800)), SamplingParams())
                req.multimodal_inputs = SimpleNamespace(
                    mm_items=[
                        SimpleNamespace(
                            is_image=lambda: True,
                            offsets=[(100, 479)],
                            model_specific_data={"types": [], "perm": []},
                        )
                    ]
                )
                cache = Mock()
                cache.swa_reprefill_tail_tokens.return_value = 0
                cache.supports_mamba.return_value = False
                cache.match_prefix.side_effect = [
                    MatchResult(
                        last_device_node=None,
                        last_host_node=None,
                        best_match_node=None,
                        device_indices=torch.arange(device),
                        host_hit_length=host,
                    ),
                    MatchResult(
                        last_device_node=None,
                        last_host_node=None,
                        best_match_node=None,
                        device_indices=torch.arange(64),
                        host_hit_length=0,
                    ),
                ]
                req.init_next_round_input(cache)
                self.assertEqual(len(req.prefix_indices), 64)
                self.assertEqual(req.host_hit_length, 0)
                self.assertEqual(cache.match_prefix.call_args.args[0].key.limit, 100)
                self.assertEqual(cache.match_prefix.call_count, 2)

    def test_page_rounded_rematch_can_cross_an_earlier_image(self):
        """Re-matching must continue if page rounding lands in an earlier image."""
        req = Req("images", "", array("q", range(800)), SamplingParams())
        req.multimodal_inputs = SimpleNamespace(
            mm_items=[
                SimpleNamespace(
                    is_image=lambda: True,
                    offsets=[(100, 379), (400, 779)],
                    model_specific_data={"types": [], "perm": []},
                )
            ]
        )
        cache = Mock()
        cache.swa_reprefill_tail_tokens.return_value = 0
        cache.supports_mamba.return_value = False
        cache.match_prefix.side_effect = [
            MatchResult(
                last_device_node=None,
                last_host_node=None,
                best_match_node=None,
                device_indices=torch.arange(length),
                host_hit_length=0,
            )
            for length in (600, 256, 0)
        ]
        req.init_next_round_input(cache)
        self.assertEqual(len(req.prefix_indices), 0)
        self.assertEqual(cache.match_prefix.call_count, 3)
        self.assertEqual(
            [call.args[0].key.limit for call in cache.match_prefix.call_args_list[1:]],
            [400, 100],
        )


if __name__ == "__main__":
    unittest.main()

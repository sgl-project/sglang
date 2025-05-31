import gc
import unittest

import torch

import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class StickySchedulerTest(unittest.TestCase):

    def init_backend(self, dp, tp):
        self.dp = dp
        self.tp = tp
        self.engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            random_seed=42,
            tp_size=tp,
            dp_size=dp,
            disable_cuda_graph=True,
        )

    def clean_up(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.engine.shutdown()


    def test_sticky_round_robin(self):
        """
        Tests the round robin with sticky scheduling. If a request desires a certain data parallel rank,
        it should be sent to the corresponding rank.
        """
        self.init_backend(2, 1)
        # first request
        print(f"Test: First request sent")
        resp = self.engine.generate(
            prompt="What is knock knock?",
            sampling_params={
                "max_new_tokens": 0,
                "temperature": 0.0,
            }
        )
        dp_rank = resp["meta_info"]["dp_rank"]
        print(f"Test: First request was sent to dp rank: {dp_rank}")
        assert dp_rank == 0

        # the subsequent request specifies the same dp_rank
        print(f"Test: Second request sent")
        resp = self.engine.generate(
            prompt="What is knock knock and why is it called that?",
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
            dp_rank=dp_rank
        )
        print(f"Test: Second request was sent to dp rank: {resp['meta_info']['dp_rank']}")
        assert resp["meta_info"]["dp_rank"] == dp_rank

        # another request without specifying dp_rank
        print(f"Test: Third request sent")
        resp = self.engine.generate(
            prompt="Where is LinkedIn's headquarters?",
            sampling_params={
                "max_new_tokens": 0,
                "temperature": 0.0,
            }
        )
        print(f"Test: Third request was sent to dp rank: {resp['meta_info']['dp_rank']}")
        assert resp["meta_info"]["dp_rank"] == 1

        self.clean_up()


if __name__ == "__main__":
    unittest.main()
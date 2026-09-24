import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.hunyuan_v4_nextn import HYV4ForCausalLMNextN
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestHunyuanV4NextNWeightLoading(unittest.TestCase):
    def test_indexer_checkpoint_layout_is_permuted(self):
        model = object.__new__(HYV4ForCausalLMNextN)
        model.config = SimpleNamespace(
            num_hidden_layers=80,
            index_n_heads=2,
            index_head_dim=4,
            qk_rope_head_dim=2,
        )
        captured = []
        object.__setattr__(
            model,
            "do_load_weights",
            lambda weights, **kwargs: captured.extend(weights),
        )
        loaded_weight = torch.arange(8).reshape(8, 1)

        model.load_weights(
            [("model.mtp_layers.0.self_attn.indexer.wq_b.weight", loaded_weight)]
        )

        self.assertEqual(
            captured[0][0], "model.layers.80.self_attn.indexer.wq_b.weight"
        )
        torch.testing.assert_close(
            captured[0][1].flatten(), torch.tensor([2, 3, 0, 1, 6, 7, 4, 5])
        )


if __name__ == "__main__":
    unittest.main()

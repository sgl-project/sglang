"""The GLM-lite NextN draft must take the NextN branch of post_load_weights when called without arguments."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.model_loader.loader import post_load_weights
from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteForCausalLM
from sglang.srt.models.glm4_moe_lite_nextn import Glm4MoeLiteForCausalLMNextN
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGlm4MoeLiteNextNPostLoad(CustomTestCase):
    def test_a_bare_post_load_runs_the_nextn_branch(self):
        """A weight-update session that writes weights without load_weights (p2p) finishes with a bare
        post_load_weights; on the draft the main model's branch walks a layer range it does not have."""
        nextn_flags = []
        draft = Glm4MoeLiteForCausalLMNextN.__new__(Glm4MoeLiteForCausalLMNextN)
        # an empty module, without the layers its __init__ would build
        torch.nn.Module.__init__(draft)

        with patch.object(
            Glm4MoeLiteForCausalLM,
            "post_load_weights",
            lambda self, is_nextn=False, weight_names=None: nextn_flags.append(
                is_nextn
            ),
        ):
            post_load_weights(draft)

        self.assertEqual(nextn_flags, [True])


if __name__ == "__main__":
    unittest.main()

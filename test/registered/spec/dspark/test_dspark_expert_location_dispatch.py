import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from torch import nn

from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
from sglang.srt.models.deepseek_v4_dspark import DSparkV4Stage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_moe(*, is_nextn: bool) -> DeepseekV2MoE:
    moe = DeepseekV2MoE.__new__(DeepseekV2MoE)
    moe.is_nextn = is_nextn
    moe._nextn_expert_location_dispatch_enabled = False
    return moe


class TestNextnExpertLocationDispatch(CustomTestCase):
    def test_non_nextn_layers_use_expert_location_dispatch(self):
        moe = _make_moe(is_nextn=False)

        self.assertTrue(moe._should_use_expert_location_dispatch())

    def test_nextn_layers_preserve_legacy_behavior_by_default(self):
        moe = _make_moe(is_nextn=True)

        self.assertFalse(moe._should_use_expert_location_dispatch())

    def test_nextn_layers_can_opt_in_to_expert_location_dispatch(self):
        moe = _make_moe(is_nextn=True)

        moe.enable_nextn_expert_location_dispatch()

        self.assertTrue(moe._should_use_expert_location_dispatch())

    def test_dspark_stage_opts_in(self):
        mlp = Mock()

        def init_decoder_layer(instance, *args, **kwargs):
            nn.Module.__init__(instance)
            instance.mlp = mlp

        with patch.object(
            DeepseekV4DecoderLayer,
            "__init__",
            new=init_decoder_layer,
        ):
            DSparkV4Stage(
                config=SimpleNamespace(hidden_size=16),
                layer_id=0,
                stage_id=1,
                num_stages=3,
                num_target_layers=1,
            )

        mlp.enable_nextn_expert_location_dispatch.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

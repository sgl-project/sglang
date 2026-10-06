import unittest
from types import SimpleNamespace

from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.models.solar import _check_skips_stay_in_stage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

# The backbone skip connections of upstage/solar-pro-preview-instruct.
CONFIG = SimpleNamespace(
    num_hidden_layers=64,
    bskcn_1=[12, 20, 32, 44],
    bskcn_2=[20, 32],
    bskcn_3=[16, 24, 36, 48],
    bskcn_4=[28, 40],
)


def _stages(pp_size):
    return [
        get_pp_indices(CONFIG.num_hidden_layers, pp_rank, pp_size)
        for pp_rank in range(pp_size)
    ]


class TestSolarPipelineSplit(CustomTestCase):
    def test_a_split_that_keeps_every_skip_in_one_stage_is_accepted(self):
        for pp_size in (1, 2):
            for start, end in _stages(pp_size):
                with self.subTest(pp_size=pp_size, stage=(start, end)):
                    _check_skips_stay_in_stage(CONFIG, start, end)

    def test_a_split_that_cuts_a_skip_is_rejected(self):
        # Four stages of 16 layers: layer 16 reads the state saved at layer 12.
        start, end = _stages(4)[1]
        with self.assertRaisesRegex(ValueError, "Solar layer 16"):
            _check_skips_stay_in_stage(CONFIG, start, end)


if __name__ == "__main__":
    unittest.main()

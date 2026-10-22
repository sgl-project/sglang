"""PPU nightly accuracy entries for DeepSeek-V3.2 FP8 (tp 8).

Two reviewed configs share this file, one checkpoint across two splits of one
dataset:

  configs/dsv3.2/fp8-144g-gsm8k.json        the default below
  configs/dsv3.2/fp8-144g-gsm8k-smoke.json  the same run, 20 samples

They are configs rather than two files for the reason the Answer line's header
gives: they claim the same eight devices and differ only in the split being
scored.  Only one runs per job -- the workflow names it in
`SGLANG_PPU_ACCURACY_TEST_CONFIG` and the class falls back to the default when
it is unset.

A smoke config per checkpoint, GSM8K only, for the reason every other file on
this line states: twenty samples prove the weight load and the report round
trip in minutes rather than hours, and a smoke report carries `limit: 20` and
its own `test_id`, so it cannot be mistaken for a full run.

The serving configuration is this repository's, not the internal case's, in two
places:

  * The btv1.5 case (deepseek-v3.2-fp8_3003.json) serves with EAGLE
    speculation and states `context_length 40000`.  The reviewed schema models
    neither a draft checkpoint nor `context_length`, so this config serves the
    case's own attention pair -- prefill `fa3`, decode `flashmla`, no unified
    backend -- with `chunked_prefill_size 32768`, `cuda_graph_max_bs 128` and
    `mem_fraction_static 0.9` as the case states, and no speculation.  A score
    is only comparable to a baseline measured on the same arguments, which is
    why `baseline` is null and the first green run fills it in.

  * The case runs non-thinking: its generation block carries no thinking switch
    and this tree's V3 detector (`reasoning_parser deepseek-v3`) defaults to
    `explicit_thinking`, so no reasoning parser is stated and the workflow
    exports no thinking variables.  GSM8K is scored four-shot, the dataset
    contract's default, exactly as the case's command does.

`dist_timeout`/`watchdog_timeout` 60000 follow the DeepSeek-V4-Flash entry's
convention rather than the case's 6000: a first weight load of this checkpoint
over the NAS has no measurement yet, and the larger budget costs nothing while
it is wrong.

The estimate is a placeholder in the spirit of the line's first entries: nothing
of this checkpoint has been measured on this board yet, so it borrows GLM-5.2's
reviewed number until the first run replaces it with one.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.accuracy_suite_kit import AccuracySuiteMixin

DATA_ROOT = Path(__file__).parent

register_ppu_ci(est_time=12000, suite="nightly-accuracy-8-dsv32-ppu", nightly=True)


class TestPPUDsv32Accuracy(AccuracySuiteMixin, unittest.TestCase):
    default_test_config_path = DATA_ROOT / "configs" / "dsv3.2" / "fp8-144g-gsm8k.json"


if __name__ == "__main__":
    unittest.main()

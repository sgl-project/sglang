"""PPU nightly accuracy entries for DeepSeek-V4-Pro MXFP4-FP8 (tp 8).

Two reviewed configs share this file, one checkpoint across two splits of one
dataset:

  configs/dsv4-pro/mxfp4-fp8-144g-gsm8k.json        the default below
  configs/dsv4-pro/mxfp4-fp8-144g-gsm8k-smoke.json  the same run, 20 samples

They are configs rather than two files for the reason the Answer line's header
gives: they claim the same eight devices and differ only in the split being
scored.  Only one runs per job -- the workflow names it in
`SGLANG_PPU_ACCURACY_TEST_CONFIG` and the class falls back to the default when
it is unset.

A smoke config per checkpoint, GSM8K only, for the reason every other file on
this line states: twenty samples prove the weight load and the report round
trip in minutes rather than hours, and a smoke report carries `limit: 20` and
its own `test_id`, so it cannot be mistaken for a full run.

The serving configuration is this repository's, not the internal case's, and
the difference is deliberate, in three places:

  * The btv1.5 case (deepseek-v4-pro-mxfp4-fp8_3003.json) serves with EAGLE
    speculation and `enable_deepseek_v4_fp4_indexer`.  The reviewed schema
    models neither a draft checkpoint nor the fp4-indexer switch, so this
    config serves plain tp 8 with `attention_backend dsv4` -- the shape the
    DeepSeek-V4-Flash entry already runs on this board -- and no speculation.
    A score is only comparable to a baseline measured on the same arguments,
    which is why `baseline` is null and the first green run fills it in.

  * The btv1.5 case asks for thinking through the request body
    (`chat_template_kwargs.thinking`, `reasoning_effort: high`), which the
    reviewed generation schema does not carry.  The workflow jobs export
    `SGLANG_DEFAULT_THINKING=1` and `SGLANG_DSV4_REASONING_EFFORT=high`
    instead, so every request thinks by server default.

  * The btv1.5 case exports `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1`, which the
    reviewed server-env schema carries after it was checked against this tree
    (`sglang.srt.environ`, read by the dsv4 attention backend).  It travels in
    the config rather than the workflow.  The two btv1.1-only names
    `SGLANG_OPT_USE_TOPK_V2` and `SGLANG_DSV4_FP4_EXPERTS` are deliberately
    absent: the PPU server already defaults the first off, and the second
    defaults on, which is what an mxfp4 checkpoint wants -- the W8A8 entry's
    workflow exports both as 0 because its checkpoint is not fp4.

Both GSM8K configs follow the internal btv1.5 EvalScope client command rather
than the dataset contract's few-shot default: a thinking model scores best
zero-shot with `filters: {remove_until: "</think>"}`, which the internal command
states and the schema's `filters` key carries.  The generation budget is the
source case's `max_tokens: 384000`, except in the smoke, whose 32768 keeps
twenty samples to minutes.

The estimate is a placeholder in the spirit of the line's first entries: nothing
of this checkpoint has been measured on this board yet, so it borrows GLM-5.2's
reviewed number until the first run replaces it with one.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.accuracy_suite_kit import AccuracySuiteMixin

DATA_ROOT = Path(__file__).parent

register_ppu_ci(est_time=12000, suite="nightly-accuracy-8-dsv4pro-ppu", nightly=True)


class TestPPUDsv4ProAccuracy(AccuracySuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "dsv4-pro" / "mxfp4-fp8-144g-gsm8k.json"
    )


if __name__ == "__main__":
    unittest.main()

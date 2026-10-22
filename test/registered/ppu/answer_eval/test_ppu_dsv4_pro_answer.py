"""PPU nightly Answer entries for DeepSeek-V4-Pro-0813 (ZW-M890P, eight devices).

One reviewed config, the MXFP4-FP8 checkpoint the btv1.5 answer_144g case names:

  configs/dsv4-pro/mxfp4-fp8-144g.json

Its server arguments come from the internal btv1.5 answer_144g case
`deepseek-v4-pro-0813-mxfp4-fp8_3001` (TC deepseek-v4-pro-0813_0013): tp 8,
`attention_backend dsv4`, `cuda_graph_max_bs` 64, `mem_fraction_static` 0.8,
`disable_piecewise_cuda_graph`, `dist_timeout`/`watchdog_timeout` 6000,
`reasoning_parser deepseek-v4`, and the served name `DeepSeek-V4-Pro-0813`.
The checkpoint path is the model registry's `/ppusw/...` entry rewritten to the
NAS mount the CI jobs read, exactly as the Kimi ppu entry was.

What the case serves with and this entry does not:

  * `tool_call_parser deepseekv4` and `enable_deepseek_v4_fp4_indexer`.  The
    Answer schema does not model the first, and the second is the same flag the
    DeepSeek-V4-Flash entry deliberately leaves off: gated on board/SM checks
    this line has not reviewed for the M890P.
  * `SGLANG_OPT_USE_TOPK_V2` and `SGLANG_DSV4_FP4_EXPERTS`.  The case does not
    export them, and that absence is correct here: the tree's PPU defaults --
    the first off, the second on -- are what an mxfp4 checkpoint wants.  The
    W8A8 entry exports both as 0 because its checkpoint is not fp4; this one
    must not.

Two environment names do come straight from the case --
`SGLANG_WARMUP_TIMEOUT 3600` and `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL 1` -- and
are both members of the Answer schema's reviewed environment set; each is
declared in `sglang.srt.environ` and read by the dsv4 path in this tree.

The reasoning switch this model's template reads is spelled `thinking`, which
the case does not pass at all: its generation block carries no thinking switch,
so the model is served non-thinking.  This tree's `DeepSeekV4Detector` declares
`reasoning_default "explicit_thinking"`, so the config asks for
`{"thinking": false}` explicitly, keeping its answers comparable with the other
non-thinking entries, the way the Kimi and GLM entries do.  Temperature is
normalized to 0 and `max_tokens` keeps the case's 32768.

The estimate is a placeholder in the spirit of the line's first entries:
nothing of this checkpoint has been measured on this board yet, so it borrows
GLM-5.2's reviewed number -- a cold NAS load plus ten non-thinking answers --
until the first measured run replaces it.  The first run also has no regression
baseline: with the baseline absent the evaluator keeps strict behaviour and any
miss reddens, so the first measured run is a probe whose verdict seeds the
baseline, exactly as the GLM-5.2, Kimi and MiniMax entries were seeded.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.answer_suite_kit import AnswerSuiteMixin

DATA_ROOT = Path(__file__).parent

# Its own suite for the same reason as the GLM file: one config variable per job.
register_ppu_ci(est_time=7200, suite="nightly-answer-8-dsv4pro-ppu", nightly=True)


class TestPPUDsv4ProAnswer(AnswerSuiteMixin, unittest.TestCase):
    """Imported source is blocked: its checkpoint path is not absolute.

    The source module description above predates this validation finding. Direct
    invocation deliberately fails schema validation until a staged path is reviewed.
    """

    data_root = DATA_ROOT
    default_test_config_path = (
        Path(__file__).resolve().parents[4]
        / "scripts/ci/ppu/blocked_configs/dsv4-pro-answer.json"
    )


if __name__ == "__main__":
    unittest.main()

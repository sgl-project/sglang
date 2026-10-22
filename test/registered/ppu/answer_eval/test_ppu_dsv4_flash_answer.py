"""PPU nightly Answer entries for DeepSeek-V4-Flash (ZW-M890P, eight devices).

One reviewed config, the W8A8-INT8 checkpoint the cluster holds for this model:

  configs/dsv4-flash/w8a8-int8-144g.json

Its server arguments come from the internal btv1.1 P0_daily case
`deepseek-v4-flash-a8w8_3001`, the reference the reviewer pointed at, and agree
with the internal btv1.5 answer_96g case `deepseek-v4-flash-w8a8-int8_3001` on
every point the two share: tp 8, `w8a8_int8`, `cuda_graph_max_bs` 64,
`mem_fraction_static` 0.8, `disable_piecewise_cuda_graph`, and the
`SGLANG_WARMUP_TIMEOUT` 3600 export.  `dist_timeout`/`watchdog_timeout` 60000
follow the btv1.1 case, and the btv1.5 `server_cmds` answer command carries the
same pair, so this entry keeps them even though the line's other configs use the
600/6000 convention.

What the btv1.1 case serves with and this entry does not:

  * The EAGLE speculative trio and the deepep/dp family (moe_a2a_backend,
    dp_size, enable_dp_attention, enable_dp_lm_head, moe_dense_tp_size).  The
    Answer schema does not model those switches, and ten corpus answers do not
    need them.
  * `tool_call_parser deepseekv4`.  The Answer schema does not model it, and the
    internal btv1.5 answer case does not set it either.
  * `SGLANG_SAIL_DSV4_USE_FLASH_MLA_SPARSE_FWD=0`.  Nothing in this tree reads
    that name (vendor-fork-only), so carrying it would state a setting no run
    honours.
  * The btv1.5 answer case's `enable_deepseek_v4_fp4_indexer`.  The referenced
    btv1.1 case does not set it, and in this tree the flag is gated on board/SM
    checks this line has not reviewed for the M890P.

Two environment names do come straight from the btv1.1 case --
`SGLANG_OPT_USE_TOPK_V2=0` and `SGLANG_DSV4_FP4_EXPERTS=0` -- and are therefore
members of the Answer schema's reviewed environment set; both are declared in
`sglang.srt.environ` and read by the dsv4 code paths in this tree.

The reasoning switch this model's template reads is spelled `thinking`, which is
what the btv1.1 case passes (`chat_template_kwargs {"thinking": true}` with
`reasoning_effort "high"`).  This tree's `DeepSeekV4Detector` declares
`reasoning_default "explicit_thinking"`: the model does not think unless asked
to.  The config therefore asks for `{"thinking": false}` explicitly, keeping its
answers comparable with the other non-thinking entries, the way the Kimi and
GLM entries do; the internal btv1.5 answer case serves the same way (its
generation block carries no thinking switch at all).

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
register_ppu_ci(est_time=7200, suite="nightly-answer-8-dsv4flash-ppu", nightly=True)


class TestPPUDsv4FlashAnswer(AnswerSuiteMixin, unittest.TestCase):
    data_root = DATA_ROOT
    default_test_config_path = (
        DATA_ROOT / "configs" / "dsv4-flash" / "w8a8-int8-144g.json"
    )


if __name__ == "__main__":
    unittest.main()

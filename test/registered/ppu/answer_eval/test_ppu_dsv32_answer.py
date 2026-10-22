"""PPU nightly Answer entries for DeepSeek-V3.2 FP8 (ZW-M890P, eight devices).

One reviewed config, the FP8 checkpoint the btv1.5 answer_144g case names:

  configs/dsv3.2/fp8-144g.json

Its server arguments come from the internal btv1.5 answer_144g case
`deepseek-v3.2-fp8_3001` (TC deepseek-v3.2_0004): tp 8, the sparse-attention
pair `flashmla_sparse` for prefill and decode, `chunked_prefill_size` 32768,
`cuda_graph_max_bs` 64, `mem_fraction_static` 0.8,
`disable_shared_experts_fusion`, `watchdog_timeout` 600 and no `dist_timeout`.
The case spells the attention family `nsa`; this tree renamed the whole family
to `dsa` and kept the old spellings only as deprecated aliases, so the config
states `attention_backend dsa` with `dsa_prefill_backend`/`dsa_decode_backend`
`flashmla_sparse` -- the case's exact pair.

What the case serves with and this entry does not:

  * The DP attention pair `enable_dp_attention`/`dp_size 8`.  The Answer schema
    does not model the DP family, and ten corpus answers do not need it.
  * `USE_ACEXT_CUDA=0`, the case's only environment export.  Nothing on this
    tree's server path reads that plain name; the server reads
    `SGLANG_SAIL_USE_ACEXT_CUDA` (declared in `sglang.srt.environ`, default
    False), which is already the value the case asks for, so dropping the
    export is behaviour-preserving.  The config carries only the line's warmup
    budget instead, `SGLANG_WARMUP_TIMEOUT` 3600, which is also the tree's own
    default.

`reasoning_parser` is required by the Answer schema even though the case does
not set one; the config states `deepseek-v3`, the V3-family detector this tree
registers for reasoning (`deepseekv32` is a tool-call parser name, not a
reasoning one).  The detector declares `reasoning_default "explicit_thinking"`,
and the case's generation block carries no thinking switch, so the config asks
for `{"thinking": false}` explicitly, keeping its answers comparable with the
other non-thinking entries.  Temperature is normalized to 0 and `max_tokens`
keeps the case's 32768.

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
register_ppu_ci(est_time=7200, suite="nightly-answer-8-dsv32-ppu", nightly=True)


class TestPPUDsv32Answer(AnswerSuiteMixin, unittest.TestCase):
    data_root = DATA_ROOT
    default_test_config_path = DATA_ROOT / "configs" / "dsv3.2" / "fp8-144g.json"


if __name__ == "__main__":
    unittest.main()

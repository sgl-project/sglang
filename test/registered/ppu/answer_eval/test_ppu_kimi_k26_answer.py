"""PPU nightly Answer entry for Kimi-K2.6 (ZW-M890P, eight devices).

  configs/kimi2.6/mxfp4-fp8-144g.json    the only single-board config

It takes its server arguments from the internal btv1.5 answer_144g case
`kimi-k2.6-w8a8-int8_3001`, which serves this model without a unified
attention backend: it names a prefill and a decode backend instead (fa3 and
flashmla). The checkpoint the cluster holds is quantized differently from
that case, so `quantization` is left unset and its own `quantization_config`
decides. The format that case does name, W8A8-INT8, is a two-node entry in
`test_ppu_kimi_k26_w8a8_answer.py`: at 968.3 GiB it does not fit one board
group.

Earlier revisions also carried a W4A8-INT8 config as the default. It is
removed rather than kept as a red entry because it can no longer serve on
this board line at all: its W-INT4 kernels route through ACEXT, which the
ZW-M890P stack does not support, so the entry reddened on
server_start_failed on every run. The MXFP4-FP8 checkpoint is the supported
single-board format.

The reasoning switch this model's chat template reads is spelled `thinking`,
not `enable_thinking`, and SGLang's `kimi_k2` detector agrees -- it declares
`reasoning_default` "thinking".  The config therefore passes
`chat_template_kwargs {"thinking": false}` to keep the answers comparable with
the other non-thinking entries.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.answer_suite_kit import AnswerSuiteMixin

DATA_ROOT = Path(__file__).parent

# Its own suite for the same reason as the GLM file: one config variable per job.
# The estimate is a cold load from the NAS plus ten non-thinking answers, and the
# first measured run replaces it.
register_ppu_ci(est_time=5400, suite="nightly-answer-8-kimi26-ppu", nightly=True)


class TestPPUKimiK26Answer(AnswerSuiteMixin, unittest.TestCase):
    data_root = DATA_ROOT
    default_test_config_path = DATA_ROOT / "configs" / "kimi2.6" / "mxfp4-fp8-144g.json"


if __name__ == "__main__":
    unittest.main()

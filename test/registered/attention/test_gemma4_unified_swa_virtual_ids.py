# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unified memory on a hybrid sliding-window model, which nothing else covered.

The unified SWA composite mints one virtual page id per allocation and binds it
on both sides, so the swa sub-pool's `virtual_to_physical` is indexed by the
FULL side's ids while it used to be sized by its own page count. A model with
few full-attention layers and many sliding ones gives the id owner more pages,
and the top of that id space then lands off the end of the swa table. On GPU
the write is unchecked (`alloc_bind_inplace`'s `tl.store`), so the symptom
surfaces later and elsewhere: a device-side index assert on the read in
`_swa_write_loc_unified`, and a dead scheduler.

The reachability of it is a KV-budget property, not just a model property: ids
come off the front of the owner's free list and freed ids return to the back, so
the cursor sweeps the owner's range over time and the failure needs cumulative
churn past `swa.num_pages`. `--max-total-tokens 60000` puts that within one
200-example GSM8K run; at this model's default budget the swa table is 3.8M
entries and it would take dozens.

Every argument below is load-bearing: a larger pool, or synthetic prompts in
place of this eval, and the narrow table passes.

    python -m pytest test/registered/attention/test_gemma4_unified_swa_virtual_ids.py -v
"""

import types
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=65, stage="base-b", runner_config="1-gpu-large")

# Healthy 0.865-0.880 (static pool 0.870); a died-mid-run server scores
# 0.05-0.21. 1 sigma over 200 examples is ~0.024.
SCORE_THRESHOLD = 0.82


class TestGemma4UnifiedSwaVirtualIds(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "google/gemma-4-E2B-it"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "triton",
                "--enable-unified-memory",
                "--disable-radix-cache",
                "--mem-fraction-static",
                "0.8",
                # The trigger: this is what brings `swa.num_pages` within one
                # eval's worth of churn.
                "--max-total-tokens",
                "60000",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k_survives_churn_past_the_swa_page_count(self):
        metrics = run_eval(
            types.SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                num_examples=200,
                num_threads=128,
            )
        )
        self.assertGreaterEqual(metrics["score"], SCORE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()

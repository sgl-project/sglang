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
"""E2E test for logical-page KV cache sharding on an MLA model.

MLA latent KV is replicated across attention-TP ranks; with
``--enable-kv-cache-sharding`` each prefill TP rank stores only its
page-interleaved stripe (~TPx unique KV capacity for the group), reads the
prefix through the assembled scratch (chunked-prefix MHA path), and sends only
its owned pages to decode (which pulls from every prefill TP rank).

Modeled on the DSA cache-layer-split E2E test; uses the PD-disaggregation
fixture because KV sharding is a PD-prefill-only feature. 4 GPUs: prefill
TP2 on GPUs 0-1, decode TP2 on GPUs 2-3.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MLA_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=900, stage="base-c", runner_config="4-gpu-h100")


class TestMLAKVCacheSharding(PDDisaggregationServerBase, GSM8KMixin):
    model = DEFAULT_MLA_MODEL_NAME_FOR_TEST

    gsm8k_accuracy_thres = 0.70
    gsm8k_num_questions = 200
    gsm8k_num_threads = 128

    extra_prefill_args = [
        "--tp",
        "2",
        "--attention-backend",
        "fa3",
        "--enable-kv-cache-sharding",
        "--page-size",
        "64",
        "--chunked-prefill-size",
        "2048",
        "--max-prefill-tokens",
        "2048",
        "--mem-fraction-static",
        "0.8",
    ]

    extra_decode_args = [
        "--tp",
        "2",
        "--attention-backend",
        "fa3",
        "--page-size",
        "64",
        "--mem-fraction-static",
        "0.8",
        "--base-gpu-id",
        "2",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.launch_all()


if __name__ == "__main__":
    unittest.main()

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
"""E2E test for logical-page KV cache sharding on a GQA model under prefill CP.

Prefill CP replicates GQA KV storage across attention-CP ranks; with
``--enable-kv-cache-sharding`` each CP rank stores only its page-interleaved
stripe (~CPx unique KV capacity for the group), extend attention reads the
assembled scratch through the translated FA3 page table, and each CP rank
sends only its owned pages to decode.

Modeled on the DSA cache-layer-split E2E test; PD-disaggregation fixture,
8 GPUs: prefill TP4 (attn-CP 2) on GPUs 0-3, decode TP4 on GPUs 4-7.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=900, stage="base-c", runner_config="8-gpu-h200")


class TestGQACPKVCacheSharding(PDDisaggregationServerBase, GSM8KMixin):
    model = "Qwen/Qwen3-30B-A3B-FP8"

    gsm8k_accuracy_thres = 0.90
    gsm8k_num_questions = 200
    gsm8k_num_threads = 128

    # Legacy zigzag prefill CP (the flag set of test_gqa_prefill_cp_legacy.py)
    # plus KV sharding across the attn-CP group.
    extra_prefill_args = [
        "--tp",
        "4",
        "--moe-dp-size",
        "2",
        "--ep-size",
        "2",
        "--attn-cp-size",
        "2",
        "--enable-prefill-context-parallel",
        "--attention-backend",
        "fa3",
        "--enable-kv-cache-sharding",
        "--page-size",
        "64",
        "--chunked-prefill-size",
        "4096",
        "--max-prefill-tokens",
        "4096",
        "--mem-fraction-static",
        "0.8",
    ]
    extra_prefill_env = {"SGLANG_ENABLE_CP_V2": "0"}

    # Decode attn-TP matches the prefill attn-TP (tp 4 / cp 2 -> attn_tp 2):
    # the owner-strided send filter pairs pages positionally per (tp, cp)
    # sender; a TP-size mismatch would add non-MLA head slicing on top and
    # fragment every page transfer.
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
        "4",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.launch_all()


if __name__ == "__main__":
    unittest.main()

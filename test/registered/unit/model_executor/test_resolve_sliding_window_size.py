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
"""`resolve_sliding_window_size` and the Inkling MTP head's window convention.

BUG REGRESSION. The resolver prefers a model's own
`get_attention_sliding_window_size`, and Inkling's trunk answers one less
than its extent. Its MTP head declared no such method, so the resolver fell
through to the config field: the head's backend ran one token wider than
the trunk's window layers, and a banded depth's per-step window rail read
one token further back than the swa sub-pool retains.

Registered on a CUDA runner, not the CPU gate: `_head` needs the real
`InklingForConditionalGenerationMTP`, whose module pulls in the flash-attn
cute interface and therefore `cutlass`, which a CPU image does not carry.

    python -m pytest test/registered/unit/model_executor/test_resolve_sliding_window_size.py -v
"""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    resolve_sliding_window_size,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _head(*, depth, local_layer_ids=(1,), local_extent=64, window=128):
    """A stand-in for the MTP head: only what the window method reads."""
    from sglang.srt.models.inkling import InklingForConditionalGenerationMTP

    head = SimpleNamespace(
        draft_model_idx=depth,
        text_config=SimpleNamespace(
            mtp_local_layer_ids=list(local_layer_ids),
            mtp_local_extent=local_extent,
            sliding_window_size=window,
        ),
    )
    head.get_attention_sliding_window_size = (
        InklingForConditionalGenerationMTP.get_attention_sliding_window_size.__get__(
            head
        )
    )
    return head


class TestInklingMTPWindowConvention(CustomTestCase):
    def test_a_banded_depth_runs_one_under_its_own_extent(self):
        self.assertEqual(_head(depth=1).get_attention_sliding_window_size(), 63)

    def test_a_full_depth_runs_one_under_the_trunk_window(self):
        self.assertEqual(_head(depth=0).get_attention_sliding_window_size(), 127)

    def test_the_resolver_prefers_the_head_over_the_config_field(self):
        """The config keeps the raw extent (the swa host admission reads it);
        the runner's window must be the head's convention, not that field."""
        model_config = SimpleNamespace(
            is_hybrid_swa=True, sliding_window_size=64, attention_chunk_size=None
        )
        self.assertEqual(resolve_sliding_window_size(_head(depth=1), model_config), 63)
        self.assertEqual(
            resolve_sliding_window_size(SimpleNamespace(), model_config), 64
        )


if __name__ == "__main__":
    unittest.main()

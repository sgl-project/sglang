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
"""`resolve_sliding_window_size`, the Inkling MTP head's window convention, and
the window of sliding layers built from an HF config.

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

import torch

from sglang.srt.layers.dp_attention import initialize_dp_attention_flags
from sglang.srt.layers.layer_boundary.factories import layer_stack
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    resolve_sliding_window_size,
)
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="extra-a", runner_config="1-gpu-small")


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


class TestHFSlidingWindowLayers(CustomTestCase):
    """BUG REGRESSION. HF's `sliding_window` counts the query and
    `sliding_window_size` does not, so a layer built from an HF config
    attended one key past the checkpoint's window."""

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        server = ServerArgs(model_path="dummy", device="cpu")
        publish(server, role="test", ranks=SpawnRanks(world_rank=0))
        initialize_dp_attention_flags(server)

    def test_sliding_layers_exclude_the_query_from_the_window(self):
        from transformers import Cohere2Config, Cohere2MoeConfig, Gemma3nTextConfig

        from sglang.srt.configs.step3p5 import Step3p5Config
        from sglang.srt.models.cohere2_moe import Cohere2MoeAttention
        from sglang.srt.models.commandr import CohereAttention
        from sglang.srt.models.gemma3n_causal import Gemma3nAttention
        from sglang.srt.models.step3p5 import Step3p5DecoderLayer
        from sglang.srt.models.transformers import TransformersBase

        window = 8
        hf = dict(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            vocab_size=128,
            max_position_embeddings=64,
            sliding_window=window,
            layer_types=["sliding_attention", "full_attention"],
        )
        step = Step3p5Config(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_attention_groups=2,
            num_hidden_layers=2,
            head_dim=16,
            vocab_size=128,
            max_position_embeddings=64,
            sliding_window=window,
            layer_types=["sliding_attention", "full_attention"],
            yarn_only_types=[],
            moe_layers_enum="2",
            rope_theta=[10000.0, 10000.0],
            partial_rotary_factors=[1.0, 1.0],
            swiglu_limits_shared=None,
            attention_other_setting={
                "num_attention_heads": 4,
                "num_attention_groups": 2,
            },
            use_head_wise_attn_gate=False,
        )
        cohere2 = Cohere2Config(**hf)
        hf_backend = SimpleNamespace(
            text_config=cohere2,
            config=cohere2,
            model=torch.nn.Module(),
            start_layer=0,
            end_layer=1,
            quant_config=None,
        )
        with layer_stack():
            layers = {
                "commandr": CohereAttention(cohere2).attn,
                "cohere2_moe": Cohere2MoeAttention(Cohere2MoeConfig(**hf)).attn,
                "gemma3n": Gemma3nAttention(
                    layer_id=0,
                    config=Gemma3nTextConfig(**hf),
                    max_position_embeddings=64,
                ).attn,
                "step3p5": Step3p5DecoderLayer(step).self_attn.attn,
                "transformers": TransformersBase._create_attention_instances(
                    hf_backend
                )[0],
            }
        for name, attn in layers.items():
            with self.subTest(model=name):
                self.assertEqual(attn.sliding_window_size, window - 1)


if __name__ == "__main__":
    unittest.main()

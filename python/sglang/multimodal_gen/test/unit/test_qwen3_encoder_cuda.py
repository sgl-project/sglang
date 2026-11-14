# SPDX-License-Identifier: Apache-2.0
"""Qwen3 text encoder RoPE fast path on CUDA; no checkpoint needed."""

import pytest
import torch

import sglang.multimodal_gen.runtime.models.encoders.qwen3 as qwen3
from sglang.multimodal_gen.configs.models.encoders.qwen3 import (
    Qwen3TextArchConfig,
    Qwen3TextConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    Flux2KleinPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import utils as rope_utils
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    set_global_server_args,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform.is_cuda())
    or rope_utils.flashinfer_apply_rope_inplace is None,
    reason="requires CUDA and FlashInfer RoPE",
)


@pytest.fixture(scope="module")
def encoder():
    set_global_server_args(
        ServerArgs(
            model_path="black-forest-labs/FLUX.2-klein-4B",
            num_gpus=1,
            pipeline_config=Flux2KleinPipelineConfig(),
            attention_backend="torch_sdpa",
        )
    )
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    config = Qwen3TextConfig(
        arch_config=Qwen3TextArchConfig(
            vocab_size=64,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
        )
    )
    # the loader fills this from the checkpoint's config.json
    config.arch_config.rope_parameters = {"rope_type": "default", "rope_theta": 1e6}
    torch.manual_seed(0)
    # the text encoder loader builds the model under its serving precision
    with set_default_torch_dtype(torch.bfloat16):
        model = qwen3.Qwen3ForCausalLM(config).cuda().eval()
    for name, param in model.named_parameters():
        if name.endswith("norm.weight"):
            torch.nn.init.ones_(param)
        else:
            torch.nn.init.normal_(param, std=0.05)
    return model


def _encode(encoder, input_ids, attention_mask):
    with set_forward_context(None, None):
        return encoder(input_ids=input_ids, attention_mask=attention_mask).hidden_states


@torch.inference_mode()
def test_padded_prompt_encodes_bitwise_like_the_native_rope(encoder, monkeypatch):
    input_ids = torch.randint(0, 64, (1, 40), device="cuda")
    attention_mask = torch.zeros(1, 40, dtype=torch.long, device="cuda")
    attention_mask[:, :23] = 1

    calls = []
    fused_rope = rope_utils.flashinfer_apply_rope_inplace

    def counting_rope(**kwargs):
        calls.append(kwargs["query"].shape)
        return fused_rope(**kwargs)

    monkeypatch.setattr(
        qwen3, "flashinfer_apply_rope_inplace", counting_rope, raising=False
    )
    fast = [_encode(encoder, input_ids, attention_mask) for _ in range(2)]
    assert len(calls) == 2 * len(encoder.layers)

    monkeypatch.setattr(qwen3, "flashinfer_apply_rope_inplace", None)
    native = _encode(encoder, input_ids, attention_mask)

    for run in fast:
        assert len(run) == len(native)
        for got, want in zip(run, native):
            assert torch.equal(got, want)

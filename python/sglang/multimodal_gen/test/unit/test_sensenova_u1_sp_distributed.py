# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --standalone --nproc-per-node=2 -m pytest -q <this file>."""

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    build_shard_plan,
    gather_seq,
    shard_like,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOLLMConfig,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
    Qwen3Attention,
    set_attn_backend,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="requires two CUDA ranks launched by torchrun",
)


@pytest.fixture(scope="module", autouse=True)
def distributed():
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=2,
        ulysses_degree=2,
        ring_degree=1,
    )
    set_attn_backend("sdpa")
    yield


def _config(*, use_sp: bool) -> NEOLLMConfig:
    config = NEOLLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        pad_token_id=0,
        attention_bias=False,
        attention_dropout=0.0,
        use_sglang_tp=True,
    )
    config.use_sglang_sp = use_sp
    config._attn_implementation = "eager"
    return config


@torch.no_grad()
def test_sensenova_u1_sp_denoise_attention_matches_full_sequence():
    torch.manual_seed(23)
    reference = Qwen3Attention(_config(use_sp=False), layer_idx=0).cuda().eval()
    sharded = Qwen3Attention(_config(use_sp=True), layer_idx=0).cuda().eval()
    sharded.load_state_dict(reference.state_dict())

    batch_size = 2
    prefix_width = 5
    image_len = 5
    hidden = torch.randn(batch_size, image_len, 64, device="cuda")
    positions = torch.arange(image_len, device="cuda").expand(batch_size, -1)
    indexes = torch.stack(
        [positions + prefix_width, positions // 3, positions % 3], dim=1
    )
    prefix_k = torch.randn(batch_size, 2, prefix_width, 16, device="cuda")
    prefix_v = torch.randn(batch_size, 2, prefix_width, 16, device="cuda")
    prefix_lengths = [3, 5]

    reference_layer = SimpleNamespace(keys=prefix_k, values=prefix_v)
    reference_cache = SimpleNamespace(layers=[reference_layer])
    prefix_mask = (
        torch.arange(prefix_width, device="cuda")[None, :]
        < torch.tensor(prefix_lengths, device="cuda")[:, None]
    )
    image_mask = torch.ones(batch_size, image_len, dtype=torch.bool, device="cuda")
    attention_mask = torch.cat([prefix_mask, image_mask], dim=1)[:, None, None, :]
    expected, _ = reference.forward_gen(
        hidden,
        indexes,
        attention_mask,
        reference_cache,
        update_cache=False,
    )

    shard = build_shard_plan(image_len)
    local_hidden = shard_like(hidden, shard, dim=1)
    local_indexes = shard_like(indexes, shard, dim=-1, pad_mode="repeat_last")
    sharded_layer = SimpleNamespace(
        keys=prefix_k,
        values=prefix_v,
        sensenova_sp_prefix_lengths=prefix_lengths,
        sensenova_sp_image_valid_len=image_len,
    )
    sharded_cache = SimpleNamespace(layers=[sharded_layer])
    with set_forward_context(current_timestep=0, attn_metadata=None):
        actual_local, _ = sharded.forward_gen(
            local_hidden,
            local_indexes,
            None,
            sharded_cache,
            update_cache=False,
        )
    actual = gather_seq(actual_local, image_len, dim=1)

    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)

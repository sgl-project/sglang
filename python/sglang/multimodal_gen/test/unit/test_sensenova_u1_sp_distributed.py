# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --standalone --nproc-per-node=2 -m pytest -q <this file>."""

import os
from types import SimpleNamespace

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

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
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
    NEOChatModel,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_fm_modules import (
    ConvDecoder,
)

_NPU_AVAILABLE = bool(
    getattr(torch, "npu", None) is not None and torch.npu.is_available()
)
_DEVICE = "cuda" if torch.cuda.is_available() else "npu"
_DTYPE = torch.bfloat16 if _DEVICE == "npu" else torch.float32
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or _NPU_AVAILABLE)
    or int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="requires two CUDA or NPU ranks launched by torchrun",
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
    reference = (
        Qwen3Attention(_config(use_sp=False), layer_idx=0)
        .to(device=_DEVICE, dtype=_DTYPE)
        .eval()
    )
    sharded = (
        Qwen3Attention(_config(use_sp=True), layer_idx=0)
        .to(device=_DEVICE, dtype=_DTYPE)
        .eval()
    )
    sharded.load_state_dict(reference.state_dict())

    batch_size = 2
    prefix_width = 5
    image_len = 5
    hidden = torch.randn(batch_size, image_len, 64, device=_DEVICE, dtype=_DTYPE)
    positions = torch.arange(image_len, device=_DEVICE).expand(batch_size, -1)
    indexes = torch.stack(
        [positions + prefix_width, positions // 3, positions % 3], dim=1
    )
    prefix_k = torch.randn(
        batch_size, 2, prefix_width, 16, device=_DEVICE, dtype=_DTYPE
    )
    prefix_v = torch.randn(
        batch_size, 2, prefix_width, 16, device=_DEVICE, dtype=_DTYPE
    )
    prefix_lengths = [3, 5]

    reference_layer = SimpleNamespace(keys=prefix_k, values=prefix_v)
    reference_cache = SimpleNamespace(layers=[reference_layer])
    prefix_mask = (
        torch.arange(prefix_width, device=_DEVICE)[None, :]
        < torch.tensor(prefix_lengths, device=_DEVICE)[:, None]
    )
    image_mask = torch.ones(batch_size, image_len, dtype=torch.bool, device=_DEVICE)
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

    tolerance = 2e-2 if _DEVICE == "npu" else 2e-4
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@torch.no_grad()
def test_sensenova_u1_sp_pixel_head_matches_full_decoder():
    torch.manual_seed(37)
    token_h, token_w, pixel_size = 9, 13, 32
    image_len = token_h * token_w
    hidden = torch.arange(image_len * 8, device=_DEVICE, dtype=torch.float32)
    hidden = (hidden.reshape(1, image_len, 8) / 100).to(_DTYPE)
    head = ConvDecoder(8, hidden_dim=8).to(device=_DEVICE, dtype=_DTYPE).eval()
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=lambda **kwargs: SimpleNamespace(
                last_hidden_state=kwargs["inputs_embeds"]
            )
        ),
        use_pixel_head=True,
        downsample_ratio=1,
        patch_size=pixel_size,
        fm_modules={"fm_head": head},
        config=SimpleNamespace(t_eps=0.02),
    )
    shard = build_shard_plan(image_len)
    local_hidden = shard_like(hidden, shard, dim=1)
    z = torch.zeros(
        1, shard.local_len, pixel_size * pixel_size * 3, device=_DEVICE, dtype=_DTYPE
    )
    t = torch.tensor(0.25, device=_DEVICE)
    actual = NEOChatModel._t2i_predict_v(
        model,
        local_hidden,
        None,
        None,
        None,
        t,
        z,
        image_token_num=shard.local_len,
        image_size=(token_w * pixel_size, token_h * pixel_size),
    )

    full = head(hidden.reshape(1, token_h, token_w, 8).permute(0, 3, 1, 2))
    expected = full.reshape(1, 3, token_h, pixel_size, token_w, pixel_size)
    expected = expected.permute(0, 2, 4, 3, 5, 1).reshape(
        1, image_len, pixel_size * pixel_size * 3
    )
    expected = shard_like(expected / (1 - t), shard, dim=1)
    tolerance = 2e-2 if _DEVICE == "npu" else 2e-4
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

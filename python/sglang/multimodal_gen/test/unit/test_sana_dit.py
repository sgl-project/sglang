# SPDX-License-Identifier: Apache-2.0
"""SANA DiT forwards prepare request-constant inputs once, not per block."""

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.multimodal_gen.configs.models.dits.sana import SanaConfig
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.sana import SanaTransformer2DModel

# the DiT's fused kernels need a GPU on a GPU platform; elsewhere run on the CPU
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _CountBoolCasts(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (
            func is torch.ops.aten._to_copy.default
            and kwargs.get("dtype") is torch.bool
        ):
            self.count += 1
        return func(*args, **kwargs)


@torch.no_grad()
def test_prompt_mask_is_cast_once_per_forward(single_process_model_parallel):
    """SANA is launch-bound in eager mode: the int64 prompt mask is cast to
    bool once per forward instead of once in every block's cross-attention."""
    config = SanaConfig()
    for key, value in dict(
        in_channels=4,
        out_channels=4,
        num_layers=3,
        attention_head_dim=8,
        num_attention_heads=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=8,
        cross_attention_dim=16,
        caption_channels=12,
    ).items():
        setattr(config.arch_config, key, value)
    model = SanaTransformer2DModel(config, {})
    generator = torch.Generator().manual_seed(0)
    for param in model.parameters():
        param.copy_(0.2 * torch.randn(param.shape, generator=generator))
    model.to(_DEVICE)
    inputs = dict(
        hidden_states=torch.randn(1, 4, 4, 4, generator=generator),
        encoder_hidden_states=torch.randn(1, 5, 12, generator=generator),
        timestep=torch.tensor([500.0]),
    )
    inputs = {key: value.to(_DEVICE) for key, value in inputs.items()}
    mask = torch.tensor([[1, 1, 1, 0, 0]], device=_DEVICE)
    with set_forward_context(current_timestep=0, attn_metadata=None):
        with _CountBoolCasts() as casts:
            out = model(**inputs, encoder_attention_mask=mask)
        reference = model(**inputs, encoder_attention_mask=mask.bool())
    assert casts.count == 1
    torch.testing.assert_close(out, reference, atol=0, rtol=0)

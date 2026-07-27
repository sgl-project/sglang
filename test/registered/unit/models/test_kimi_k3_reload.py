from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.attn_residual import get_cw, refresh_cw
from sglang.srt.models.kimi_k3 import KimiK3MultiModalProjector
from sglang.srt.models.kimi_k25 import MoonVision3dPatchEmbed
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_attn_res_combined_weight_refreshes_every_cached_dtype_in_place():
    """A weight reload re-runs post_load_weights, but cuda graph capture has
    already baked in the cached cw address, so the refresh has to write through
    the existing tensors -- for every cached dtype, since the fast kernel reads
    the bf16 slot and the triton fallback the fp32 one. Rebinding a cache entry,
    or refreshing only one dtype, leaves replays reading the weights from before
    the reload."""
    proj = SimpleNamespace(weight=nn.Parameter(torch.tensor([[1.0, 2.0, 3.0]])))
    norm = SimpleNamespace(weight=nn.Parameter(torch.tensor([4.0, 5.0, 6.0])))

    cached = {dtype: get_cw(proj, norm, dtype=dtype) for dtype in (torch.float32, torch.bfloat16)}
    data_ptrs = {dtype: cw.data_ptr() for dtype, cw in cached.items()}

    proj.weight.data.copy_(torch.tensor([[2.0, 3.0, 4.0]]))
    norm.weight.data.copy_(torch.tensor([5.0, 6.0, 7.0]))
    with torch.no_grad():
        refresh_cw(proj, norm)

    for dtype, previous in cached.items():
        refreshed = get_cw(proj, norm, dtype=dtype)
        assert refreshed is previous
        assert refreshed.data_ptr() == data_ptrs[dtype]
        torch.testing.assert_close(refreshed, torch.tensor([10.0, 18.0, 28.0], dtype=dtype))


def test_attn_res_refresh_before_first_get_is_a_noop():
    """refresh_cw runs ahead of the warm-up get_cw calls on the very first load,
    when nothing is cached yet; it must not fabricate a cache entry that get_cw
    would then hand out instead of computing from the loaded weights."""
    proj = SimpleNamespace(weight=nn.Parameter(torch.tensor([[1.0, 2.0, 3.0]])))
    norm = SimpleNamespace(weight=nn.Parameter(torch.tensor([4.0, 5.0, 6.0])))

    refresh_cw(proj, norm)

    assert not hasattr(proj, "_attn_res_cw_cache")
    torch.testing.assert_close(get_cw(proj, norm), torch.tensor([4.0, 10.0, 18.0]))


def test_kimi_k3_patch_embed_disables_bias():
    patch_embed = MoonVision3dPatchEmbed(
        out_dim=4,
        patch_size=2,
        patch_embed_proj_bias=False,
    )

    assert patch_embed.proj.bias is None


def test_kimi_k3_projector_matches_checkpoint_structure():
    config = SimpleNamespace(
        merge_kernel_size=(2, 2),
        vt_hidden_size=2,
        text_hidden_size=4,
        projector_ln_eps=1e-5,
    )
    projector = KimiK3MultiModalProjector(config)

    assert set(projector.state_dict()) == {
        "linear_1.weight",
        "linear_2.weight",
        "post_norm.weight",
    }
    output = projector(torch.randn(3, 4, 2))
    assert output.shape == (3, 4)

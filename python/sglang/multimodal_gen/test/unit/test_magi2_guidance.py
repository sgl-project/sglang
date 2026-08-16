# SPDX-License-Identifier: Apache-2.0
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2.guidance import (
    apply_guidance,
    skimming_mask,
)


def _inputs(seed: int = 0):
    torch.manual_seed(seed)
    return (
        torch.randn(4, 8),
        torch.randn(4, 8),
        torch.randn(4, 8),
    )


def test_plain_guidance_is_the_textbook_formula():
    latent, cond, uncond = _inputs()
    out = apply_guidance(latent=latent, cond=cond, uncond=uncond, guidance_scale=5.0)
    torch.testing.assert_close(out, uncond + 5.0 * (cond - uncond))


def test_skimming_changes_the_result_but_only_where_masked():
    latent, cond, uncond = _inputs()
    plain = apply_guidance(latent=latent, cond=cond, uncond=uncond, guidance_scale=5.0)
    skimmed = apply_guidance(
        latent=latent,
        cond=cond,
        uncond=uncond,
        guidance_scale=5.0,
        skimmed_scale=3.0,
    )

    # Two rounds, so the bound is the union of both masks: the second round
    # recomputes from the damped uncond with the mask's arguments swapped, which
    # is how it reaches elements the first round's sign tests excluded.
    first = skimming_mask(latent=latent, cond=cond, uncond=uncond, guidance_scale=5.0)
    damped = torch.where(first, torch.lerp(cond, uncond, 0.5), uncond)
    second = skimming_mask(latent=latent, cond=damped, uncond=cond, guidance_scale=5.0)
    touched = first | second

    assert first.any() and not first.all()
    assert touched.any() and not touched.all()
    torch.testing.assert_close(plain[~touched], skimmed[~touched])
    assert not torch.allclose(plain[touched], skimmed[touched])

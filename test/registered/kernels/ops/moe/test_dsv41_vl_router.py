import itertools
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.multimodal.dsv41.vl_routing import vision_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def route(
    logits,
    text,
    image,
    tokens,
    *,
    image_token_id,
    num_token_non_padded=None,
    **config,
):
    moe = SimpleNamespace(
        gate=SimpleNamespace(
            e_score_correction_bias=text, e_score_correction_bias_vl=image
        ),
        config=SimpleNamespace(image_token_id=image_token_id),
        topk=SimpleNamespace(topk_config=SimpleNamespace(**config)),
    )
    out = vision_topk(moe, logits, tokens, num_token_non_padded)
    return out.topk_weights, out.topk_ids


def reference(
    logits,
    text,
    image,
    tokens,
    *,
    image_token_id,
    top_k,
    renormalize,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output,
    num_token_non_padded=None,
):
    scores = F.softplus(logits.float()).sqrt()
    bias = (
        text
        if tokens is None
        else torch.where((tokens == image_token_id)[:, None], image, text)
    )
    indices = (scores + bias).topk(top_k, dim=-1).indices
    weights = scores.gather(-1, indices)
    if renormalize and top_k > 1:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    if apply_routed_scaling_factor_on_output:
        weights = weights * routed_scaling_factor
    indices = indices.int()
    if num_token_non_padded is not None:
        pad = (
            torch.arange(logits.shape[0], device=logits.device) >= num_token_non_padded
        )
        indices[pad] = -1
        weights[pad] = 0
    return weights, indices


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestV41VLRouter(unittest.TestCase):
    def make_case(self, rows, experts, dtype, k=6):
        logits = torch.randn(rows, experts * 2, device="cuda", dtype=dtype)[:, ::2]
        text = torch.randn(experts * 2, device="cuda", dtype=dtype)[::2]
        image = -text
        tokens = (torch.arange(rows * 2, device="cuda", dtype=torch.int64) % 4)[::2]
        kwargs = dict(
            image_token_id=0,
            top_k=k,
            renormalize=True,
            routed_scaling_factor=1.5,
            apply_routed_scaling_factor_on_output=True,
        )
        return (logits, text, image, tokens), kwargs

    def assert_parity(self, args, kwargs):
        expected = reference(*args, **kwargs)
        actual = route(*args, **kwargs)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
        torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=3e-6)
        return actual

    def test_modality_dtype_padding(self):
        torch.manual_seed(114)
        for rows, experts, dtype, k in itertools.product(
            (0, 1, 5, 17), (128, 384), (torch.float32, torch.bfloat16), (1, 3, 6)
        ):
            with self.subTest(rows=rows, experts=experts, dtype=dtype, k=k):
                args, kwargs = self.make_case(rows, experts, dtype, k)
                self.assert_parity(args, kwargs)
                for real in (0, max(0, rows - 1)):
                    kwargs["num_token_non_padded"] = torch.tensor(
                        real, device="cuda", dtype=torch.int32
                    )
                    self.assert_parity(args, kwargs)


if __name__ == "__main__":
    unittest.main()

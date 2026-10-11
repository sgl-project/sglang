# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits import qwen_image


@pytest.mark.parametrize("num_img, num_txt", [(1, 1), (2, 1)])  # 2:1 is zero_cond_t
def test_gather_modulation_shards_matches_full_projection(
    monkeypatch, num_img, num_txt
):
    tp_size, dim = 2, 8
    torch.manual_seed(0)
    w_img, w_txt = torch.randn(6 * dim, dim), torch.randn(6 * dim, dim)
    x_img, x_txt = torch.randn(num_img, dim), torch.randn(num_txt, dim)

    def local(rank):
        rows = slice(rank * 6 * dim // tp_size, (rank + 1) * 6 * dim // tp_size)
        return x_img @ w_img[rows].T, x_txt @ w_txt[rows].T

    shards = [torch.cat(local(rank), dim=0) for rank in range(tp_size)]

    def fake_all_gather(tensor, dim=-1):
        assert torch.equal(tensor, shards[0])
        return torch.cat(shards, dim=dim)

    monkeypatch.setattr(qwen_image, "tensor_model_parallel_all_gather", fake_all_gather)
    img, txt = qwen_image._gather_modulation_shards(*local(0))
    torch.testing.assert_close(img, x_img @ w_img.T)
    torch.testing.assert_close(txt, x_txt @ w_txt.T)

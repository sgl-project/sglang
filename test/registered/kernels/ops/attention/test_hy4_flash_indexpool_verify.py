import unittest

import torch

from sglang.kernels.ops.attention.dsa import hy4_flash_indexpool as K
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

RATIO = 4
PAGE = 64


def _reference(keys, norm_w, out_loc, starts, pages, raw, weights, buf):
    """The unfused verify path: RMSNorm + store of every row, then pool the
    groups each request's block completes; padded requests (out_loc 0) skip."""
    bs = starts.shape[0]
    width = keys.shape[0] // bs
    normed = torch.nn.functional.rms_norm(keys, (128,), norm_w, weights["key_eps"]).to(
        torch.bfloat16
    )
    raw[out_loc.clamp(0, raw.shape[0] - 1)] = normed
    starts = starts.to(torch.int64)
    groups = starts[:, None] // RATIO + torch.arange(width // RATIO + 1, device="cuda")
    live = (groups + 1) * RATIO <= starts[:, None] + width
    live &= (out_loc[torch.arange(bs, device="cuda") * width] != 0)[:, None]
    members = groups[..., None] * RATIO + torch.arange(RATIO, device="cuda")
    page_rows = (torch.arange(bs, device="cuda") * width)[:, None, None]
    slots = pages[page_rows, members // PAGE].to(torch.int64) * PAGE + members % PAGE
    slots = slots.view(-1, RATIO)
    write = torch.where(live.view(-1), slots[:, 0], -1)
    K.indexpool_pool_write(
        raw,
        slots,
        members[..., 0].reshape(-1),
        write,
        weights["wv"],
        weights["wg"],
        weights["ape"],
        weights["norm"],
        weights["eps"],
        weights["inv_freq"],
        buf,
        PAGE,
    )


class TestHy4FlashIndexPoolVerifyUpdate(CustomTestCase):
    def test_matches_unfused_path(self):
        """Fused verify update stores every row's raw key and pools exactly the
        groups the block completes, for every block start offset mod 4,
        block widths up to 17, and CUDA-graph padded requests."""
        torch.manual_seed(0)
        num_pages = 256
        slots = num_pages * PAGE
        weights = dict(
            wv=torch.randn(128, 128, device="cuda") / 11,
            wg=torch.randn(128, 128, device="cuda") / 11,
            ape=torch.randn(RATIO, 128, device="cuda"),
            norm=torch.rand(128, device="cuda") + 0.5,
            eps=1e-6,
            key_eps=1e-6,
            inv_freq=1.0
            / (10000 ** (torch.arange(64, device="cuda", dtype=torch.float32) / 64)),
        )
        norm_w = (torch.rand(128, device="cuda") + 0.5).to(torch.bfloat16)
        for width in (2, 4, 6, 9, 17):
            for bs in (1, 5):
                starts = torch.randint(8, 3000, (bs,), device="cuda")
                starts[: min(bs, 4)] = starts[: min(bs, 4)] // 4 * 4 + torch.arange(
                    min(bs, 4), device="cuda"
                )
                max_pages = (int(starts.max()) + width) // PAGE + 1
                req_pages = torch.stack(
                    [
                        torch.randperm(num_pages - 1, device="cuda")[:max_pages] + 1
                        for _ in range(bs)
                    ]
                ).to(torch.int32)
                pages = req_pages.repeat_interleave(width, dim=0)
                pos = starts[:, None] + torch.arange(width, device="cuda")
                out_loc = (
                    req_pages[torch.arange(bs, device="cuda")[:, None], pos // PAGE]
                    .to(torch.int64)
                    .mul(PAGE)
                    .add(pos % PAGE)
                    .view(-1)
                )
                if bs > 1:
                    out_loc[(bs - 1) * width :] = 0
                keys = torch.randn(bs * width, 160, device="cuda").to(torch.bfloat16)
                keys = keys[:, :128]
                raw0 = torch.randn(slots, 128, device="cuda").to(torch.bfloat16)
                buf0 = torch.randint(
                    0, 255, (num_pages, PAGE * 132), device="cuda", dtype=torch.uint8
                )
                raw_ref, buf_ref = raw0.clone(), buf0.clone()
                _reference(
                    keys, norm_w, out_loc, starts, pages, raw_ref, weights, buf_ref
                )
                if bs > 1:
                    raw_ref[0] = raw0[0]
                raw, buf = raw0.clone(), buf0.clone()
                K.indexpool_verify_update(
                    keys,
                    norm_w,
                    weights["key_eps"],
                    out_loc,
                    starts,
                    pages,
                    raw,
                    weights["wv"],
                    weights["wg"],
                    weights["ape"],
                    weights["norm"],
                    weights["eps"],
                    weights["inv_freq"],
                    buf,
                    PAGE,
                )
                with self.subTest(width=width, bs=bs):
                    self.assertTrue(
                        torch.equal(raw.view(torch.int16), raw_ref.view(torch.int16))
                    )
                    self.assertTrue(torch.equal(buf, buf_ref))


if __name__ == "__main__":
    unittest.main()

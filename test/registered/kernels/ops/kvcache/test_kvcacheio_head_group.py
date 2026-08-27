"""Correctness of transfer_kv_per_layer_pfdhg_lf (head-group-major H2D).

When the unified L3 grid cuts the kv-head axis (cross-TP GQA reuse), a host page
reassembled from L3 chunks is head-group-major rather than page_first_direct's
own order -- the same bytes, permuted:

    host   (page_num, HG, L, P, hg, D)      HG = head groups, hg = heads/group
    device (rows, H, D) per layer           H  = HG * hg      (NHD)

The kernel absorbs that permutation into the H2D it was already running, so the
contract is exactly:

    device[d] == host[s // P, :, layer, s % P, :, :].reshape(H, D)

The oracle below is built from the *logical* host tensor with torch indexing and
compared against what the CUDA kernel wrote to the device, so it catches drift
between the offset functor and the layout it claims to read (a Python
transcription of the functor would not).
"""

import itertools
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


def _has_op() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from sgl_kernel import kvcacheio  # noqa: F401
    except ImportError:
        return False
    return hasattr(kvcacheio, "transfer_kv_per_layer_pfdhg_lf")


# (head_num, head_groups, layers, page_size, head_dim)
CONFIGS = [
    # The motivating cross-TP GQA case: one kv head per group => 256 B runs,
    # at DeepSeek-V3's prime layer count.
    (8, 8, 61, 16, 128),
    (8, 4, 4, 8, 64),
    (16, 2, 5, 16, 128),
    (16, 4, 61, 8, 128),
    # No head cut: must reduce to page_first_direct's own order.
    (8, 1, 3, 4, 32),
    # Degenerate shapes.
    (4, 4, 1, 4, 64),
]
DTYPES = [torch.float16, torch.bfloat16]


@unittest.skipUnless(
    _has_op(),
    "requires CUDA and transfer_kv_per_layer_pfdhg_lf",
)
class TestTransferKVHeadGroup(CustomTestCase):
    def _run_case(
        self,
        head_num,
        head_groups,
        layers,
        page_size,
        head_dim,
        dtype,
        block_quota=2,
        num_pages=3,
    ):
        hg = head_num // head_groups
        self.assertEqual(head_num % head_groups, 0)
        rows = num_pages * page_size
        itemsize = torch.tensor([], dtype=dtype).element_size()

        def make(seed):
            n = num_pages * head_groups * layers * page_size * hg * head_dim
            # distinct, exactly-representable values so any mis-index is visible
            vals = (torch.arange(n, dtype=torch.int32) + seed) % 2048
            return (
                vals.to(dtype)
                .reshape(num_pages, head_groups, layers, page_size, hg, head_dim)
                .contiguous()
                .pin_memory()
            )

        host_k, host_v = make(0), make(1024)
        dev_k = torch.zeros(
            layers, rows, head_num, head_dim, dtype=dtype, device="cuda"
        )
        dev_v = torch.zeros_like(dev_k)

        # non-identity permutation so the index tensors are actually exercised
        src = torch.arange(rows, dtype=torch.int64)
        dst = torch.flip(src, [0]).contiguous()
        src_c, dst_c = src.cuda(), dst.cuda()

        item_size = head_num * head_dim * itemsize
        layout_dim = item_size * layers

        from sgl_kernel.kvcacheio import transfer_kv_per_layer_pfdhg_lf

        for layer in range(layers):
            transfer_kv_per_layer_pfdhg_lf(
                src_k=host_k.view(-1),
                dst_k=dev_k[layer],
                src_v=host_v.view(-1),
                dst_v=dev_v[layer],
                src_indices=src_c,
                dst_indices=dst_c,
                layer_id=layer,
                item_size=item_size,
                src_layout_dim=layout_dim,
                page_size=page_size,
                head_num=head_groups,
                block_quota=block_quota,
            )
        torch.cuda.synchronize()

        for layer in range(layers):
            exp_k = torch.empty(rows, head_num, head_dim, dtype=dtype)
            exp_v = torch.empty(rows, head_num, head_dim, dtype=dtype)
            for i in range(rows):
                s, d = int(src[i]), int(dst[i])
                page, tok = s // page_size, s % page_size
                exp_k[d] = host_k[page, :, layer, tok, :, :].reshape(head_num, head_dim)
                exp_v[d] = host_v[page, :, layer, tok, :, :].reshape(head_num, head_dim)
            self.assertTrue(
                torch.equal(dev_k[layer].cpu(), exp_k),
                f"K mismatch at layer {layer} for H={head_num} HG={head_groups} "
                f"L={layers} P={page_size} D={head_dim} {dtype}",
            )
            self.assertTrue(
                torch.equal(dev_v[layer].cpu(), exp_v),
                f"V mismatch at layer {layer} for H={head_num} HG={head_groups} "
                f"L={layers} P={page_size} D={head_dim} {dtype}",
            )

    def test_matches_oracle(self):
        for (h, hgs, l, p, d), dtype in itertools.product(CONFIGS, DTYPES):
            with self.subTest(
                head_num=h,
                head_groups=hgs,
                layers=l,
                page_size=p,
                head_dim=d,
                dtype=dtype,
            ):
                self._run_case(h, hgs, l, p, d, dtype)

    def test_block_quota_does_not_change_results(self):
        """block_quota only sizes the grid; it must not affect the bytes moved."""
        for quota in (2, 16, 132):
            with self.subTest(block_quota=quota):
                self._run_case(8, 4, 4, 8, 64, torch.float16, block_quota=quota)

    def test_no_head_cut_is_page_first_direct_order(self):
        """At head_groups == 1 the layout degenerates to page_first_direct.

        The host block is then (page, 1, L, P, H, D), i.e. exactly
        page_first_direct's (L, P, H, D) page block, so the transfer must be a
        pure token gather with no permutation.
        """
        head_num, layers, page_size, head_dim, num_pages = 8, 3, 4, 32, 5
        rows = num_pages * page_size
        dtype = torch.float16
        itemsize = 2

        n = num_pages * layers * page_size * head_num * head_dim
        host = (
            ((torch.arange(n, dtype=torch.int32)) % 2048)
            .to(dtype)
            .reshape(num_pages, 1, layers, page_size, head_num, head_dim)
            .contiguous()
            .pin_memory()
        )
        dev = torch.zeros(layers, rows, head_num, head_dim, dtype=dtype, device="cuda")
        src = torch.arange(rows, dtype=torch.int64)
        dst = torch.flip(src, [0]).contiguous()

        from sgl_kernel.kvcacheio import transfer_kv_per_layer_pfdhg_lf

        for layer in range(layers):
            transfer_kv_per_layer_pfdhg_lf(
                src_k=host.view(-1),
                dst_k=dev[layer],
                src_v=host.view(-1),
                dst_v=dev[layer],
                src_indices=src.cuda(),
                dst_indices=dst.cuda(),
                layer_id=layer,
                item_size=head_num * head_dim * itemsize,
                src_layout_dim=head_num * head_dim * itemsize * layers,
                page_size=page_size,
                head_num=1,
            )
        torch.cuda.synchronize()

        # page_first_direct view: (page, L, P, H, D) indexed straight through
        pfd = host.squeeze(1)
        for layer in range(layers):
            exp = torch.empty(rows, head_num, head_dim, dtype=dtype)
            for i in range(rows):
                s, d = int(src[i]), int(dst[i])
                exp[d] = pfd[s // page_size, layer, s % page_size]
            self.assertTrue(torch.equal(dev[layer].cpu(), exp))

    def test_invalid_launch_geometry_is_rejected(self):
        """Reject invalid divisors and copy widths instead of truncating."""
        h = torch.zeros(4 * 4 * 1 * 4 * 1 * 64, dtype=torch.float16).pin_memory()
        d = torch.zeros(16, 4, 64, dtype=torch.float16, device="cuda")
        idx = torch.arange(16, dtype=torch.int64, device="cuda")

        def launch(
            *,
            item_size=4 * 64 * 2,
            src_layout_dim=4 * 64 * 2,
            page_size=4,
            head_num=4,
            block_quota=2,
            indices=idx,
        ):
            torch.ops.sgl_kernel.transfer_kv_per_layer_pfdhg_lf(
                h.view(-1),
                d,
                h.view(-1),
                d,
                indices,
                indices,
                0,
                item_size,
                src_layout_dim,
                page_size,
                head_num,
                block_quota,
                32,
            )

        cases = [
            ("block quota", {"block_quota": 0}, "block_quota"),
            ("page size", {"page_size": 0}, "page_size"),
            ("head count", {"head_num": 0}, "head_num"),
            ("head division", {"head_num": 3}, "item_size"),
            (
                "copy width",
                {"item_size": 24, "src_layout_dim": 24, "head_num": 2},
                "Per-head-group",
            ),
            ("layout division", {"src_layout_dim": 513}, "src_layout_dim"),
        ]
        for name, kwargs, error in cases:
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, error):
                launch(**kwargs)

        # An empty transfer is a valid no-op, not a grid-size division by zero.
        launch(indices=idx[:0])


if __name__ == "__main__":
    unittest.main()

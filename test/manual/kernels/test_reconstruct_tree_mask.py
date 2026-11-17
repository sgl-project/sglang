"""Test the Triton implementation of ``reconstruct_indices_from_tree_mask``.

``reconstruct_indices_from_tree_mask_triton`` is the device-native fallback for
the compiled ``sgl_kernel.speculative.reconstruct_indices_from_tree_mask`` op on
platforms that ship no such op (e.g. Intel XPU). It reconstructs NGRAM verify
metadata (``positions`` / ``retrieve_index`` / ``retrieve_next_token`` /
``retrieve_next_sibling``) from a per-batch ``n x n`` tree mask. Run manually;
it exercises whichever GPU-like backends (CUDA, XPU) are locally available.

This test pins the kernel to the op's documented contract two ways:

1. **External-literal anchor** -- the worked example from the compiled op's own
   test (``kernels/aot/tests/speculative/test_ngram_utils.py``). Those output
   values are the CUDA/CPU op's ground truth, independent of any torch port.

2. **Differential sweep vs an independent oracle** -- a plain-python
   reconstruction (no torch/triton vectorization shared with the kernel) over
   randomly generated *valid* trees, across power-of-two and non-power-of-two
   ``n`` (exercises the ``BLOCK_N`` padding mask) and multiple batch sizes.

Failure modes guarded (no other test covers these):
  - root nodes (``parent < 0``) must NOT link as siblings to one another;
  - ``n`` not a power of two must mask out the padded ``BLOCK_N`` tail;
  - the two reduction axes (parent/depth over columns, child/sibling over rows)
    must not be transposed.
"""

import unittest

import numpy as np
import torch

from sglang.test.test_utils import CustomTestCase

try:
    from sglang.kernels.ops.speculative.reconstruct_tree import (
        reconstruct_indices_from_tree_mask_triton,
    )

    _HAS_KERNEL = True
except Exception:  # pragma: no cover - import guarded for hardware-less CI shards
    _HAS_KERNEL = False

_AVAILABLE_DEVICES = [
    device
    for device, available in (
        ("cuda", torch.cuda.is_available()),
        ("xpu", hasattr(torch, "xpu") and torch.xpu.is_available()),
    )
    if available
]


def _make_valid_tree_mask(bs: int, n: int, seed: int) -> np.ndarray:
    """Random *valid* tree mask [bs, n, n]: mask[b, i, j] == node j is an
    ancestor of node i (transitive closure; the diagonal is set). Each node
    either roots (~30%) or attaches under a uniformly random earlier node."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((bs, n, n), dtype=bool)
    for b in range(bs):
        ancestors = [set() for _ in range(n)]
        for i in range(n):
            ancestors[i].add(i)
            if i > 0 and rng.random() >= 0.3:
                parent = int(rng.integers(0, i))
                ancestors[i] |= ancestors[parent]
            for j in ancestors[i]:
                mask[b, i, j] = True
    return mask


def _reference(mask: np.ndarray, seq_lens: np.ndarray, bs: int, n: int):
    """Plain-python oracle for the documented reconstruction contract. Returns
    (positions, retrieve_index, next_token, next_sibling) as int64 arrays."""
    positions = np.empty(bs * n, dtype=np.int64)
    retrieve_index = np.empty(bs * n, dtype=np.int64)
    next_token = np.full(bs * n, -1, dtype=np.int64)
    next_sibling = np.full(bs * n, -1, dtype=np.int64)

    for b in range(bs):
        parent = [-1] * n
        for tid in range(n):
            ancestors = [j for j in range(tid) if mask[b, tid, j]]
            positions[b * n + tid] = len(ancestors) + int(seq_lens[b])
            retrieve_index[b * n + tid] = b * n + tid
            parent[tid] = max(ancestors) if ancestors else -1

        for tid in range(n):
            children = [k for k in range(tid + 1, n) if mask[b, k, tid]]
            if children:
                next_token[b * n + tid] = min(children)
            if parent[tid] >= 0:
                siblings = [k for k in range(tid + 1, n) if parent[k] == parent[tid]]
                if siblings:
                    next_sibling[b * n + tid] = min(siblings)

    return positions, retrieve_index, next_token, next_sibling


@unittest.skipUnless(_HAS_KERNEL, "reconstruct_tree Triton kernel import required")
@unittest.skipUnless(_AVAILABLE_DEVICES, "CUDA or XPU device required")
class TestReconstructTreeMask(CustomTestCase):
    def _run_kernel(self, device, mask_bool_cpu, seq_lens_cpu, bs, n):
        tree_mask = mask_bool_cpu.reshape(-1).contiguous().to(device)
        seq_lens = seq_lens_cpu.to(device)
        positions = torch.empty(bs * n, dtype=torch.int64, device=device)
        retrieve_index = torch.full((bs, n), -1, dtype=torch.int64, device=device)
        next_token = torch.full((bs, n), -1, dtype=torch.int64, device=device)
        next_sibling = torch.full((bs, n), -1, dtype=torch.int64, device=device)
        reconstruct_indices_from_tree_mask_triton(
            tree_mask,
            seq_lens,
            positions,
            retrieve_index,
            next_token,
            next_sibling,
            bs,
            n,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        else:
            torch.xpu.synchronize()
        return (
            positions.cpu().numpy(),
            retrieve_index.reshape(-1).cpu().numpy(),
            next_token.reshape(-1).cpu().numpy(),
            next_sibling.reshape(-1).cpu().numpy(),
        )

    def test_matches_compiled_op_worked_example(self):
        # Literal contract from the compiled op's test (test_ngram_utils.py):
        # tree over 4 nodes, verified_seq_len == 12.
        bs, n = 1, 4
        tree_mask = torch.tensor(
            [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1],
            dtype=torch.bool,
        )
        seq_lens = torch.tensor([12], dtype=torch.int64)
        for device in _AVAILABLE_DEVICES:
            with self.subTest(device=device):
                positions, retrieve_index, next_token, next_sibling = self._run_kernel(
                    device, tree_mask.reshape(bs, n, n), seq_lens, bs, n
                )
                self.assertEqual(positions.tolist(), [12, 13, 13, 14])
                self.assertEqual(retrieve_index.tolist(), [0, 1, 2, 3])
                self.assertEqual(next_token.tolist(), [1, -1, 3, -1])
                self.assertEqual(next_sibling.tolist(), [-1, 2, -1, -1])

    def test_matches_reference_over_random_trees(self):
        # Power-of-two and non-power-of-two n (BLOCK_N tail masking) x batch sizes.
        for device in _AVAILABLE_DEVICES:
            for n in (1, 2, 3, 7, 8, 16, 17, 32, 63, 64):
                for bs in (1, 3, 16, 64):
                    for seed in range(3):
                        mask = _make_valid_tree_mask(bs, n, seed * 1000 + bs * 100 + n)
                        seq_lens = torch.from_numpy(
                            np.random.default_rng(seed).integers(1, 200, size=bs)
                        ).to(torch.int64)
                        got = self._run_kernel(
                            device, torch.from_numpy(mask), seq_lens, bs, n
                        )
                        ref = _reference(mask, seq_lens.numpy(), bs, n)
                        names = (
                            "positions",
                            "retrieve_index",
                            "retrieve_next_token",
                            "retrieve_next_sibling",
                        )
                        for name, g, r in zip(names, got, ref):
                            np.testing.assert_array_equal(
                                g,
                                r,
                                err_msg=(
                                    f"{name} mismatch at device={device} "
                                    f"bs={bs} n={n} seed={seed}"
                                ),
                            )


if __name__ == "__main__":
    unittest.main()

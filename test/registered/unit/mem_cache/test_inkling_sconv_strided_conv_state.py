# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inkling SConv kernels must accept a STRIDED (page-major / unified) conv-state.

Bug regression (fixed by relaxing 7 TensorMatcher sites): every conv-state cache
matcher in ``kernels/jit/csrc/inkling/*.cuh`` used the bare
``TensorMatcher({-1, W1s, D})`` form, whose default is a hard
``view.is_contiguous()`` RuntimeCheck (``sgl_kernel/tensor.h``). The kernel
BODIES are already stride-aware — they index via ``cache.stride(0)`` /
``cache.stride(1)`` and only require the channel dim contiguous — so the matcher
was strictly stronger than the kernel's real contract. Under the unified
tri-pool the conv-state is served as a page-major envelope view (slot pitch
spans all layers), which is non-contiguous, and the matcher rejection kills the
forward.

The fix chains ``.with_strides({-1, -1, 1})``: slot/window strides wildcarded,
channel stride pinned to 1 (the one contract the vectorized loads rely on).

Two layers of guard:

  1. SOURCE SCAN (CPU, always runs — the portable red/green, same precedent as
     ``test_unified_free_no_host_sync.py``): every ``.verify(cache)`` matcher in
     the inkling kernel sources must carry the stride relaxation. Fails the
     moment a site is reverted to the contiguity-default form or a new
     conv-state matcher lands without it.

  2. FUNCTIONAL (CUDA + JIT, skipped elsewhere): drive the real
     ``update_sconv_cache`` kernel with a page-major strided cache view; on
     pre-fix sources this raises ``Tensor is not contiguous as expected``;
     post-fix it must run AND be bit-identical to the same op on a contiguous
     clone.

    python -m pytest test/registered/unit/mem_cache/test_inkling_sconv_strided_conv_state.py -v
"""

import re
import unittest
from pathlib import Path

import torch

import sglang.kernels.jit as _jit_pkg
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_KERNEL_DIR = Path(_jit_pkg.__file__).parent / "csrc" / "inkling"

# The 7 known conv-state cache matcher sites (file -> expected count). A new
# file/site is still caught: the scan sweeps every *.cuh, and any
# `.verify(cache)` without the relaxation fails regardless of this table.
_KNOWN_SITES = {
    "update_sconv_cache.cuh": 1,
    "causal_conv1d.cuh": 1,
    "draft_extend_sconv.cuh": 1,
    "fused_decode_update.cuh": 1,
    "gather_scatter_sconv.cuh": 1,
    "inkling_ar_fused_decode.cuh": 2,
}

_RELAXATION = "with_strides({-1, -1, 1})"


def _cache_matcher_lines():
    """Every TensorMatcher line that verifies a tensor named `cache`."""
    hits = []
    for cuh in sorted(_KERNEL_DIR.glob("*.cuh")):
        for lineno, line in enumerate(cuh.read_text().splitlines(), 1):
            if "TensorMatcher" in line and re.search(r"\.verify\(cache\)", line):
                hits.append((cuh.name, lineno, line.strip()))
    return hits


class TestConvStateMatchersAcceptStrided(unittest.TestCase):
    def test_every_cache_matcher_carries_the_stride_relaxation(self):
        hits = _cache_matcher_lines()
        bad = [(f, n, l) for f, n, l in hits if _RELAXATION not in l]
        self.assertEqual(
            bad,
            [],
            msg=(
                "conv-state cache matcher(s) without the stride relaxation "
                f"{_RELAXATION!r} — the TensorMatcher default enforces "
                "is_contiguous(), which rejects the unified/page-major "
                f"conv-state view the stride-aware kernel bodies accept: {bad}"
            ),
        )

    def test_all_known_sites_still_present(self):
        """Completeness guard: the relaxation must not be 'fixed' by deleting
        the matcher (losing shape/dtype/device verification entirely)."""
        by_file = {}
        for f, _, _ in _cache_matcher_lines():
            by_file[f] = by_file.get(f, 0) + 1
        for fname, expected in _KNOWN_SITES.items():
            self.assertGreaterEqual(
                by_file.get(fname, 0),
                expected,
                msg=f"{fname}: conv-state matcher site(s) disappeared",
            )

    def test_channel_dim_stays_pinned_contiguous(self):
        """The relaxation must wildcard ONLY slot/window: a fully-wildcarded
        stride spec ({-1, -1, -1}) would drop the channel-contiguity contract
        the vectorized state loads rely on."""
        for f, n, line in _cache_matcher_lines():
            self.assertNotIn(
                "with_strides({-1, -1, -1})",
                line,
                msg=f"{f}:{n} wildcards the channel stride",
            )


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA + JIT for the real kernel")
class TestUpdateSconvCacheStridedFunctional(unittest.TestCase):
    """The real kernel on a page-major strided view == on a contiguous clone.

    Red on pre-fix sources: the matcher raises
    'Tensor is not contiguous as expected' for the strided view.
    """

    _SLOTS, _LAYERS, _W1, _D = 4, 2, 3, 64

    def _run(self, cache: torch.Tensor) -> torch.Tensor:
        from sglang.kernels.ops.mamba.inkling_sconv import update_sconv_cache

        torch.manual_seed(0)
        dev = cache.device
        tokens = 10
        x = torch.randn(tokens, self._D, dtype=cache.dtype, device=dev)
        # 2 sequences: [0:6) -> slot 1 (has state), [6:10) -> slot 3 (fresh)
        cache_indices = torch.tensor([1, 3], dtype=torch.int32, device=dev)
        has_initial_state = torch.tensor([True, False], device=dev)
        query_start_loc = torch.tensor([0, 6, tokens], dtype=torch.int32, device=dev)
        update_sconv_cache(x, cache, cache_indices, has_initial_state, query_start_loc)
        return cache

    def test_strided_view_matches_contiguous(self):
        dev = "cuda"
        torch.manual_seed(1)
        # Page-major envelope: (slots, LAYERS, W1, D); the per-layer view
        # cache = env[:, 1] has stride(0) = LAYERS*W1*D != W1*D -> non-contiguous.
        env = torch.randn(
            self._SLOTS,
            self._LAYERS,
            self._W1,
            self._D,
            dtype=torch.bfloat16,
            device=dev,
        )
        strided = env[:, 1]
        self.assertFalse(strided.is_contiguous(), "precondition: view is strided")
        contiguous = strided.clone()
        self.assertTrue(contiguous.is_contiguous())

        out_c = self._run(contiguous)
        out_s = self._run(strided)  # pre-fix: matcher rejection raises here

        self.assertTrue(
            torch.equal(out_s, out_c),
            "strided-view kernel result differs from the contiguous reference",
        )


if __name__ == "__main__":
    unittest.main()

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
"""The flashinfer-side contract that unified-memory MLA decode relies on.

Under cuda graph, `FlashInferMLAIndicesUpdaterDecode` translates kv_indices to
DENSE ids by writing back IN PLACE, because the tensor it is handed is the
capture-stable buffer the captured kernel reads and sglang's replacement plan
(`fast_mla_decode_plan`) never consumes its `kv_indices` argument.

That is only correct while two things hold in flashinfer:

  1. `BatchMLAPagedAttentionWrapper(use_cuda_graph=True, kv_indices=buf)` keeps
     `buf` itself as `_kv_indices_buf` -- it must not clone or reallocate, or the
     in-place write would land somewhere the kernel never reads.
  2. `plan()` under cuda graph copies INTO that buffer rather than rebinding it
     (the eager branch does rebind, which is why the eager path may rebind too).

A flashinfer upgrade that changes either one turns unified-memory flashinfer
decode into silently-wrong attention -- no crash, just wrong KV pages. It cost
GSM8K 0.000 before the in-place fix, so it is worth failing loudly here instead.

    python -m pytest test/registered/unit/mem_cache/test_unified_mla_flashinfer_graph_contract.py -v
"""

import unittest

import torch

from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

_HAS_CUDA = torch.cuda.is_available()
_HAS_FI = is_flashinfer_available()


@unittest.skipUnless(_HAS_CUDA and _HAS_FI, "requires CUDA + flashinfer")
class TestFlashInferMLAGraphBufferContract(unittest.TestCase):
    def _wrapper(self, kv_indices_buf):
        from flashinfer import BatchMLAPagedAttentionWrapper

        dev = "cuda"
        max_bs = 8
        return BatchMLAPagedAttentionWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev),
            use_cuda_graph=True,
            qo_indptr=torch.zeros(max_bs + 1, dtype=torch.int32, device=dev),
            kv_indptr=torch.zeros(max_bs + 1, dtype=torch.int32, device=dev),
            kv_indices=kv_indices_buf,
            kv_len_arr=torch.ones(max_bs, dtype=torch.int32, device=dev),
            backend="auto",
        )

    def test_wrapper_keeps_the_caller_buffer(self):
        """The buffer handed in at construction must be the one the wrapper holds
        -- the unified remap writes into it directly."""
        buf = torch.zeros(4096, dtype=torch.int32, device="cuda")
        wrapper = self._wrapper(buf)
        self.assertIs(
            wrapper._kv_indices_buf,
            buf,
            "flashinfer cloned the kv_indices buffer; the in-place dense remap in "
            "FlashInferMLAIndicesUpdaterDecode would no longer reach the kernel",
        )

    def test_in_place_write_is_visible_through_the_wrapper(self):
        """Writing dense ids into the caller's buffer must be observable through
        `_kv_indices_buf`, which is what `run()` reads."""
        buf = torch.zeros(4096, dtype=torch.int32, device="cuda")
        wrapper = self._wrapper(buf)
        dense = torch.arange(1, 65, dtype=torch.int32, device="cuda") * 24
        buf[:64].copy_(dense)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(wrapper._kv_indices_buf[:64], dense))

    def test_fast_decode_plan_ignores_its_kv_indices_argument(self):
        """`fast_mla_decode_plan` skips the buffer copy that the real `plan()`
        does, which is precisely why the remap must be in place. If it ever
        starts consuming the argument, the in-place write becomes redundant
        rather than load-bearing -- worth noticing either way."""
        import inspect

        from sglang.srt.layers.attention.flashinfer_mla_backend import (
            fast_mla_decode_plan,
        )

        src = inspect.getsource(fast_mla_decode_plan)
        body = src.split(")", 1)[1] if ")" in src else src
        self.assertNotIn(
            "_kv_indices_buf",
            body,
            "fast_mla_decode_plan now touches _kv_indices_buf; re-check whether "
            "the in-place dense remap in the decode updater is still required",
        )


if __name__ == "__main__":
    unittest.main()

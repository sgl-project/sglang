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
"""`--enable-page-major-kv-layout` full-attention backend allowlist.

The page-major envelope K/V views are strided, which only the Triton attention
kernels read. The one exception is the unified-memory MLA pool: it exposes each
layer as a DENSE contiguous view (`build_dense_mla_views`), so the paged MLA
backends can read it directly once their kv_indices / block tables are remapped
to dense ids -- `fa3`, `flashinfer`'s MLA backend, and `trtllm_mla` with its
`cutedsl_mla` / `tokenspeed_mla` subclasses.

Pinned here so the exception cannot silently widen to a backend that has no
dense-id remapping (`flashmla`, `cutlass_mla`, ...) or leak into the MHA path.
`fa3` matters most: it is the resolved default on pre-Blackwell hosts, so it is
the one entry whose absence used to make `--enable-unified-memory` fail to boot
under its own default configuration.

    python -m pytest test/registered/unit/server_args/test_page_major_backend_allowlist.py -v
"""

import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _accepts(backend: str, *, use_mla: bool, unified: bool = True) -> bool:
    """Run just `_handle_page_major_kv_layout` against a minimal stand-in.

    ServerArgs' real constructor pulls in a model config; this exercises the
    single handler under test with the fields it reads.
    """
    sa = ServerArgs.__new__(ServerArgs)
    for name, value in {
        "enable_unified_memory": unified,
        # The unified pool sets this itself; without it the flag must be explicit
        # or the handler returns before reaching the allowlist.
        "enable_page_major_kv_layout": not unified,
        "attention_backend": backend,
        "prefill_attention_backend": None,
        "decode_attention_backend": None,
        "linear_attn_backend": "triton",
        "linear_attn_decode_backend": None,
        "linear_attn_prefill_backend": None,
        "mamba_backend": "triton",
    }.items():
        object.__setattr__(sa, name, value)
    sa.use_mla_backend = lambda: use_mla
    sa._resolved_attention_backends = lambda: [backend]
    try:
        ServerArgs._handle_page_major_kv_layout(sa)
        return True
    except AssertionError:
        return False


class TestPageMajorBackendAllowlist(unittest.TestCase):
    # Wired for the dense per-layer MLA views (see the module docstring).
    DENSE_MLA_BACKENDS = (
        "fa3",
        "trtllm_mla",
        "flashinfer",
        "cutedsl_mla",
        "tokenspeed_mla",
    )
    # No dense-id remapping: must stay rejected until they get one.
    UNWIRED_BACKENDS = ("flashmla", "cutlass_mla", "trtllm_mha", "aiter")

    def test_triton_always_allowed(self):
        for use_mla in (True, False):
            self.assertTrue(_accepts("triton", use_mla=use_mla))

    def test_dense_mla_backends_allowed_under_unified_mla(self):
        for backend in self.DENSE_MLA_BACKENDS:
            self.assertTrue(
                _accepts(backend, use_mla=True),
                f"{backend} should be allowed with the unified-memory MLA pool",
            )

    def test_dense_mla_backends_rejected_for_mha(self):
        """The dense-view exception is MLA-only -- MHA sub-pools stay strided."""
        for backend in self.DENSE_MLA_BACKENDS:
            self.assertFalse(
                _accepts(backend, use_mla=False),
                f"{backend} must stay rejected for a non-MLA model",
            )

    def test_dense_mla_backends_rejected_without_unified_memory(self):
        """Plain --enable-page-major-kv-layout (no unified pool) keeps the
        strided views, so only Triton can read them."""
        for backend in self.DENSE_MLA_BACKENDS:
            self.assertFalse(
                _accepts(backend, use_mla=True, unified=False),
                f"{backend} must stay rejected without --enable-unified-memory",
            )

    def test_unwired_backends_always_rejected(self):
        for backend in self.UNWIRED_BACKENDS:
            for use_mla in (True, False):
                self.assertFalse(
                    _accepts(backend, use_mla=use_mla),
                    f"{backend} has no dense-id remapping and must be rejected",
                )


if __name__ == "__main__":
    unittest.main()

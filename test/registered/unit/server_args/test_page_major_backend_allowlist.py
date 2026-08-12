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

Three-way gate (see `_handle_page_major_kv_layout`):
  * unified-memory MLA models expose DENSE per-layer views
    (`build_dense_mla_views`), so the whole wired paged MLA family is allowed
    -- `fa3`, `flashinfer`'s MLA backend, `trtllm_mla` with its `cutedsl_mla`
    / `tokenspeed_mla` subclasses, and `flashmla` (ps=64 snap);
  * unified-memory MHA/SWA models get dense views iff every KV sub-pool has
    uniform rows (`not has_asymmetric_kv`) and the strided A/B escape
    (`SGLANG_FORCE_STRIDED_UNIFIED_MHA`) is off -- then `fa3` / `fa4` /
    `flashinfer` / `trtllm_mha` join Triton; the decision is the SAME shared
    helper the pool factories flip on (`unified_dense_mha_enabled_for_model`),
    so the gate and the factories cannot diverge;
  * everything else (asymmetric-K/V models like MiMoV2, env-forced strided,
    plain page-major without the unified pool) keeps the envelope-strided 4-D
    views only the stride-aware Triton kernels read.

Pinned here so no arm silently widens to an unwired backend (`cutlass_mla`,
`aiter`) and no arm silently narrows: `fa3` is the resolved default on
pre-Blackwell hosts, so its absence from an arm makes `--enable-unified-memory`
fail to boot under its own default configuration.

    python -m pytest test/registered/unit/server_args/test_page_major_backend_allowlist.py -v
"""

import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _accepts(
    backend: str,
    *,
    use_mla: bool,
    unified: bool = True,
    has_asymmetric_kv: bool = False,
    force_strided: bool = False,
) -> bool:
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
    sa.get_model_config = lambda: SimpleNamespace(has_asymmetric_kv=has_asymmetric_kv)
    try:
        with envs.SGLANG_FORCE_STRIDED_UNIFIED_MHA.override(force_strided):
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
        "flashmla",
    )
    # Wired for the dense per-layer MHA/SWA views (uniform-row models).
    DENSE_MHA_BACKENDS = ("fa3", "fa4", "flashinfer", "trtllm_mha")
    # MLA-family kernels that must never leak into the MHA arm.
    MLA_ONLY_BACKENDS = ("trtllm_mla", "cutedsl_mla", "tokenspeed_mla", "flashmla")
    # No dense-id wiring anywhere: must stay rejected until they get one.
    UNWIRED_BACKENDS = ("cutlass_mla", "aiter")

    def test_triton_always_allowed(self):
        for use_mla in (True, False):
            for asym in (False, True):
                for forced in (False, True):
                    self.assertTrue(
                        _accepts(
                            "triton",
                            use_mla=use_mla,
                            has_asymmetric_kv=asym,
                            force_strided=forced,
                        )
                    )

    def test_dense_mla_backends_allowed_under_unified_mla(self):
        for backend in self.DENSE_MLA_BACKENDS:
            self.assertTrue(
                _accepts(backend, use_mla=True),
                f"{backend} should be allowed with the unified-memory MLA pool",
            )

    def test_dense_mha_backends_allowed_for_uniform_row_models(self):
        for backend in self.DENSE_MHA_BACKENDS:
            self.assertTrue(
                _accepts(backend, use_mla=False),
                f"{backend} should be allowed for a uniform-row MHA model",
            )

    def test_mla_only_backends_rejected_for_mha(self):
        for backend in self.MLA_ONLY_BACKENDS:
            self.assertFalse(
                _accepts(backend, use_mla=False),
                f"{backend} is an MLA kernel and must stay out of the MHA arm",
            )

    def test_asymmetric_kv_model_is_triton_only(self):
        """head_dim != v_head_dim (MiMoV2): no uniform rows, no dense views —
        the strided-Triton fallback arm."""
        for backend in self.DENSE_MHA_BACKENDS:
            self.assertFalse(
                _accepts(backend, use_mla=False, has_asymmetric_kv=True),
                f"{backend} must be rejected for an asymmetric-K/V model",
            )

    def test_env_forced_strided_is_triton_only(self):
        """SGLANG_FORCE_STRIDED_UNIFIED_MHA is the dense-vs-strided A/B
        escape; forcing strided must narrow the gate exactly like the
        factories stop building dense views."""
        for backend in self.DENSE_MHA_BACKENDS:
            self.assertFalse(
                _accepts(backend, use_mla=False, force_strided=True),
                f"{backend} must be rejected when the env forces strided views",
            )

    def test_dense_backends_rejected_without_unified_memory(self):
        """Plain --enable-page-major-kv-layout (no unified pool) keeps the
        strided views, so only Triton can read them."""
        for backend in set(self.DENSE_MLA_BACKENDS + self.DENSE_MHA_BACKENDS):
            self.assertFalse(
                _accepts(backend, use_mla=True, unified=False),
                f"{backend} must stay rejected without --enable-unified-memory",
            )

    def test_unwired_backends_always_rejected(self):
        for backend in self.UNWIRED_BACKENDS:
            for use_mla in (True, False):
                self.assertFalse(
                    _accepts(backend, use_mla=use_mla),
                    f"{backend} has no dense-id wiring and must be rejected",
                )


if __name__ == "__main__":
    unittest.main()

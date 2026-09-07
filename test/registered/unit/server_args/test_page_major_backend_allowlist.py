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

`handle_page_major_kv_layout` gates two ways, because the per-layer views the
unified pool exposes are all the allowlisted backends can read: an MLA arm
(`build_mla_views`), an MHA/SWA arm (`build_mha_views`), and no page-major arm
at all without the unified pool. `fa3` is the resolved default on pre-Blackwell
hosts, so its absence from an arm makes `--enable-unified-memory` fail to boot
under its own default configuration.

The same handler screens the pool itself: the MHA/SWA per-layer views need
uniform K/V rows, so an asymmetric-K/V model (MiMoV2: head_dim 192 !=
v_head_dim 128) is rejected on EVERY backend, Triton included. MLA models are
exempt -- their sub-pool keeps one latent row per layer, and real MLA configs
(Kimi-Linear: head_dim 72, v_head_dim 128) report asymmetric dims while running
the unified pool today.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.kv_cache_hook import handle_page_major_kv_layout
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _accepts(
    backend: str,
    *,
    use_mla: bool,
    unified: bool = True,
    linear_decode: str | None = None,
    linear_prefill: str | None = None,
    has_asymmetric_kv: bool = False,
) -> bool:
    """Run just `handle_page_major_kv_layout` against a minimal stand-in, since
    ServerArgs' real constructor pulls in a model config."""
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
        "linear_attn_decode_backend": linear_decode,
        "linear_attn_prefill_backend": linear_prefill,
        "mamba_backend": "triton",
    }.items():
        object.__setattr__(sa, name, value)
    object.__setattr__(
        sa,
        "_model_config",
        SimpleNamespace(
            attention_arch=AttentionArch.MLA if use_mla else AttentionArch.MHA,
            has_asymmetric_kv=has_asymmetric_kv,
            head_dim=192 if has_asymmetric_kv else 128,
            v_head_dim=128,
            swa_head_dim=128,
            swa_v_head_dim=128,
        ),
    )
    try:
        handle_page_major_kv_layout(sa)
        return True
    except AssertionError:
        return False


class TestPageMajorBackendAllowlist(unittest.TestCase):
    # Wired for the per-layer MLA views (see the module docstring).
    PER_LAYER_VIEW_MLA_BACKENDS = (
        "fa3",
        "trtllm_mla",
        "flashinfer",
        "cutedsl_mla",
        "tokenspeed_mla",
        "flashmla",
    )
    # Wired for the per-layer MHA/SWA views (uniform-row models).
    PER_LAYER_VIEW_MHA_BACKENDS = ("fa3", "fa4", "flashinfer", "trtllm_mha")
    # MLA-family kernels that must never leak into the MHA arm.
    MLA_ONLY_BACKENDS = ("trtllm_mla", "cutedsl_mla", "tokenspeed_mla", "flashmla")
    # No kernel-facing-id wiring anywhere: must stay rejected until they get one.
    UNWIRED_BACKENDS = ("cutlass_mla", "aiter")

    def test_triton_allowed_on_every_arm(self):
        """Triton reads both view families, so it is the one backend neither
        the MLA nor the MHA arm can narrow away."""
        for use_mla in (True, False):
            self.assertTrue(_accepts("triton", use_mla=use_mla))
        # The uniform-row screen is a property of the model, not of the
        # backend, so it rejects even Triton.
        self.assertFalse(_accepts("triton", use_mla=False, has_asymmetric_kv=True))

    def test_per_layer_view_mla_backends_allowed_under_unified_mla(self):
        for backend in self.PER_LAYER_VIEW_MLA_BACKENDS:
            self.assertTrue(
                _accepts(backend, use_mla=True),
                f"{backend} should be allowed with the unified-memory MLA pool",
            )

    def test_per_layer_view_mha_backends_allowed_for_uniform_row_models(self):
        for backend in self.PER_LAYER_VIEW_MHA_BACKENDS:
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

    def test_asymmetric_kv_mha_model_cannot_use_unified_memory(self):
        """The rejection is the POOL's, not a backend's, so it must fire on
        every backend -- Triton included."""
        for backend in ("triton",) + self.PER_LAYER_VIEW_MHA_BACKENDS:
            self.assertFalse(
                _accepts(backend, use_mla=False, has_asymmetric_kv=True),
                f"--enable-unified-memory + {backend} must be rejected for an "
                "asymmetric-K/V model",
            )

    def test_asymmetric_dims_do_not_screen_out_mla(self):
        """Screening on `has_asymmetric_kv` alone would lock every real MLA
        config out of the unified pool."""
        for backend in ("triton",) + self.PER_LAYER_VIEW_MLA_BACKENDS:
            self.assertTrue(
                _accepts(backend, use_mla=True, has_asymmetric_kv=True),
                f"{backend} must stay allowed for an MLA model with asymmetric "
                "K/V head dims",
            )

    def test_page_major_rejected_without_unified_memory(self):
        """There is no static page-major arm today, so the flag alone is
        rejected outright -- Triton included."""
        for backend in ("triton",) + tuple(
            set(self.PER_LAYER_VIEW_MLA_BACKENDS + self.PER_LAYER_VIEW_MHA_BACKENDS)
        ):
            for use_mla in (True, False):
                self.assertFalse(
                    _accepts(backend, use_mla=use_mla, unified=False),
                    f"{backend} must stay rejected without --enable-unified-memory",
                )

    def test_unwired_backends_always_rejected(self):
        for backend in self.UNWIRED_BACKENDS:
            for use_mla in (True, False):
                self.assertFalse(
                    _accepts(backend, use_mla=use_mla),
                    f"{backend} has no kernel-facing-id wiring and must be rejected",
                )

    def test_helion_linear_attention_is_kda_only(self):
        for phase in ("decode", "prefill"):
            kwargs = {f"linear_{phase}": "helion"}
            self.assertTrue(_accepts("triton", use_mla=True, **kwargs))
            self.assertFalse(_accepts("triton", use_mla=False, **kwargs))


if __name__ == "__main__":
    unittest.main()

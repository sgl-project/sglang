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
"""`--enable-unified-memory` speculative-decoding allow-list.

Two audited arms (see `handle_unified_memory_pool`), each with its own
constraints because each rides a different draft-KV story:
  * DSPARK: chain draft with a private draft pool; verify runs on the MLA
    backend family (`triton` / `trtllm_mla` / `cutedsl_mla` / `tokenspeed_mla`
    / `flashmla` / `flashinfer` / `fa3`)
    and the draft chain must be linear (`--speculative-eagle-topk` in
    {None, 1}).
  * EAGLE/EAGLE3: unified targets (hybrid-SWA or mamba hybrids, either
    full-pool kind) -- the draft's KV lives fused inside the full pool's
    page envelope (`DenseDraftRegion`), with an automatic private-pool
    fallback when no region resolves. MLA hosts verify on the MLA backend
    family; MHA hosts on the translated MHA rails. Chain
    only, and verify is audited on `triton` / `flashinfer` / `fa3` --
    demanded EXPLICITLY (an unset backend could resolve to an unaudited
    default), for the draft worker too (its backend resolves separately:
    explicit flag first, else it inherits the target's). The MLA verify
    backends must not leak into this arm.

Everything else (NGRAM / STANDALONE / registered customs) stays refused.
NGRAM's refusal is load-bearing, not pending: like tree verify it relocates
accepted tokens one at a time inside the target pool, which the unified
pool's page-granular `move_kv_cache` cannot express. Pinned so no arm silently
widens to an unaudited algorithm, family, tree shape, or backend -- and so the
EAGLE arm's addition never perturbs the DSPARK arm.

    python -m pytest test/registered/unit/server_args/test_unified_memory_spec_gate.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.configs.hybrid_arch as hybrid_arch
from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _PlainHFConfig:
    """Matches none of the hybrid-arch isinstance probes."""

    def get_text_config(self):
        return self


def _accepts(
    algorithm: str | None,
    *,
    is_hybrid_swa: bool = True,
    topk: int | None = 1,
    backend: str | None = "triton",
    draft_backend: str | None = None,
    attention_arch: AttentionArch = AttentionArch.MHA,
) -> bool:
    """Run just `handle_unified_memory_pool` against a minimal stand-in.

    ServerArgs' real constructor pulls in a model config; this exercises the
    single handler under test with the fields it reads (disaggregation off, so
    only the speculative / cache / dcp / cuda-graph checks run).
    """
    sa = ServerArgs.__new__(ServerArgs)
    for name, value in {
        "enable_unified_memory": True,
        "disaggregation_mode": "null",
        "speculative_algorithm": algorithm,
        "speculative_eagle_topk": topk,
        "speculative_draft_attention_backend": draft_backend,
        "enable_hierarchical_cache": False,
        "enable_lmcache": False,
        "dcp_size": 1,
        "cuda_graph_config": None,
        "attention_backend": backend,
        "prefill_attention_backend": None,
        "decode_attention_backend": None,
    }.items():
        object.__setattr__(sa, name, value)
    object.__setattr__(
        sa,
        "_model_config",
        SimpleNamespace(
            is_hybrid_swa=is_hybrid_swa,
            attention_arch=attention_arch,
            hf_config=_PlainHFConfig(),
            linear_attn_registry_result=None,
        ),
    )
    try:
        handle_unified_memory_pool(sa)
        return True
    except AssertionError:
        return False


class TestUnifiedMemorySpecGate(unittest.TestCase):
    # Verify-audited backends for the DSPARK (MLA-family) arm.
    DSPARK_BACKENDS = (
        "triton",
        "trtllm_mla",
        "cutedsl_mla",
        "tokenspeed_mla",
        "flashmla",
        "flashinfer",
        "fa3",
    )
    # Verify-audited backends for the EAGLE (fused-draft) arm.
    EAGLE_BACKENDS = ("triton", "flashinfer", "fa3")
    # Algorithms with no audited unified-pool verify rails.
    # "NEXTN" is deliberately absent: the CLI alias collapses it to
    # "EAGLE" in handle_speculative_decoding BEFORE this gate runs
    # (arg_groups/pipeline.py orders the hooks), so the raw string can
    # never reach the gate -- a case on it would test an impossible
    # input. The aliased spelling is covered by the EAGLE cases.
    UNAUDITED_ALGORITHMS = ("NGRAM", "STANDALONE")

    def test_dspark_admitted_on_every_audited_backend(self):
        """Each entry is a claim that the backend's spec verify path
        translates. Adding one here without the code is how a unified run
        silently reads the pool with virtual ids."""
        for backend in self.DSPARK_BACKENDS:
            self.assertTrue(
                _accepts("DSPARK", backend=backend),
                f"DSPARK should pass on verify-audited backend {backend}",
            )

    def test_dspark_refused_on_unaudited_backends(self):
        """fa4 and trtllm_mha have no translated spec verify path."""
        for backend in ("fa4", "trtllm_mha"):
            self.assertFalse(_accepts("DSPARK", backend=backend))

    def test_tree_drafting_refused(self):
        """Tree verify needs a per-token move inside the TARGET pool, which
        the unified pool's page-granular `move_kv_cache` cannot express.
        DSPARK is the only admitted algorithm today, so it is the only one
        that can demonstrate the rule; the check itself is unconditional, so
        it keeps holding as further arms are admitted."""
        for topk in (2, 4, 8):
            self.assertFalse(_accepts("DSPARK", topk=topk))
        # The shape knob must not disturb spec-off.
        self.assertTrue(_accepts(None, topk=None))

    def test_draft_backend_is_audited_too(self):
        """The draft worker runs its OWN forward on its OWN backend, resolved
        from `--speculative-draft-attention-backend` before it falls back to
        inheriting the target's. Checking only the target therefore leaves an
        unaudited backend reachable: every target arm can be audited while the
        draft translates nothing. Unset must still inherit and pass."""
        self.assertTrue(_accepts("DSPARK", draft_backend=None))
        for draft_backend in self.DSPARK_BACKENDS:
            self.assertTrue(
                _accepts("DSPARK", draft_backend=draft_backend),
                f"audited draft backend {draft_backend} should pass",
            )
        for draft_backend in ("fa4", "trtllm_mha", "torch_native"):
            self.assertFalse(
                _accepts("DSPARK", draft_backend=draft_backend),
                f"unaudited draft backend {draft_backend} must be refused",
            )

    def test_eagle_family_admitted_on_hybrid_swa(self):
        """EAGLE/EAGLE3 chain on a hybrid-SWA target with audited verify
        backends is the fused-draft-KV configuration -- both spellings,
        resolved or unset topk, every audited backend."""
        for algorithm in ("EAGLE", "EAGLE3"):
            for topk in (None, 1):
                for backend in self.EAGLE_BACKENDS:
                    self.assertTrue(
                        _accepts(algorithm, topk=topk, backend=backend),
                        f"{algorithm} topk={topk} backend={backend} should "
                        "pass on a hybrid-SWA target",
                    )

    def test_eagle_refused_on_dense_targets(self):
        """No fused draft region outside the unified composites: a dense
        (non-hybrid) target must be refused, not fail at boot."""
        for algorithm in ("EAGLE", "EAGLE3"):
            self.assertFalse(_accepts(algorithm, is_hybrid_swa=False))

    def test_eagle_admitted_on_mamba_mha(self):
        """A mamba hybrid off the MLA backend provisions the fused draft
        region in its MHA full sub-pool; refusing it here strands the whole
        mamba x EAGLE matrix."""
        with patch.object(hybrid_arch, "mambaish_config", return_value=object()):
            for algorithm in ("EAGLE", "EAGLE3"):
                self.assertTrue(_accepts(algorithm, is_hybrid_swa=False))

    def test_eagle_on_mla_mamba_verifies_on_the_mla_family(self):
        """An MLA mamba hybrid (Kimi) verifies on the MLA backend family;
        the MHA-only rails (and unset/unaudited backends) still refuse.
        Boot falls back to the private draft pool when no region resolves."""
        with patch.object(hybrid_arch, "mambaish_config", return_value=object()):
            for backend in self.DSPARK_BACKENDS:
                self.assertTrue(
                    _accepts(
                        "EAGLE",
                        is_hybrid_swa=False,
                        attention_arch=AttentionArch.MLA,
                        backend=backend,
                    ),
                    f"EAGLE on an MLA host should pass on {backend}",
                )
            self.assertFalse(
                _accepts(
                    "EAGLE",
                    is_hybrid_swa=False,
                    attention_arch=AttentionArch.MLA,
                    backend=None,
                )
            )
            self.assertFalse(
                _accepts(
                    "EAGLE",
                    is_hybrid_swa=False,
                    attention_arch=AttentionArch.MLA,
                    backend="fa4",
                )
            )

    def test_eagle_refused_tree_topk(self):
        """Tree verify is not audited for the unified pool; only a linear
        chain passes."""
        for topk in (2, 4, 8):
            self.assertFalse(_accepts("EAGLE", topk=topk))

    def test_eagle_refused_unaudited_backends(self):
        """Unaudited backends stay out, and the MLA verify set from the
        DSPARK arm must not leak into the EAGLE arm."""
        for backend in ("fa4", "trtllm_mha") + tuple(
            b for b in self.DSPARK_BACKENDS if b not in self.EAGLE_BACKENDS
        ):
            self.assertFalse(_accepts("EAGLE", backend=backend))

    def test_eagle_refused_unset_backend(self):
        """An unset backend defaults to fa3/flashinfer later in resolution --
        the EAGLE arm demands an explicit triton, never a None slip."""
        self.assertFalse(_accepts("EAGLE", backend=None))

    def test_eagle_draft_backend_pinned(self):
        """The draft worker resolves its own backend: unset inherits the
        target's triton, explicit triton passes, anything else refuses."""
        self.assertTrue(_accepts("EAGLE", draft_backend=None))
        for draft_backend in self.EAGLE_BACKENDS:
            self.assertTrue(_accepts("EAGLE", draft_backend=draft_backend))
        for draft_backend in ("fa4", "trtllm_mha"):
            self.assertFalse(_accepts("EAGLE", draft_backend=draft_backend))

    def test_dspark_arm_unchanged(self):
        """The EAGLE addition must not perturb DSPARK: its MLA verify set
        passes, its chain constraint holds, and unaudited backends refuse."""
        for backend in self.DSPARK_BACKENDS:
            self.assertTrue(
                _accepts("DSPARK", backend=backend),
                f"DSPARK should pass on verify-audited backend {backend}",
            )
        self.assertFalse(_accepts("DSPARK", backend="fa4"))
        self.assertFalse(_accepts("DSPARK", topk=4))

    def test_ngram_refused_because_it_relocates_target_kv_per_token(self):
        """BUG REGRESSION (eval_568). NGRAM was admitted on the reasoning that
        it carries no draft KV, so only the target-verify rails matter. But a
        tree-shaped drafter finalizes a batch through
        `move_accept_tokens_to_target_kvcache`, which relocates INDIVIDUAL
        accepted tokens inside the target pool -- and the unified pool's
        `move_kv_cache` is compaction-only (whole page envelopes). Every
        unified NGRAM cell died there: a reshape past the end on the MHA/mamba
        hosts, and the SWA composite's explicit refusal on gpt-oss. Admitting
        it again without a per-token move primitive re-breaks every family."""
        for backend in self.DSPARK_BACKENDS:
            self.assertFalse(
                _accepts("NGRAM", backend=backend),
                f"NGRAM must stay refused (backend={backend})",
            )

    def test_spec_off_admitted(self):
        """The gate constrains only speculative configurations; spec-off must
        keep booting regardless of family."""
        for is_hybrid_swa in (True, False):
            self.assertTrue(_accepts(None, is_hybrid_swa=is_hybrid_swa))

    def test_unaudited_algorithms_refused(self):
        """Every other algorithm stays out until its verify id rails are
        audited -- on every family."""
        for algorithm in self.UNAUDITED_ALGORITHMS:
            for is_hybrid_swa in (True, False):
                self.assertFalse(_accepts(algorithm, is_hybrid_swa=is_hybrid_swa))


if __name__ == "__main__":
    unittest.main()

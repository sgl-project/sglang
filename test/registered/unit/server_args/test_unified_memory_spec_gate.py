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

Three audited arms (see `handle_unified_memory_pool`), each with its own
backend constraints:
  * DSPARK: the target verifies on the audited verify set (`triton` /
    `trtllm_mla` / `cutedsl_mla` / `tokenspeed_mla` / `flashmla` /
    `flashinfer` / `fa3`, which includes the MLA family). Its draft fuses
    into the target's pages when the fused-draft decision allows, else keeps
    a private pool, so the gate leaves the draft backend unconstrained.
  * DFLASH: the target verifies on `triton` / `fa3` / `flashinfer`; the MLA
    verify family must not leak into this arm.
  * EAGLE/EAGLE3: unified targets only (hybrid-SWA or mamba hybrids, either
    full-pool kind) -- the draft's KV lives fused inside the full pool's
    page envelope (`DenseDraftRegion`), with an automatic private-pool
    fallback when no region resolves. The target's verify set follows the
    host kind: the audited verify set on an MLA host, `triton` /
    `flashinfer` / `fa3` on an MHA host, and an unresolved backend is
    refused. The draft worker (its backend resolves separately: explicit
    flag first, else it inherits the target's) runs on `triton` /
    `flashinfer` / `fa3`.

Every arm drafts a linear chain (`--speculative-eagle-topk` in {None, 1}).
Everything else (NGRAM / STANDALONE / registered customs) stays refused.
NGRAM's refusal is load-bearing, not pending: like tree verify it relocates
accepted tokens one at a time inside the target pool, which the unified
pool's page-granular `move_kv_cache` cannot express. Pinned so no arm silently
widens to an unaudited algorithm, family, tree shape, or backend -- and so one
arm's addition never perturbs another.

    python -m pytest test/registered/unit/server_args/test_unified_memory_spec_gate.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec

import sglang.srt.arg_groups.kv_cache_hook as kv_cache_hook
from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _PlainHFConfig:
    """Matches none of the hybrid-arch isinstance probes."""

    # `mambaish_config` screens on the architecture name, so a stub that
    # declares none is what "not a hybrid arch" looks like to it.
    architectures = ["LlamaForCausalLM"]

    def get_text_config(self):
        return self


def _accepts(
    algorithm: str | None,
    *,
    is_hybrid_swa: bool = True,
    topk: int | None = 1,
    backend: str | None = "triton",
    draft_backend: str | None = None,
    dcp_size: int = 1,
    retraction_backup: str | None = None,
    fields: dict | None = None,
    attention_arch: AttentionArch = AttentionArch.MHA,
) -> bool:
    """Run just `handle_unified_memory_pool` against a minimal stand-in.

    ServerArgs' real constructor pulls in a model config; this exercises the
    single handler under test with the fields it reads (disaggregation off, so
    only the speculative / cache / dcp / cuda-graph checks run). ServerArgs is
    a msgspec Struct, so a field is restated with `force_setattr`, which the
    Struct's own `__setattr__` guard allows and `object.__setattr__` does not.
    """
    sa = ServerArgs(model_path="dummy")
    for name, value in {
        "enable_unified_memory": True,
        "disaggregation_mode": "null",
        "speculative_algorithm": algorithm,
        "speculative_eagle_topk": topk,
        "speculative_draft_attention_backend": draft_backend,
        "enable_hierarchical_cache": False,
        "enable_lmcache": False,
        "disaggregation_decode_retraction_backup": retraction_backup,
        "dcp_size": dcp_size,
        "cuda_graph_config": None,
        "attention_backend": backend,
        "prefill_attention_backend": None,
        "decode_attention_backend": None,
        **(fields or {}),
    }.items():
        msgspec.structs.force_setattr(sa, name, value)
    sa._model_config = SimpleNamespace(
        is_hybrid_swa=is_hybrid_swa,
        attention_arch=attention_arch,
        hf_config=_PlainHFConfig(),
        linear_attn_registry_result=None,
    )
    try:
        handle_unified_memory_pool(sa)
        return True
    except AssertionError:
        return False


class TestUnifiedMemorySpecGate(unittest.TestCase):
    # Verify-audited backends for the DSPARK arm (includes the MLA family).
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
    # Verify-audited backends for the DFLASH arm.
    DFLASH_BACKENDS = ("triton", "fa3", "flashinfer")
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
        DSPARK demonstrates the rule here; the check itself is unconditional,
        so it holds for every admitted arm."""
        for topk in (2, 4, 8):
            self.assertFalse(_accepts("DSPARK", topk=topk))
        # The shape knob must not disturb spec-off.
        self.assertTrue(_accepts(None, topk=None))

    def test_dspark_draft_backend_is_not_constrained(self):
        """The gate does not constrain DSPARK's draft backend: a draft whose
        backend is off the translated rails keeps a private pool (the
        fused-draft decision declines fusion), where its translator is a
        passthrough and any backend reads correctly. Model hooks declare one
        on the operator's behalf -- Kimi-Linear / Kimi-K3 + DSPARK on SM100
        declares `trtllm_mha` -- so refusing it would refuse a configuration
        the operator never touched."""
        for draft_backend in (None, *self.DSPARK_BACKENDS, "fa4", "trtllm_mha"):
            self.assertTrue(
                _accepts("DSPARK", draft_backend=draft_backend),
                f"DSPARK draft backend {draft_backend} should pass",
            )

    def test_dcp_keeps_flashinfer_off_the_verify_list(self):
        """Under --dcp-size > 1 the read ids stay DCP-widened for the consumer
        to finish; flashinfer's spec verify gathers its CSR args with no DCP
        read translation, so it is refused there while the MLA verify family,
        which builds its own DCP block table, still passes."""
        self.assertTrue(
            _accepts(
                "DSPARK",
                backend="trtllm_mla",
                dcp_size=2,
                is_hybrid_swa=False,
                attention_arch=AttentionArch.MLA,
            )
        )
        self.assertFalse(
            _accepts(
                "DSPARK",
                backend="flashinfer",
                dcp_size=2,
                is_hybrid_swa=False,
                attention_arch=AttentionArch.MLA,
            )
        )
        # Without DCP flashinfer stays admitted.
        self.assertTrue(
            _accepts(
                "DSPARK",
                backend="flashinfer",
                is_hybrid_swa=False,
                attention_arch=AttentionArch.MLA,
            )
        )

    def test_dflash_admitted_only_on_its_verify_backends(self):
        """DFLASH's target verifies on triton / fa3 / flashinfer. fa4 and
        trtllm_mha have no translated spec verify path, and the MLA verify
        family the DSPARK arm admits must not leak into this arm."""
        for backend in self.DFLASH_BACKENDS:
            self.assertTrue(
                _accepts("DFLASH", backend=backend),
                f"DFLASH should pass on verify-audited backend {backend}",
            )
        for backend in ("fa4", "trtllm_mha") + tuple(
            b for b in self.DSPARK_BACKENDS if b not in self.DFLASH_BACKENDS
        ):
            self.assertFalse(
                _accepts("DFLASH", backend=backend),
                f"DFLASH must be refused on backend {backend}",
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
        with patch.object(kv_cache_hook, "mambaish_config", return_value=object()):
            for algorithm in ("EAGLE", "EAGLE3"):
                self.assertTrue(_accepts(algorithm, is_hybrid_swa=False))

    def test_eagle_on_mla_mamba_verifies_on_the_mla_family(self):
        """An MLA mamba hybrid fuses into MLA pages, so the target verifies on
        the MLA backend family, while the fused draft is MHA-shaped and must run
        on the translated MHA rails: an MLA-only backend is refused for the
        draft whether named or inherited."""
        with patch.object(kv_cache_hook, "mambaish_config", return_value=object()):
            for backend in self.DSPARK_BACKENDS:
                self.assertTrue(
                    _accepts(
                        "EAGLE",
                        is_hybrid_swa=False,
                        attention_arch=AttentionArch.MLA,
                        backend=backend,
                        draft_backend="triton",
                    ),
                    f"EAGLE on an MLA host should pass on {backend}",
                )
            for backend in ("trtllm_mla", "flashmla"):
                for draft_backend in (None, backend):
                    self.assertFalse(
                        _accepts(
                            "EAGLE",
                            is_hybrid_swa=False,
                            attention_arch=AttentionArch.MLA,
                            backend=backend,
                            draft_backend=draft_backend,
                        ),
                        f"MLA-only draft backend {draft_backend or backend}",
                    )
            for backend in (None, "fa4"):
                self.assertFalse(
                    _accepts(
                        "EAGLE",
                        is_hybrid_swa=False,
                        attention_arch=AttentionArch.MLA,
                        backend=backend,
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
        """A real boot resolves the default backend before this gate runs, so
        an unset one here is unresolved: the gate refuses it rather than
        trusting it."""
        self.assertFalse(_accepts("EAGLE", backend=None))

    def test_eagle_draft_backend_pinned(self):
        """The draft worker resolves its own backend: unset inherits the
        target's (triton here), an explicit triton / flashinfer / fa3 passes,
        and anything else refuses."""
        self.assertTrue(_accepts("EAGLE", draft_backend=None))
        for draft_backend in self.EAGLE_BACKENDS:
            self.assertTrue(_accepts("EAGLE", draft_backend=draft_backend))
        for draft_backend in ("fa4", "trtllm_mha"):
            self.assertFalse(_accepts("EAGLE", draft_backend=draft_backend))

    def test_hierarchical_cache_refused_with_an_eagle_draft(self):
        """HiCache builds a draft host pool off the draft's device pool, and a
        fused draft view has no transfer surface of its own: the page-envelope
        host pool refuses per-layer draft loads. Without a draft HiCache is
        not this gate's to refuse, and with DSPARK HiCache declines draft
        fusion, so the draft keeps its private pool."""
        hicache = {"enable_hierarchical_cache": True}
        for algorithm in ("EAGLE", "EAGLE3"):
            self.assertFalse(_accepts(algorithm, fields=hicache))
        self.assertTrue(_accepts(None, fields=hicache))
        self.assertTrue(_accepts("DSPARK", fields=hicache))

    def test_pd_decode_refusals_raise_their_own_assertions(self):
        """The PD-decode branches screen the model with `mambaish_config`. A
        function-local import of that name further down made it local to the
        whole handler, so these branches raised UnboundLocalError instead of
        their refusal."""
        pd_decode = {
            "disaggregation_mode": "decode",
            "disaggregation_transfer_backend": "mooncake",
            "pp_size": 1,
            "enable_hisparse": False,
        }
        for extra in (
            {
                "disaggregation_decode_enable_offload_kvcache": True,
                "hicache_storage_backend": "file",
            },
            {"disaggregation_decode_retraction_backup": "host_pool"},
        ):
            with patch.object(kv_cache_hook, "mambaish_config", return_value=object()):
                self.assertFalse(
                    _accepts(None, is_hybrid_swa=False, fields={**pd_decode, **extra})
                )

    def test_host_pool_retraction_refused_with_a_fused_draft(self):
        """The host-pool retraction backup builds host pools off the draft's
        device pool, and a fused draft view has no transfer surface. Without a
        draft the backup is not this gate's to refuse."""
        for algorithm in ("EAGLE", "EAGLE3", "DFLASH", "DSPARK"):
            with self.subTest(algorithm=algorithm):
                self.assertTrue(_accepts(algorithm))
                self.assertFalse(_accepts(algorithm, retraction_backup="host_pool"))
        self.assertTrue(_accepts(None, retraction_backup="host_pool"))

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
        """BUG REGRESSION. NGRAM was admitted on the reasoning that
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

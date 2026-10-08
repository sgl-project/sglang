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

Two audited arms, each with its own constraints because each rides a different
draft-KV story:
  * DSPARK: a chain draft whose KV lives in a pool of its own; verify runs on
    every backend whose spec path builds through the KV-index translator.
  * EAGLE/EAGLE3: the draft's KV lives FUSED inside the target's full-attention
    page envelope, which only the hybrid-SWA unified composite provisions, so
    the target family is constrained too. Its backends are the MHA-shaped
    subset -- the MLA verify family cannot serve an MHA draft -- and they are
    demanded EXPLICITLY, for the draft worker as well (it resolves its own
    backend: explicit flag first, else it inherits the target's).

Pinned so no arm silently widens to an unaudited algorithm, tree shape, or
backend. The backend set in particular is a claim about code that exists: a
backend listed here MUST have a translated spec path, or a unified run on it
reads the pool with virtual ids and silently returns wrong tokens.

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
    topk: int | None = 1,
    backend: str | None = "triton",
    is_hybrid_swa: bool = True,
    attention_arch: AttentionArch = AttentionArch.MHA,
    draft_backend: str | None = None,
    dcp_size: int = 1,
    retraction_backup: str | None = None,
    fields: dict | None = None,
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
    # Backends whose speculative verify path builds through the translator.
    DSPARK_BACKENDS = (
        "triton",
        "trtllm_mla",
        "cutedsl_mla",
        "tokenspeed_mla",
        "flashmla",
        "flashinfer",
        "fa3",
    )
    # Algorithms with no audited unified-pool verify rails.
    # Backends the FUSED draft arm can verify on (MHA-shaped).
    EAGLE_BACKENDS = ("triton", "flashinfer", "fa3")
    # Algorithms with no audited unified-pool verify rails.
    UNAUDITED_ALGORITHMS = ("DFLASH", "NGRAM", "STANDALONE")

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

    def test_dspark_draft_backend_is_not_constrained(self):
        """DSPARK's draft owns a KV pool indexed directly by virtual id, so
        its translator is a passthrough and any draft backend reads correctly.
        Model hooks declare one on the operator's behalf -- Kimi-Linear /
        Kimi-K3 + DSPARK on SM100 declares `trtllm_mha` -- so refusing it
        would refuse a configuration the operator never touched."""
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

    def test_eagle_family_admitted_on_hybrid_swa(self):
        """EAGLE/EAGLE3 chain on a hybrid-SWA target is the fused-draft-KV
        configuration -- both spellings, resolved or unset topk, on every
        MHA-shaped audited backend."""
        for algorithm in ("EAGLE", "EAGLE3"):
            for topk in (None, 1):
                for backend in self.EAGLE_BACKENDS:
                    self.assertTrue(
                        _accepts(algorithm, topk=topk, backend=backend),
                        f"{algorithm} topk={topk} backend={backend} should pass",
                    )

    def test_eagle_refused_on_dense_targets(self):
        """No fused draft region outside the unified composites: a dense
        (non-hybrid) target must be refused, not fail at boot."""
        for algorithm in ("EAGLE", "EAGLE3"):
            self.assertFalse(_accepts(algorithm, is_hybrid_swa=False))

    def test_eagle_admitted_on_mamba_mha(self):
        """A mamba hybrid off the MLA backend provisions the region in its
        MHA full sub-pool; refusing it strands the whole mamba x EAGLE
        matrix."""
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

    def test_eagle_refused_unaudited_and_unset_backends(self):
        """The MLA verify family must not leak into the MHA-shaped arm. A real
        boot resolves the default backend before this gate runs, so it checks
        that resolved default; an unresolved one is refused rather than
        trusted."""
        for backend in ("fa4", "trtllm_mha", "trtllm_mla", "flashmla"):
            self.assertFalse(_accepts("EAGLE", backend=backend))
        self.assertFalse(_accepts("EAGLE", backend=None))

    def test_eagle_draft_backend_pinned(self):
        """The draft worker resolves its own backend: unset inherits the
        target's, explicit audited passes, anything else refuses."""
        self.assertTrue(_accepts("EAGLE", draft_backend=None))
        for draft_backend in self.EAGLE_BACKENDS:
            self.assertTrue(_accepts("EAGLE", draft_backend=draft_backend))
        for draft_backend in ("fa4", "trtllm_mha"):
            self.assertFalse(_accepts("EAGLE", draft_backend=draft_backend))

    def test_hierarchical_cache_refused_with_an_eagle_draft(self):
        """HiCache builds a draft host pool off the draft's device pool, and a
        fused draft view has no transfer surface of its own: the page-envelope
        host pool refuses per-layer draft loads. Without a draft, or with
        DSPARK's private pool, HiCache is not this gate's to refuse."""
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

    def test_spec_off_admitted(self):
        """The gate constrains only speculative configurations; spec-off must
        keep booting."""
        self.assertTrue(_accepts(None))

    def test_unaudited_algorithms_refused(self):
        """Every other algorithm stays out until its verify id rails are
        audited -- on every backend."""
        for algorithm in self.UNAUDITED_ALGORITHMS:
            for backend in ("triton", "fa3"):
                self.assertFalse(_accepts(algorithm, backend=backend))


if __name__ == "__main__":
    unittest.main()

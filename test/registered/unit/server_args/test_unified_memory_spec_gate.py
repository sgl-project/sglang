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

One audited arm today: DSPARK, a chain draft whose draft KV lives in a pool of
its own. Its verify runs on the backends whose spec paths build their read
tables through the KV-index translator -- the MLA verify family plus
`flashmla` / `flashinfer` / `fa3`. Everything else stays refused until its
verify id rails are audited.

Pinned so no arm silently widens to an unaudited algorithm, tree shape, or
backend. The backend set in particular is a claim about code that exists: a
backend listed here MUST have a translated spec path, or a unified run on it
reads the pool with virtual ids and silently returns wrong tokens.

    python -m pytest test/registered/unit/server_args/test_unified_memory_spec_gate.py -v
"""

import unittest
from types import SimpleNamespace

import msgspec

from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _accepts(
    algorithm: str | None,
    *,
    topk: int | None = 1,
    backend: str | None = "triton",
    draft_backend: str | None = None,
    dcp_size: int = 1,
    mla: bool = False,
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
        "dcp_size": dcp_size,
        "cuda_graph_config": None,
        "attention_backend": backend,
        "prefill_attention_backend": None,
        "decode_attention_backend": None,
    }.items():
        msgspec.structs.force_setattr(sa, name, value)
    sa._model_config = SimpleNamespace(
        is_hybrid_swa=not mla,
        attention_arch=AttentionArch.MLA if mla else AttentionArch.MHA,
    )
    try:
        handle_unified_memory_pool(sa)
        return True
    except AssertionError:
        return False


class TestUnifiedMemorySpecGate(unittest.TestCase):
    # Backends whose speculative verify path builds through the translator.
    AUDITED_BACKENDS = (
        "triton",
        "trtllm_mla",
        "cutedsl_mla",
        "tokenspeed_mla",
        "flashmla",
        "flashinfer",
        "fa3",
    )
    # Algorithms with no audited unified-pool verify rails.
    UNAUDITED_ALGORITHMS = ("EAGLE", "EAGLE3", "DFLASH", "NGRAM", "STANDALONE")

    def test_dspark_admitted_on_every_audited_backend(self):
        """Each entry is a claim that the backend's spec verify path
        translates. Adding one here without the code is how a unified run
        silently reads the pool with virtual ids."""
        for backend in self.AUDITED_BACKENDS:
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
        for draft_backend in (None, *self.AUDITED_BACKENDS, "fa4", "trtllm_mha"):
            self.assertTrue(
                _accepts("DSPARK", draft_backend=draft_backend),
                f"DSPARK draft backend {draft_backend} should pass",
            )

    def test_dcp_keeps_flashinfer_off_the_verify_list(self):
        """Under --dcp-size > 1 the read ids stay DCP-widened for the consumer
        to finish; flashinfer's spec verify gathers its CSR args with no DCP
        read translation, so it is refused there while the MLA verify family,
        which builds its own DCP block table, still passes."""
        self.assertTrue(_accepts("DSPARK", backend="trtllm_mla", dcp_size=2, mla=True))
        self.assertFalse(_accepts("DSPARK", backend="flashinfer", dcp_size=2, mla=True))
        # Without DCP flashinfer stays admitted.
        self.assertTrue(_accepts("DSPARK", backend="flashinfer", mla=True))

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

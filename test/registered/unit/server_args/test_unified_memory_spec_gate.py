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
        SimpleNamespace(is_hybrid_swa=True, attention_arch=AttentionArch.MHA),
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

    def test_dspark_refused_tree_topk(self):
        """Tree verify needs a per-token move inside the target pool, which
        the unified pool's page-granular `move_kv_cache` cannot express."""
        for topk in (2, 4, 8):
            self.assertFalse(_accepts("DSPARK", topk=topk))

    def test_draft_backend_is_audited_too(self):
        """The draft worker runs its OWN forward on its OWN backend, resolved
        from `--speculative-draft-attention-backend` before it falls back to
        inheriting the target's. Checking only the target therefore leaves an
        unaudited backend reachable: every target arm can be audited while the
        draft translates nothing. Unset must still inherit and pass."""
        self.assertTrue(_accepts("DSPARK", draft_backend=None))
        for draft_backend in self.AUDITED_BACKENDS:
            self.assertTrue(
                _accepts("DSPARK", draft_backend=draft_backend),
                f"audited draft backend {draft_backend} should pass",
            )
        for draft_backend in ("fa4", "trtllm_mha", "torch_native"):
            self.assertFalse(
                _accepts("DSPARK", draft_backend=draft_backend),
                f"unaudited draft backend {draft_backend} must be refused",
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

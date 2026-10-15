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
"""`--enable-unified-memory` speculative-decoding backend allow-lists.

A speculative backend that does not read through the KV-index translator
reads the unified pool with virtual ids and produces wrong tokens without
crashing, so each arm's admitted backend set is pinned exactly.

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

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _PlainHFConfig:
    """Matches none of the hybrid-arch probes (`mambaish_config` screens on
    the architecture name)."""

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
    attention_arch: AttentionArch = AttentionArch.MHA,
) -> bool:
    """Run just `handle_unified_memory_pool` on a ServerArgs whose fields are
    restated with `force_setattr` (ServerArgs is a msgspec Struct)."""
    sa = ServerArgs(model_path="dummy")
    for name, value in {
        "enable_unified_memory": True,
        "disaggregation_mode": "null",
        "speculative_algorithm": algorithm,
        "speculative_eagle_topk": topk,
        "speculative_draft_attention_backend": draft_backend,
        "enable_hierarchical_cache": False,
        "enable_lmcache": False,
        "disaggregation_decode_retraction_backup": None,
        "dcp_size": dcp_size,
        "cuda_graph_config": None,
        "attention_backend": backend,
        "prefill_attention_backend": None,
        "decode_attention_backend": None,
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


# The audited verify set (includes the MLA family).
_VERIFY_BACKENDS = (
    "triton",
    "trtllm_mla",
    "cutedsl_mla",
    "tokenspeed_mla",
    "flashmla",
    "flashinfer",
    "fa3",
)
_MHA_RAILS = ("triton", "flashinfer", "fa3")


class TestUnifiedMemorySpecGate(unittest.TestCase):
    VERIFY_BACKENDS = _VERIFY_BACKENDS
    MHA_RAILS = _MHA_RAILS
    MLA_ONLY = tuple(b for b in _VERIFY_BACKENDS if b not in _MHA_RAILS)
    UNAUDITED = ("fa4", "trtllm_mha")

    def test_dspark_verify_backends(self):
        for backend in self.VERIFY_BACKENDS:
            self.assertTrue(_accepts("DSPARK", backend=backend), backend)
        for backend in self.UNAUDITED:
            self.assertFalse(_accepts("DSPARK", backend=backend), backend)

    def test_dcp_keeps_flashinfer_off_the_verify_list(self):
        """Under DCP flashinfer's spec verify reads DCP-widened ids; the MLA
        verify family builds its own DCP block table."""
        mla = dict(is_hybrid_swa=False, attention_arch=AttentionArch.MLA)
        self.assertTrue(_accepts("DSPARK", backend="trtllm_mla", dcp_size=2, **mla))
        self.assertFalse(_accepts("DSPARK", backend="flashinfer", dcp_size=2, **mla))
        self.assertTrue(_accepts("DSPARK", backend="flashinfer", **mla))

    def test_dflash_verify_backends(self):
        """DFLASH verifies on the MHA rails only; the MLA family DSPARK admits
        must not leak into this arm."""
        for backend in self.MHA_RAILS:
            self.assertTrue(_accepts("DFLASH", backend=backend), backend)
        for backend in self.UNAUDITED + self.MLA_ONLY:
            self.assertFalse(_accepts("DFLASH", backend=backend), backend)

    def test_eagle_verify_backends_on_an_mha_host(self):
        for algorithm in ("EAGLE", "EAGLE3"):
            for topk in (None, 1):
                for backend in self.MHA_RAILS:
                    self.assertTrue(
                        _accepts(algorithm, topk=topk, backend=backend),
                        f"{algorithm} topk={topk} {backend}",
                    )
            for backend in self.UNAUDITED + self.MLA_ONLY:
                self.assertFalse(
                    _accepts(algorithm, backend=backend, draft_backend="triton"),
                    f"{algorithm} {backend}",
                )

    def test_eagle_draft_backend(self):
        """An unset draft backend inherits the target's; an explicit one must
        be on the MHA rails."""
        self.assertTrue(_accepts("EAGLE", draft_backend=None))
        for draft_backend in self.MHA_RAILS:
            self.assertTrue(_accepts("EAGLE", draft_backend=draft_backend))
        for draft_backend in self.UNAUDITED:
            self.assertFalse(_accepts("EAGLE", draft_backend=draft_backend))

    def test_eagle_on_an_mla_host(self):
        """An MLA mamba hybrid verifies on the audited verify set, while its
        fused draft is MHA-shaped: an MLA-only draft backend is refused whether
        named or inherited."""
        mla = dict(is_hybrid_swa=False, attention_arch=AttentionArch.MLA)
        with patch.object(kv_cache_hook, "mambaish_config", return_value=object()):
            for backend in self.VERIFY_BACKENDS:
                self.assertTrue(
                    _accepts("EAGLE", backend=backend, draft_backend="triton", **mla),
                    backend,
                )
            self.assertFalse(
                _accepts("EAGLE", backend="fa4", draft_backend="triton", **mla)
            )
            for backend in ("trtllm_mla", "flashmla"):
                for draft_backend in (None, backend):
                    self.assertFalse(
                        _accepts(
                            "EAGLE", backend=backend, draft_backend=draft_backend, **mla
                        ),
                        draft_backend or backend,
                    )

    def test_unaudited_algorithms_and_tree_drafting_refused(self):
        """NGRAM and tree drafting relocate accepted tokens one at a time
        inside the target pool, which the unified pool's page-granular move
        cannot express."""
        for algorithm in ("NGRAM", "STANDALONE"):
            for is_hybrid_swa in (True, False):
                self.assertFalse(_accepts(algorithm, is_hybrid_swa=is_hybrid_swa))
        for algorithm in ("DSPARK", "EAGLE"):
            for topk in (2, 4):
                self.assertFalse(_accepts(algorithm, topk=topk), f"{algorithm} {topk}")


if __name__ == "__main__":
    unittest.main()

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
"""`--enable-unified-memory` speculative-decoding gates.

The target verifies on whatever backend the layout gates admit: every
backend reads the unified pool through the KV-index runner_ids. What is
still gated here is the fused EAGLE draft's backend, DCP's verify on
flashinfer, and the algorithms and tree shapes the pool cannot serve.

    python -m pytest test/registered/unit/server_args/test_unified_memory_spec_gate.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec

import sglang.srt.configs.hybrid_arch as hybrid_arch
from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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


# The backends the layout gates admit for a unified target.
_MLA_VERIFY = ("triton", "trtllm_mla", "cutedsl_mla", "flashmla", "flashinfer", "fa3")
_MHA_VERIFY = ("triton", "flashinfer", "fa3", "fa4", "trtllm_mha")


class TestUnifiedMemorySpecGate(CustomTestCase):
    def test_target_backend_is_not_gated(self):
        """No target backend translates ids itself, so none is refused for
        the algorithm it verifies."""
        mla = dict(is_hybrid_swa=False, attention_arch=AttentionArch.MLA)
        for algorithm in ("DSPARK", "DFLASH", "EAGLE", "EAGLE3"):
            for backend in _MHA_VERIFY:
                self.assertTrue(
                    _accepts(algorithm, backend=backend, draft_backend="triton"),
                    f"{algorithm} {backend}",
                )
        for algorithm in ("DSPARK", "DFLASH"):
            for backend in _MLA_VERIFY:
                self.assertTrue(
                    _accepts(algorithm, backend=backend, **mla),
                    f"{algorithm} {backend}",
                )

    def test_dcp_keeps_flashinfer_off_the_verify_list(self):
        """Under DCP flashinfer's spec verify reads DCP-widened ids; the MLA
        verify family builds its own DCP block table."""
        mla = dict(is_hybrid_swa=False, attention_arch=AttentionArch.MLA)
        for algorithm in ("DSPARK", "DFLASH"):
            for backend in ("trtllm_mla", "cutedsl_mla"):
                self.assertTrue(
                    _accepts(algorithm, backend=backend, dcp_size=2, **mla),
                    f"{algorithm} {backend}",
                )
            self.assertFalse(
                _accepts(algorithm, backend="flashinfer", dcp_size=2, **mla)
            )
            self.assertTrue(_accepts(algorithm, backend="flashinfer", **mla))

    def test_draft_backend_is_not_gated(self):
        """A fused draft is parts of the target's sub-pools read through views,
        so no draft backend, named or inherited, is refused for it."""
        mla = dict(is_hybrid_swa=False, attention_arch=AttentionArch.MLA)
        for draft_backend in (None, "triton", "flashinfer", "fa3", "fa4", "trtllm_mha"):
            self.assertTrue(_accepts("EAGLE", draft_backend=draft_backend))
        self.assertTrue(_accepts("EAGLE", backend="fa4", draft_backend=None))
        # The gate reads the lifetimes the model declares, its state from here.
        with patch.object(hybrid_arch, "mambaish_config", return_value=object()):
            for backend in _MLA_VERIFY:
                for draft_backend in (None, "triton"):
                    self.assertTrue(
                        _accepts(
                            "EAGLE", backend=backend, draft_backend=draft_backend, **mla
                        ),
                        f"{backend} {draft_backend}",
                    )

    def test_eagle_needs_a_unified_target(self):
        """EAGLE's draft fuses into the target's sub-pools, so the target must
        declare one beside the full one."""
        mla = dict(is_hybrid_swa=False, attention_arch=AttentionArch.MLA)
        with patch.object(hybrid_arch, "mambaish_config", return_value=None):
            self.assertFalse(_accepts("EAGLE", backend="triton", **mla))

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

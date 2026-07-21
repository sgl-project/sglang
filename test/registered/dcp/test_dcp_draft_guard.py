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
"""Regression tests: every draft-side forward/metadata entry point runs under
draft_forward_guard (get_parallel().dcp_enabled == False).

Root cause pinned 2026-07-21 (probe-v2 dump pair, jobs 1495062 off /
1496158 dcp8): the EAGLE chain-decode step built its attention metadata
OUTSIDE ModelRunner.forward, so it never saw the runner's
draft_forward_guard. With DCP on, the TRTLLMMLA-family multi-step draft
backend took its `dcp_enabled() and is_decode_or_idle()` branches
(dcp-granular page size, rank-local lens) against the draft's UNSHARDED
replicated pool; the guarded forward then consumed that poisoned metadata,
each rank attended a wrong ~1/D slice, and the draft's TP allreduce merged
the garbage identically on every rank. Result: the 2nd chain token
degenerated to a copy of the 1st (verify input_ids showed d2 == d1 on 5/9
cycles vs 0/9 off; d2 auto-rejected) => the EAGLE3 accept-length x0.83
deficit under DCP. d1 stayed clean because the draft-extend path builds its
metadata inside the guarded ModelRunner.forward, and DRAFT_EXTEND_V2
metadata has no DCP branch.

These tests drive the real worker code with SimpleNamespace fakes and
assert the guard state at every recorded draft-side call site while the
ambient context has dcp_enabled=True (the scheduler's state under DCP).
The graph-replay site matters as much as the eager one: replay-prep
(_apply_cuda_graph_metadata) has the same decode-mode DCP branch.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")

from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative import eagle_worker_common as common_mod
from sglang.srt.speculative import eagle_worker_v2 as worker_mod


class TestDraftChainGuard(unittest.TestCase):
    """EagleDraftWorker.draft: chain metadata init, eager forwards, and
    graph replay-prep must all run with dcp_enabled == False."""

    def _drive_draft(self, can_cuda_graph):
        records = {}

        def rec(key, ret=None):
            records[key] = get_parallel().dcp_enabled
            return ret

        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            mark_forward_metadata_ready=lambda: rec("mark_ready"),
        )
        fake_worker = SimpleNamespace(
            req_to_token_pool=None,
            cuda_graph_runner=SimpleNamespace(
                execute=lambda fb: rec("graph_execute", (1, 2, 3, 4))
            ),
            draft_runner=SimpleNamespace(canary_manager=None),
            topk=1,
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            seed_dsa_topk_from_draft_extend=False,
            draft_attn_backend=SimpleNamespace(
                init_forward_metadata=lambda fb: rec("init_metadata")
            ),
            draft_forward=lambda fb: rec("draft_forward", (1, 2, 3, 4)),
            target_worker=None,
            tree_mask_mode=None,
            device="cpu",
        )
        batch = SimpleNamespace(spec_info=SimpleNamespace(dsa_topk_indices=None))
        with (
            mock.patch.object(
                worker_mod,
                "prepare_for_draft",
                return_value=(forward_batch, can_cuda_graph),
            ),
            mock.patch.object(
                worker_mod, "build_eagle_verify_input", return_value="verify-input"
            ),
            get_parallel().override(dcp_enabled=True),
        ):
            out = worker_mod.EagleDraftWorker.draft(fake_worker, batch)
        self.assertEqual(out, "verify-input")
        return records

    def test_eager_chain_metadata_and_forward_guarded(self):
        records = self._drive_draft(can_cuda_graph=False)
        # The eager path must hit metadata init + the chain forward.
        self.assertIn("init_metadata", records)
        self.assertIn("draft_forward", records)
        for key, dcp_enabled in records.items():
            self.assertFalse(
                dcp_enabled,
                f"draft-side call site '{key}' ran with dcp_enabled=True "
                "(draft_forward_guard missing on the chain-decode path)",
            )

    def test_graph_replay_prep_guarded(self):
        records = self._drive_draft(can_cuda_graph=True)
        self.assertIn("graph_execute", records)
        for key, dcp_enabled in records.items():
            self.assertFalse(
                dcp_enabled,
                f"draft-side call site '{key}' ran with dcp_enabled=True "
                "(draft_forward_guard missing on the graph replay path)",
            )

    def test_gate_off_reproduces_bypass(self):
        """SGLANG_DCP_DRAFT_CHAIN_GUARD=0 must revert to the pre-fix
        behavior (chain metadata built with ambient dcp_enabled=True) — the
        red leg proving this suite would have caught the original bug, and
        the revert switch the fix-on/fix-off A/B uses."""
        from sglang.srt import environ

        with environ.envs.SGLANG_DCP_DRAFT_CHAIN_GUARD.override(False):
            records = self._drive_draft(can_cuda_graph=False)
        self.assertTrue(
            records["init_metadata"],
            "gate-off should reproduce the unguarded (dcp_enabled=True) "
            "chain metadata build; if this fails the A/B revert switch is dead",
        )


class TestDraftExtendMetadataGuard(unittest.TestCase):
    """prepare_for_draft_extend's eager metadata init must run under the
    guard. DRAFT_EXTEND_V2 metadata has no DCP branch today, so this is the
    contract pin that keeps future extend-mode DCP branches from recreating
    the chain-decode bug on the extend path."""

    def test_eager_extend_metadata_guarded(self):
        records = {}

        def rec(key):
            records[key] = get_parallel().dcp_enabled

        batch = SimpleNamespace(
            seq_lens=torch.tensor([5, 9]),
            seq_lens_cpu=None,
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            input_ids=None,
            model_config=SimpleNamespace(vocab_size=100),
            device="cpu",
        )
        forward_batch = SimpleNamespace(
            seq_lens=torch.tensor([5, 9]),
            seq_lens_cpu=None,
            mark_forward_metadata_ready=lambda: rec("mark_ready"),
        )
        draft_runner = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_standalone=lambda: False),
            attn_backend=SimpleNamespace(
                init_forward_metadata=lambda fb: rec("init_metadata")
            ),
        )
        draft_extend_input = SimpleNamespace(
            num_front_tokens=0, hidden_states=None, positions=None
        )
        predict = torch.tensor([1, 2], dtype=torch.int64)
        with (
            mock.patch.object(
                common_mod.ForwardBatch,
                "init_new",
                staticmethod(lambda *a, **k: forward_batch),
            ),
            get_parallel().override(dcp_enabled=True),
        ):
            out = common_mod.prepare_for_draft_extend(
                draft_extend_input,
                batch,
                predict,
                3,
                draft_runner,
                None,
                return_hidden_states_before_norm=False,
            )
        self.assertIs(out, forward_batch)
        self.assertIn("init_metadata", records)
        for key, dcp_enabled in records.items():
            self.assertFalse(
                dcp_enabled,
                f"draft-extend call site '{key}' ran with dcp_enabled=True "
                "(draft_forward_guard missing in prepare_for_draft_extend)",
            )


if __name__ == "__main__":
    unittest.main()

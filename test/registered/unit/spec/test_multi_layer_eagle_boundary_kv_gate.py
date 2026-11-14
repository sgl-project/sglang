"""Which multi-layer EAGLE drafts widen the draft-extend window.

The boundary-KV fix re-runs front rows below the committed length so the draft
KV there is recomputed from confirmed tokens. A chain draft's step k at
position q embeds the token at q + k + 1, still a draft guess when q was
written, so every chain draft needs the K - 1 rows; a hybrid (state-slot) cache
adds the conv warm-up rows on top. Only a non-chain draft on a plain cache
skips the fix.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleDraftWorker,
)

NUM_STEPS = 4
CONV_WARMUP = 3
HIDDEN = 8
REQ_POOL = 5


def _draft_worker(*, chain: bool, hybrid: bool) -> MultiLayerEagleDraftWorker:
    # skip __init__: only the attributes _init_boundary_kv_fix_state reads are set
    worker = object.__new__(MultiLayerEagleDraftWorker)
    worker.topk = 1
    worker.speculative_num_steps = NUM_STEPS
    worker.chain_mtp_hidden_states = chain
    worker.device = "cpu"
    worker.req_to_token_pool = SimpleNamespace(
        holds_state_slots=hybrid,
        req_to_token=torch.zeros((REQ_POOL, 32), dtype=torch.int32),
    )
    worker.draft_runner_list = [
        SimpleNamespace(
            model_config=SimpleNamespace(hidden_size=HIDDEN), dtype=torch.float32
        )
    ]
    # min(conv[0].shape[2:]) is the conv warm-up width.
    conv = [torch.zeros(2, 6, CONV_WARMUP)]
    state_part = SimpleNamespace(mamba_cache=SimpleNamespace(conv=conv))
    worker.target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model_config=SimpleNamespace(hidden_size=HIDDEN),
            token_to_kv_pool=SimpleNamespace(
                layer_router=SimpleNamespace(part_view=lambda part: state_part)
            ),
        )
    )
    return worker


class TestBoundaryKvFixGate(CustomTestCase):
    def test_front_rows_by_draft_kind_and_cache(self):
        chain_depth = NUM_STEPS - 1
        cases = (
            # (chain, hybrid, front rows)
            (True, False, chain_depth),
            (True, True, chain_depth + CONV_WARMUP),
            (False, True, chain_depth + CONV_WARMUP),
            (False, False, 0),
        )
        for chain, hybrid, front in cases:
            with self.subTest(chain=chain, hybrid=hybrid):
                worker = _draft_worker(chain=chain, hybrid=hybrid)
                with envs.SGLANG_ENABLE_MTP_BOUNDARY_KV_FIX.override(True):
                    worker._init_boundary_kv_fix_state()
                self.assertEqual(worker.draft_extend_num_front_tokens, front)
                if front == 0:
                    self.assertIsNone(worker.boundary_kv_stash_tokens)
                    continue
                self.assertEqual(
                    worker.draft_extend_num_warmup_tokens,
                    CONV_WARMUP if hybrid else 0,
                )
                self.assertEqual(
                    tuple(worker.boundary_kv_stash_tokens.shape), (REQ_POOL, front)
                )
                self.assertEqual(
                    tuple(worker.boundary_kv_stash_hiddens.shape),
                    (REQ_POOL, front, HIDDEN),
                )

    def test_disabled_by_env(self):
        worker = _draft_worker(chain=True, hybrid=False)
        with envs.SGLANG_ENABLE_MTP_BOUNDARY_KV_FIX.override(False):
            worker._init_boundary_kv_fix_state()
        self.assertEqual(worker.draft_extend_num_front_tokens, 0)


if __name__ == "__main__":
    unittest.main()

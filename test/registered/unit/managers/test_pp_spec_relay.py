"""The PP+spec relay: the trees the last stage drafts reach the next round of
the same requests through their slots, and the slot's steps stay as built."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler_components.pp_ring import PPRing  # noqa: E402
from sglang.srt.managers.scheduler_components.pp_spec_forward import (  # noqa: E402
    PPSpecForward,
)
from sglang.srt.managers.utils import GenerationBatchResult  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.runtime_context import get_spec  # noqa: E402
from sglang.srt.speculative.pp_spec_relay import PPSpecRelay  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang.test.step_freeze import freeze_step, release_step  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _frozen_step(forward_mode, req_pool_indices, **fields) -> ScheduleBatch:
    step = ScheduleBatch.__new__(ScheduleBatch)
    step.reqs = [object() for _ in req_pool_indices]
    step.forward_mode = forward_mode
    step.req_pool_indices = torch.tensor(req_pool_indices)
    step.spec_algorithm = SpeculativeAlgorithm.EAGLE
    for name, value in fields.items():
        setattr(step, name, value)
    freeze_step(step)
    return step


class TestPPSpecRelay(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology(role="scheduler", pp_size=2))
        # topk = 2: the relayed topology differs from the chain of a root.
        enter_scope(
            self,
            get_spec().override(
                speculative_num_steps=2,
                speculative_eagle_topk=2,
                speculative_num_draft_tokens=4,
            ),
        )
        self.s = object.__new__(PPRing)
        self.s.pp_spec_relay = PPSpecRelay(req_pool_size=8, device="cpu")
        # The forward verifies the trees the ring's relay wrote.
        self.fwd = object.__new__(PPSpecForward)
        self.fwd.pp_spec_relay = self.s.pp_spec_relay

    def commit(self, step: ScheduleBatch, tensors) -> None:
        """What the ring does when the relay brings the step's outputs back."""
        self.s.pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
        PPRing._prep_batch_result(
            self.s,
            step,
            GenerationBatchResult(can_run_cuda_graph=False),
            PPProxyTensors(tensors),
        )
        release_step(step)

    def launch(self, step: ScheduleBatch, is_last_rank: bool):
        """Launch a verify round; return the verify input's tree inputs and
        the batch the forward ran on."""
        s = self.fwd
        s.pp_group = SimpleNamespace(is_last_rank=is_last_rank)
        s.device = "cpu"
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(verify_mask=None),
                model_config=SimpleNamespace(context_len=64),
            ),
            forward_model=Mock(return_value=GenerationBatchResult()),
        )
        s.model_worker = SimpleNamespace(
            forward_batch_generation=Mock(return_value=GenerationBatchResult())
        )
        s.req_to_token_pool = object()
        build_tree = Mock(return_value=(None,) * 6)
        verify_out_cache_loc = torch.tensor([9])
        prepare_verify = Mock(
            return_value=(SimpleNamespace(out_cache_loc=verify_out_cache_loc), False)
        )
        with (
            patch(
                "sglang.srt.speculative.eagle_utils.build_tree_kernel_efficient",
                build_tree,
            ),
            patch(
                "sglang.srt.speculative.eagle_utils.eagle_prepare_for_verify",
                prepare_verify,
            ),
        ):
            result = PPSpecForward.spec_forward(s, step, pp_proxy_tensors=None)
        if is_last_rank:
            (forward_batch,), _ = s.model_worker.forward_batch_generation.call_args
        else:
            verify_input, _, forward_batch, _ = prepare_verify.call_args.args
            self.assertIs(forward_batch.spec_info, verify_input)
            # Kept for this stage's commit of the accepted path.
            self.assertIs(result.spec_verify_out_cache_loc, verify_out_cache_loc)
        return build_tree.call_args.args, forward_batch

    def test_next_round_verifies_the_trees_relayed_to_its_slots(self):
        # A prompt's last chunk roots the request's tree at its sampled token;
        # a middle chunk samples nothing.
        self.commit(
            _frozen_step(
                ForwardMode.EXTEND,
                [7],
                contains_last_prefill_chunk=True,
                return_logprob=False,
            ),
            {"next_token_ids": torch.tensor([42])},
        )
        self.commit(
            _frozen_step(
                ForwardMode.EXTEND,
                [3],
                contains_last_prefill_chunk=False,
                return_logprob=False,
            ),
            {"next_token_ids": torch.tensor([99])},
        )
        # A verify round relays the trees drafted for its rows' next round.
        self.commit(
            _frozen_step(ForwardMode.DECODE, [5, 2], return_logprob=False),
            {
                "next_token_ids": torch.tensor([11, 12, 0, 21, 0, 0]),
                "spec_accept_lens": torch.tensor([2, 1]),
                "spec_new_seq_lens": torch.tensor([10, 20]),
                "spec_bonus_tokens": torch.tensor([12, 21]),
                "spec_next_chain": torch.tensor([12, 13, 14, 15, 21, 23, 24, 25]),
                "spec_next_parents": torch.tensor([[-1, 0, 1], [-1, 0, 2]]),
                "spec_next_top_scores": torch.tensor([[0, 1, 3], [0, 2, 3]]),
            },
        )
        self.assertEqual(self.s.pp_spec_relay.tokens[3].tolist(), [0, 0, 0, 0])

        # The next round runs another composition of the slot's requests.
        for is_last_rank in (False, True):
            with self.subTest(is_last_rank=is_last_rank):
                seq_lens = torch.tensor([20, 8, 10])
                step = _frozen_step(
                    ForwardMode.DECODE,
                    [2, 7, 5],
                    seq_lens=seq_lens,
                    seq_lens_cpu=seq_lens.clone(),
                    sampling_info=None,
                    spec_info=None,
                )
                tree_inputs, forward_batch = self.launch(step, is_last_rank)
                release_step(step)

                self.assertIsNot(forward_batch, step)
                bonus, parents, top_scores, drafts, build_seq_lens = tree_inputs[:5]
                self.assertEqual(bonus.tolist(), [21, 42, 12])
                self.assertEqual(
                    drafts.tolist(), [[23, 24, 25], [0, 0, 0], [13, 14, 15]]
                )
                # A root's zero drafts hang on the chain topology.
                self.assertEqual(parents.tolist(), [[-1, 0, 2], [-1, 0, 1], [-1, 0, 1]])
                self.assertEqual(top_scores.tolist(), [[0, 2, 3], [0, 1, 2], [0, 1, 3]])
                self.assertIs(build_seq_lens, seq_lens)


if __name__ == "__main__":
    unittest.main()

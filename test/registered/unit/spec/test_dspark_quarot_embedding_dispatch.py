"""Exercise the real proposer and both embedding-to-forward paths.

This checks module selection and preserves the existing V4-style selection;
it does not execute draft attention, graph replay, or acceptance.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _ForwardBoundaryReached(Exception):
    pass


class _CaptureLayer(nn.Module):
    def forward(self, positions, hidden_states, forward_batch, residual):
        self.hidden_states = hidden_states.detach().clone()
        raise _ForwardBoundaryReached


class TestDSparkQuaRotEmbeddingDispatch(CustomTestCase):
    def _check_selection(self, *, config_state, sample_from_anchor, own_vocab):
        draft_embedding = nn.Embedding(4, 2)
        target_embedding = nn.Embedding(4, 2)
        draft_model = SimpleNamespace(
            embed_tokens=draft_embedding,
            uses_own_vocab_modules=own_vocab,
        )
        if config_state != "absent":
            draft_model._glm_dspark_quarot_config = (
                object() if config_state == "active" else None
            )
        target_model = SimpleNamespace(
            get_input_embeddings=Mock(return_value=target_embedding)
        )
        proposer = DraftBlockProposer.__new__(DraftBlockProposer)
        proposer.draft_model = draft_model
        proposer.sample_from_anchor = sample_from_anchor
        proposer._draft_sampler = None
        proposer._run_forward = Mock(side_effect=_ForwardBoundaryReached)

        with self.assertRaises(_ForwardBoundaryReached):
            proposer.propose(
                batch=None,
                draft_input=None,
                verify_window=None,
                bs=1,
                device="cpu",
                target_model=target_model,
                sampling_info=None,
            )

        proposer._run_forward.assert_called_once()
        use_draft = config_state == "active" or not sample_from_anchor
        self.assertIs(
            proposer._run_forward.call_args.kwargs["embed_module"],
            draft_embedding if use_draft else target_embedding,
        )
        if use_draft:
            target_model.get_input_embeddings.assert_not_called()
        else:
            target_model.get_input_embeddings.assert_called_once_with()

    def test_only_active_quarot_config_overrides_anchor_embedding(self):
        for config_state in ("absent", "none", "active"):
            for sample_from_anchor in (False, True):
                with self.subTest(
                    config_state=config_state, sample_from_anchor=sample_from_anchor
                ):
                    self._check_selection(
                        config_state=config_state,
                        sample_from_anchor=sample_from_anchor,
                        own_vocab=False,
                    )

    def test_existing_own_vocab_flag_does_not_change_selection(self):
        # V4 already exposes this flag. It is not the opt-in for the GLM path.
        for config_state in ("absent", "none"):
            for sample_from_anchor in (False, True):
                with self.subTest(
                    config_state=config_state, sample_from_anchor=sample_from_anchor
                ):
                    self._check_selection(
                        config_state=config_state,
                        sample_from_anchor=sample_from_anchor,
                        own_vocab=True,
                    )

    def test_external_and_in_model_embedding_reach_the_same_layer_input(self):
        for embed_in_graph in (False, True):
            with (
                self.subTest(embed_in_graph=embed_in_graph),
                envs.SGLANG_DSPARK_EMBED_IN_GRAPH.override(embed_in_graph),
                get_context().override_server_args(
                    device="cpu", speculative_algorithm="DSpark"
                ),
            ):
                # Use the production embedding and forward methods, stopping at
                # the first layer instead of executing attention or graph replay.
                model = DSparkDraftModel.__new__(DSparkDraftModel)
                nn.Module.__init__(model)
                model.embed_tokens = nn.Embedding(8, 4)
                with torch.no_grad():
                    model.embed_tokens.weight.copy_(torch.arange(32).reshape(8, 4))
                model._glm_dspark_quarot_config = object()
                capture = _CaptureLayer()
                model.layers = nn.ModuleList([capture])
                forwarded = []

                def forward(batch):
                    self.assertIsInstance(batch, ForwardBatch)
                    forwarded.append(batch)
                    return DFlashDraftModel.forward(
                        model,
                        batch.input_ids,
                        batch.positions,
                        batch,
                        input_embeds=batch.input_embeds,
                    )

                proposer = DraftBlockProposer.__new__(DraftBlockProposer)
                proposer.draft_model = model
                proposer.draft_model_runner = SimpleNamespace(
                    device=torch.device("cpu"),
                    decode_cuda_graph_runner=None,
                    forward=forward,
                )
                proposer.sample_from_anchor = True
                proposer.gamma = proposer.query_token_num = 3
                proposer._mask_token_id = 7
                proposer._draft_block_ids_buf = None
                proposer._draft_block_spec_info = None
                proposer._draft_sampler = None
                proposer._dp_moe_sync = False
                proposer._num_token_non_padded = None
                target = SimpleNamespace(
                    get_input_embeddings=Mock(
                        side_effect=AssertionError("target embedding must not be used")
                    )
                )
                with self.assertRaises(_ForwardBoundaryReached):
                    proposer.propose(
                        batch=SimpleNamespace(
                            seq_lens=torch.tensor([5, 9]),
                            seq_lens_cpu=torch.tensor([5, 9]),
                            req_pool_indices=torch.tensor([0, 1]),
                            can_run_decode_cuda_graph=False,
                        ),
                        draft_input=SimpleNamespace(bonus_tokens=torch.tensor([1, 2])),
                        verify_window=SimpleNamespace(
                            positions_2d=torch.tensor([[5, 6, 7], [9, 10, 11]]),
                            verify_cache_loc_2d=torch.arange(6).reshape(2, 3),
                        ),
                        bs=2,
                        device="cpu",
                        target_model=target,
                        sampling_info=None,
                    )

                target.get_input_embeddings.assert_not_called()
                self.assertEqual(len(forwarded), 1)
                if embed_in_graph:
                    self.assertIsNone(forwarded[0].input_embeds)
                else:
                    self.assertIsNotNone(forwarded[0].input_embeds)
                expected = model.embed_tokens.weight[torch.tensor([1, 7, 7, 2, 7, 7])]
                torch.testing.assert_close(
                    capture.hidden_states, expected, rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()

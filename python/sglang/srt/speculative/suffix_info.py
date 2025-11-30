"""
Data structures for suffix decoding speculative method.

Most of the logic is reused from ngram_info.py since both use tree-based verification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpecInputType

logger = logging.getLogger(__name__)


@dataclass
class SuffixVerifyInput(NgramVerifyInput):
    """
    Data structure for suffix decoding verification.

    Inherits from NgramVerifyInput as both use identical tree-based verification
    without a separate draft model. The only differences are the SpecInputType
    and debug logging for accepted tokens.
    """

    def __init__(
        self,
        draft_token: torch.Tensor,
        tree_mask: torch.Tensor,
        positions: torch.Tensor,
        retrive_index: torch.Tensor,
        retrive_next_token: torch.Tensor,
        retrive_next_sibling: torch.Tensor,
        draft_token_num: int,
    ):
        # Call parent init to reuse all initialization logic
        super().__init__(
            draft_token,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_token_num,
        )
        # Override to set correct SpecInputType for suffix decoding
        self.spec_input_type = SpecInputType.SUFFIX_VERIFY

    def _fill_requests(
        self,
        batch: ScheduleBatch,
        logits_output: torch.Tensor,
    ):
        """
        Fill requests with accepted tokens.

        Overrides parent to add debug logging of accepted tokens.
        """
        accept_index_cpu = self.accept_index.tolist()
        predict_cpu = self.predict.tolist()
        has_finished = False

        # Iterate every accepted token and check if req has finished after append the token
        # should be checked BEFORE free kv cache slots
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            accepted_tokens = []
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                id = predict_cpu[idx]
                accepted_tokens.append(id)
                req.output_ids.append(id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    # set all tokens after finished token to -1 and break
                    self.accept_index[i, j + 1 :] = -1
                    break
                else:
                    if req.grammar is not None:
                        try:
                            req.grammar.accept_token(id)
                        except ValueError as e:
                            logger.info(
                                f"{i=}, {req=}\n"
                                f"{self.accept_index=}\n"
                                f"{self.predict=}\n"
                            )
                            raise e
            if accepted_tokens:
                logger.debug(
                    f"[DEBUG SUFFIX VERIFY] req={req.rid}: Accepted {len(accepted_tokens)} tokens: {accepted_tokens[:10]}"
                )
            req.spec_verify_ct += 1
        if has_finished:
            self.accept_length = (self.accept_index != -1).sum(dim=1) - 1
        self.accept_index = self.accept_index[self.accept_index != -1]

        logits_output.next_token_logits = logits_output.next_token_logits[
            self.accept_index
        ]
        if logits_output.hidden_states:
            logits_output.hidden_states = logits_output.hidden_states[self.accept_index]
        self.verified_id = self.predict[self.accept_index]

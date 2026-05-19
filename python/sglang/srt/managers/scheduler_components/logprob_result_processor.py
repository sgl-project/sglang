from __future__ import annotations

from dataclasses import dataclass
from typing import (
    List,
    Tuple,
)

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import (
    MIS_DELIMITER_TOKEN_ID,
    ServerArgs,
)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLogprobResultProcessor:
    server_args: ServerArgs
    model_config: ModelConfig

    def _process_input_token_logprobs(
        self, req: Req, input_token_logprobs: List
    ) -> None:
        """Process input token logprobs values and indices."""
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Process logprob values - handle multi-item scoring vs regular requests
        if is_multi_item_scoring:
            # Multi-item scoring: use all logprobs as-is
            req.logprob.input_token_logprobs_val = input_token_logprobs
        else:
            # Regular request: add None at start, remove last (sampling token)
            req.logprob.input_token_logprobs_val = [None] + input_token_logprobs[:-1]

        # Process logprob indices based on scoring type
        if is_multi_item_scoring:
            # MIS scores come from input_token_ids_logprobs, not input_token_logprobs.
            # But the shared pipeline requires input_token_logprobs_idx to be the same
            # length as input_token_logprobs_val (validated at line 816). We fill with
            # MIS_DELIMITER_TOKEN_ID as a dummy — score_request() ignores this field.
            delimiter_count = len(req.multi_item_delimiter_indices)
            input_token_logprobs_idx = [MIS_DELIMITER_TOKEN_ID] * delimiter_count
        else:
            # Regular request: include all tokens from logprob_start_len onwards
            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len :]

        # Clip padded hash values from image tokens to prevent detokenization errors
        req.logprob.input_token_logprobs_idx = [
            x if x < self.model_config.vocab_size - 1 else 0
            for x in input_token_logprobs_idx
        ]

    def _process_input_top_logprobs(self, req: Req) -> None:
        """Process input top logprobs."""
        if req.logprob.top_logprobs_num <= 0:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        req.logprob.input_top_logprobs_val = [] if is_multi_item_scoring else [None]
        req.logprob.input_top_logprobs_idx = [] if is_multi_item_scoring else [None]

        # Extend arrays with temp values
        for val, idx in zip(
            req.temp_input_top_logprobs_val,
            req.temp_input_top_logprobs_idx,
            strict=True,
        ):
            req.logprob.input_top_logprobs_val.extend(val)
            req.logprob.input_top_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        if not is_multi_item_scoring:
            req.logprob.input_top_logprobs_val.pop()
            req.logprob.input_top_logprobs_idx.pop()

        # Clean up temp storage
        req.temp_input_top_logprobs_idx = None
        req.temp_input_top_logprobs_val = None

    def _process_input_token_ids_logprobs(self, req: Req) -> None:
        """Process input token IDs logprobs."""
        if req.logprob.token_ids_logprob is None:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        req.logprob.input_token_ids_logprobs_val = (
            [] if is_multi_item_scoring else [None]
        )
        req.logprob.input_token_ids_logprobs_idx = (
            [] if is_multi_item_scoring else [None]
        )

        # Process temp values - convert tensors to lists and extend arrays
        for val, idx in zip(
            req.temp_input_token_ids_logprobs_val,
            req.temp_input_token_ids_logprobs_idx,
            strict=True,
        ):
            val_list = val.tolist() if isinstance(val, torch.Tensor) else val
            req.logprob.input_token_ids_logprobs_val.extend(
                val_list if isinstance(val_list, list) else [val_list]
            )
            req.logprob.input_token_ids_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        if not is_multi_item_scoring:
            req.logprob.input_token_ids_logprobs_val.pop()
            req.logprob.input_token_ids_logprobs_idx.pop()

        # Clean up temp storage
        req.temp_input_token_ids_logprobs_idx = None
        req.temp_input_token_ids_logprobs_val = None

    def _calculate_relevant_tokens_len(self, req: Req) -> int:
        """Calculate the expected length of logprob arrays based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions from logprob_start_len onwards have logprobs.
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        if is_multi_item_scoring:
            return len(req.multi_item_delimiter_indices)
        else:
            return len(req.origin_input_ids[req.logprob_start_len :])

    def calculate_num_input_logprobs(
        self,
        req: Req,
        extend_input_len: int,
        extend_logprob_start_len: int,
    ) -> int:
        """Calculate the number of input logprobs based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions in the range have logprobs.
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        if is_multi_item_scoring:
            # Count pre-computed delimiter indices within the extend range
            return sum(
                1
                for idx in req.multi_item_delimiter_indices
                if extend_logprob_start_len <= idx < extend_input_len
            )
        else:
            # Regular request: all tokens in the range
            return extend_input_len - extend_logprob_start_len

    def _is_multi_item_scoring(self, req: Req) -> bool:
        """Check if request uses multi-item scoring.

        Multi-item scoring applies to prefill-only requests when a delimiter
        token is configured. In this mode, only positions containing the
        delimiter token receive logprobs.
        """
        return (
            self.server_args.enable_mis
            and req.is_prefill_only
            and req.multi_item_delimiter_indices is not None
        )

    def add_input_logprob_return_values(
        self,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,  # If True, it means prefill is finished.
    ):
        """Incrementally add input logprobs to `req`.

        Args:
            i: The request index in a batch.
            req: The request. Input logprobs inside req are modified as a
                consequence of the API
            fill_ids: The prefill ids processed.
            output: Logit processor output that's used to compute input logprobs
            last_prefill_chunk: True if it is the last prefill (when chunked).
                Some of input logprob operation should only happen at the last
                prefill (e.g., computing input token logprobs).
        """
        assert output.input_token_logprobs is not None
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.logprob.input_token_logprobs_val is not None:
            # The input logprob has been already computed. It only happens
            # upon retract.
            if req.logprob.top_logprobs_num > 0:
                assert req.logprob.input_token_logprobs_val is not None
            return

        # Important for the performance.
        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: Tuple[int] = output.input_token_logprobs
        input_token_logprobs = input_token_logprobs[
            logprob_pt : logprob_pt + num_input_logprobs
        ]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.logprob.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.logprob.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(
                output.input_token_ids_logprobs_val[i]
            )
            req.temp_input_token_ids_logprobs_idx.append(
                output.input_token_ids_logprobs_idx[i]
            )

        if last_prefill_chunk:
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.logprob.input_token_logprobs_val is None
            assert req.logprob.input_token_logprobs_idx is None
            assert req.logprob.input_top_logprobs_val is None
            assert req.logprob.input_top_logprobs_idx is None

            # Process all input logprob types using helper functions
            self._process_input_token_logprobs(req, input_token_logprobs)
            self._process_input_top_logprobs(req)

            self._process_input_token_ids_logprobs(req)

            if req.return_logprob:
                relevant_tokens_len = self._calculate_relevant_tokens_len(req)
                assert len(req.logprob.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.logprob.input_token_logprobs_idx) == relevant_tokens_len
                if req.logprob.top_logprobs_num > 0:
                    assert (
                        len(req.logprob.input_top_logprobs_val) == relevant_tokens_len
                    )
                    assert (
                        len(req.logprob.input_top_logprobs_idx) == relevant_tokens_len
                    )
                if req.logprob.token_ids_logprob is not None:
                    assert (
                        len(req.logprob.input_token_ids_logprobs_val)
                        == relevant_tokens_len
                    )
                    assert (
                        len(req.logprob.input_token_ids_logprobs_idx)
                        == relevant_tokens_len
                    )

    def add_logprob_return_values(
        self,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        if output.next_token_logprobs is not None:
            req.logprob.output_token_logprobs_val.append(output.next_token_logprobs[i])
            req.logprob.output_token_logprobs_idx.append(next_token_ids[i])

        # Only add input logprobs if there are input tokens to process
        # Note: For prefill-only requests with default logprob_start_len, this will be 0,
        # meaning we only compute output logprobs (which is the intended behavior)
        if num_input_logprobs > 0:
            self.add_input_logprob_return_values(
                i,
                req,
                output,
                pt,
                num_input_logprobs,
                last_prefill_chunk=True,
            )
        else:
            self._initialize_empty_logprob_containers(req)

        if req.logprob.top_logprobs_num > 0:
            req.logprob.output_top_logprobs_val.append(
                output.next_token_top_logprobs_val[i]
            )
            req.logprob.output_top_logprobs_idx.append(
                output.next_token_top_logprobs_idx[i]
            )

        if (
            req.logprob.token_ids_logprob is not None
            and output.next_token_token_ids_logprobs_val is not None
        ):
            # Convert GPU tensor to list if needed
            logprobs_val = output.next_token_token_ids_logprobs_val[i]
            if isinstance(logprobs_val, torch.Tensor):
                logprobs_val = logprobs_val.tolist()
            req.logprob.output_token_ids_logprobs_val.append(logprobs_val)
            req.logprob.output_token_ids_logprobs_idx.append(
                output.next_token_token_ids_logprobs_idx[i]
            )

        return num_input_logprobs

    def _initialize_empty_logprob_containers(self, req: Req) -> None:
        """
        Initialize logprob fields to empty lists if unset.

        This is needed for prefill-only requests where the normal initialization
        flow might be bypassed, but downstream code expects these fields to be lists.
        """
        if req.logprob.input_token_logprobs_val is None:
            req.logprob.input_token_logprobs_val = []
        if req.logprob.input_token_logprobs_idx is None:
            req.logprob.input_token_logprobs_idx = []
        if req.logprob.input_top_logprobs_val is None:
            req.logprob.input_top_logprobs_val = []
        if req.logprob.input_top_logprobs_idx is None:
            req.logprob.input_top_logprobs_idx = []
        if req.logprob.input_token_ids_logprobs_val is None:
            req.logprob.input_token_ids_logprobs_val = []
        if req.logprob.input_token_ids_logprobs_idx is None:
            req.logprob.input_token_ids_logprobs_idx = []

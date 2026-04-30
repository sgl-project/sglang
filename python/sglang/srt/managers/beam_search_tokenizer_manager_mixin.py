# Copyright 2026 SGLang Team
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
"""Mixin class for handling beam search in TokenizerManager."""

import logging
from typing import Any, Dict, Optional, Union

from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
)

logger = logging.getLogger(__name__)


class BeamSearchTokenizerManagerMixin:
    def build_beam_search_out(self, out: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a beam search out dict (containing beam_results) to a regular out dict.

        Takes the first beam's meta_info as the top-level meta_info, and stores
        the full beam_results list inside meta_info so callers can access all beams.
        All post-processing (logging, metrics, abort handling, timing, etc.) is handled
        by the shared _wait_one_response logic after this conversion.
        """
        beam_results = out.get("beam_results", [])
        if not beam_results:
            return out
        first_beam = beam_results[0]
        # Use the first beam's fields as the top-level out, put all beams in meta_info.
        converted = {
            "text": first_beam.get("text", ""),
            "output_ids": first_beam.get("output_ids", []),
            "meta_info": first_beam.get("meta_info", {}).copy(),
        }
        converted["meta_info"]["beam_results"] = beam_results
        return converted

    def try_build_beam_search_out_dict(
        self,
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchTokenIDOutput,
        ],
        i: int,
        meta_info: Dict[str, Any],
    ) -> Optional[dict]:
        """If this item is a beam search result, build and return the out_dict.
        Returns None if not a beam search item.
        """
        # Only support BatchTokenIDOutput or BatchStrOutput for beam search
        if not isinstance(recv_obj, (BatchTokenIDOutput, BatchStrOutput)):
            return None

        beam_search_output = (
            recv_obj.beam_search_output[i]
            if recv_obj.beam_search_output and i < len(recv_obj.beam_search_output)
            else None
        )
        has_beam_search = (
            beam_search_output is not None
            and hasattr(beam_search_output, "sequences")
            and beam_search_output.sequences
        )
        if not has_beam_search or recv_obj.finished_reasons[i] is None:
            return None

        return self._build_beam_search_out_dict(beam_search_output, meta_info, recv_obj)

    def _build_beam_search_out_dict(
        self,
        beam_search_output: Any,
        meta_info: Dict[str, Any],
        recv_obj: Union[BatchStrOutput, BatchTokenIDOutput],
    ) -> dict:
        """Build the out_dict for a beam search result."""
        beam_results = []
        total_completion_tokens = sum(
            len(beam_seq.tokens) for beam_seq in beam_search_output.sequences
        )
        for idx, beam_seq in enumerate(beam_search_output.sequences):
            if isinstance(recv_obj, BatchStrOutput):
                beam_out_dict = {
                    "text": beam_seq.text if beam_seq.text else "",
                    "output_ids": beam_seq.tokens.copy(),
                }
            elif isinstance(recv_obj, BatchTokenIDOutput):
                beam_out_dict = {
                    "output_ids": beam_seq.tokens.copy(),
                }
            else:
                continue
            if idx == 0:
                beam_meta_info = meta_info.copy()
                # Override completion_tokens with the sum of all beam sequences,
                # since recv_obj.completion_tokens[i] only counts the first beam.
                beam_meta_info["completion_tokens"] = total_completion_tokens
            else:
                beam_meta_info = {}
            beam_meta_info["finish_reason"] = beam_seq.finish_reason
            beam_meta_info["sequence_score"] = beam_seq.beam_score
            beam_out_dict["meta_info"] = beam_meta_info

            beam_results.append(beam_out_dict)

        return {"beam_results": beam_results}

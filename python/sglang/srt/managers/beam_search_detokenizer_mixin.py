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
"""Mixin class for beam search detokenization logic."""

from sglang.srt.managers.io_struct import BatchTokenIDOutput


class BeamSearchDetokenizerMixin:
    def is_beam_search_batch(self, recv_obj: BatchTokenIDOutput) -> bool:
        """Check if the batch contains beam search requests."""
        return (
            recv_obj.beam_search_output is not None
            and len(recv_obj.beam_search_output) > 0
        )

    def decode_beam_search_output(self, recv_obj: BatchTokenIDOutput):
        """Decode beam search candidate sequences to text."""
        if self.disable_tokenizer_batch_decode:
            for i, beam_output in enumerate(recv_obj.beam_search_output):
                for beam in beam_output.sequences:
                    trimmed_tokens = self.trim_matched_stop(
                        beam.tokens,
                        recv_obj.finished_reasons[i],
                        recv_obj.no_stop_trim[i],
                    )
                    beam.text = self.tokenizer.decode(
                        trimmed_tokens,
                        skip_special_tokens=recv_obj.skip_special_tokens[i],
                        spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[
                            i
                        ],
                    )
        else:
            beam_ids = []
            for i, beam_output in enumerate(recv_obj.beam_search_output):
                for beam in beam_output.sequences:
                    beam_ids.append(
                        self.trim_matched_stop(
                            beam.tokens,
                            recv_obj.finished_reasons[i],
                            recv_obj.no_stop_trim[i],
                        )
                    )
            beam_texts = self.tokenizer.batch_decode(
                beam_ids,
                skip_special_tokens=recv_obj.skip_special_tokens[0],
                spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
            )

            i = 0
            for beam_output in recv_obj.beam_search_output:
                for beam in beam_output.sequences:
                    beam.text = beam_texts[i]
                    i += 1

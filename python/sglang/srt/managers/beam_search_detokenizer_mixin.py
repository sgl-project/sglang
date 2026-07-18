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
                if beam_output is None:
                    # Mixed batch: this item is not a beam request.
                    continue
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
            # batch_decode only accepts scalar skip_special_tokens /
            # spaces_between_special_tokens flags, so decode per request (each
            # request's beams share that request's flags). Batching across the whole
            # batch would apply request 0's flags to every request's beams.
            for i, beam_output in enumerate(recv_obj.beam_search_output):
                if beam_output is None:
                    # Mixed batch: this item is not a beam request.
                    continue
                trimmed_tokens = [
                    self.trim_matched_stop(
                        beam.tokens,
                        recv_obj.finished_reasons[i],
                        recv_obj.no_stop_trim[i],
                    )
                    for beam in beam_output.sequences
                ]
                beam_texts = self.tokenizer.batch_decode(
                    trimmed_tokens,
                    skip_special_tokens=recv_obj.skip_special_tokens[i],
                    spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[
                        i
                    ],
                )
                for beam, text in zip(beam_output.sequences, beam_texts):
                    beam.text = text

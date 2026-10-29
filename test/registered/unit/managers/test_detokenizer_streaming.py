"""Regression coverage for streamed text after detokenizer state eviction."""

import unittest
from array import array

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.detokenizer_manager import (
    DecodeStatus,
    DetokenizerManager,
    LimitedCapacityDict,
)
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDetokenizerStreaming(CustomTestCase):
    def test_state_eviction_preserves_streamed_text(self):
        """Chunk offsets preserve fresh text and stop trimming after state eviction."""
        # Use the real byte-level BPE decoder without downloading a model.
        vocab = {
            piece: index
            for index, piece in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
        }
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, clean_up_tokenization_spaces=False
        )
        tokenizer.additional_stop_token_ids = []
        ascii_chunks = [tokenizer.encode(text) for text in ("A", "BC", "D")]
        unicode_chunks = [[token] for token in tokenizer.encode("A🦄B")]
        stop_chunks = [tokenizer.encode(text) for text in ("A", "ST", "OP", "suffix")]
        cases = [
            ("ascii", ascii_chunks, (None, 1, 2), None, False, "ABCD"),
            # Split the four-byte character across chunks. Evict before or
            # after it; its incomplete state must survive the intervening steps.
            ("utf8", unicode_chunks, (None, 1, 5), None, False, "A🦄B"),
            ("stop_trim", stop_chunks, (None, 1, 2), "STOP", False, "A"),
            ("stop_keep", stop_chunks, (None, 1, 2), "STOP", True, "ASTOP"),
        ]
        for name, chunks, evictions, stop, no_stop_trim, expected in cases:
            for prompt in ("hi", "long prompt"):
                for evict_before in evictions:
                    for disable_batch_decode in (False, True):
                        with self.subTest(
                            case=name,
                            prompt=prompt,
                            evict_before=evict_before,
                            disable_batch_decode=disable_batch_decode,
                        ):
                            params = SamplingParams(
                                stop=stop, no_stop_trim=no_stop_trim
                            )
                            params.normalize(tokenizer)
                            req = Req(
                                rid="streaming",
                                origin_input_text=prompt,
                                origin_input_ids=array("q", tokenizer.encode(prompt)),
                                sampling_params=params,
                                stream=True,
                            )
                            req.tokenizer = tokenizer
                            detokenizer = DetokenizerManager.__new__(DetokenizerManager)
                            detokenizer.tokenizer = tokenizer
                            detokenizer.vocab_size = len(tokenizer)
                            detokenizer.decode_status = LimitedCapacityDict(capacity=1)
                            detokenizer.disable_tokenizer_batch_decode = (
                                disable_batch_decode
                            )
                            detokenizer.is_tool_call_parser_gpt_oss = False

                            emitted = []
                            for index, chunk in enumerate(chunks):
                                if index == evict_before:
                                    # A different request fills the only state slot.
                                    detokenizer.decode_status["other"] = DecodeStatus(
                                        decoded_text="",
                                        decode_ids=[],
                                        surr_offset=0,
                                        read_offset=0,
                                    )
                                req.output_ids.extend(chunk)
                                req.update_finish_state(new_accepted_len=len(chunk))
                                if index == len(chunks) - 1 and not req.finished():
                                    req.finished_reason = FINISH_LENGTH(
                                        len(req.output_ids)
                                    )
                                accumulator = _GenerationStreamAccumulator(
                                    return_logprob=False,
                                    return_hidden_states=False,
                                    return_routed_experts=False,
                                    return_indexer_topk=False,
                                    spec_algorithm=SpeculativeAlgorithm.NONE,
                                    disaggregation_mode=DisaggregationMode.NULL,
                                    default_stream_interval=1,
                                    default_force_stream_interval=1,
                                    get_cached_tokens_details=lambda req: None,
                                    is_request_level_customized_info_key=lambda key: (
                                        False
                                    ),
                                    current_weight_version=None,
                                )
                                accumulator.accept(req=req)
                                payload = accumulator.to_payload(
                                    dp_rank=0, is_idle_batch=False
                                )
                                if payload is not None:
                                    emitted.extend(
                                        detokenizer._decode_batch_token_id_output(
                                            payload
                                        )
                                    )
                                if req.finished():
                                    break

                            self.assertEqual("".join(emitted), expected)


if __name__ == "__main__":
    unittest.main()

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    DSV4Metadata,
    _tail_rows,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import BlockIds
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def metadata(rows):
    return DSV4Metadata(
        core_attn_metadata=Mock(),
        indexer_metadata=None,
        tail_metadata=DSV4Metadata(
            core_attn_metadata=None,
            indexer_metadata=None,
            late_layer_tail=SimpleNamespace(
                cp_metadata=None,
                extend_seq_lens_cpu=[min(128, n) for n in rows],
            ),
        ),
    )


class TestTailMetadata(CustomTestCase):
    def test_contiguous_tail_excludes_graph_padding(self):
        source = torch.arange(128)
        result = _tail_rows(source, token_indices=torch.arange(44), contiguous_start=0)
        self.assertEqual(result.tolist(), list(range(44)))
        self.assertEqual(result.data_ptr(), source.data_ptr())

    def test_replay_uses_its_own_tail_after_an_eager_batch(self):
        backend = object.__new__(DeepseekV4AttnBackend)
        captured = metadata([128])
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, max_seq_len_override=512
        )
        for rows in ([44], [24, 24], [80], [0, 128]):
            with self.subTest(rows=rows):
                eager = metadata([512, 256])
                backend.forward_metadata = eager
                backend._build_forward_metadata = Mock(return_value=metadata(rows))
                backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
                    captured, batch, static_forward_batch=batch
                )
                published = BlockIds(torch.arange(sum(rows)).view(-1, 1), rows)
                backend._publish_candidate_metadata(published)
                self.assertIs(backend.forward_metadata, captured)
                self.assertIs(backend.tail_forward_metadata, captured.tail_metadata)
                torch.testing.assert_close(
                    captured.tail_metadata.candidate_metadata.blocks, published.blocks
                )
                self.assertIsNone(eager.tail_metadata.candidate_metadata)

    def test_forward_switch_does_not_inherit_a_tail(self):
        backend = object.__new__(DeepseekV4AttnBackend)
        previous = metadata([512])
        backend.forward_metadata = previous
        self.assertIs(backend.tail_forward_metadata, previous.tail_metadata)
        backend.forward_metadata = DSV4Metadata(None, None)
        self.assertIsNone(backend.tail_forward_metadata)
        backend.forward_metadata = previous.tail_metadata
        self.assertIs(backend.tail_forward_metadata, previous.tail_metadata)

    def test_context_parallel_candidate_tail_uses_local_rows(self):
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.forward_metadata = metadata([5, 0, 3])
        tail = backend.tail_forward_metadata.late_layer_tail
        tail.cp_metadata = object()
        tail.local_lens_cpu = [1, 0, 2]
        published = BlockIds(torch.arange(8).view(8, 1), [5, 0, 3])
        backend._publish_candidate_metadata(published)
        self.assertEqual(
            backend.tail_forward_metadata.candidate_metadata.blocks.flatten().tolist(),
            [4, 6, 7],
        )


class TestBoundedPrefillReaders(CustomTestCase):
    def test_prompt_logprobs_rejected_before_dispatch(self):
        from sglang.srt.managers.io_struct import GenerateReqInput
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.runtime_context import get_context

        manager = TokenizerManager.__new__(TokenizerManager)
        manager.context_len = 1024
        manager.num_reserved_tokens = 0
        manager.allow_auto_truncate = False
        manager.validate_total_tokens = False
        manager.is_generation = True
        for encoder, decoder, embedded in (
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, True, True),
        ):
            for start in (0, 511, 512, -1, None):
                request = GenerateReqInput(
                    input_ids=None if embedded else [42] * 512,
                    input_embeds=[[0.0]] * 512 if embedded else None,
                    sampling_params={"max_new_tokens": 1},
                    return_logprob=True,
                    logprob_start_len=start,
                )
                with (
                    self.subTest(
                        encoder=encoder, decoder=decoder, embedded=embedded, start=start
                    ),
                    get_context().override_server_args(
                        enable_encoder_swa_bounded_replay=encoder,
                        enable_decoder_swa_bounded_replay=decoder,
                    ),
                ):
                    if (encoder or decoder) and start in (0, 511):
                        with self.assertRaisesRegex(ValueError, "prompt logprobs"):
                            manager._validate_one_request(request, request.input_ids)
                    else:
                        manager._validate_one_request(request, request.input_ids)

    def test_graph_replay_rejects_uncomputed_prompt_rows(self):
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            enable_breakable_cuda_graph,
        )
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4Model

        model = SimpleNamespace(late_layer_start=2, dspark_layers_to_capture=None)
        for name in ("is_bounded_prefill", "_check_late_layer_tail_readers"):
            setattr(model, name, MethodType(getattr(DeepseekV4Model, name), model))
        network = SimpleNamespace(vision=None, model=model)
        for hidden, logprob, message in (
            (CaptureHiddenMode.FULL, False, "hidden states"),
            (CaptureHiddenMode.NULL, True, "logprobs"),
        ):
            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                capture_hidden_mode=hidden,
                return_logprob=logprob,
                extend_logprob_start_lens_cpu=[0],
                extend_seq_lens_cpu=[44],
            )
            with (
                self.subTest(hidden=hidden, logprob=logprob),
                enable_breakable_cuda_graph(),
                self.assertRaisesRegex(ValueError, message),
            ):
                DeepseekV4ForCausalLM.forward(
                    network, torch.arange(44), torch.arange(44), batch
                )


if __name__ == "__main__":
    unittest.main()

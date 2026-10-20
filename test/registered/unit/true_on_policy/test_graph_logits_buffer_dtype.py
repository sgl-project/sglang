import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    get_next_token_logits_dtype,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
from sglang.srt.true_on_policy import QWEN3_DENSE_TRUE_ON_POLICY_V1
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

_SERVER_ARGS_TARGET = "sglang.srt.true_on_policy.config._get_global_server_args"
_GET_EXEC_TARGET = "sglang.srt.layers.logits_processor.get_exec"

VOCAB = 37
ROWS = 4


def _server_args(*, contract):
    return SimpleNamespace(
        rl_on_policy_target=None, true_on_policy_contract=contract, tp_size=1
    )


def _exec(*, enable_fp32_lm_head=False):
    return SimpleNamespace(
        features=SimpleNamespace(enable_fp32_lm_head=enable_fp32_lm_head)
    )


def _copy_logits(config, logits, logits_buffer):
    processor = LogitsProcessor.__new__(LogitsProcessor)
    processor.vocab_size = config.vocab_size
    processor.next_token_logits_dtype = get_next_token_logits_dtype(config)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.DECODE, next_token_logits_buffer=logits_buffer
    )
    return processor._copy_logits_to_buffer(logits, metadata)


class TestGraphLogitsBufferDtype(unittest.TestCase):
    """CUDA-graph replay copies logits into the shared buffer, eager forward does
    not; both must hand the sampler identical logits, or true-on-policy top-k and
    sampled-token logprobs drift from the trainer on decode only."""

    def _graph_and_eager_logits(
        self, *, contract, enable_fp32_lm_head=False, final_logit_softcapping=None
    ):
        config = SimpleNamespace(
            vocab_size=VOCAB, final_logit_softcapping=final_logit_softcapping
        )
        logits = torch.randn(ROWS, VOCAB).to(torch.bfloat16)
        with (
            patch(_SERVER_ARGS_TARGET, return_value=_server_args(contract=contract)),
            patch(
                _GET_EXEC_TARGET,
                return_value=_exec(enable_fp32_lm_head=enable_fp32_lm_head),
            ),
        ):
            shared = GraphSharedOutput(device=torch.device("cpu"), max_rows=ROWS)
            logits_buffer = shared.get_logits_buffer(
                VOCAB, rows=ROWS, dtype=get_next_token_logits_dtype(config)
            )
            graph = _copy_logits(config, logits, logits_buffer)
            eager = _copy_logits(config, logits, None)
        self.assertEqual(graph.data_ptr(), logits_buffer.data_ptr())
        return graph, eager

    def test_true_on_policy_graph_logits_match_eager_bf16(self):
        graph, eager = self._graph_and_eager_logits(
            contract=QWEN3_DENSE_TRUE_ON_POLICY_V1
        )
        self.assertEqual(graph.dtype, torch.bfloat16)
        self.assertEqual(eager.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(graph, eager))
        self.assertTrue(
            torch.equal(torch.log_softmax(graph, -1), torch.log_softmax(eager, -1))
        )

    def test_default_fp32_lm_head_and_softcap_keep_fp32_buffer(self):
        # A logit softcap is fused with an fp32 cast, so the buffer stays fp32 under the contract.
        for contract, enable_fp32_lm_head, softcap in (
            (None, False, None),
            (QWEN3_DENSE_TRUE_ON_POLICY_V1, True, None),
            (QWEN3_DENSE_TRUE_ON_POLICY_V1, False, 30.0),
        ):
            with self.subTest(
                contract=contract, fp32_lm_head=enable_fp32_lm_head, softcap=softcap
            ):
                graph, eager = self._graph_and_eager_logits(
                    contract=contract,
                    enable_fp32_lm_head=enable_fp32_lm_head,
                    final_logit_softcapping=softcap,
                )
                self.assertEqual(graph.dtype, torch.float)
                self.assertEqual(eager.dtype, torch.float)
                self.assertTrue(torch.equal(graph, eager))

    def test_multi_layer_eagle_draft_extend_buffer_follows_logits_dtype(self):
        # This runner owns its next_token_logits_buffer and holds per-step runners,
        # not a model_runner of its own.
        from sglang.srt.speculative import (
            multi_layer_eagle_draft_extend_cuda_graph_runner as runner_module,
        )

        model_config = SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                vocab_size=VOCAB, final_logit_softcapping=None
            ),
        )
        runner = object.__new__(
            runner_module.MultiLayerEagleMultiStepDraftExtendCudaGraphRunner
        )
        runner.runners = [
            None,
            SimpleNamespace(model_runner=SimpleNamespace(model_config=model_config)),
        ]
        runner.max_bs = 2
        runner.captured_req_width = 2
        runner.seq_len_fill_value = 1
        runner.device = "cpu"
        runner.prune_draft_extend_logits = False
        runner.rotates_in_graph = False
        runner.require_gathered_buffer = False
        runner._vocab_size = lambda: VOCAB
        with (
            patch(
                _SERVER_ARGS_TARGET,
                return_value=_server_args(contract=QWEN3_DENSE_TRUE_ON_POLICY_V1),
            ),
            patch(_GET_EXEC_TARGET, return_value=_exec()),
            patch.object(
                runner_module,
                "get_draft_input_from_target_hidden_dim",
                return_value=8,
            ),
            patch.object(
                runner_module,
                "get_exec",
                return_value=SimpleNamespace(
                    mamba=SimpleNamespace(enable_mamba_extra_buffer=False)
                ),
            ),
        ):
            buffers = runner._allocate_buffers()
        self.assertEqual(buffers.next_token_logits_buffer.dtype, torch.bfloat16)

    def test_shared_buffer_is_keyed_by_dtype(self):
        shared = GraphSharedOutput(device=torch.device("cpu"), max_rows=ROWS)
        fp32 = shared.get_logits_buffer(VOCAB, rows=ROWS, dtype=torch.float)
        bf16 = shared.get_logits_buffer(VOCAB, rows=2, dtype=torch.bfloat16)
        self.assertEqual(fp32.dtype, torch.float)
        self.assertEqual(bf16.dtype, torch.bfloat16)
        self.assertEqual(
            shared.get_logits_buffer(VOCAB, rows=ROWS, dtype=torch.bfloat16).data_ptr(),
            bf16.data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()

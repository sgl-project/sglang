"""CPU ownership tests for Frozen-KV-MTP KDA CUDA-graph telemetry.

FrozenKVMTPCudaGraphRunner intentionally bypasses DecodeCudaGraphRunner.__init__,
so these tests pin its explicit transaction boundary.  Metadata/provenance
refresh happens before that boundary; if it fails, no KDA graph was launched and
the correct replay receipt is therefore zero events.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.linear.kda_route_telemetry import (
    KDACudaGraphRoutePlans,
)
from sglang.srt.speculative import draft_utils
from sglang.srt.speculative import frozen_kv_mtp_cuda_graph_runner as runner_module
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner import (
    FrozenKVMTPCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFrozenKVMTPKDATelemetryOwnership(unittest.TestCase):
    def test_eagle_hybrid_draft_backends_are_full_attention_only(self):
        """EAGLE custom runners cannot reach the KDA terminal wrapper.

        Both ordinary and multi-layer EAGLE use DraftBackendFactory.  Its
        ``hybrid_linear_attn`` rail deliberately selects a full-attention draft
        backend on every device family; KDA-capable MTP uses FrozenKVMTP instead.
        """
        factory = Mock()
        cpu_backend = object()
        blackwell_backend = object()
        other_gpu_backend = object()
        factory._create_intel_amx_decode_backend.return_value = cpu_backend
        factory._create_triton_decode_backend.return_value = blackwell_backend
        factory._create_fa3_decode_backend.return_value = other_gpu_backend
        factory._create_intel_amx_prefill_backend.return_value = cpu_backend
        factory._create_triton_prefill_backend.return_value = blackwell_backend
        factory._create_fa3_prefill_backend.return_value = other_gpu_backend

        cases = (
            (True, True, True, cpu_backend),
            (False, False, True, blackwell_backend),
            (False, False, False, other_gpu_backend),
        )
        for is_cpu, has_amx, is_blackwell, expected in cases:
            with self.subTest(
                is_cpu=is_cpu, has_amx=has_amx, is_blackwell=is_blackwell
            ), patch.object(draft_utils, "is_cpu", return_value=is_cpu), patch.object(
                draft_utils, "cpu_has_amx_support", return_value=has_amx
            ), patch.object(
                draft_utils, "is_blackwell", return_value=is_blackwell
            ):
                self.assertIs(
                    DraftBackendFactory._create_hybrid_linear_attn_decode_backend(
                        factory
                    ),
                    expected,
                )
                self.assertIs(
                    DraftBackendFactory._create_hybrid_linear_attn_prefill_backend(
                        factory
                    ),
                    expected,
                )

    def test_direct_backend_replay_is_owned_by_route_transaction(self):
        runner = FrozenKVMTPCudaGraphRunner.__new__(FrozenKVMTPCudaGraphRunner)
        runner.backend = Mock()
        runner.backend.replay.return_value = "output"
        runner.kda_cuda_graph_route_plans = KDACudaGraphRoutePlans()
        shape_key = object()
        forward_batch = object()

        with patch.object(
            runner_module,
            "replay_kda_route_plan",
            side_effect=lambda shape, mode, replay, **kwargs: replay(),
        ) as replay_transaction:
            output = runner._replay_graph(shape_key, forward_batch)

        self.assertEqual(output, "output")
        replay_transaction.assert_called_once()
        args, kwargs = replay_transaction.call_args
        self.assertEqual(args[:2], (shape_key, "decode"))
        self.assertIs(kwargs["plans"], runner.kda_cuda_graph_route_plans)
        runner.backend.replay.assert_called_once_with(shape_key, forward_batch)

    def test_metadata_provenance_failure_precedes_replay_transaction(self):
        runner = FrozenKVMTPCudaGraphRunner.__new__(FrozenKVMTPCudaGraphRunner)
        runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
        runner.captured_req_width = 1
        runner.topk = 1
        runner.require_mlp_tp_gather = False
        runner.require_gathered_buffer = False
        runner.capture_bs = [1]
        runner.seq_len_fill_value = 1
        runner._pad_to_bucket = lambda raw_bs, capture_bs: raw_bs
        runner.model_runner = SimpleNamespace(device_timer=None)
        runner.kda_cuda_graph_route_plans = KDACudaGraphRoutePlans()
        runner.backend = Mock()
        runner.buffers = SimpleNamespace(
            seq_lens=torch.empty(1, dtype=torch.int64),
            positions=torch.empty(1, dtype=torch.int64),
            mrope_positions=torch.empty(3, 1, dtype=torch.int64),
            bonus_tokens=torch.empty(1, dtype=torch.int64),
            hidden_states=torch.empty(1, 2),
            req_pool_indices=torch.empty(1, dtype=torch.int64),
            seq_lens_cpu=torch.empty(1, dtype=torch.int64),
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_gpu=None,
        )
        provenance_failure = RuntimeError("stale frozen-KV provenance")
        runner.frozen_kv_mtp_worker = SimpleNamespace(
            _init_frozen_kv_metadata_replay_cuda_graph=Mock(
                side_effect=provenance_failure
            )
        )
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens=torch.tensor([5], dtype=torch.int64),
            positions=torch.tensor([4], dtype=torch.int64),
            mrope_positions=None,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([5], dtype=torch.int64),
            seq_lens_sum=5,
            spec_info=SimpleNamespace(
                bonus_tokens=torch.tensor([1], dtype=torch.int64),
                hidden_states=torch.zeros(1, 2),
            ),
        )

        with patch.object(runner_module, "replay_kda_route_plan") as transaction:
            with self.assertRaisesRegex(RuntimeError, "stale frozen-KV provenance"):
                runner.execute(forward_batch)

        transaction.assert_not_called()
        runner.backend.replay.assert_not_called()


if __name__ == "__main__":
    unittest.main()

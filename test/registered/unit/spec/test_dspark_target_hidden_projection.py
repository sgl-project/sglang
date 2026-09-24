import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Attention:
    num_kv_heads = 1
    head_dim = 3

    def __init__(self) -> None:
        self.attn = SimpleNamespace(k_scale=0.5, v_scale=0.25)
        self.input = None

    def kv_proj_only(self, hidden_states: torch.Tensor):
        self.input = hidden_states
        return hidden_states + 1, hidden_states + 2

    def apply_k_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 3

    def apply_k_rope(
        self, _positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return hidden_states + 4


class DSparkTargetHiddenProjectionTest(CustomTestCase):
    def test_nonfinal_prefill_stage_only_forwards_target_proxies(self) -> None:
        proxy = object()
        target = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=object(), spec_algorithm=None),
            device="cpu",
            forward_batch_generation=lambda batch, *, pp_proxy_tensors, capture_hidden_mode: (
                SimpleNamespace(pp_hidden_states_proxy_tensors=pp_proxy_tensors)
            ),
        )
        with (
            get_parallel().override(pp_group=SimpleNamespace(is_last_rank=False)),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_schedule",
                return_value=SimpleNamespace(page_size=1),
            ),
        ):
            worker = DSparkWorkerV2(None, 0, 0, target)
        worker.alloc_memory_pool()
        worker.init_attention_backends()
        worker.init_cuda_graphs()
        batch = SimpleNamespace(seq_lens=torch.tensor([8]))
        result = worker.forward_batch_generation(batch, pp_proxy_tensors=proxy)
        self.assertIs(result.pp_hidden_states_proxy_tensors, proxy)
        self.assertIs(result.new_seq_lens, batch.seq_lens)
        self.assertIsNone(worker.get_confidence_budget_prepare())
        self.assertIsNone(worker.primary_draft_kv_pool)
        self.assertEqual(worker.preloaded_weights_bytes, 0)
        self.assertEqual(
            worker.spec_v2_attn_backends, (target.model_runner.attn_backend,)
        )

    def test_single_aux_hidden_state_is_returned_without_copy(self) -> None:
        hidden_states = torch.empty(2, 3)

        self.assertIs(hidden_states, pack_aux_hidden_states([hidden_states]))

    def test_preprojected_hidden_is_not_projected_again(self) -> None:
        attention = _Attention()
        draft_model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention)],
            project_target_hidden=mock.Mock(
                side_effect=AssertionError("projection must not run twice")
            ),
            _fused_kv_write_bundle=lambda _pool: None,
            _stacked_ctx_kv_params=lambda: None,
        )
        draft_model.write_target_hidden_kv = MethodType(
            DSparkDraftMixin.write_target_hidden_kv, draft_model
        )
        pool = SimpleNamespace(set_kv_buffer=mock.Mock())
        injector = TargetHiddenKvInjector(
            draft_model=draft_model,
            draft_model_runner=SimpleNamespace(token_to_kv_pool=pool),
            model_runner=SimpleNamespace(device=torch.device("cpu")),
            device=torch.device("cpu"),
            verify_num_draft_tokens=2,
            block_pos_offsets=torch.arange(2),
        )
        projected_hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        injector.inject_target_hidden(
            target_hidden=projected_hidden,
            cache_loc=torch.arange(2),
            positions=torch.arange(2),
            target_hidden_is_projected=True,
        )

        draft_model.project_target_hidden.assert_not_called()
        self.assertIs(attention.input, projected_hidden)

    def test_cache_only_replay_seeds_decode_local_draft_state(self) -> None:
        hidden_states = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        logits_output = SimpleNamespace(
            hidden_states=hidden_states,
            hidden_states_token_indices=None,
        )
        batch_output = SimpleNamespace(
            logits_output=logits_output,
            next_token_ids=torch.tensor([42]),
            next_draft_input=None,
            new_seq_lens=None,
        )
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(),
                prefill_attention_backend_str="dsv4",
            ),
            forward_batch_generation=mock.Mock(return_value=batch_output),
        )
        worker = object.__new__(DSparkWorkerV2)
        worker._target_worker = target_worker
        worker._target_hidden_projection_enabled = False
        worker._tp_sync = SimpleNamespace(sync=mock.Mock())
        worker._kv_injector = SimpleNamespace(inject_target_hidden=mock.Mock())
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            dsv41_cache_only_replay=True,
            seq_lens=torch.tensor([256]),
            extend_lens=[128],
            prefix_lens=[128],
            out_cache_loc=torch.arange(4),
            req_pool_indices=torch.tensor([7]),
        )

        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.compute_position",
                return_value=(torch.arange(4), None),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.is_unified_kv_triton",
                return_value=False,
            ),
        ):
            result = DSparkWorkerV2._forward_prefill(worker, batch, on_publish=None)

        worker._kv_injector.inject_target_hidden.assert_called_once()
        call = worker._kv_injector.inject_target_hidden.call_args.kwargs
        self.assertIs(call["target_hidden"], hidden_states)
        self.assertIs(call["cache_loc"], batch.out_cache_loc)
        self.assertTrue(torch.equal(call["positions"], torch.arange(4)))
        self.assertIsNone(call["state_slot"])
        self.assertIsNone(call["final_pos"])
        self.assertFalse(call["target_hidden_is_projected"])
        self.assertEqual(result.next_draft_input.bonus_tokens.tolist(), [42])
        self.assertEqual(result.next_draft_input.new_seq_lens.tolist(), [256])
        self.assertIsNone(logits_output.hidden_states)


if __name__ == "__main__":
    unittest.main()

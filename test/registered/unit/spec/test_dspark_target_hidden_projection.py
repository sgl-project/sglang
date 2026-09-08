import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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


if __name__ == "__main__":
    unittest.main()

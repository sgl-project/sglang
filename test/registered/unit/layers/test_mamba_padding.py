import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.attention.mamba.mamba as mamba_module
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMambaCudaGraphPadding(unittest.TestCase):
    def test_decode_zeroes_skipped_convolution_rows_before_ssm(self):
        num_tokens = 4
        state_indices = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
        projected_states = torch.zeros(num_tokens, 8)
        ssm_inputs = {}

        def fake_causal_conv1d_update(hidden_states, *args, **kwargs):
            return torch.full_like(hidden_states, 7.0)

        def fake_selective_state_update(*args, **kwargs):
            ssm_inputs["hidden_states"] = args[1].detach().clone()
            ssm_inputs["B"] = args[4].detach().clone()
            ssm_inputs["C"] = args[5].detach().clone()
            kwargs["out"].zero_()

        mixer = SimpleNamespace(
            in_proj=lambda hidden_states: (projected_states.clone(), None),
            intermediate_size=2,
            conv_dim=4,
            num_heads=2,
            tp_size=1,
            conv1d=SimpleNamespace(
                weight=torch.ones(4, 1, 1),
                bias=torch.zeros(4),
            ),
            activation="silu",
            groups_ssm_state_size=1,
            head_dim=1,
            n_groups=1,
            ssm_state_size=1,
            A=torch.ones(2),
            dt_bias=torch.ones(2),
            D=torch.ones(2),
            norm=lambda hidden_states, gate: hidden_states,
            out_proj=lambda hidden_states: (hidden_states, None),
        )
        layer_cache = SimpleNamespace(
            conv=(torch.zeros(2, 4, 1),),
            temporal=torch.zeros(2, 2, 1, 1),
        )
        metadata = SimpleNamespace(
            mamba_cache_indices=state_indices,
            query_start_loc=torch.tensor([0], dtype=torch.int32),
            num_prefills=0,
            num_decodes=num_tokens,
            draft_token_num=1,
            is_target_verify=False,
            num_prefill_tokens=0,
        )

        with (
            patch.object(
                mamba_module,
                "causal_conv1d_update_triton",
                side_effect=fake_causal_conv1d_update,
                create=True,
            ),
            patch.object(
                mamba_module,
                "selective_state_update",
                side_effect=fake_selective_state_update,
            ),
        ):
            MambaMixer2.forward(
                mixer,
                hidden_states=torch.zeros(num_tokens, 2),
                layer_cache=layer_cache,
                metadata=metadata,
                forward_batch=SimpleNamespace(mamba_track_mask=None),
                use_triton_causal_conv=True,
            )

        combined_ssm_input = torch.cat(
            [
                ssm_inputs["hidden_states"].flatten(1),
                ssm_inputs["B"].flatten(1),
                ssm_inputs["C"].flatten(1),
            ],
            dim=-1,
        )
        torch.testing.assert_close(
            combined_ssm_input[:2],
            torch.full((2, 4), 7.0),
        )
        self.assertTrue(
            torch.equal(
                combined_ssm_input[2:],
                torch.zeros_like(combined_ssm_input[2:]),
            )
        )


if __name__ == "__main__":
    unittest.main()

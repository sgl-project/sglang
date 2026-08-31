import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

try:
    from sglang.srt.hardware_backend.npu.attention.ascend_kda_backend import (
        AscendKDAAttnBackend,
    )
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("sgl_kernel_npu"):
        raise
    AscendKDAAttnBackend = None
from sglang.srt.layers.attention.linear.kda_cp import (
    KDAFLACPContext,
    compose_kda_cp_affine_states,
    kda_use_fla_prefill_cp,
    prepare_kda_cp_conv_states,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ForwardMode:
    def is_context_parallel_extend(self):
        return True

    def is_mixed(self):
        return False

    def is_target_verify(self):
        return False

    def is_draft_extend_v2(self):
        return False


class _FakeFixedShapeGatherGroup:
    def __init__(self, all_rank_tensors, rank):
        self.all_rank_tensors = all_rank_tensors
        self.rank = rank

    def all_gather_into_tensor(self, output, input_tensor):
        torch.testing.assert_close(input_tensor, self.all_rank_tensors[self.rank])
        torch.cat(self.all_rank_tensors, dim=0, out=output)


def _context(group, rank, local_lens=(1, 1), split_list=(1, 1, 1, 1)):
    return KDAFLACPContext(
        group=group,
        cp_size=2,
        cp_rank=rank,
        batch_size=1,
        split_list=split_list,
        local_segment_lens=local_lens,
        local_cu_seqlens=torch.tensor(
            [0, local_lens[0], sum(local_lens)], dtype=torch.int32
        ),
        local_segment_slots=((0, 6), (2, 4))[rank],
        rank_segment_slots=((0, 6), (2, 4)),
        fixed_segment_sources=(0, -1, 0, -1, 1, -1, 1, -1),
        max_rank_segments=2,
        fixed_segment_lens=tuple(
            value for length in split_list for value in (length, 0)
        ),
        track_after_slots=(-1,),
        track_state_indices=torch.tensor([-1]),
    )


class TestKDAPrefillCP(unittest.TestCase):
    def test_fla_affine_composes_natural_zigzag_order(self):
        # Natural transforms are 2x+1, 3x+2, 4x+3, 5x+4.
        all_rank_affine = [
            torch.tensor([[[[1.0, 2.0]]], [[[4.0, 5.0]]]]),
            torch.tensor([[[[2.0, 3.0]]], [[[3.0, 4.0]]]]),
        ]
        expected_initial = [
            torch.tensor([[[[7.0]]], [[[191.0]]]]),
            torch.tensor([[[[15.0]]], [[[47.0]]]]),
        ]
        for rank in range(2):
            state_pool = torch.tensor([[[[7.0]]]])
            context = _context(_FakeFixedShapeGatherGroup(all_rank_affine, rank), rank)
            local_initial = compose_kda_cp_affine_states(
                all_rank_affine[rank],
                state_pool,
                torch.tensor([0], dtype=torch.int32),
                context,
            )
            torch.testing.assert_close(local_initial, expected_initial[rank])
            torch.testing.assert_close(state_pool, torch.tensor([[[[959.0]]]]))

    def test_fla_affine_supports_value_major_npu_state_pool(self):
        all_rank_affine = [
            torch.tensor([[[[1.0, 2.0]]], [[[4.0, 5.0]]]]),
            torch.tensor([[[[2.0, 3.0]]], [[[3.0, 4.0]]]]),
        ]
        state_pool = torch.tensor([[[[7.0]]]])
        context = _context(_FakeFixedShapeGatherGroup(all_rank_affine, 0), 0)
        compose_kda_cp_affine_states(
            all_rank_affine[0],
            state_pool,
            torch.tensor([0], dtype=torch.int32),
            context,
            state_value_major=True,
        )
        torch.testing.assert_close(state_pool, torch.tensor([[[[959.0]]]]))

    def test_fla_conv_uses_only_segment_tails(self):
        local_inputs = [
            torch.tensor([[1.0], [2.0], [7.0], [8.0]]),
            torch.tensor([[3.0], [4.0], [5.0], [6.0]]),
        ]
        all_rank_tails = [
            torch.tensor([[[0.0], [1.0], [2.0]], [[0.0], [7.0], [8.0]]]),
            torch.tensor([[[0.0], [0.0], [3.0]], [[4.0], [5.0], [6.0]]]),
        ]
        local_lens = [(2, 2), (1, 3)]
        expected = [
            torch.tensor([[[-2.0, -1.0, 0.0]], [[4.0, 5.0, 6.0]]]),
            torch.tensor([[[0.0, 1.0, 2.0]], [[1.0, 2.0, 3.0]]]),
        ]
        for rank in range(2):
            state_pool_seed = torch.tensor([[[-2.0, -1.0, 0.0]]])
            context = _context(
                _FakeFixedShapeGatherGroup(all_rank_tails, rank),
                rank,
                local_lens[rank],
                (2, 1, 3, 2),
            )
            state_pool = state_pool_seed.clone()
            local_initial = prepare_kda_cp_conv_states(
                local_inputs[rank],
                state_pool,
                torch.tensor([0], dtype=torch.int32),
                context,
            ).clone()
            torch.testing.assert_close(local_initial, expected[rank])
            torch.testing.assert_close(state_pool, torch.tensor([[[6.0, 7.0, 8.0]]]))

    def test_fla_cp_is_npu_prefill_only(self):
        forward_batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(split_list=[1, 1, 1, 1]),
            forward_mode=_ForwardMode(),
        )
        with (
            patch(
                "sglang.srt.layers.attention.linear.kda_cp.get_parallel",
                return_value=SimpleNamespace(
                    enable_prefill_context_parallel=True,
                    attn_cp_size=2,
                ),
            ),
            patch(
                "sglang.srt.layers.attention.linear.kda_cp.is_npu",
                return_value=True,
            ),
        ):
            self.assertTrue(kda_use_fla_prefill_cp(forward_batch))

    @unittest.skipIf(
        AscendKDAAttnBackend is None,
        "requires the matching sgl-kernel-npu KDA affine-state PR",
    )
    def test_ascend_extend_uses_local_cp_conv_boundaries(self):
        context = _context(None, rank=0)
        conv_states = torch.zeros(1, 3, 2)
        cache = SimpleNamespace(conv=[conv_states], temporal=torch.zeros(1, 1, 1, 1))

        backend = object.__new__(AscendKDAAttnBackend)
        backend.forward_metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([0], dtype=torch.int32),
            has_mamba_track_mask=False,
        )
        backend.req_to_token_pool = SimpleNamespace(
            mamba2_layer_cache=lambda _layer_id: cache
        )
        backend.kernel_dispatcher = SimpleNamespace(
            extend=Mock(return_value=torch.zeros(1, 2, 1, 1))
        )
        backend._causal_conv1d_extend = Mock(
            side_effect=lambda x, *_args, **_kwargs: x.transpose(0, 1)
        )
        backend._prepare_extend_gate_inputs = Mock(
            side_effect=lambda _layer, a, b: (a, b, None, None)
        )

        layer = SimpleNamespace(
            layer_id=0,
            q_dim=1,
            k_dim=1,
            v_dim=1,
            head_q_dim=1,
            head_k_dim=1,
            head_v_dim=1,
            conv_weights=torch.zeros(3, 2),
            bias=None,
            A_log=None,
            dt_bias=None,
            lower_bound=None,
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_target_verify=lambda: False,
                is_draft_extend_v2=lambda: False,
            ),
            extend_prefix_lens=torch.tensor([0]),
            extend_seq_lens_cpu=[4],
        )
        local_conv_states = torch.zeros(2, 3, 2)

        with (
            patch(
                "sglang.srt.hardware_backend.npu.attention.ascend_kda_backend."
                "kda_use_fla_prefill_cp",
                return_value=True,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.attention.ascend_kda_backend."
                "build_kda_fla_cp_context",
                return_value=context,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.attention.ascend_kda_backend."
                "prepare_kda_cp_conv_states",
                return_value=local_conv_states,
            ),
        ):
            backend.forward_extend(
                layer,
                forward_batch,
                mixed_qkv=torch.zeros(2, 3),
                a=torch.zeros(1, 2, 1),
                b=torch.zeros(1, 2, 1),
            )

        self.assertEqual(backend._causal_conv1d_extend.call_count, 3)
        for call in backend._causal_conv1d_extend.call_args_list:
            torch.testing.assert_close(
                call.kwargs["query_start_loc"], context.local_cu_seqlens
            )
            self.assertEqual(call.kwargs["seq_lens_cpu"], [1, 1])

        kernel_call = backend.kernel_dispatcher.extend.call_args
        torch.testing.assert_close(
            kernel_call.kwargs["query_start_loc"], context.local_cu_seqlens
        )
        self.assertIs(kernel_call.kwargs["cp_context"], context)


if __name__ == "__main__":
    unittest.main()

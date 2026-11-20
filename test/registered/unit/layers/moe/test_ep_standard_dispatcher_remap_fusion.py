"""CPU unit tests for EP StandardDispatcher local expert remap fusion and guards."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe import topk as topk_mod
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    StandardTopKOutputLocalMapped,
    TopKConfig,
    _build_ep_remap_kwargs,
    _can_fuse_ep_local_expert_remap,
)
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.runtime_context import get_context, get_exec, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEpStandardDispatcherRemapFusion(CustomTestCase):
    def setUp(self):
        self.enterContext(
            patch(
                "sglang.srt.layers.moe.token_dispatcher.standard.get_device",
                return_value="cpu",
            )
        )
        self.enterContext(
            patch(
                "sglang.srt.layers.moe.token_dispatcher.standard."
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            )
        )
        self.enterContext(
            patch(
                "sglang.srt.layers.moe.utils."
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            )
        )
        self.enterContext(
            patch(
                "sglang.srt.layers.moe.topk."
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            )
        )
        self.enterContext(
            get_context().override_server_args(model_path="dummy", device="cpu")
        )
        self.enterContext(
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.TRITON,
                a2a_backend=MoeA2ABackend.NONE,
            )
        )
        self.enterContext(
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                tp_group=None,
                attn_dp_size=1,
                attn_dp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_ep_size=2,
                moe_ep_rank=1,
                moe_tp_size=1,
                moe_tp_rank=0,
                dwdp_size=1,
            )
        )
        self.enterContext(
            patch.dict(
                os.environ,
                {
                    "SGLANG_DISABLE_FUSED_EP_REMAP": "0",
                    "SGLANG_EP_INTERLEAVED_EXPERTS": "0",
                },
            )
        )

    def test_eligibility_guards_and_remap_kwargs(self):
        base_cfg = TopKConfig(
            top_k=8, runner_backend=MoeRunnerBackend.TRITON, layer_id=0
        )
        with (
            patch.object(topk_mod, "_is_cuda", True),
            patch.object(topk_mod, "_use_aiter", False),
            patch.object(topk_mod, "is_expert_remapping_enabled", return_value=False),
            patch.object(topk_mod, "get_global_experts_capturer", return_value=None),
        ):
            self.assertTrue(_can_fuse_ep_local_expert_remap(base_cfg, 128, None, False))
            self.assertEqual(
                _build_ep_remap_kwargs(128), {"ep_start": 64, "ep_end": 128}
            )

            with patch.dict(os.environ, {"SGLANG_EP_INTERLEAVED_EXPERTS": "1"}):
                self.assertEqual(
                    _build_ep_remap_kwargs(128),
                    {"ep_interleaved_size": 2, "ep_rank": 1},
                )

            with patch.dict(os.environ, {"SGLANG_DISABLE_FUSED_EP_REMAP": "1"}):
                self.assertFalse(
                    _can_fuse_ep_local_expert_remap(base_cfg, 128, None, False)
                )

            # Non-divisible routed expert count must not fuse.
            self.assertFalse(
                _can_fuse_ep_local_expert_remap(base_cfg, 129, None, False)
            )

            # DWDP collapses EP after weight binding.
            with get_parallel().override(dwdp_size=2):
                self.assertFalse(
                    _can_fuse_ep_local_expert_remap(base_cfg, 128, None, False)
                )

            # Waterfill shared experts require standard post-processing.
            wf_cfg = TopKConfig(
                top_k=8,
                runner_backend=MoeRunnerBackend.TRITON,
                shared_experts=SharedExpertsConfig(use_waterfill=True),
                layer_id=0,
            )
            self.assertFalse(_can_fuse_ep_local_expert_remap(wf_cfg, 128, None, False))

            # EP join-mode="scale" scales local outputs differently.
            with get_exec().moe.override(ep_join_mode="scale"):
                self.assertFalse(
                    _can_fuse_ep_local_expert_remap(base_cfg, 128, None, False)
                )

            # Backends keeping global expert IDs must not remap in-router.
            for global_backend in (
                MoeRunnerBackend.FLASHINFER_TRTLLM,
                MoeRunnerBackend.FLASHINFER_CUTLASS,
            ):
                bad_cfg = TopKConfig(top_k=8, runner_backend=global_backend, layer_id=0)
                self.assertFalse(
                    _can_fuse_ep_local_expert_remap(bad_cfg, 128, None, False)
                )

    def test_rank_ge_1_no_double_remap_and_custom_mapping_override(self):
        cfg = MoeRunnerConfig(
            num_experts=128,
            num_local_experts=64,
            top_k=8,
            num_fused_shared_experts=0,
            runner_backend=MoeRunnerBackend.TRITON,
        )
        dispatcher = StandardDispatcher(cfg)

        # Warm lazy default mapping on an un-fused StandardTopKOutput (e.g. M=0).
        dispatcher.dispatch(
            torch.empty((0, 16)),
            StandardTopKOutput(
                torch.empty((0, 8), dtype=torch.float32),
                torch.full((0, 8), -1, dtype=torch.int32),
                torch.empty((0, 128)),
            ),
        )
        self.assertIsNotNone(dispatcher.local_expert_mapping)
        self.assertTrue(dispatcher._owns_default_local_mapping)

        # Rank 1 local IDs [0..63] must not be double-remapped through [64..127]->[0..63].
        local_ids = torch.tensor(
            [[0, 1, -1, -1, 63, -1, -1, 2], [5, -1, 10, 20, -1, 30, -1, 63]],
            dtype=torch.int32,
        )
        out = dispatcher.dispatch(
            torch.ones((2, 16)),
            StandardTopKOutputLocalMapped(
                torch.ones((2, 8), dtype=torch.float32) / 8.0,
                local_ids.clone(),
                torch.ones((2, 128)),
            ),
        )
        self.assertTrue(torch.equal(out.topk_output.topk_ids, local_ids))

        # External custom local_expert_mapping must NOT be bypassed.
        dispatcher_custom = StandardDispatcher(cfg)
        custom_map = torch.full((128,), -1, dtype=torch.int32)
        custom_map[:64] = torch.arange(63, -1, -1, dtype=torch.int32)
        dispatcher_custom.local_expert_mapping = custom_map
        self.assertFalse(dispatcher_custom._owns_default_local_mapping)
        out_custom = dispatcher_custom.dispatch(
            torch.ones((2, 16)),
            StandardTopKOutputLocalMapped(
                torch.ones((2, 8), dtype=torch.float32) / 8.0,
                local_ids.clone(),
                torch.ones((2, 128)),
            ),
        )
        expected = torch.where(local_ids < 0, -1, custom_map[local_ids.clamp_min(0)])
        self.assertTrue(torch.equal(out_custom.topk_output.topk_ids, expected))

    def test_downstream_local_expert_mapping_and_dwdp_reset(self):
        cfg = MoeRunnerConfig(
            num_experts=129,
            num_local_experts=65,
            top_k=9,
            num_fused_shared_experts=1,
            runner_backend=MoeRunnerBackend.MARLIN,
        )
        dispatcher = StandardDispatcher(cfg)
        local_ids = torch.tensor([[0, 1, -1, -1, 63, -1, -1, 2, 64]], dtype=torch.int32)
        out = dispatcher.dispatch(
            torch.ones((1, 16)),
            StandardTopKOutputLocalMapped(
                torch.ones((1, 9), dtype=torch.float32) / 8.0,
                local_ids.clone(),
                torch.ones((1, 129)),
            ),
        )
        self.assertTrue(torch.equal(out.topk_output.topk_ids, local_ids))
        self.assertIsNotNone(dispatcher.local_expert_mapping)
        self.assertTrue(
            torch.equal(
                dispatcher.local_expert_mapping[:64],
                torch.full((64,), -1, dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                dispatcher.local_expert_mapping[64:128],
                torch.arange(64, dtype=torch.int32),
            )
        )
        self.assertEqual(int(dispatcher.local_expert_mapping[128].item()), 64)

        fake_layer = SimpleNamespace(
            moe_ep_size=2,
            moe_ep_rank=1,
            _num_global_routed=128,
            _num_local_routed=64,
            num_local_experts=65,
            num_experts=129,
            moe_runner_config=cfg,
            dispatcher=dispatcher,
            replace_expert_tensor=lambda *a, **kw: None,
            _dwdp_bound=False,
        )
        FusedMoE.bind_full_expert_weights(fake_layer, {})
        self.assertIsNone(dispatcher.local_expert_mapping)
        self.assertFalse(dispatcher._owns_default_local_mapping)

    def test_interleaved_ep_weight_loader_mapping(self):
        with patch.dict(os.environ, {"SGLANG_EP_INTERLEAVED_EXPERTS": "1"}):
            fake_moe = SimpleNamespace(
                moe_ep_size=2,
                _expert_storage_rank=1,
                _num_global_routed=8,
                _num_local_routed=4,
                num_experts=9,
                num_local_experts=5,
                _has_fused_shared=True,
                scheme=None,
                quant_method=SimpleNamespace(apply=None),
                quant_config=None,
                use_flashinfer_trtllm_moe=False,
            )
            self.assertEqual(
                FusedMoE._map_global_expert_id_to_local_expert_id(fake_moe, 0), -1
            )
            self.assertEqual(
                FusedMoE._map_global_expert_id_to_local_expert_id(fake_moe, 1), 0
            )
            self.assertEqual(
                FusedMoE._map_global_expert_id_to_local_expert_id(fake_moe, 7), 3
            )
            self.assertEqual(
                FusedMoE._map_global_expert_id_to_local_expert_id(fake_moe, 8), 4
            )
            self.assertEqual(
                FusedMoE._local_routed_expert_ids(fake_moe, list(range(9))),
                [1, 3, 5, 7],
            )
            full_w = (
                torch.arange(9, dtype=torch.float32)[:, None, None]
                .expand(9, 4, 4)
                .clone()
            )
            param = torch.nn.Parameter(torch.zeros((5, 4, 4)), requires_grad=False)
            fake_moe._load_model_weight_or_group_weight_scale = (
                lambda shard_id, shard_dim, loaded_weight, expert_data, is_bias: (
                    expert_data.copy_(loaded_weight)
                )
            )
            FusedMoE.weight_loader_fused(fake_moe, param, full_w, "w13_weight", "w13")
            self.assertTrue(
                torch.equal(param[:, 0, 0], torch.tensor([1.0, 3.0, 5.0, 7.0, 8.0]))
            )


if __name__ == "__main__":
    unittest.main()

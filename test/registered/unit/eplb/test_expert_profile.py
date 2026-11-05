"""Expert profile validation and sparse logical-to-physical routing on CPU."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
)
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
    transform_select_experts_inputs,
)
from sglang.srt.eplb.expert_profile import (
    get_num_physical_experts,
    load_expert_profile,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestExpertProfile(CustomTestCase):
    def test_unprofiled_capacity_preserves_redundant_experts(self):
        with get_context().override_server_args(
            moe_expert_profile=None, ep_num_redundant_experts=2
        ):
            self.assertEqual(get_num_physical_experts(8), 10)

    def test_profile_path_reads_rewritten_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text('{"num_experts": 8, "keep": [[7, 1, 5]]}')
            first = load_expert_profile(str(path))
            path.write_text('{"num_experts": 8, "keep": [[7, 1]]}')
            second = load_expert_profile(str(path))
            self.assertEqual(first.num_kept, 3)
            self.assertEqual(second.num_kept, 2)

    def test_server_args_accepts_profile_and_resolver_rejects_unsupported_modes(self):
        from sglang.srt.arg_groups import parallel_hook

        source = '{"num_experts": 8, "keep": [[7, 1, 5]]}'
        args = ServerArgs(model_path="dummy", moe_expert_profile=source)
        self.assertEqual(args.moe_expert_profile, source)
        config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen3MoeForCausalLM"]),
            hf_text_config=SimpleNamespace(
                num_hidden_layers=1, num_experts=8, num_experts_per_tok=2
            ),
            quantization=None,
            dtype=torch.bfloat16,
        )
        options = dict(
            moe_expert_profile=source,
            moe_runner_backend="triton",
            moe_a2a_backend="none",
            ep_size=1,
            enable_eplb=False,
            ep_num_redundant_experts=0,
            init_expert_location="trivial",
            ep_dispatch_algorithm=None,
            ep_join_mode=None,
            enable_paged_experts=False,
            enable_lora=False,
            speculative_algorithm=None,
            expert_distribution_recorder_mode=None,
            expert_balancedness_report_mode="off",
            enforce_shared_experts_fusion=False,
        )
        cfg = SimpleNamespace(**options)
        with (
            patch.object(parallel_hook, "resolving_view", return_value=cfg),
            patch.object(parallel_hook, "resolved_view", return_value=cfg),
            patch.object(parallel_hook, "model_config_of", return_value=config),
            patch.object(parallel_hook, "declare_resolution") as declare,
        ):
            parallel_hook.handle_eplb_and_dispatch(args)
            self.assertEqual(
                declare.call_args.kwargs["ep_dispatch_algorithm"], "dynamic"
            )
            self.assertEqual(
                load_expert_profile(
                    declare.call_args.kwargs["moe_expert_profile"]
                ).num_kept,
                3,
            )
            for key, value in [
                ("moe_runner_backend", "flashinfer_trtllm"),
                ("ep_size", 4),
                ("enable_eplb", True),
                ("enable_paged_experts", True),
                ("speculative_algorithm", "EAGLE"),
            ]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    setattr(cfg, key, value)
                    try:
                        parallel_hook.handle_eplb_and_dispatch(args)
                    finally:
                        setattr(cfg, key, options[key])

    def test_profile_round_trip_preserves_physical_slot_order(self):
        profile = load_expert_profile(
            '{"num_experts": 8, "keep": [[7, 1, 5], [2, 0, 6]]}'
        )
        profile.validate_model(num_layers=2, num_experts=8, top_k=2)
        self.assertEqual(profile.num_kept, 3)
        self.assertEqual(load_expert_profile(profile.to_json()), profile)
        self.assertEqual(profile.keep[0], (7, 1, 5))

    def test_rejects_invalid_profiles(self):
        invalid = [
            {"num_experts": True, "keep": [[0]]},
            {"num_experts": 0, "keep": [[0]]},
            {"num_experts": 4, "keep": []},
            {"num_experts": 4, "keep": [[]]},
            {"num_experts": 4, "keep": [[True]]},
            {"num_experts": 4, "keep": [[-1]]},
            {"num_experts": 4, "keep": [[4]]},
            {"num_experts": 4, "keep": [[1, 1]]},
            {"num_experts": 4, "keep": [[1], [1, 2]]},
            {"num_experts": 4, "keep": [[1]], "kepp": [[1]]},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_expert_profile(json.dumps(value))

    def test_rejects_model_mismatch_and_insufficient_experts(self):
        profile = load_expert_profile('{"num_experts": 8, "keep": [[7, 1, 5]]}')
        for dimensions in [(2, 8, 2), (1, 4, 2), (1, 8, 4)]:
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                profile.validate_model(*dimensions)

    def test_sparse_routing_masks_missing_experts_and_maps_compact_slots(self):
        source = '{"num_experts": 8, "keep": [[7, 1, 5]]}'
        profile = load_expert_profile(source)
        model_shape = ModelConfigForExpertLocation(num_layers=1, num_logical_experts=8)
        with (
            get_context().override_server_args(
                device="cpu",
                ep_size=1,
                tp_size=1,
                moe_expert_profile=source,
                moe_a2a_backend="none",
                ep_dispatch_algorithm="dynamic",
            ),
            patch.object(
                ModelConfigForExpertLocation,
                "from_model_config",
                return_value=model_shape,
            ),
        ):
            metadata = ExpertLocationMetadata.init_by_mapping(
                None, profile.keep, moe_ep_rank=0
            )
            self.assertEqual(metadata.num_logical_experts, 8)
            self.assertEqual(metadata.num_physical_experts, 3)
            self.assertEqual(metadata.logical_to_all_physical(0, 0), [])
            self.assertEqual(metadata.logical_to_all_physical(0, 7), [0])
            info = ExpertLocationDispatchInfo(
                ep_dispatch_algorithm="dynamic",
                partial_logical_to_rank_dispatch_physical_map=None,
                partial_logical_to_all_physical_map=metadata.logical_to_all_physical_map[
                    0
                ],
                partial_logical_to_all_physical_map_num_valid=metadata.logical_to_all_physical_map_num_valid[
                    0
                ],
                num_physical_experts=3,
                rank_invariant=True,
            )
            logits = torch.tensor([[100.0, 2.0, 99.0, 98.0, 97.0, 3.0, 96.0, 4.0]])
            masked, _ = transform_select_experts_inputs(logits, None, info)
            logical_ids = masked.topk(2, dim=-1).indices
            self.assertEqual(logical_ids.tolist(), [[7, 5]])
            physical_ids = topk_ids_logical_to_physical(logical_ids, info)
            self.assertEqual(physical_ids.tolist(), [[0, 2]])
            self.assertEqual(logits[0, 0].item(), 100.0)
            compact_scores = (
                logits[:, list(profile.keep[0])].softmax(dim=-1).topk(2).values
            )
            self.assertTrue(
                torch.allclose(masked.softmax(dim=-1).topk(2).values, compact_scores)
            )


if __name__ == "__main__":
    unittest.main()

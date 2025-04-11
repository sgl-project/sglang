import tempfile
import unittest
from pathlib import Path
from typing import List

import sglang as sgl
from sglang.srt.managers.expert_distribution_storage import ExpertDistributionStorage
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_NUM_ROUTED_EXPERTS = 64  # DeepSeek-Coder-V2-Lite-Instruct
_EP_NUM_REDUNDANT_EXPERTS = 4
_NUM_OVERALL_PHYSICAL_EXPERTS = _NUM_ROUTED_EXPERTS + _EP_NUM_REDUNDANT_EXPERTS
_TRIVIAL_EXPERT_LOCATIONS = list(
    x % _NUM_ROUTED_EXPERTS for x in range(_NUM_OVERALL_PHYSICAL_EXPERTS)
)


class TestEPLB(CustomTestCase):
    # def test_eplb_e2e(self):
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         engine_kwargs = dict(
    #             model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    #             trust_remote_code=True,
    #             enable_eplb=True,
    #             eplb_storage_dir=tmpdir,
    #             ep_num_redundant_experts=_EP_NUM_REDUNDANT_EXPERTS,
    #             enable_deepep_moe=True,
    #             deepep_mode="normal",
    #             disable_cuda_graph=True,
    #             enable_scheduler_input_blocker=True,
    #             disable_overlap_schedule=True,  # TODO
    #             tp_size=2,
    #             log_level="info",
    #         )
    #
    #         print(f"Action: start engine")
    #         engine = sgl.Engine(**engine_kwargs)
    #         ref_output = self._engine_generate(engine)
    #         self._assert_behavior(engine, ref_output, "equal_trivial")
    #
    #         print(f"Action: eplb_rebalance")
    #         engine.eplb_rebalance()
    #         physical_to_logical_map_layer_0_after_first_rebalance = (
    #             self._assert_behavior(engine, ref_output, "not_equal_trivial")
    #         )
    #
    #         print(f"Action: shutdown engine")
    #         engine.shutdown()
    #         del engine
    #
    #         print(f"Action: start engine")
    #         engine = sgl.Engine(**engine_kwargs)
    #         self._assert_behavior(
    #             engine,
    #             ref_output,
    #             physical_to_logical_map_layer_0_after_first_rebalance,
    #         )
    #
    #         print(f"Action: shutdown engine")
    #         engine.shutdown()
    #         del engine

    def test_eplb_init_expert_location_and_save_expert_distribution(self):
        with tempfile.TemporaryDirectory() as eplb_storage_dir_a, tempfile.TemporaryDirectory() as eplb_storage_dir_b:
            engine_kwargs = dict(
                model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
                trust_remote_code=True,
                enable_eplb=True,
                ep_num_redundant_experts=_EP_NUM_REDUNDANT_EXPERTS,
                enable_deepep_moe=True,
                deepep_mode="normal",
                disable_cuda_graph=True,
                tp_size=2,
                log_level="info",
            )

            print(f"Action: start engine")
            engine = sgl.Engine(**engine_kwargs, eplb_storage_dir=eplb_storage_dir_a)
            ref_output = self._engine_generate(engine)
            self._assert_behavior(engine, ref_output, "equal_trivial")

            print(f"Action: eplb_save_expert_distribution")
            engine.eplb_save_expert_distribution()
            snapshot_path = ExpertDistributionStorage.get_last_snapshot_path(
                Path(eplb_storage_dir_a) / "expert_distribution_storage"
            )
            assert snapshot_path is not None
            print(f"{snapshot_path=} {snapshot_path.read_text()=}")

            print(f"Action: shutdown engine")
            engine.shutdown()
            del engine

            print(f"Action: start engine with init_expert_location")
            engine = sgl.Engine(
                **engine_kwargs,
                eplb_storage_dir=eplb_storage_dir_b,
                init_expert_location=str(snapshot_path),
            )
            self._assert_behavior(engine, ref_output, "not_equal_trivial")
            print(f"Action: shutdown engine")
            engine.shutdown()
            del engine

            print(
                f"Action: start engine to check automatically loading from storage dir"
            )
            engine = sgl.Engine(**engine_kwargs, eplb_storage_dir=eplb_storage_dir_a)
            self._assert_behavior(engine, ref_output, "not_equal_trivial")
            print(f"Action: shutdown engine")
            engine.shutdown()
            del engine

    def _assert_behavior(
            self, engine: sgl.Engine, ref_output: List[str], expect_physical_to_local_map
    ):
        ret = engine.flush_cache()
        assert ret.success

        actual_output = self._engine_generate(engine)
        self.assertEqual(actual_output, ref_output)

        physical_to_logical_map = (
            engine.tokenizer_manager.expert_location_metadata.physical_to_logical_map
        )
        physical_to_logical_map_layer_0 = physical_to_logical_map[0, :].tolist()
        print(f"{physical_to_logical_map_layer_0=}")

        if expect_physical_to_local_map == "equal_trivial":
            self.assertEqual(physical_to_logical_map_layer_0, _TRIVIAL_EXPERT_LOCATIONS)
        elif expect_physical_to_local_map == "not_equal_trivial":
            self.assertNotEqual(
                physical_to_logical_map_layer_0, _TRIVIAL_EXPERT_LOCATIONS
            )
        else:
            self.assertEqual(
                physical_to_logical_map_layer_0, expect_physical_to_local_map
            )

        return physical_to_logical_map_layer_0

    def _engine_generate(self, engine: sgl.Engine):
        output = engine.generate(
            prompt=["1+1=2, 2+2=4", "One plus one is two, two plus two is four"],
            sampling_params=dict(max_new_tokens=8, temperature=0.0),
        )
        print(f"engine_generate {output=}")
        return [x["text"] for x in output]


if __name__ == "__main__":
    unittest.main()

import asyncio
import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np
import sglang as sgl
import torch
from python.sglang.srt.managers.expert_location import compute_logical_to_rank_dispatch_physical_map
from sglang.srt.managers.expert_distribution_storage import ExpertDistributionStorage
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# DeepSeek-Coder-V2-Lite-Instruct
_NUM_ROUTED_EXPERTS = 64
_NUM_HIDDEN_LAYERS = 27
_REF_OUTPUT = [", 4+4=8,", ", four plus four is eight, eight"]


class TestEPLB(CustomTestCase):
    def test_eplb_many_rebalances(self):
        print("Action: test_eplb_many_rebalances")

        num_rebalance = 20
        request_rate = 20
        content_duplicate_num = 20
        contents_raw = [
            dict(
                prompt="1+1=2, 1+2=3, 1+3=4, 1+4=5, 1+5=",
                expect_output="TODO",
            ),
            dict(
                prompt="2*1=2, 2*2=4, 2*3=6, 2*4=",
                expect_output="TODO",
            ),
            dict(
                prompt="10*1=10, 10*2=20, 10*3=30, 10*4=40, 10*5=50, 10*6=",
                expect_output="TODO",
            ),
            dict(
                prompt="2/2=1, 4/2=2, 6/2=3, 8/2=",
                expect_output="TODO",
            ),
            dict(
                prompt="One plus one is two, one plus two is three, one plus three is",
                expect_output="TODO",
            ),
        ]

        async def _main_async():
            await asyncio.gather(
                asyncio.create_task(_task_generate()),
                asyncio.create_task(_task_rebalance()),
            )

        async def _task_generate():
            contents_duplicated = contents_raw * content_duplicate_num
            random.shuffle(contents_duplicated)

            tasks = []
            async for content in _yield_with_poisson_process(contents_duplicated, action_rate=request_rate):
                tasks.append(asyncio.create_task(engine.async_generate(
                    prompt=content["prompt"],
                    sampling_params=dict(temperature=0, max_new_tokens=8),
                )))

            actual_outputs = await asyncio.gather(*tasks)
            actual_output_texts = [x["text"] for x in actual_outputs]
            expect_output_texts = [x["expect_output"] for x in contents_duplicated]
            print(f"{actual_output_texts=} {expect_output_texts=}")
            self.assertEqual(actual_output_texts, expect_output_texts)

        async def _task_rebalance():
            for i in range(num_rebalance):
                await engine.tokenizer_manager.eplb_rebalance()

        with tempfile.TemporaryDirectory() as tmpdir:
            engine_kwargs = dict(
                model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
                trust_remote_code=True,
                enable_eplb=True,
                eplb_storage_dir=tmpdir,
                ep_num_redundant_experts=4,
                enable_dp_attention=True,
                enable_deepep_moe=True,
                deepep_mode="normal",
                disable_cuda_graph=True,
                enable_scheduler_input_blocker=True,
                disable_overlap_schedule=True,
                tp_size=2,
                dp_size=2,
                log_level="info",
                disable_radix_cache=True,
            )

            print(f"Action: start engine")
            engine = sgl.Engine(**engine_kwargs)

            loop = asyncio.get_event_loop()
            loop.run_until_complete(_main_async)

            print(f"Action: shutdown engine")
            engine.shutdown()
            del engine

    def test_eplb_start_rebalance_restart(self):
        print("Action: test_eplb_start_rebalance_restart")
        with tempfile.TemporaryDirectory() as tmpdir:
            engine_kwargs = dict(
                model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
                trust_remote_code=True,
                enable_eplb=True,
                eplb_storage_dir=tmpdir,
                ep_num_redundant_experts=4,
                enable_dp_attention=True,
                enable_deepep_moe=True,
                deepep_mode="normal",
                disable_cuda_graph=True,
                enable_scheduler_input_blocker=True,
                disable_overlap_schedule=True,
                tp_size=2,
                dp_size=2,
                log_level="info",
            )

            print(f"Action: start engine")
            engine = sgl.Engine(**engine_kwargs)
            self._assert_behavior(engine, "equal_trivial")

            print(f"Action: eplb_rebalance")
            engine.eplb_rebalance()
            self._engine_flush_cache(engine)
            physical_to_logical_map_layer_0_after_first_rebalance = (
                self._assert_behavior(engine, "not_equal_trivial")
            )

            print(f"Action: shutdown engine")
            engine.shutdown()
            del engine

            print(f"Action: start engine")
            engine = sgl.Engine(**engine_kwargs, port=21000)
            self._assert_behavior(
                engine,
                physical_to_logical_map_layer_0_after_first_rebalance,
            )

            print(f"Action: shutdown engine")
            engine.shutdown()
            del engine

    def test_eplb_init_expert_location_and_save_expert_distribution(self):
        print("Action: test_eplb_init_expert_location_and_save_expert_distribution")
        with tempfile.TemporaryDirectory() as eplb_storage_dir_a, tempfile.TemporaryDirectory() as eplb_storage_dir_b:
            engine_kwargs = dict(
                model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
                trust_remote_code=True,
                enable_eplb=True,
                ep_num_redundant_experts=4,
                enable_dp_attention=True,
                enable_deepep_moe=True,
                deepep_mode="normal",
                disable_cuda_graph=True,
                disable_overlap_schedule=True,
                tp_size=2,
                dp_size=2,
                log_level="info",
            )

            print(f"Action: start engine")
            engine = sgl.Engine(**engine_kwargs, eplb_storage_dir=eplb_storage_dir_a)
            self._assert_behavior(engine, "equal_trivial")

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
                port=21000,
            )
            self._assert_behavior(engine, "not_equal_trivial")
            print(f"Action: shutdown engine")
            engine.shutdown()
            del engine

            print(
                f"Action: start engine to check automatically loading from storage dir"
            )
            engine = sgl.Engine(
                **engine_kwargs, eplb_storage_dir=eplb_storage_dir_a, port=22000
            )
            self._assert_behavior(engine, "not_equal_trivial")
            print(f"Action: shutdown engine")
            engine.shutdown()
            del engine

    def test_nontrivial_location(self):
        print("Action: test_nontrivial_location")
        ep_num_redundant_experts = 4
        engine_kwargs = dict(
            model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
            trust_remote_code=True,
            ep_num_redundant_experts=ep_num_redundant_experts,
            enable_dp_attention=True,
            enable_deepep_moe=True,
            deepep_mode="normal",
            disable_cuda_graph=True,
            tp_size=2,
            dp_size=2,
            log_level="info",
        )

        offset = 3
        physical_to_logical_map = (
                                      offset
                                      + torch.arange(0, _NUM_ROUTED_EXPERTS + ep_num_redundant_experts).repeat(
                                      _NUM_HIDDEN_LAYERS, 1
                                  )
                                  ) % _NUM_ROUTED_EXPERTS
        init_expert_location = dict(
            physical_to_logical_map=physical_to_logical_map.tolist()
        )

        engine = sgl.Engine(
            **engine_kwargs, init_expert_location=json.dumps(init_expert_location)
        )
        self._assert_behavior(engine, physical_to_logical_map[0].tolist())
        engine.shutdown()
        del engine

    def test_trivial_with_redundant_experts(self):
        print("Action: test_trivial_with_redundant_experts")
        engine_kwargs = dict(
            model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
            trust_remote_code=True,
            ep_num_redundant_experts=4,
            enable_dp_attention=True,
            enable_deepep_moe=True,
            deepep_mode="normal",
            disable_cuda_graph=True,
            tp_size=2,
            dp_size=2,
            log_level="info",
        )

        engine = sgl.Engine(**engine_kwargs)
        self._assert_behavior(engine, "equal_trivial")
        engine.shutdown()
        del engine

    def _assert_behavior(self, engine: sgl.Engine, expect_physical_to_local_map):
        actual_output = self._engine_generate(engine)
        self.assertEqual(actual_output, _REF_OUTPUT)

        physical_to_logical_map = (
            engine.tokenizer_manager.expert_location_metadata.physical_to_logical_map
        )
        physical_to_logical_map_layer_0 = physical_to_logical_map[0, :].tolist()
        print(f"{physical_to_logical_map_layer_0=}")

        trivial_expert_locations = _compute_trivial_expert_locations(
            engine.server_args.ep_num_redundant_experts
        )

        if expect_physical_to_local_map == "equal_trivial":
            self.assertEqual(physical_to_logical_map_layer_0, trivial_expert_locations)
        elif expect_physical_to_local_map == "not_equal_trivial":
            self.assertNotEqual(
                physical_to_logical_map_layer_0, trivial_expert_locations
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

    def _engine_flush_cache(self, engine: sgl.Engine):
        ret = engine.flush_cache()
        assert ret.success


def _compute_trivial_expert_locations(ep_num_redundant_experts: int):
    return list(
        x % _NUM_ROUTED_EXPERTS
        for x in range(_NUM_ROUTED_EXPERTS + ep_num_redundant_experts)
    )


async def _yield_with_poisson_process(items: List, action_rate: float):
    for item in items:
        yield item
        interval = np.random.exponential(1.0 / action_rate)
        await asyncio.sleep(interval)


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage


class TestDecodingStageParallelism(unittest.TestCase):
    def test_cfg_parallel_uses_replicated_decode_when_decode_group_has_multiple_ranks(
        self,
    ):
        stage = object.__new__(DecodingStage)
        stage.vae = SimpleNamespace(use_parallel_decode=True)

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.get_global_server_args",
                return_value=SimpleNamespace(enable_cfg_parallel=True),
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.get_decode_parallel_world_size",
                return_value=2,
            ),
        ):
            self.assertEqual(
                stage.parallelism_type,
                StageParallelismType.REPLICATED,
            )

    def test_cfg_parallel_keeps_main_rank_decode_without_parallel_decode(self):
        stage = object.__new__(DecodingStage)
        stage.vae = SimpleNamespace(use_parallel_decode=False)

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.get_global_server_args",
                return_value=SimpleNamespace(enable_cfg_parallel=True),
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.get_decode_parallel_world_size",
                return_value=2,
            ),
        ):
            self.assertEqual(
                stage.parallelism_type,
                StageParallelismType.MAIN_RANK_ONLY,
            )

    def test_cfg_parallel_keeps_main_rank_decode_when_decode_group_is_single_rank(
        self,
    ):
        stage = object.__new__(DecodingStage)
        stage.vae = SimpleNamespace(use_parallel_decode=True)

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.get_global_server_args",
                return_value=SimpleNamespace(enable_cfg_parallel=True),
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.get_decode_parallel_world_size",
                return_value=1,
            ),
        ):
            self.assertEqual(
                stage.parallelism_type,
                StageParallelismType.MAIN_RANK_ONLY,
            )

    def test_non_cfg_parallel_keeps_replicated_decode(self):
        stage = object.__new__(DecodingStage)
        stage.vae = SimpleNamespace(use_parallel_decode=True)

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.get_global_server_args",
            return_value=SimpleNamespace(enable_cfg_parallel=False),
        ):
            self.assertEqual(
                stage.parallelism_type,
                StageParallelismType.REPLICATED,
            )


if __name__ == "__main__":
    unittest.main()

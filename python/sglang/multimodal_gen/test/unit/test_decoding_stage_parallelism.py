import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch.nn as nn

from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av import (
    LTX2RefinementStage,
)


class FakeVAE(nn.Module):
    def decode(self, latents):
        return latents


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

    def test_torch_compile_decode_cache_is_replaced_for_new_vae_instance(self):
        vae = FakeVAE()
        stage = DecodingStage(vae)
        server_args = SimpleNamespace(enable_torch_compile=True)

        with patch(
            "torch.compile",
            side_effect=lambda fn, **_: fn,
        ) as compile_fn:
            compiled_vae_decode = stage._get_vae_decode_fn(vae, server_args)
            self.assertIs(
                stage._get_vae_decode_fn(vae, server_args), compiled_vae_decode
            )
            self.assertEqual(compile_fn.call_count, 1)

            new_vae = FakeVAE()
            new_compiled_vae_decode = stage._get_vae_decode_fn(new_vae, server_args)
            self.assertIsNot(new_compiled_vae_decode, compiled_vae_decode)
            self.assertEqual(compile_fn.call_count, 2)
            self.assertEqual(stage._compiled_vae_decode.target_id, id(new_vae))
            self.assertIs(
                stage._compiled_vae_decode.compiled_module, new_compiled_vae_decode
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

    def test_ltx2_refinement_broadcasts_when_decode_uses_all_ranks(self):
        stage = object.__new__(LTX2RefinementStage)
        stage.server_args = SimpleNamespace(enable_cfg_parallel=True)
        stage.vae = SimpleNamespace(use_parallel_decode=True)

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av.get_decode_parallel_world_size",
                return_value=2,
            ),
        ):
            self.assertEqual(
                stage.parallelism_type,
                StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS,
            )

    def test_ltx2_refinement_stays_main_rank_only_without_parallel_decode(self):
        stage = object.__new__(LTX2RefinementStage)
        stage.server_args = SimpleNamespace(enable_cfg_parallel=True)
        stage.vae = SimpleNamespace(use_parallel_decode=False)

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av.get_decode_parallel_world_size",
                return_value=2,
            ),
        ):
            self.assertEqual(
                stage.parallelism_type,
                StageParallelismType.MAIN_RANK_ONLY,
            )


if __name__ == "__main__":
    unittest.main()

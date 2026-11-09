# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap


def test_external_diffusion_package_registration(tmp_path):
    package_dir = tmp_path / "external_diffusion_plugin"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        textwrap.dedent("""
            from sglang.multimodal_gen.registry import register_pipeline
            from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

            from .plugin import (
                ExternalPipeline,
                ExternalPipelineConfig,
                ExternalSamplingParams,
            )

            ModelRegistry.register_model(
                "ExternalTransformer",
                "external_diffusion_plugin.plugin:ExternalTransformer",
            )
            register_pipeline(
                ExternalPipeline,
                sampling_param_cls=ExternalSamplingParams,
                pipeline_config_cls=ExternalPipelineConfig,
                hf_model_paths=["external-org/external-checkpoint"],
                model_detectors=[lambda value: "external-checkpoint" in value],
            )
            """),
        encoding="utf-8",
    )
    (package_dir / "plugin.py").write_text(
        textwrap.dedent("""
            from torch import nn

            from sglang.multimodal_gen.configs.pipeline_configs.base import (
                PipelineConfig,
            )
            from sglang.multimodal_gen.configs.sample.sampling_params import (
                SamplingParams,
            )
            from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
                ComposedPipelineBase,
            )


            class ExternalTransformer(nn.Module):
                pass


            class ExternalPipelineConfig(PipelineConfig):
                pass


            class ExternalSamplingParams(SamplingParams):
                pass


            class ExternalPipeline(ComposedPipelineBase):
                pipeline_name = "ExternalPipeline"

                def create_pipeline_stages(self, server_args):
                    pass
            """),
        encoding="utf-8",
    )

    model_dir = tmp_path / "external-checkpoint"
    model_dir.mkdir()

    script = textwrap.dedent("""
        import sys

        from sglang.cli.utils import get_is_diffusion_model
        from sglang.multimodal_gen.registry import (
            get_model_info,
            get_pipeline_config_classes,
        )
        from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

        assert get_is_diffusion_model(sys.argv[1])
        model_cls, architecture = ModelRegistry.resolve_model_cls(
            "ExternalTransformer"
        )
        assert model_cls.__name__ == "ExternalTransformer"
        assert architecture == "ExternalTransformer"

        model_info = get_model_info(sys.argv[1], backend="sglang")
        assert model_info.pipeline_cls.__name__ == "ExternalPipeline"
        assert model_info.pipeline_config_cls.__name__ == "ExternalPipelineConfig"
        assert model_info.sampling_param_cls.__name__ == "ExternalSamplingParams"

        pipeline_config_cls, sampling_param_cls = get_pipeline_config_classes(
            "ExternalPipeline"
        )
        assert pipeline_config_cls is model_info.pipeline_config_cls
        assert sampling_param_cls is model_info.sampling_param_cls
        """)
    env = os.environ.copy()
    env["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "external_diffusion_plugin"
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(tmp_path), env.get("PYTHONPATH")) if path
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(model_dir)],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

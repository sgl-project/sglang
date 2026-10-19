# SPDX-License-Identifier: Apache-2.0
"""GPU-free package and registration tests for FLUX.3 Image Klein."""

import argparse
import dataclasses
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
    Flux3ImagePipelineConfig,
    Flux3ImageSamplingParams,
)

_MODEL = "flux-3-image-klein"
_SNAPSHOT_DOWNLOAD = (
    "sglang.multimodal_gen.runtime.pipelines.flux3_image_package.snapshot_download"
)
_FP8R_MANIFEST = {
    "dtype": "float8_e4m3fn",
    "scale_convention": "weight.float() * weight_scale[:, None]",
    "fp8_matrices": 234,
    "weight_scales": 234,
}
_NVFP4_MANIFEST = {
    "dtype": "nvfp4: weight uint8",
    "group_size": 16,
    "scale_convention": (
        "e2m1(weight) * weight_scale[:, k // 16] * weight_scale_2; "
        "weight_scale_2 = amax(weight) / (448 * 6); "
        "input_scale = p99.9 of per-call input amax / (448 * 6)"
    ),
    "nvfp4_matrices": 234,
    "weight_scales": 234,
    "weight_scale_2": 234,
    "input_scales": 234,
}


def _config_dict():
    return {
        "format_version": 1,
        "model": _MODEL,
        "quantization": None,
        "torch_dtype": "bfloat16",
        "dit": {
            "hidden": 3072,
            "heads": 24,
            "depth": 5,
            "depth_single": 28,
            "mlp_ratio": 3.0,
            "axes_dim": [32, 32, 32, 32],
            "theta": 10000,
            "in_channels": {"image": 128, "image_cond": 128, "text": 20480},
            "out_modes": ["image"],
            "vec_in_dim": None,
            "image_rope_scale": 0.5,
        },
        "text_encoder": {
            "path": "text_encoder",
            "tokenizer": "tokenizer",
            "layers": [4, 8, 12, 16, 20, 24, 28, 32],
            "pad_multiple": 80,
            "max_tokens": 8192,
            "dtype": "bfloat16",
        },
        "vae": {"path": "ae.safetensors", "dtype": "float32"},
        "sampler": {
            "steps": 50,
            "guidance": 3.5,
            "alpha": 6.93,
            "size": [1024, 1024],
        },
    }


def _write_dit(path: Path, flow_map: bool = False) -> None:
    from safetensors.torch import save_file

    # Resolution reads only the key set: ref_time_in marks a flow-map DiT.
    tensors = {"img_in.weight": torch.zeros(1)}
    if flow_map:
        tensors["ref_time_in.in_layer.weight"] = torch.zeros(1)
    save_file(tensors, str(path), metadata={"model": _MODEL})


def _write_package(
    root: Path, config=None, flow_map=False, assets=True, manifest=None
) -> None:
    config = config or _config_dict()
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config))
    (root / "manifest.json").write_text(
        json.dumps({"format_version": 1, **(manifest or {})})
    )
    _write_dit(root / "model.safetensors", flow_map=flow_map)
    if config["quantization"] is not None:
        # Resolution only checks that it exists; the weight loader parses it.
        (root / "hf_quant_config.json").write_text("{}")
    if assets:
        (root / "ae.safetensors").touch()
        (root / "text_encoder").mkdir()
        (root / "tokenizer").mkdir()


def _write_variant(
    directory: Path, quantization=None, manifest=None, flow_map=False, **sampler
) -> None:
    config = dict(_config_dict(), quantization=quantization)
    config["sampler"].update(sampler)
    _write_package(directory, config, flow_map, assets=False, manifest=manifest)


class TestFlux3ServedRegistration(unittest.TestCase):
    def test_batch_admission_uses_the_packed_vae_token_grid(self):
        config = Flux3ImagePipelineConfig()
        request = SimpleNamespace(
            n_tokens=None,
            width=1024,
            height=512,
            num_frames=1,
            num_outputs_per_prompt=1,
        )
        self.assertEqual(config.estimate_request_cost(request), 64 * 32)

    def test_release_ids_and_renamed_packages_select_flux3_image(self):
        """Hub ids and local packages of any name must reach the FLUX.3 image
        pipeline without the model_index.json probe, which downloads a whole
        repo that has none."""
        from sglang.multimodal_gen.registry import (
            get_model_info,
            get_non_diffusers_pipeline_name,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
            LoRAPipeline,
        )

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "sglang.multimodal_gen.registry.maybe_download_model_index",
                side_effect=AssertionError("probed model_index.json"),
            ),
        ):
            local = Path(directory) / "any-name"
            local.mkdir()
            (local / "config.json").write_text(json.dumps(_config_dict()))
            # A Diffusers export may coexist with the native package. The native
            # config still identifies it without probing/downloading the export.
            (local / "modular_model_index.json").write_text("{}")
            for model_path in (
                "herzogenhorn/f3-image-small",
                "black-forest-labs/flux-3-image-klein",
                "black-forest-labs/flux-3-image-klein-base",
                str(local),
            ):
                info = get_model_info(model_path)
                self.assertEqual(
                    (
                        info.pipeline_cls.__name__,
                        info.pipeline_config_cls.__name__,
                        info.sampling_param_cls.__name__,
                    ),
                    (
                        "Flux3ImagePipeline",
                        "Flux3ImagePipelineConfig",
                        "Flux3ImageSamplingParams",
                    ),
                )
        # Without the mixin a startup --lora-path is silently ignored.
        self.assertTrue(issubclass(info.pipeline_cls, LoRAPipeline))
        self.assertEqual(
            get_non_diffusers_pipeline_name("black-forest-labs/flux-3-action-droid"),
            "Flux3ActionPipeline",
        )

    def test_sampler_defaults_follow_the_served_package(self):
        """The distilled flow map runs 16 steps without CFG; fixed 50-step CFG
        defaults in the pipeline config used to override the package sampler."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
            _steps,
        )

        def validated(root, variant, enable_cfg_parallel=False, **fields):
            config = Flux3ImagePipelineConfig(**fields)
            config.validate_server_args(
                SimpleNamespace(
                    model_path=str(root),
                    model_variant=variant,
                    revision=None,
                    enable_cfg_parallel=enable_cfg_parallel,
                )
            )
            return config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root)
            _write_variant(root / "variants" / "sd", flow_map=True, steps=16)
            for variant, default_steps in ((None, 50), ("sd", 16)):
                package = validated(root, variant).flux3_package.config
                for requested, expected in ((None, default_steps), (4, 4)):
                    batch = SimpleNamespace(num_inference_steps=requested)
                    self.assertEqual(_steps(batch, package), expected)
            # The package alpha applies only when --flux3-alpha is unset.
            self.assertEqual(validated(root, None).flux3_alpha, 6.93)
            self.assertEqual(validated(root, None, flux3_alpha=6.5).flux3_alpha, 6.5)
            with self.assertRaisesRegex(ValueError, "cfg-parallel"):
                validated(root, "sd", enable_cfg_parallel=True)

    def test_cli_registers_flux3_flags_for_a_detected_model_path(self):
        """Without --pipeline-class-name the FLUX.3 flags (--flux3-alpha, ...)
        must still parse when --model-path is a detected FLUX.3 image model."""
        from sglang.multimodal_gen.runtime.entrypoints.cli.utils import (
            add_resolved_pipeline_config_cli_args,
        )

        args = ["--model-path", "black-forest-labs/flux-3-image-klein"]
        args += ["--flux3-alpha", "6.5"]
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-path")
        with mock.patch.object(sys, "argv", ["sglang", *args]):
            add_resolved_pipeline_config_cli_args(parser)
        self.assertEqual(parser.parse_args(args).flux3_alpha, 6.5)


class TestFlux3ServedOutput(unittest.TestCase):
    def test_decode_output_is_normalized_once_before_worker_materialization(self):
        from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
            Flux3ImageDecodeStage,
        )

        image = torch.tensor([[[-1.0]], [[0.0]], [[1.0]]])
        displayed = Flux3ImageDecodeStage._to_display(image)
        req = SimpleNamespace(
            enable_frame_interpolation=False,
            enable_upscaling=False,
        )

        frame = GPUWorker._materialize_frame_output(displayed[0], OutputBatch(), req)

        self.assertTrue(displayed.dtype.is_floating_point)
        self.assertEqual(tuple(displayed.shape), (1, 3, 1, 1))
        self.assertEqual(frame[0, 0, 0].tolist(), [0, 127, 255])


class TestFlux3Package(unittest.TestCase):
    def test_config_and_dit_mapping(self):
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_package import (
            Flux3ImagePackageConfig,
        )
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_weights import (
            flux3_image_dit_config_from_package,
        )

        raw = _config_dict()
        raw["dit"]["flow_map"] = True
        package = SimpleNamespace(config=Flux3ImagePackageConfig.from_dict(raw))
        arch = flux3_image_dit_config_from_package(package).arch_config
        # Text is the DiT context, not a content stream with its own mode blocks.
        self.assertEqual(set(arch.in_channels), {"image", "image_cond"})
        self.assertEqual(arch.context_in_dim, 20480)
        self.assertEqual(arch.out_modes, ["image"])
        self.assertEqual(arch.num_channels_latents, 128)
        self.assertTrue(arch.flow_map)

    def test_resolver_selects_quantized_variants_with_shared_root_assets(self):
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_package import (
            resolve_flux3_image_package,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root)
            fp8r = root / "variants" / "fp8r"
            _write_variant(fp8r, "fp8r", _FP8R_MANIFEST)
            package = resolve_flux3_image_package(root, variant="fp8r")
            self.assertEqual(
                (package.variant, package.weights, package.quant_config),
                ("fp8r", fp8r / "model.safetensors", fp8r / "hf_quant_config.json"),
            )
            self.assertEqual(
                (package.vae, package.text_encoder, package.tokenizer),
                (root / "ae.safetensors", root / "text_encoder", root / "tokenizer"),
            )
            self.assertEqual(resolve_flux3_image_package(fp8r), package)

            # A standalone quantized repo keeps the same files at its root.
            standalone = root / "fp8r-repo"
            config = dict(_config_dict(), quantization="fp8r")
            _write_package(standalone, config, manifest=_FP8R_MANIFEST)
            package = resolve_flux3_image_package(standalone)
            self.assertEqual(
                (package.variant, package.config.quantization, package.quant_config),
                (None, "fp8r", standalone / "hf_quant_config.json"),
            )

            nvfp4 = root / "variants" / "nvfp4"
            _write_variant(nvfp4, "nvfp4", _NVFP4_MANIFEST)

            def resolve_nvfp4(**manifest):
                (nvfp4 / "manifest.json").write_text(
                    json.dumps({"format_version": 1, **_NVFP4_MANIFEST, **manifest})
                )
                return resolve_flux3_image_package(root, variant="nvfp4")

            self.assertEqual(resolve_nvfp4().vae, root / "ae.safetensors")
            # Re-exports keep more layers in BF16 and add calibration-only clauses.
            reexport = dict(
                scale_convention=_NVFP4_MANIFEST["scale_convention"].replace(
                    "; input_scale",
                    "; weight_scale = the positive finite e4m3 code minimizing "
                    "the block's squared error; input_scale",
                ),
                weight_scales={"method": "mse_fp8_scale_sweep", "candidates": 126},
                weight_scale_tensors=198,
                nvfp4_matrices=198,
                weight_scale_2=198,
                input_scales=198,
            )
            resolve_nvfp4(**reexport)
            with self.assertRaisesRegex(ValueError, "one scale set each"):
                resolve_nvfp4(**dict(reexport, input_scales=197))
            with self.assertRaisesRegex(ValueError, "scale_convention lacks"):
                resolve_nvfp4(
                    scale_convention=_NVFP4_MANIFEST["scale_convention"].replace(
                        "(448 * 6)", "448", 1
                    )
                )


class TestFlux3ReleasePackages(unittest.TestCase):
    def test_release_metadata_does_not_block_loading(self):
        """The BFL reference reads only the weights, so a renamed model, extra
        keys, a stale manifest name or a missing dit.flow_map must still load;
        the flow map is told from its ref_time_in weights."""
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_package import (
            resolve_flux3_image_package,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config_dict()
            config.update(model="flux-3-image-klein", license="non-commercial")
            config["dit"]["notes"] = "release"
            manifest = {"model": "flux-3-image-rc"}
            _write_package(root, config, flow_map=True, manifest=manifest)
            package = resolve_flux3_image_package(root)
            self.assertTrue(package.config.dit.flow_map)
            # A flow map has no CFG branch, whatever the sampler says.
            self.assertIsNone(package.config.sampler.guidance)

            _write_dit(root / "model.safetensors", flow_map=False)
            package = resolve_flux3_image_package(root)
            self.assertFalse(package.config.dit.flow_map)
            self.assertEqual(package.config.sampler.guidance, 3.5)

    def test_split_distilled_repo_takes_assets_from_the_base_repo(self):
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_package import (
            resolve_flux3_image_package,
        )

        with tempfile.TemporaryDirectory() as directory:
            namespace = Path(directory) / "black-forest-labs"
            base = namespace / "flux-3-image-klein-base"
            _write_package(base)
            klein = namespace / "flux-3-image-klein"
            config = _config_dict()
            config["model"] = "flux-3-image-klein"
            hub = "hf://black-forest-labs/flux-3-image-klein-base"
            config["text_encoder"].update(
                path=f"{hub}/text_encoder", tokenizer=f"{hub}/tokenizer"
            )
            config["vae"]["path"] = f"{hub}/ae.safetensors"
            _write_package(klein, config, flow_map=True, assets=False)

            package = resolve_flux3_image_package(klein)
            self.assertEqual(
                (package.vae, package.text_encoder, package.tokenizer),
                (base / "ae.safetensors", base / "text_encoder", base / "tokenizer"),
            )

            # Without a sibling checkout the assets come from the Hub.
            snapshot = Path(directory) / "snapshot"
            shutil.move(base, snapshot)
            with mock.patch(_SNAPSHOT_DOWNLOAD, return_value=str(snapshot)) as fetch:
                package = resolve_flux3_image_package(klein)
            self.assertEqual(package.vae, snapshot / "ae.safetensors")
            self.assertEqual(
                {call.kwargs["repo_id"] for call in fetch.call_args_list},
                {"black-forest-labs/flux-3-image-klein-base"},
            )

    def test_quantized_variant_names_never_cross_models(self):
        """A distilled repo may keep the sd-fp8r name, which fp8r then selects;
        sd-fp8r must never fall back to a base repo's fp8r DiT."""
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_package import (
            resolve_flux3_image_package,
        )

        with tempfile.TemporaryDirectory() as directory:
            klein = Path(directory) / "klein"
            _write_package(klein, flow_map=True)
            _write_variant(
                klein / "variants" / "sd-fp8r", "fp8r", _FP8R_MANIFEST, flow_map=True
            )
            package = resolve_flux3_image_package(klein, variant="fp8r")
            self.assertEqual(package.variant, "sd-fp8r")
            self.assertTrue(package.config.dit.flow_map)

            base = Path(directory) / "base"
            _write_package(base)
            _write_variant(base / "variants" / "fp8r", "fp8r", _FP8R_MANIFEST)
            with self.assertRaisesRegex(ValueError, "sd-fp8r.*does not exist"):
                resolve_flux3_image_package(base, variant="sd-fp8r")

    def test_hub_repo_id_fetches_only_the_selected_dit(self):
        """Both release repos are self-contained and use identical variant names.
        Serving one DiT must not download another model or the Diffusers export."""
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_package import (
            resolve_flux3_image_package,
        )

        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            for flow_map in (False, True):
                repo = "black-forest-labs/flux-3-image-klein" + (
                    "" if flow_map else "-base"
                )
                root = snapshot / repo
                _write_package(root, flow_map=flow_map)
                (root / "modular_model_index.json").write_text("{}")
                for variant, manifest in (
                    (None, None),
                    ("fp8r", _FP8R_MANIFEST),
                    ("nvfp4", _NVFP4_MANIFEST),
                ):
                    with self.subTest(repo=repo, variant=variant):
                        prefix = f"variants/{variant}/" if variant else ""
                        if variant:
                            _write_variant(
                                root / prefix, variant, manifest, flow_map=flow_map
                            )
                        with mock.patch(
                            _SNAPSHOT_DOWNLOAD, return_value=str(root)
                        ) as fetch:
                            package = resolve_flux3_image_package(
                                repo, variant=variant, revision="r1"
                            )
                        self.assertEqual(package.config.model, _MODEL)
                        self.assertEqual(package.config.dit.flow_map, flow_map)
                        self.assertEqual(
                            package.weights, root / prefix / "model.safetensors"
                        )
                        self.assertEqual(package.vae, root / "ae.safetensors")
                        patterns = [
                            pattern
                            for call in fetch.call_args_list
                            for pattern in call.kwargs["allow_patterns"]
                        ]
                        self.assertIn(prefix + "model.safetensors", patterns)
                        self.assertIn("text_encoder/*", patterns)
                        self.assertNotIn("modular_model_index.json", patterns)
                        if variant:
                            self.assertNotIn("model.safetensors", patterns)
                        self.assertFalse(
                            [p for p in patterns if "*" in p.split("/")[0]]
                        )
                        self.assertEqual(
                            {
                                (c.kwargs["repo_id"], c.kwargs["revision"])
                                for c in fetch.call_args_list
                            },
                            {(repo, "r1")},
                        )


class TestFlux3RequestControls(unittest.TestCase):
    def test_flow_map_rejects_cfg_while_building_the_request(self):
        """On the flow map any CFG scale but 1 (as in the reference) must fail
        while building the request, which the API maps to 400, not as an
        opaque HTTP 500 from the denoise stage."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
            _cfg_scale,
        )

        flow_map = SimpleNamespace(dit=SimpleNamespace(flow_map=True))
        server_args = SimpleNamespace(
            pipeline_config=Flux3ImagePipelineConfig(
                flux3_package=SimpleNamespace(config=flow_map)
            ),
            output_path=None,
            comfyui_mode=False,
        )
        for scale, accepted in ((None, True), (1.0, True), (0.5, False), (3.5, False)):
            params = Flux3ImageSamplingParams(prompt="a fox", guidance_scale=scale)
            batch = SimpleNamespace(true_cfg_scale=None, guidance_scale=scale)
            if accepted:
                # Raises "drop negative_prompt" if the default negative is not empty.
                params._adjust(server_args)
                self.assertIsNone(_cfg_scale(batch, flow_map))
                continue
            with self.assertRaisesRegex(ValueError, "without CFG"):
                params._adjust(server_args)
            with self.assertRaisesRegex(ValueError, "without CFG"):
                _cfg_scale(batch, flow_map)
        params = Flux3ImageSamplingParams(prompt="a fox", negative_prompt="blurry")
        with self.assertRaisesRegex(ValueError, "drop negative_prompt"):
            params._adjust(server_args)
        with self.assertRaisesRegex(ValueError, "flux3_mode"):
            Flux3ImageSamplingParams(flux3_mode="unsupported")

    def test_base_cfg_scale_is_per_request_with_the_package_default(self):
        """Only an omitted scale takes the package default; explicit 1 is not 3.5."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
            _cfg_scale,
        )

        base = SimpleNamespace(
            dit=SimpleNamespace(flow_map=False), sampler=SimpleNamespace(guidance=3.5)
        )
        for scale, expected in ((None, 3.5), (1.0, 1.0), (5.0, 5.0)):
            batch = SimpleNamespace(true_cfg_scale=None, guidance_scale=scale)
            self.assertEqual(_cfg_scale(batch, base), expected)
        server_args = SimpleNamespace(
            pipeline_config=Flux3ImagePipelineConfig(
                flux3_package=SimpleNamespace(config=base)
            ),
            output_path=None,
            comfyui_mode=False,
        )
        for requested, expected in ((None, 3.5), (1.0, 1.0), (5.0, 5.0)):
            params = Flux3ImageSamplingParams(prompt="a fox", guidance_scale=requested)
            params._adjust(server_args)
            self.assertEqual(_cfg_scale(params, base), expected)
        params = Flux3ImageSamplingParams(prompt="a fox", true_cfg_scale=1.0)
        params._adjust(server_args)
        self.assertEqual(_cfg_scale(params, base), 1.0)

    def test_synthetic_warmup_keeps_a_reference_only_for_i2i(self):
        """Under breakable CUDA graphs no warmup reference is synthesized; an
        i2i warmup must then run t2i instead of failing on the missing image."""
        server_args = SimpleNamespace(pipeline_config=SimpleNamespace())
        for mode, image_path, expected in (
            ("t2i", "warmup.png", (None, "t2i")),
            (None, "warmup.png", (None, "t2i")),
            ("i2i", "warmup.png", ("warmup.png", "i2i")),
            ("i2i", None, (None, "t2i")),
        ):
            params = Flux3ImageSamplingParams(flux3_mode=mode)
            req = SimpleNamespace(image_path=image_path, extra={})
            params.prepare_synthetic_warmup_request_for_queue(req, server_args)
            self.assertEqual((req.image_path, params.flux3_mode), expected)

    def test_warmup_resolves_the_default_step_target(self):
        """A request without steps means the server default. Server warmup
        (e.g. for breakable CUDA graphs) copies the request's steps as the
        full-workload target, and ``int(None)`` there failed the warmup."""
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

        params = Flux3ImageSamplingParams(flux3_mode="t2i")
        req = Req(sampling_params=params)
        req.set_as_warmup(1)
        package = SimpleNamespace(
            config=SimpleNamespace(sampler=SimpleNamespace(steps=28))
        )
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(flux3_package=package)
        )
        params.prepare_synthetic_warmup_request_for_queue(req, server_args)
        self.assertEqual(req.num_inference_steps, 1)
        self.assertEqual(req.extra["cache_dit_num_inference_steps"], 28)

    def test_image_api_accepts_the_request_mode(self):
        """The image endpoints only forward fields the model declares, so an
        undeclared ``flux3_mode`` is silently dropped and the inferred mode
        runs instead."""
        from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
            get_sampling_request_extra_fields,
        )

        fields = {
            f.name for f in dataclasses.fields(Flux3ImageSamplingParams) if f.init
        }
        self.assertLessEqual(
            Flux3ImageSamplingParams.image_request_extra_fields(), fields
        )
        self.assertIn(
            "flux3_mode",
            get_sampling_request_extra_fields(Flux3ImageSamplingParams, "image"),
        )

    def test_reference_image_without_mode_selects_i2i(self):
        """One server serves both modes: the image-edit endpoint uploads a
        reference but cannot name a mode, so references select i2i."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
            _mode,
        )

        config = Flux3ImagePipelineConfig()
        edit = SimpleNamespace(flux3_mode=None, image_path=["reference.png"])
        self.assertEqual(_mode(edit, config), "i2i")
        plain = SimpleNamespace(flux3_mode=None, image_path=None)
        self.assertEqual(_mode(plain, config), "t2i")
        explicit = SimpleNamespace(flux3_mode="t2i", image_path=["reference.png"])
        self.assertEqual(_mode(explicit, config), "t2i")


class TestFlux3Lora(unittest.TestCase):
    _SUPPORTS_LORA = (
        "sglang.multimodal_gen.runtime.pipelines.flux3_image."
        "flux3_prims_ts_supports_lora"
    )

    @staticmethod
    def _pipeline_with_one_prims_ts_layer():
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase
        from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
            Flux3PrimsTsMethodMixin,
        )
        from sglang.multimodal_gen.runtime.pipelines.flux3_image import (
            Flux3ImagePipeline,
        )

        layer = LinearBase.__new__(LinearBase)
        torch.nn.Module.__init__(layer)
        layer.quant_method = Flux3PrimsTsMethodMixin()
        pipeline = Flux3ImagePipeline.__new__(Flux3ImagePipeline)
        pipeline.modules = {"transformer": torch.nn.Sequential(layer)}
        pipeline.lora_target_modules = None
        return pipeline, layer

    def test_rejects_late_lora_before_wrapping_prims_ts_weights(self):
        from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
            Flux3PrimsTsRole,
        )

        pipeline, layer = self._pipeline_with_one_prims_ts_layer()
        with mock.patch(self._SUPPORTS_LORA, return_value=False):
            pipeline._reject_lora_on_packed_weights()
            layer.quant_method.prims_ts_role = Flux3PrimsTsRole.SWIGLU
            with self.assertRaisesRegex(ValueError, "--lora-path"):
                pipeline._reject_lora_on_packed_weights()
        self.assertIs(pipeline.get_module("transformer")[0], layer)

    def test_accepts_late_lora_when_prims_ts_takes_a_delta(self):
        from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
            Flux3PrimsTsRole,
        )

        pipeline, layer = self._pipeline_with_one_prims_ts_layer()
        layer.quant_method.prims_ts_role = Flux3PrimsTsRole.SWIGLU
        with mock.patch(self._SUPPORTS_LORA, return_value=True):
            pipeline._reject_lora_on_packed_weights()

    def test_prefixed_adapter_keys_reach_the_native_modules(self):
        """Prefixed adapters (BFL/ostris ``dit.``, diffusers ``transformer.``)
        must map onto the native DiT modules, not load onto 0 layers, and load
        as standard LoRA rather than through the FLUX.1 ai-toolkit conversion."""
        from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
            flux3_image_dit_config,
        )
        from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
        from sglang.multimodal_gen.runtime.models.dits.flux3_image import (
            Flux3ImageDiT,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
            LoRAFormat,
            detect_lora_format_from_state_dict,
            normalize_lora_state_dict,
        )

        # Resolved as LoRAPipeline.load_lora_adapter does: config, else the DiT.
        arch = flux3_image_dit_config().arch_config
        lora_map = get_param_names_mapping(
            arch.lora_param_names_mapping or Flux3ImageDiT.lora_param_names_mapping
        )
        param_map = get_param_names_mapping(
            arch.param_names_mapping or Flux3ImageDiT.param_names_mapping
        )
        for prefix in ("", "dit.", "transformer."):
            targets = {
                f"{prefix}single_blocks.0.q_proj.lora_A.weight": (
                    "single_blocks.0.to_qkv.lora_A"
                ),
                f"{prefix}txt_mode_blocks.4.mlp_in.lora_B.weight": (
                    "txt_mode_blocks.4.mlp_in.lora_B"
                ),
            }
            for key, target in targets.items():
                name, _, _ = lora_map(key.removesuffix(".weight"))
                self.assertEqual(param_map(name)[0], target)
            state = {key: torch.zeros(1) for key in targets}
            self.assertEqual(
                detect_lora_format_from_state_dict(state), LoRAFormat.STANDARD
            )
            self.assertEqual(set(normalize_lora_state_dict(state)), set(state))
        self.assertEqual(
            detect_lora_format_from_state_dict(
                {"single_blocks.0.linear1.lora_A.weight": torch.zeros(1)}
            ),
            LoRAFormat.AI_TOOLKIT_FLUX,
        )


class TestFlux3TextEncoder(unittest.TestCase):
    def test_batch_encodes_each_prompt_with_reference_positions(self):
        """Batch only equal pad buckets; preserve each prompt's positions and order."""
        from sglang.multimodal_gen.runtime.models.encoders.flux3_text_encoder import (
            Flux3TextEncoder,
        )

        class FakeTokenizer:
            lengths = {"short": 3, "long": 6, "tiny": 2}

            def __call__(self, text, **kwargs):
                if "return_tensors" not in kwargs:
                    length = min(self.lengths[text], kwargs["max_length"])
                    return {"input_ids": list(range(length))}
                length = kwargs["max_length"]
                attention_mask = torch.zeros((len(text), length), dtype=torch.long)
                for row, prompt in enumerate(text):
                    attention_mask[row, : self.lengths[prompt]] = 1
                return {
                    "input_ids": torch.arange(length)[None].expand(len(text), -1),
                    "attention_mask": attention_mask,
                }

        class FakeProcessor:
            tokenizer = FakeTokenizer()

            @staticmethod
            def apply_chat_template(message, **_kwargs):
                return message[0]["content"]

        class FakeBaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.call_kwargs = []

            def forward(self, **kwargs):
                self.call_kwargs.append(kwargs)
                hidden = kwargs["input_ids"].unsqueeze(-1).expand(-1, -1, 2)
                return SimpleNamespace(hidden_states=(hidden,))

        base_model = FakeBaseModel()
        encoder = Flux3TextEncoder.__new__(Flux3TextEncoder)
        torch.nn.Module.__init__(encoder)
        encoder.embed_dtype = torch.bfloat16
        encoder.output_layers = (0,)
        encoder.max_length = 16
        encoder.pad_multiple = 4
        encoder.processor = FakeProcessor()
        encoder.model = base_model

        result = encoder.encode_batch(["short", "long", "tiny", "short"])

        self.assertEqual(
            [tuple(call["input_ids"].shape) for call in base_model.call_kwargs],
            [(2, 4), (1, 8)],
        )
        positions = base_model.call_kwargs[0]["position_ids"]
        self.assertEqual(tuple(positions.shape), (4, 2, 4))
        self.assertEqual(positions[:, 0].tolist(), [[0, 1, 2, 1]] * 4)
        self.assertEqual(positions[:, 1].tolist(), [[0, 1, 1, 1]] * 4)
        self.assertEqual(
            base_model.call_kwargs[1]["position_ids"][0, 0].tolist(),
            [0, 1, 2, 3, 4, 5, 1, 1],
        )
        self.assertEqual(
            [tuple(tokens.shape) for tokens, _ in result],
            [(4, 2), (8, 2), (4, 2), (4, 2)],
        )
        self.assertEqual(
            [tuple(ids.shape) for _, ids in result], [(4, 4), (8, 4), (4, 4), (4, 4)]
        )
        self.assertIs(result[0], result[3])
        self.assertEqual(result[0][1][-1, -1].item(), 3)
        self.assertEqual(result[1][1][-1, -1].item(), 7)

    def test_constructor_truncates_and_compiles_language_model(self):
        from sglang.multimodal_gen.runtime.models.encoders.flux3_text_encoder import (
            Flux3TextEncoder,
        )

        class FakeLanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Identity() for _ in range(36)]
                )
                self.norm = torch.nn.Linear(1, 1)

            def forward(self, value):
                return value

        class FakeConditionalGeneration(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleNamespace(
                    language_model=FakeLanguageModel(), visual=object()
                )

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        tokenizer = SimpleNamespace(padding_side="right")
        processor_cls = mock.Mock()
        processor_cls.from_pretrained.return_value = SimpleNamespace(
            tokenizer=tokenizer
        )
        transformers = SimpleNamespace(
            AutoProcessor=processor_cls,
            Qwen3VLForConditionalGeneration=FakeConditionalGeneration,
        )

        with (
            mock.patch.dict(sys.modules, {"transformers": transformers}),
            mock.patch(
                "torch.compile", side_effect=lambda function, **_kwargs: function
            ) as compile_mock,
        ):
            encoder = Flux3TextEncoder(
                "model",
                output_layers=(4, 8),
                tokenizer_spec="tokenizer",
                compile_language_model=True,
            )

        language_model = encoder.model.language_model
        self.assertEqual(len(language_model.layers), 8)
        self.assertIsInstance(language_model.norm, torch.nn.Identity)
        self.assertFalse(hasattr(encoder.model, "visual"))
        processor_cls.from_pretrained.assert_called_once_with("tokenizer")
        self.assertTrue(compile_mock.call_args.kwargs["dynamic"])


class TestFlux3PackedCFGParallel(unittest.TestCase):
    def test_one_sample_per_rank_is_gathered_in_cfg_order(self):
        from sglang.multimodal_gen.runtime.models.dits.flux3_image import TEXT, pack
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
            flux3_image as stages,
        )

        image_ids, text = torch.zeros(2, 4), (torch.zeros(1, 3), torch.zeros(1, 4))
        packed = pack(
            [
                {"image": (fill(2, 2), image_ids), "image_cond": None, TEXT: text}
                for fill in (torch.zeros, torch.ones)
            ],
            {"image": 2, "image_cond": 2, TEXT: 3},
        )
        with mock.patch.object(
            stages,
            "run_cfg_parallel",
            side_effect=lambda policy, predict: [
                predict(branch) for branch in policy.branches
            ],
        ):
            model = stages._CFGParallelModel(
                lambda *, packed: {"image": packed.data["image"]}
            )
            output = model(packed=packed)
        torch.testing.assert_close(
            output["image"], torch.cat((torch.zeros(2, 2), torch.ones(2, 2)))
        )


if __name__ == "__main__":
    unittest.main()

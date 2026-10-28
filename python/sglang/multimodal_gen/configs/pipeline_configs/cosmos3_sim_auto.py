# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Nano-Sim-Auto pipeline configuration.

A Sim-Auto export is a distilled causal Multiview-AV student: the same Nano
weights, Wan VAE, tokenizer and LiDAR VAE as Cosmos3-Nano-Transfer-Auto, plus
a ``cosmos3_nano_sim_auto`` artifact in ``transformer/config.json`` and a
fixed-step scheduler. The config reuses ``Cosmos3Config`` and derives the
multiview stage contract from the artifact so the Multiview-AV stages serve
this pipeline unchanged; only the transformer class and the denoising stage
differ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from sglang.multimodal_gen.configs.models.dits.cosmos3_sim_auto import (
    Cosmos3SimAutoManifest,
    load_cosmos3_sim_auto_manifest,
    multiview_deployment_from_manifest,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import (
    Cosmos3Config,
    _transformer_config,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    Cosmos3MultiviewDeploymentConfig,
    resolve_multiview_backend_request,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class Cosmos3SimAutoConfig(Cosmos3Config):
    """Cosmos3-Nano-Sim-Auto: causal rolling multiview WSM transfer, optional joint LiDAR."""

    # Single-task pipeline; the parent's multi-task declaration does not apply.
    supported_task_types: ClassVar[tuple[ModelTaskType, ...] | None] = None
    transformer_class_override: str | None = "Cosmos3SimAutoTransformer"
    # The multiview tokenization stage owns prompt formatting (system prompt,
    # duration and resolution sentences, per-camera rig headers).
    use_duration_template: bool = True
    use_system_prompt: bool = True
    # Kernel behind the replay attention: "triton", "fa4", or None for the env
    # override / device default; the same switch as Cosmos3-Nano-Transfer-Auto.
    multiview_attention_backend: str | None = None
    # Parsed once from transformer/config.json in update_config_from_dict.
    sim_auto_manifest: Cosmos3SimAutoManifest | None = None
    # (query, key/value) head counts, for the TP x Ulysses divisibility rule.
    sim_auto_heads: tuple[int, int] | None = None
    # The stage contract derived from the manifest; the multiview sampling
    # params and stages read it under this name.
    multiview_deployment: Cosmos3MultiviewDeploymentConfig | None = None

    def update_config_from_dict(self, args, prefix: str = "") -> None:
        super().update_config_from_dict(args, prefix)
        if self.model_path:
            transformer_config = _transformer_config(self.model_path)
            manifest = load_cosmos3_sim_auto_manifest(transformer_config)
            self._validate_scheduler(manifest)
            self.sim_auto_manifest = manifest
            self.sim_auto_heads = (
                int(transformer_config["num_attention_heads"]),
                int(transformer_config["num_key_value_heads"]),
            )
            self.multiview_deployment = multiview_deployment_from_manifest(manifest)
            # Fail on a bad backend name at launch, not on the first request.
            backend, source = resolve_multiview_backend_request(
                self.multiview_attention_backend
            )
            logger.info(
                "Cosmos3-Nano-Sim-Auto attention backend: %s (from %s)", backend, source
            )

    def _validate_scheduler(self, manifest: Cosmos3SimAutoManifest) -> None:
        if self.distilled_sigmas is None:
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto requires a distilled fixed-step scheduler "
                "(FlowMatchEulerDiscreteScheduler with fixed_step_sampler_config.t_list)."
            )
        if len(self.distilled_sigmas) != len(manifest.t_list) or any(
            not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8)
            for a, b in zip(self.distilled_sigmas, manifest.t_list, strict=True)
        ):
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto scheduler and transformer manifests define different "
                f"fixed-step schedules: scheduler={self.distilled_sigmas}, "
                f"transformer={list(manifest.t_list)}."
            )

    def resolved_multiview_backend(self) -> str:
        """``triton``, ``fa4``, or ``auto`` (the device decides)."""
        return resolve_multiview_backend_request(self.multiview_attention_backend)[0]

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        # The student runs one branch per step, so CFG parallel would only
        # duplicate the rollout; the per-camera decode follows the rollout, so
        # the VAE stays resident next to the DiT.
        return replace(
            super().get_model_deployment_config(),
            auto_enable_cfg_parallel=False,
            supports_cfg_parallel=False,
            keep_resident_components=("dit", "vae"),
        )

    def validate_server_args(self, server_args: Any) -> None:
        super().validate_server_args(server_args)
        # Ulysses shards the chunk over ranks and trades tokens for heads around
        # every replay attention; the mask spans the whole chunk, so ring
        # attention and the K/V-gather exchange are not supported.
        for name, value in (
            ("ring_degree", server_args.ring_degree),
            ("kv_gather_degree", server_args.kv_gather_degree),
        ):
            if int(value or 1) > 1:
                raise ValueError(
                    f"{type(self).__name__} does not support {name} > 1; pass "
                    "--ulysses-degree explicitly for sequence parallelism."
                )
        ulysses = int(server_args.ulysses_degree or 1)
        sp_degree = int(server_args.sp_degree or 1)
        if sp_degree != ulysses:
            raise ValueError(
                f"{type(self).__name__} sequence parallelism is Ulysses only: sp_degree "
                f"({sp_degree}) must equal ulysses_degree ({ulysses}); pass "
                "--ulysses-degree explicitly (with two GPUs the launcher otherwise "
                "picks the K/V-gather exchange)."
            )
        if ulysses > 1 and self.sim_auto_heads is not None:
            tp_size = int(server_args.tp_size or 1)
            query_heads, kv_heads = self.sim_auto_heads
            if query_heads % (tp_size * ulysses) or kv_heads % (tp_size * ulysses):
                raise ValueError(
                    f"{type(self).__name__} query ({query_heads}) and key/value "
                    f"({kv_heads}) heads must divide by tp_size x ulysses_degree "
                    f"({tp_size} x {ulysses})."
                )

    def supports_action_endpoint(self) -> bool:
        return False

    def supports_dynamic_batching(self) -> bool:
        return False


def register():
    from sglang.multimodal_gen.configs.sample.cosmos3_sim_auto import (
        Cosmos3SimAutoSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    # The hf path must sort before the shorter "nvidia/Cosmos3-Nano" in the
    # partial-path match, so local folders named after the release resolve here.
    register_configs(
        sampling_param_cls=Cosmos3SimAutoSamplingParams,
        pipeline_config_cls=Cosmos3SimAutoConfig,
        hf_model_paths=["nvidia/Cosmos3-Nano-Sim-Auto"],
        # Matches the ``Cosmos3NanoSimAutoPipeline`` ``_class_name`` of the export.
        model_detectors=[lambda hf_id: "cosmos3nanosimautopipeline" in hf_id.lower()],
    )

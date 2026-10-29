# SPDX-License-Identifier: Apache-2.0
"""Physis-AI H3 same-step checkpoint on SGLang's native encoder and DiT."""

import sys
import time
from contextlib import ExitStack, contextmanager, nullcontext

import torch
import torch.distributed as dist

from sglang.multimodal_gen.configs.pipeline_configs.physis_h3 import (
    PhysisH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.physis_h3 import (
    PhysisH3SamplingParams,
    validate_physis_h3_sampling,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed.causal_pipeline import CausalPipelinePlan
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_world_group,
    get_world_rank,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.physis_denoising import (
    PhysisH3Denoiser,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
    minimax_h3_text_only_ids,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_video_latent_t,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import (
    autocast_context,
    resolve_decode_precision,
)

logger = init_logger(__name__)


def _raise_if_any_rank_failed(error):
    """Agree on local failures before posting the next device transfer."""
    world = get_world_group()
    rank = world.rank_in_group
    failed = torch.tensor(
        rank if error is not None else world.world_size, dtype=torch.int64
    )
    dist.all_reduce(failed, op=dist.ReduceOp.MIN, group=world.cpu_group)
    source = int(failed.item())
    if source < world.world_size:
        message = world.broadcast_object(error if rank == source else None, src=source)
        raise RuntimeError(f"Physis H3 rank {source} failed: {message}")


def _distribute_tensor(tensor, *, src, dst=None):
    """Allocate all receivers before a broadcast or paired device transfer."""
    world = get_world_group()
    rank = world.rank_in_group
    metadata = (tuple(tensor.shape), tensor.dtype) if rank == src else None
    shape, dtype = world.broadcast_object(metadata, src=src)
    error = None
    buffer = None
    try:
        if rank == src:
            buffer = tensor.contiguous()
        elif dst is None or rank == dst:
            buffer = torch.empty(shape, dtype=dtype, device="cuda")
        torch.cuda.synchronize()
    except Exception as exc:
        error = str(exc)
    _raise_if_any_rank_failed(error)
    if dst is None:
        dist.broadcast(buffer, src=world.ranks[src], group=world.device_group)
    elif rank in (src, dst):
        operation = dist.isend if rank == src else dist.irecv
        peer = dst if rank == src else src
        for work in dist.batch_isend_irecv(
            [dist.P2POp(operation, buffer, world.ranks[peer], world.device_group)]
        ):
            work.wait()
    return buffer if dst is None or rank == dst else None


@contextmanager
def _synchronized_component_scope(scope):
    stack = ExitStack()
    try:
        error = None
        try:
            stack.enter_context(scope)
        except Exception as exc:
            error = str(exc)
        _raise_if_any_rank_failed(error)
        yield
    finally:
        error = None
        try:
            stack.__exit__(*sys.exc_info())
        except Exception as exc:
            error = str(exc)
        _raise_if_any_rank_failed(error)


class PhysisH3TextStage(PipelineStage):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def component_uses(self, server_args, stage_name=None):
        if self.pipeline.get_module("text_encoder") is None:
            return []
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "text_encoder",
                start_at_stage_entry=False,
            )
        ]

    @torch.no_grad()
    def forward(self, batch, server_args):
        # Validate on every rank before entering any collective.
        if not isinstance(batch.prompt, str) or not batch.prompt.strip():
            raise ValueError("Physis H3 requires one non-empty text prompt")
        validate_physis_h3_sampling(batch)
        encoder = self.pipeline.get_module("text_encoder")
        embeddings = None
        error = None
        try:
            if encoder is not None:
                with self.use_declared_component(
                    component_name="text_encoder", module=encoder
                ):
                    ids = minimax_h3_text_only_ids(
                        self.pipeline.get_module("tokenizer"), batch.prompt
                    )
                    with set_forward_context(
                        current_timestep=0, attn_metadata=None, forward_batch=batch
                    ):
                        embeddings = encoder.encode_ids(ids).to("cuda")
            torch.cuda.synchronize()
        except Exception as exc:
            error = str(exc)
        _raise_if_any_rank_failed(error)
        embeddings = _distribute_tensor(embeddings, src=self.pipeline.decoder_rank)
        batch.prompt_embeds = [embeddings]
        return batch


def _run_wavefront(runner, plan, rank, num_chunks, start_chunk=1):
    """All ranks, including the decoder, agree before each bounded transfer."""
    slices = plan.slices(rank) if rank < plan.degree else ()
    latent, hidden = None, None
    finished = []
    group = get_world_group().device_group
    for tick in range(num_chunks - start_chunk + plan.degree - 1):
        chunk = tick - rank + start_chunk
        active = rank < plan.degree and start_chunk <= chunk < num_chunks
        operations = []
        incoming_latent = incoming_hidden = None
        error = None
        try:
            if active:
                latent, hidden = runner.run_chunk(chunk, slices, latent, hidden)
                if rank == plan.degree - 1:
                    finished.append(latent)
            if active and rank < plan.degree - 1:
                operations.append(dist.P2POp(dist.isend, latent, rank + 1, group))
                if hidden is not None:
                    operations.append(dist.P2POp(dist.isend, hidden, rank + 1, group))
            if 0 < rank < plan.degree and start_chunk <= chunk + 1 < num_chunks:
                incoming_latent = torch.empty(
                    runner.shape, device=runner.device, dtype=torch.bfloat16
                )
                operations.append(
                    dist.P2POp(dist.irecv, incoming_latent, rank - 1, group)
                )
                if plan.slices(rank - 1)[-1].block_end != plan.num_blocks:
                    incoming_hidden = torch.empty(
                        runner.padded_tokens,
                        runner.model.arch.hidden_size,
                        device=runner.device,
                        dtype=torch.bfloat16,
                    )
                    operations.append(
                        dist.P2POp(dist.irecv, incoming_hidden, rank - 1, group)
                    )
            torch.cuda.synchronize()
        except Exception as exc:
            error = str(exc)
        _raise_if_any_rank_failed(error)
        if operations:
            for work in dist.batch_isend_irecv(operations):
                work.wait()
        latent, hidden = incoming_latent, incoming_hidden
    return finished


class PhysisH3DenoisingStage(PipelineStage):
    role_affinity = RoleType.DENOISER

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def component_uses(self, server_args, stage_name=None):
        if self.pipeline.get_module("transformer") is None:
            return []
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "transformer",
                start_at_stage_entry=False,
            )
        ]

    @torch.no_grad()
    def forward(self, batch, server_args):
        world = get_world_group()
        world.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        model = self.pipeline.get_module("transformer")
        parallel = server_args.causal_pipeline_mode != "off"
        rank = get_world_rank()
        num_chunks = minimax_h3_video_latent_t(batch.num_frames)
        num_blocks = None
        error = None
        try:
            if rank == 0:
                num_blocks = len(model.blocks)
        except Exception as exc:
            error = str(exc)
        _raise_if_any_rank_failed(error)
        num_blocks = world.broadcast_object(num_blocks, src=0)
        plan = CausalPipelinePlan(
            4,
            num_blocks,
            server_args.causal_pipeline_parallel_size if parallel else 1,
            server_args.causal_pipeline_mode if parallel else "timestep",
        )
        bootstrap_frames = min(
            server_args.pipeline_config.physis_bootstrap_chunks, num_chunks
        )
        runner = opening = None
        scope = (
            self.use_declared_component(component_name="transformer", module=model)
            if model is not None
            else nullcontext()
        )
        with (
            _synchronized_component_scope(scope),
            set_forward_context(
                current_timestep=0, attn_metadata=None, forward_batch=batch
            ),
        ):
            error = None
            try:
                if model is not None:
                    sp = get_sp_group()
                    runner = PhysisH3Denoiser(
                        model,
                        batch.prompt_embeds[0],
                        (
                            1,
                            model.arch.latents_dim,
                            1,
                            batch.height // 16,
                            batch.width // 16,
                        ),
                        batch.seed,
                        history=server_args.pipeline_config.physis_history_chunks,
                        sp_size=sp.world_size,
                        sp_rank=sp.rank_in_group,
                    )
                    opening = runner.bootstrap(
                        bootstrap_frames,
                        owned_slices=plan.slices(rank if parallel else 0),
                    )
                torch.cuda.synchronize()
            except Exception as exc:
                if not parallel:
                    raise
                error = str(exc)
            _raise_if_any_rank_failed(error)
            if parallel:
                chunks = _run_wavefront(
                    runner, plan, rank, num_chunks, start_chunk=bootstrap_frames
                )
                latents = None
                error = None
                try:
                    if rank == plan.degree - 1:
                        latents = torch.cat([opening, *chunks], dim=2)
                except Exception as exc:
                    error = str(exc)
                _raise_if_any_rank_failed(error)
                batch.latents = _distribute_tensor(
                    latents, src=plan.degree - 1, dst=self.pipeline.decoder_rank
                )
            else:
                chunks = [
                    runner.run_chunk(chunk, plan.slices(0))[0]
                    for chunk in range(bootstrap_frames, num_chunks)
                ]
                if rank == 0:
                    batch.latents = torch.cat([opening, *chunks], dim=2)
            del runner
        torch.cuda.synchronize()
        world.barrier()
        logger.info(
            "Physis H3 DiT: %.3f s, %d chunks, %s",
            time.perf_counter() - start,
            num_chunks,
            server_args.causal_pipeline_mode,
        )
        batch.prompt_embeds = []
        return batch


class PhysisH3DecodingStage(PipelineStage):
    role_affinity = RoleType.DECODER

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def component_uses(self, server_args, stage_name=None):
        if self.pipeline.get_module("vae") is None:
            return []
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "vae",
                target_dtype=torch.float32,
                start_at_stage_entry=False,
            )
        ]

    @torch.no_grad()
    def forward(self, batch, server_args):
        world = get_world_group()
        output = None
        vae = self.pipeline.get_module("vae")
        error = None
        try:
            if vae is not None:
                with self.use_declared_component(component_name="vae", module=vae):
                    latent = batch.latents.float()
                    mean = latent.new_tensor(vae.config.latents_mean).view(
                        1, -1, 1, 1, 1
                    )
                    std = latent.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
                    decode_dtype = resolve_decode_precision(server_args, "vae")
                    with autocast_context(decode_dtype, server_args.disable_autocast):
                        output = vae.decode(latent * std + mean).sample.float()
                    pixel_std = output.new_tensor((0.229, 0.224, 0.225)).view(
                        1, 3, 1, 1, 1
                    )
                    pixel_mean = output.new_tensor((0.485, 0.456, 0.406)).view(
                        1, 3, 1, 1, 1
                    )
                    output = (output * pixel_std + pixel_mean).clamp_(0, 1)
                    if output.shape[2] != batch.num_frames:
                        raise RuntimeError(
                            f"H3 VAE produced {output.shape[2]} frames, expected {batch.num_frames}"
                        )
            torch.cuda.synchronize()
        except Exception as exc:
            error = str(exc)
        _raise_if_any_rank_failed(error)
        if self.pipeline.decoder_rank != 0:
            output = _distribute_tensor(output, src=self.pipeline.decoder_rank, dst=0)
        batch.latents = None
        world.barrier()
        return OutputBatch(output=output, fps=batch.fps, metrics=batch.metrics)


class PhysisH3Pipeline(ComposedPipelineBase):
    pipeline_name = "PhysisH3Pipeline"
    pipeline_config_cls = PhysisH3PipelineConfig
    sampling_params_cls = PhysisH3SamplingParams
    is_video_pipeline = True
    _required_config_modules = ["text_encoder", "tokenizer", "transformer", "vae"]

    def __init__(self, model_path, server_args, **kwargs):
        import diffusers

        if not hasattr(diffusers, "AutoencoderKLMiniMaxH3"):
            raise RuntimeError("Physis H3 VAE requires diffusers>=0.40.0")
        rank = get_world_rank()
        parallel = server_args.causal_pipeline_mode != "off"
        self.decoder_rank = server_args.causal_pipeline_parallel_size if parallel else 0
        required = []
        if rank == self.decoder_rank:
            required.extend(("text_encoder", "tokenizer", "vae"))
        if not parallel or rank != self.decoder_rank:
            required.append("transformer")
        super().__init__(
            model_path, server_args, required_config_modules=required, **kwargs
        )

    def _load_config(self):
        config = super()._load_config()
        # Diffusers ModularPipeline adds a third, descriptive item per component.
        return {
            key: value[:2] if isinstance(value, list) else value
            for key, value in config.items()
        }

    def create_pipeline_stages(self, server_args):
        self.add_stage(PhysisH3TextStage(self))
        self.add_stage(PhysisH3DenoisingStage(self))
        self.add_stage(PhysisH3DecodingStage(self))


EntryClass = PhysisH3Pipeline

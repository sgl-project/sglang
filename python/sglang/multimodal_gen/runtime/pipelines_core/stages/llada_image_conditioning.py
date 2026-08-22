# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from array import array
from contextlib import nullcontext
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_parallel_rank,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def ensure_conditioning_mask_active(attn_backend) -> None:
    """Fail closed if the conditioning block mask did not reach the backend."""
    if not attn_backend.conditioning_mask_active:
        raise RuntimeError(
            "LLaDA-Image conditioning ran without its block attention mask "
            "because the forward batch lost the conditioning text lengths"
        )


def format_llada_image_prompt(prompt: str | None) -> str:
    if prompt is None:
        return "<role>HUMAN</role> Generate an image.\n<role>ASSISTANT</role>\n<IMAGE1>"
    return (
        f"<role>HUMAN</role> Generate an image: {prompt.strip()}\n"
        "<role>ASSISTANT</role>\n<IMAGE1>"
    )


class LLaDAImageTextEncoderRunner:
    """One-shot LLaDA2 prefill runner used to produce diffusion cap features."""

    def __init__(
        self,
        model_root: str,
        queryformer,
        text_projection,
        tokenizer,
        server_args: ServerArgs,
    ) -> None:
        import sglang.multimodal_gen.runtime.distributed.parallel_state as mm_parallel_state
        import sglang.srt.distributed.parallel_state as srt_parallel_state
        from sglang.srt.distributed.parallel_state_wrapper import ParallelState
        from sglang.srt.managers.tp_worker import TpModelWorker
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.chunk_cache import ChunkCache
        from sglang.srt.runtime_context import publish as srt_publish
        from sglang.srt.server_args import ServerArgs as SRTServerArgs

        # Keep the SRT world aligned with the pure-Ulysses ranks. Within that
        # world, SP ranks become data replicas so conditioning matches SP1.
        text_tp_size = int(server_args.sp_degree)
        text_tp_rank = get_sp_parallel_rank()
        text_dp_attention = text_tp_size > 1

        self.queryformer = queryformer
        self.text_projection = text_projection
        self.tokenizer = tokenizer
        text_encoder_path = str(Path(model_root) / "text_encoder")
        device = get_local_torch_device()
        gpu_id = device.index
        if gpu_id is None:
            gpu_id = int(os.environ.get("LOCAL_RANK", "0"))
        srt_args = SRTServerArgs(
            model_path=text_encoder_path,
            tokenizer_path=str(Path(model_root) / "tokenizer"),
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            tp_size=text_tp_size,
            dp_size=text_tp_size if text_dp_attention else 1,
            enable_dp_attention=text_dp_attention,
            enable_dp_lm_head=text_dp_attention,
            dcp_size=1,
            attn_cp_size=1,
            ep_size=1,
            moe_dp_size=text_tp_size if text_dp_attention else 1,
            moe_dense_tp_size=1 if text_dp_attention else None,
            moe_a2a_backend="none",
            pp_size=1,
            attention_backend="llada2_cfg_flashinfer",
            disable_cuda_graph=True,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            max_prefill_tokens=8192,
            max_total_tokens=8192,
            max_running_requests=4 if text_dp_attention else 2,
            mem_fraction_static=(
                server_args.pipeline_config.text_encoder_mem_fraction_static
            ),
        )
        # TpModelWorker reads the runtime_context config bags the srt
        # scheduler process publishes before construction.
        srt_publish(srt_args, role="scheduler")
        # The diffusion runtime mirrors its TP group into the srt globals.
        # Clear it so the worker can install the encoder's real srt groups,
        # then restore the mirror and keep the encoder group for forwards.
        saved_srt_tp = srt_parallel_state._TP
        saved_srt_attn_tp = srt_parallel_state._ATTN_TP
        try:
            mm_parallel_state._clear_srt_tp_group()
            self.worker = TpModelWorker(
                server_args=srt_args,
                gpu_id=gpu_id,
                ps=ParallelState.trivial(
                    tp_rank=text_tp_rank,
                    tp_size=text_tp_size,
                    dp_rank=text_tp_rank if text_dp_attention else 0,
                    dp_size=text_tp_size if text_dp_attention else 1,
                    attn_tp_rank=0 if text_dp_attention else text_tp_rank,
                    attn_tp_size=1 if text_dp_attention else text_tp_size,
                    attn_cp_rank=0,
                    attn_cp_size=1,
                    attn_dcp_rank=0,
                    attn_dcp_size=1,
                    attn_dp_rank=text_tp_rank if text_dp_attention else 0,
                    attn_dp_size=text_tp_size if text_dp_attention else 1,
                    moe_ep_rank=0,
                    moe_ep_size=1,
                    moe_dp_rank=text_tp_rank if text_dp_attention else 0,
                    moe_dp_size=text_tp_size if text_dp_attention else 1,
                    gpu_id=gpu_id,
                ),
                nccl_port=server_args.nccl_port or 29500,
            )
            # Run the post-construction init phases the srt scheduler drives.
            self.worker.alloc_memory_pool()
            self.worker.init_attention_backends()
            self.worker.init_cuda_graphs()
            self.encoder_tp_group = srt_parallel_state._TP
            self.encoder_attn_tp_group = srt_parallel_state._ATTN_TP
        finally:
            srt_parallel_state._TP = saved_srt_tp
            srt_parallel_state._ATTN_TP = saved_srt_attn_tp
        self.server_args = srt_args
        self.text_dp_attention = text_dp_attention
        self.model_runner = self.worker.model_runner
        self.page_size = self.model_runner.page_size
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.worker.get_memory_pool()
        )
        self.tree_cache = ChunkCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=self.page_size,
            )
        )

    def _prepare_input_embeds(
        self,
        input_ids: torch.Tensor,
        sequence_lengths: list[int],
        text_lengths: list[int],
        component_context=None,
    ) -> torch.Tensor:
        model = self.model_runner.model
        embeddings = model.get_input_embeddings()
        inputs = []
        offset = 0
        context = (
            nullcontext(self.queryformer)
            if component_context is None
            else component_context(
                component_name="queryformer", module=self.queryformer
            )
        )
        with context as queryformer:
            assert queryformer is not None
            queryformer_dtype = queryformer.dtype
            for sequence_length, text_length in zip(
                sequence_lengths, text_lengths, strict=True
            ):
                text_ids = input_ids[offset : offset + text_length]
                text_embeds = embeddings(text_ids)
                text_mask = torch.ones(
                    (1, text_length), dtype=torch.bool, device=text_embeds.device
                )
                query_embeds = queryformer(
                    text_embeds.unsqueeze(0).to(dtype=queryformer_dtype),
                    text_mask,
                ).query_embeds
                query_count = sequence_length - text_length
                if query_embeds.shape[1] != query_count:
                    raise RuntimeError(
                        f"QueryFormer returned {query_embeds.shape[1]} queries, "
                        f"expected {query_count}"
                    )
                inputs.append(
                    torch.cat(
                        [text_embeds, query_embeds.squeeze(0).to(text_embeds.dtype)],
                        dim=0,
                    )
                )
                offset += sequence_length
        if offset != input_ids.numel():
            raise ValueError(
                f"Conditioning spans {offset} tokens, got {input_ids.numel()}"
            )
        return torch.cat(inputs, dim=0)

    @torch.no_grad()
    def encode(
        self, prompts: list[str], max_sequence_length: int, component_context=None
    ):
        import sglang.multimodal_gen.runtime.distributed.parallel_state as mm_parallel_state
        import sglang.srt.distributed.parallel_state as srt_parallel_state

        # srt modules resolve their TP group through the srt globals, which
        # hold the diffusion mirror between encodes.
        with mm_parallel_state.use_tensor_parallel_group(self.encoder_tp_group):
            saved_attn_tp = srt_parallel_state._ATTN_TP
            try:
                srt_parallel_state._ATTN_TP = self.encoder_attn_tp_group
                return self._encode_impl(
                    prompts, max_sequence_length, component_context=component_context
                )
            finally:
                srt_parallel_state._ATTN_TP = saved_attn_tp

    def _encode_impl(
        self, prompts: list[str], max_sequence_length: int, component_context=None
    ):
        from sglang.srt.managers.overlap_utils import resolve_forward_inputs
        from sglang.srt.managers.schedule_batch import Req as SRTReq
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
        from sglang.srt.mem_cache.common import release_kv_cache
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.sampling.sampling_params import (
            SamplingParams as SRTSamplingParams,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        tokenized = self.tokenizer(
            prompts,
            add_special_tokens=True,
            padding=False,
            truncation=True,
            max_length=max_sequence_length,
        ).input_ids
        query_count = int(self.queryformer.config.num_queries)
        reqs = []
        text_lengths = []
        sequence_lengths = []
        for index, text_input_ids in enumerate(tokenized):
            sampling_params = SRTSamplingParams(max_new_tokens=1, temperature=0.0)
            sampling_params.normalize(None)
            sequence_ids = list(text_input_ids) + [0] * query_count
            req = SRTReq(
                rid=f"llada-image-cap-{index}",
                origin_input_text="",
                origin_input_ids=array("q", sequence_ids),
                sampling_params=sampling_params,
                vocab_size=self.worker.model_config.vocab_size,
                eos_token_ids=set(),
                return_hidden_states=True,
                dllm_config=None,
            )
            req.skip_radix_cache_insert = True
            reqs.append(req)
            text_lengths.append(len(text_input_ids))
            sequence_lengths.append(len(sequence_ids))

        chunked_prefill_size = self.server_args.chunked_prefill_size
        if chunked_prefill_size <= 0:
            chunked_prefill_size = None

        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            None,
            0.5,
            self.server_args.max_prefill_tokens,
            chunked_prefill_size,
            prefill_max_requests=len(reqs),
            dllm_config=None,
        )
        for req in reqs:
            req.init_next_round_input(self.tree_cache)
            result = adder.add_one_req(
                req, has_chunked_req=False, truncation_align_size=None
            )
            if req not in adder.can_run_list:
                raise RuntimeError(
                    "Insufficient prefill capacity for LLaDA-Image conditioning"
                )
            if result != AddReqResult.CONTINUE:
                break

        if len(adder.can_run_list) != len(reqs):
            raise RuntimeError("Could not batch all LLaDA-Image conditioning prompts")

        batch = ScheduleBatch.init_new(
            reqs=adder.can_run_list,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.worker.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            dllm_config=None,
        )
        if self.text_dp_attention:
            token_count = sum(sequence_lengths)
            batch.global_num_tokens = [token_count] * self.server_args.dp_size
            batch.global_num_tokens_for_logprob = [len(reqs)] * (
                self.server_args.dp_size
            )
        try:
            batch.prepare_for_extend()
            batch.forward_mode = ForwardMode.EXTEND
            # The FutureMap argument is only consulted on decode/spec paths.
            resolve_forward_inputs(batch, None)
            forward_batch = ForwardBatch.init_new(
                batch,
                self.model_runner,
                return_hidden_states_before_norm=False,
            )
            forward_batch.input_embeds = self._prepare_input_embeds(
                forward_batch.input_ids,
                sequence_lengths,
                text_lengths,
                component_context=component_context,
            )
            forward_batch.llada_image_conditioning_text_lens_cpu = text_lengths
            model_output = self.model_runner.forward(forward_batch=forward_batch)
            ensure_conditioning_mask_active(self.model_runner.attn_backend)
            hidden_states = model_output.logits_output.hidden_states
            if not isinstance(hidden_states, torch.Tensor):
                raise TypeError(
                    "SGLang did not return full LLaDA-Image conditioning states"
                )

            projected = []
            offset = 0
            context = (
                nullcontext(self.text_projection)
                if component_context is None
                else component_context(
                    component_name="text_projection", module=self.text_projection
                )
            )
            with context as text_projection:
                assert text_projection is not None
                projection_dtype = text_projection.dtype
                for sequence_length in sequence_lengths:
                    output = text_projection(
                        hidden_states[offset : offset + sequence_length]
                        .unsqueeze(0)
                        .to(dtype=projection_dtype)
                    ).hidden_states
                    projected.append(output.squeeze(0))
                    offset += sequence_length
            return projected
        finally:
            for req in reqs:
                if req.kv is not None:
                    release_kv_cache(req, self.tree_cache, is_insert=False)
                elif req.req_pool_idx is not None:
                    # KV alloc failed after the slot grab. Freeing the bare
                    # slot avoids the release_kv_cache lifecycle assert.
                    self.req_to_token_pool.free(req)


class LLaDAImageTextConditioningStage(PipelineStage):
    def __init__(self, runner: LLaDAImageTextEncoderRunner) -> None:
        super().__init__()
        self.runner = runner

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        del server_args
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(stage_name, "queryformer"),
            ComponentUse(stage_name, "text_projection"),
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not isinstance(batch.prompt, str):
            raise TypeError("LLaDA-Image currently supports one prompt per request")
        do_cfg = batch.guidance_scale > 1.0
        prompts = [format_llada_image_prompt(batch.prompt)]
        if do_cfg:
            prompts.append(format_llada_image_prompt(batch.negative_prompt))

        outputs = self.runner.encode(
            prompts,
            max_sequence_length=batch.max_sequence_length or 2048,
            component_context=self.use_declared_component,
        )
        masks = [
            torch.ones(output.shape[0], dtype=torch.bool, device=output.device)
            for output in outputs
        ]
        output_count = batch.num_outputs_per_prompt
        batch.prompt_embeds = [outputs[0]] * output_count
        batch.prompt_attention_mask = [masks[0]] * output_count
        batch.do_classifier_free_guidance = do_cfg
        if do_cfg:
            batch.negative_prompt_embeds = [outputs[1]] * output_count
            batch.negative_attention_mask = [masks[1]] * output_count
        else:
            batch.negative_prompt_embeds = []
            batch.negative_attention_mask = []
        return batch

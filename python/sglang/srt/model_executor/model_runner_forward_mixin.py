# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ModelRunner mixin extracted from model_runner.py.

This module is a behavior-preserving file split; methods keep their original bodies.
"""

from __future__ import annotations

from sglang.srt.model_executor.model_runner_common import *


class ModelRunnerForwardMixin:
    def init_threads_binding(self):
        omp_cpuids = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", "all")
        cpu_ids_by_node = get_cpu_ids_by_node()
        n_numa_node = len(cpu_ids_by_node)
        if omp_cpuids == "all":
            assert self.tp_size <= n_numa_node, (
                f"SGLANG_CPU_OMP_THREADS_BIND is not set, in this case, "
                f"tp_size {self.tp_size} should be smaller than or equal to number of numa node on the machine {n_numa_node}. "
                f"If you need tp_size to be larger than number of numa node, please set the CPU cores for each tp rank via SGLANG_CPU_OMP_THREADS_BIND explicitly. "
                f"For example, on a machine with 2 numa nodes, where core 0-31 are on numa node 0 and core 32-63 are on numa node 1, "
                f"it is suggested to use -tp 2 and bind tp rank 0 to core 0-31 and tp rank 1 to core 32-63. "
                f"This is the default behavior if SGLANG_CPU_OMP_THREADS_BIND is not set and it is the same as setting SGLANG_CPU_OMP_THREADS_BIND=0-31|32-63. "
                f"If you do need tp_size to be larger than the number of numa nodes, you could set SGLANG_CPU_OMP_THREADS_BIND explicitly for example SGLANG_CPU_OMP_THREADS_BIND=0-15|16-31|32-47|48-63 and run with -tp 4. "
                f"If you don't want each tp rank to use all the cores on one numa node, you could set for example SGLANG_CPU_OMP_THREADS_BIND=0-15|32-47 and run with -tp 2."
            )
            if self.tp_size < n_numa_node:
                logger.warning(
                    f"Detected the current machine has {n_numa_node} numa nodes available, but tp_size is set to {self.tp_size}, so only {self.tp_size} numa nodes are used."
                )
            self.local_omp_cpuid = cpu_ids_by_node[self.tp_rank]
        else:
            threads_bind_list = omp_cpuids.split("|")
            assert self.tp_size == len(threads_bind_list), (
                f"SGLANG_CPU_OMP_THREADS_BIND setting must be aligned with TP size parameter ({self.tp_size}). "
                f"Please double check your settings."
            )
            self.local_omp_cpuid = threads_bind_list[self.tp_rank]
            if self.tp_size > n_numa_node:
                logger.warning(
                    f"TP size ({self.tp_size})is larger than numa node number ({n_numa_node}), "
                    f"in this case the available memory amount of each rank cannot be determined in prior. "
                    f"Please set proper `--max-total-tokens` to avoid the out-of-memory error."
                )

    def apply_torch_tp(self):
        logger.info(f"Enabling torch tensor parallelism on {self.tp_size} devices.")
        from sglang.srt.layers.model_parallel import tensor_parallel

        device_mesh = torch.distributed.init_device_mesh(self.device, (self.tp_size,))
        tensor_parallel(self.model, device_mesh)

    def update_decode_attn_backend(self, stream_idx: int):
        self.decode_attn_backend = self.decode_attn_backend_group[stream_idx]

    def _ensure_eager_registry(
        self,
        cache: _EagerBufferRegistry,
        raw_bs: int,
        raw_num_tokens: int,
        build: Callable[[int, int], CudaGraphBufferRegistry],
    ) -> CudaGraphBufferRegistry:
        # Built on first use and grown (next power of two) when a batch exceeds
        # the current capacity.
        if (
            cache.registry is not None
            and raw_bs <= cache.max_bs
            and raw_num_tokens <= cache.max_num_tokens
        ):
            return cache.registry
        cache.max_bs = next_power_of_2(max(raw_bs, cache.max_bs))
        cache.max_num_tokens = next_power_of_2(
            max(raw_num_tokens, cache.max_num_tokens)
        )
        cache.registry = build(cache.max_bs, cache.max_num_tokens)
        return cache.registry

    def _ensure_eager_decode_registry(
        self, raw_bs: int, raw_num_tokens: int
    ) -> CudaGraphBufferRegistry:
        is_encoder_decoder = self.model_config.is_encoder_decoder
        return self._ensure_eager_registry(
            self._eager_decode_registry,
            raw_bs,
            raw_num_tokens,
            lambda bs, num_tokens: build_decode_registry(
                device=self.device,
                max_bs=bs,
                max_num_token=num_tokens,
                # Eager has no padding so this sentinel is never read; 0 avoids the
                # cuda-graph-only fill-value method that some backends lack.
                seq_len_fill_value=0,
                cache_loc_dtype=torch.int64,
                enable_mamba_track=(
                    self.server_args.enable_mamba_extra_buffer()
                    and self.spec_algorithm.is_none()
                ),
                is_encoder_decoder=is_encoder_decoder,
                encoder_len_fill_value=(
                    getattr(self.model_config.hf_config, "max_source_positions", 0)
                    if is_encoder_decoder
                    else 0
                ),
                enable_num_token_non_padded=False,
                register_global_num_tokens=False,
                require_gathered_buffer=False,
                require_mlp_tp_gather=False,
                dp_size=self.server_args.dp_size,
                share_pool=False,
                source=None,
            ),
        )

    def _ensure_eager_prefill_registry(
        self, raw_bs: int, raw_num_tokens: int
    ) -> CudaGraphBufferRegistry:
        return self._ensure_eager_registry(
            self._eager_prefill_registry,
            raw_bs,
            raw_num_tokens,
            lambda bs, num_tokens: build_prefill_registry(
                device=self.device,
                max_bs=bs,
                max_num_token=num_tokens,
                cache_loc_dtype=torch.int64,
                is_multimodal=self.is_multimodal,
                enable_mamba_track=False,
                register_input_embeds=False,
                share_pool=False,
                source=None,
            ),
        )

    def _eager_fb_view(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> ForwardBatch:
        if envs.SGLANG_EAGER_INPUT_NO_COPY.get():
            return replace(forward_batch)
        raw_bs = forward_batch.batch_size
        raw_num_tokens = forward_batch.input_ids.shape[0]
        ensure = (
            self._ensure_eager_prefill_registry
            if forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
            else self._ensure_eager_decode_registry
        )
        registry = ensure(raw_bs, raw_num_tokens)
        registry.fill_from(
            forward_batch,
            raw_bs=raw_bs,
            padded_bs=raw_bs,
            raw_num_tokens=raw_num_tokens,
            padded_num_tokens=raw_num_tokens,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return registry.extract_buffer(
            padded_bs=raw_bs,
            padded_num_tokens=raw_num_tokens,
            forward_batch_template=forward_batch,
        )

    def _prepare_eager_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Pad / normalize a batch for the eager (non-cuda-graph) forward.

        Runs the DP/MLP-sync padding, the attn-tp num_token_non_padded
        normalization, and the hisparse-coordinator refresh that the eager
        forward path needs — the cuda-graph path does the equivalent inside the
        runner's capture/replay, so this is skipped there.
        """
        if forward_batch.global_num_tokens_cpu is not None:
            forward_batch.prepare_mlp_sync_batch(self)
        else:
            forward_batch.prepare_attn_tp_scatter_input(self)

        # Normalize num_token_non_padded to be local to this attention TP rank if needed.
        # The skip is scoped to DSACPLayerCommunicator-style CP (DSA, MLA): those
        # flavors already feed a zigzag-split rank-local layout whose token count
        # should not be further divided by attn_tp_size. MHA-arch prefill CP
        # (Qwen3/Qwen2 MoE) keeps the attn_tp-replicated layout and wants the
        # adjustment to run — see docs/design/prefill-cp-mla.md §Phase 5.
        if (
            forward_batch.num_token_non_padded is not None
            and forward_batch.global_num_tokens_gpu is not None
            and require_gathered_buffer(self.server_args)
            and not is_dsa_enable_prefill_cp()
            and not is_mla_prefill_cp_enabled()
        ):
            forward_batch.adjust_num_token_non_padded_for_attn_tp(
                server_args=self.server_args,
            )

        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

    def _pp_kwargs(self, pp_proxy_tensors) -> dict:
        """Build the pp_proxy_tensors forward kwarg, in one place.

        Pipeline-parallel proxy tensors are threaded into model.forward only
        when the model accepts them (``support_pp``).
        """
        return {"pp_proxy_tensors": pp_proxy_tensors} if self.support_pp else {}

    def forward_decode(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        if not self.server_args.enable_pdmux:
            forward_batch = self._eager_fb_view(forward_batch, pp_proxy_tensors)
        # Set extra arguments
        pdmux_override = False
        if forward_batch.needs_forward_metadata_init():
            if hasattr(self.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                self.model.prepare_forward_batch(forward_batch)
            if self.server_args.enable_pdmux:
                self.decode_attn_backend.init_forward_metadata(forward_batch)
                # PDmux selects a per-stream backend; publish it to model-layer
                # readers via the active ForwardContext so RadixAttention etc.
                # dispatch against the right backend for this forward.
                pdmux_override = True
            else:
                self.attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = self._pp_kwargs(pp_proxy_tensors)

        # Launch forward
        ctx = (
            self.device_timer.wrap(metadata={"category": "decode"})
            if self.device_timer
            else contextlib.nullcontext()
        )

        def _do_forward():
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

        with ctx:
            if pdmux_override:
                with forward_context(
                    ForwardContext(attn_backend=self.decode_attn_backend)
                ):
                    return _do_forward()
            return _do_forward()

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors=None,
    ) -> Tuple[
        Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput], bool
    ]:
        # Setup extra arguments
        kwargs = self._pp_kwargs(pp_proxy_tensors)
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if (
            forward_batch.replace_embeds is not None
            and forward_batch.replace_positions is not None
        ):
            # Token embedding overrides: get base embeddings, scatter replacements
            if "input_embeds" not in kwargs:
                embed_layer = self.model.get_input_embeddings()
                kwargs["input_embeds"] = embed_layer(forward_batch.input_ids)
            kwargs["input_embeds"][forward_batch.replace_positions] = (
                forward_batch.replace_embeds.to(kwargs["input_embeds"].dtype)
            )
        if not self.is_generation:
            kwargs["get_embedding"] = True

        # Check piecewies cuda graph
        can_run_graph = (
            self.prefill_cuda_graph_runner is not None
            and self.prefill_cuda_graph_runner.can_run(forward_batch)
        )
        if get_cp_strategy() is not None:
            can_run_graph = False
        if can_run_graph:
            # TODO: device_timer.wrap is too broad here — it also includes
            # replay_prepare time. Move timing into the prefill cuda graph
            # runner to capture only the model.forward part.
            ctx = (
                self.device_timer.wrap(metadata={"category": "extend"})
                if self.device_timer
                else contextlib.nullcontext()
            )
            with ctx:
                ret = self.prefill_cuda_graph_runner.replay(forward_batch, **kwargs)
            return (ret, can_run_graph)

        if not self.server_args.enable_pdmux:
            forward_batch = self._eager_fb_view(forward_batch, pp_proxy_tensors)

        # Launch model forward
        if forward_batch.needs_forward_metadata_init():
            if hasattr(self.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                self.model.prepare_forward_batch(forward_batch)
            self.attn_backend.init_forward_metadata(forward_batch)
        cp_v2_active = is_cp_v2_active(forward_batch)
        forward_positions = forward_batch.positions
        if cp_v2_active:
            prepare_cp_forward(forward_batch)
            complete_hidden_states = kwargs.get("input_embeds")
            if complete_hidden_states is None:
                embed_layer = self.model.get_input_embeddings()
                complete_hidden_states = embed_layer(forward_batch.input_ids)
            sharded_hidden_states, sharded_positions = cp_split_before_forward(
                complete_hidden_states,
                forward_batch.positions,
                forward_batch,
            )
            kwargs["input_embeds"] = sharded_hidden_states
            forward_positions = sharded_positions

        ctx = (
            self.device_timer.wrap(metadata={"category": "extend"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            if (
                _is_hip
                and self.prefill_cuda_graph_runner is not None
                and not cp_v2_active
            ):
                # AMD/HIP: when PCG is enabled but the batch exceeds max captured
                # size, run eagerly under enable_tc_piecewise_cuda_graph() and
                # set_tc_piecewise_forward_context() so that (a) Dynamo guards on
                # _in_tc_piecewise_cuda_graph stay consistent with the PCG-traced
                # graph (preventing runtime recompilation) and (b) PCG-specific
                # code paths (MoE, attention) can access their layer objects.
                with (
                    enable_tc_piecewise_cuda_graph(),
                    set_tc_piecewise_forward_context(
                        forward_batch,
                        self.attention_layers,
                        getattr(self.model, "quant_config", None),
                        self.moe_layers,
                        self.moe_fusions,
                        dsa_indexers=self.dsa_indexers,
                    ),
                ):
                    ret = self.model.forward(
                        forward_batch.input_ids,
                        forward_positions,
                        forward_batch,
                        **kwargs,
                    )
            elif cp_v2_active:
                hidden_states = self.model.model(
                    forward_batch.input_ids,
                    forward_positions,
                    forward_batch,
                    input_embeds=kwargs.get("input_embeds"),
                    pp_proxy_tensors=kwargs.get("pp_proxy_tensors"),
                )

                aux_hidden_states = None
                capture_aux_hidden_states = getattr(
                    self.model, "capture_aux_hidden_states", False
                )
                if capture_aux_hidden_states:
                    hidden_states, aux_hidden_states = hidden_states

                if self.model.pp_group.is_last_rank:
                    hidden_states = cp_gather_after_forward(
                        hidden_states,
                        forward_batch,
                        torch.cuda.current_stream(),
                    )
                    ret = self.model.logits_processor(
                        forward_batch.input_ids,
                        hidden_states,
                        self.model.lm_head,
                        forward_batch,
                        aux_hidden_states,
                    )
                elif capture_aux_hidden_states:
                    ret = hidden_states, aux_hidden_states
                else:
                    ret = hidden_states
            else:
                ret = self.model.forward(
                    forward_batch.input_ids,
                    forward_positions,
                    forward_batch,
                    **kwargs,
                )
        return (ret, can_run_graph)

    def forward_idle(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        # In DP Attention, IDLE batches may be padded (batch_size > 0) for MLP
        # sync. Reinit metadata for the padded case so attention kernels see
        # the right batch_size (e.g. DSA Indexer). For the unpadded case
        # (batch_size == 0) explicitly drop any stale forward_metadata left
        # over from the previous forward — without this, attention layers
        # called from the idle path can re-read a prior batch's req_pool
        # indices and trigger SWA mapping use-after-free.
        if forward_batch.batch_size > 0:
            if not self.server_args.enable_pdmux:
                forward_batch = self._eager_fb_view(forward_batch, pp_proxy_tensors)
            self.attn_backend.init_forward_metadata(forward_batch)
        else:
            self.attn_backend.forward_metadata = None

        kwargs = self._pp_kwargs(pp_proxy_tensors)
        ctx = (
            self.device_timer.wrap(metadata={"category": "idle"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

    def forward_split_prefill(
        self,
        forward_batch: ForwardBatch,
        reinit_attn_backend: bool = False,
        forward_count: int = 1,
    ) -> LogitsProcessorOutput:
        if forward_batch.split_index == 0 or reinit_attn_backend:
            self.attn_backend.init_forward_metadata(forward_batch)
        next_split_index = min(
            forward_batch.split_index + forward_count,
            self.model_config.num_hidden_layers,
        )
        ctx = (
            self.device_timer.wrap(metadata={"category": "split_prefill"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            ret = self.model.forward_split_prefill(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                (forward_batch.split_index, next_split_index),
            )
        forward_batch.split_index = next_split_index
        return ret

    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: Optional[bool] = None,  # deprecated
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # Deprecated kwarg: pre-planners mark the batch themselves now.
        forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)

        self.forward_pass_id += 1

        # Try msprob debugger
        if self.msprobe_debugger is not None:
            rank_id = (
                self.gpu_id if self.dp_size is not None and self.dp_size > 1 else None
            )
            self.msprobe_debugger.start(model=self.model, rank_id=rank_id)

        # Step span
        step_span_ctx = profile_range(_build_step_span_name(forward_batch))

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if not self.is_draft_worker and ((c := self.canary_manager) is not None)
            else contextlib.nullcontext()
        )

        with (
            canary_ctx,
            step_span_ctx,
            get_global_expert_distribution_recorder().with_forward_pass(
                self.forward_pass_id,
                forward_batch,
            ) as recorder_outputs,
        ):
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            if self.enable_elastic_ep:
                output = self._maybe_rebalance_after_rank_fault(
                    output,
                    forward_batch,
                    pp_proxy_tensors,
                    reinit_attn_backend,
                    split_forward_count,
                )
        output.expert_distribution_metrics = recorder_outputs.get("metrics")

        no_copy_to_cpu = not self.server_args.disable_overlap_schedule
        if (experts_capturer := get_global_experts_capturer()) is not None:
            output.routed_experts_output = experts_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.decode_cuda_graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if (indexer_capturer := get_global_indexer_capturer()) is not None:
            output.indexer_topk_output = indexer_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.decode_cuda_graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()

        if dumper.may_enable:
            dumper.step()

        if self.msprobe_debugger is not None:
            self.msprobe_debugger.stop()
            self.msprobe_debugger.step()

        if self.server_args.elastic_ep_backend is not None:
            self.maybe_recover_ep_ranks()

        return output

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        if has_forward_context():
            ctx_mgr = contextlib.nullcontext()
        else:
            ctx_mgr = forward_context(ForwardContext(attn_backend=self.attn_backend))
        with ctx_mgr:
            mode_check = (
                forward_batch.forward_mode.is_cpu_graph
                if self.device == "cpu"
                else forward_batch.forward_mode.is_cuda_graph
            )
            can_run_graph = bool(
                mode_check()
                and self.decode_cuda_graph_runner
                and self.decode_cuda_graph_runner.can_run(forward_batch)
            )

            if (
                forward_batch.forward_mode.is_decode()
                and self.hisparse_coordinator is not None
            ):
                forward_batch.hisparse_coordinator = self.hisparse_coordinator
                self.hisparse_coordinator.wait_for_pending_backup()
                self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

            # Replay cuda graph if applicable
            if can_run_graph:
                ret = self.decode_cuda_graph_runner.replay(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

            # DP / MLP-sync padding + attn-tp normalization that the eager
            # (non-graph) forward needs. The graph path skips it: capture/replay
            # pads inside the runner.
            self._prepare_eager_forward_batch(forward_batch)

            # Forward without cuda graph
            if forward_batch.forward_mode.is_decode():
                ret = self.forward_decode(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            elif forward_batch.forward_mode.is_split_prefill():
                ret = self.forward_split_prefill(
                    forward_batch,
                    reinit_attn_backend=reinit_attn_backend,
                    forward_count=split_forward_count,
                )
            elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
                ret, can_run_graph = self.forward_extend(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            elif forward_batch.forward_mode.is_idle():
                ret = self.forward_idle(
                    forward_batch, pp_proxy_tensors=pp_proxy_tensors
                )
            else:
                raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

            if (
                forward_batch.global_num_tokens_cpu is not None
                and self.pp_group.is_last_rank
            ):
                forward_batch.post_forward_mlp_sync_batch(ret)

            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # NOTE: In overlap mode, the function update_regex_vocab_mask (in sample)
        #       was executed after we processed last batch's results.

        # Calculate logits bias and apply it to next_token_logits.
        sampling_info.update_regex_vocab_mask()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

        # Release the vocab_mask GPU tensor immediately after it has been applied
        # to the logits. In overlap scheduling, the sampling_info (and its
        # vocab_mask) can be kept alive by the delay_sample_func closure and
        # batch_record_buf until the next iteration, causing a steady VRAM leak
        # when structured output (grammar) is used.
        sampling_info.vocab_mask = None

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            # For prefill, we only use the position of the last token.
            (
                forward_batch.positions
                if forward_batch.forward_mode.is_decode()
                else forward_batch.seq_lens - 1
            ),
        )
        self.maybe_update_ngram_token_table(next_token_ids, forward_batch)
        return next_token_ids

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> None:
        """
        Compute token_ids_logprobs without performing sampling.

        Optimized path for prefill-only requests that need token_ids_logprobs but don't
        require next token generation. Skips expensive sampling operations
        while still providing requested probability information.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
        """
        if not forward_batch.token_ids_logprobs:
            return

        # Preprocess logits (same as in sample method)
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Delegate to sampler for logprob-only computation
        # This populates logits_output with requested token probabilities
        self.sampler.compute_logprobs_only(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )

    def save_remote_model(self, url: str):
        from sglang.srt.model_loader.loader import RemoteModelLoader

        logger.info(f"Saving model to {url}")
        RemoteModelLoader.save_model(self.model, self.model_config.model_path, url)

    def save_sharded_model(
        self, path: str, pattern: Optional[str] = None, max_size: Optional[int] = None
    ):
        from sglang.srt.model_loader.loader import ShardedStateLoader

        logger.info(
            f"Save sharded model to {path} with pattern {pattern} and max_size {max_size}"
        )
        ShardedStateLoader.save_model(self.model, path, pattern, max_size)

    def check_weights(self, action: str):
        return self._weight_checker.handle(action=action)

    def update_weights_from_ipc(self, recv_req):
        """Update weights from IPC for checkpoint-engine integration."""
        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self)
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)

    def prealloc_symmetric_memory_pool(self):
        # PyTorch mempools never de-fragment memory in OOM scenarios, so we need to pre-allocate a large chunk of memory to limit fragmentation.
        if (
            self.is_draft_worker
            or not self.server_args.enable_symm_mem
            or envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() <= 0
        ):
            return

        # Memory allocation is tied to a cuda stream, use the forward stream
        with torch.get_device_module(self.device).stream(self.forward_stream):
            logger.info(
                f"Pre-allocating symmetric memory pool with {envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get()} GiB"
            )
            with use_symmetric_memory(get_tp_group()):
                torch.empty(
                    (envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() * 1024 * 1024 * 1024,),
                    dtype=torch.uint8,
                    device=self.device,
                )

    def _maybe_rebalance_after_rank_fault(
        self,
        output: ModelRunnerOutput,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool,
        split_forward_count: int,
    ) -> ModelRunnerOutput:
        elastic_ep_state = ElasticEPStateManager.instance()
        if elastic_ep_state is not None and not elastic_ep_state.is_active_equal_last():
            elastic_ep_state.snapshot_active_to_last()
            elastic_ep_state.sync_active_to_cpu()
            logging.info("EPLB due to rank faults")
            gen = self.eplb_manager.rebalance()
            while True:
                try:
                    next(gen)
                except StopIteration:
                    break
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
        return output



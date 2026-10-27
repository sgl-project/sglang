"""NPUCudaGraphBackend — Ascend NPU full-graph capture (torch.npu.NPUGraph).

Mirrors FullCudaGraphBackend with two differences:
  - Captures via torch.npu.graph(...) into torch.npu.NPUGraph.
  - replay_with_input_update(shape_key, seq_lens, attr_name) rebinds
    the recorded graph's input bindings for variable seq_lens at replay
    time (NPU's NPUGraph.update(...) API).

torch.npu is imported lazily inside methods so the module loads on
non-NPU hosts.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np
import torch

from sglang.srt.configs.model_config import is_deepseek_dsa, is_deepseek_v4
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import empty_context, get_bool_env_var, get_compiler_backend
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


logger = logging.getLogger(__name__)


class NPUCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.npu.NPUGraph per shape; attention metadata captured
    inside the graph. replay_with_input_update substitutes fresh
    seq_lens without re-recording."""

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
    ) -> None:
        self._decode_runner = cuda_graph_runner
        self._graphs: Dict[Any, Any] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._device_id = self._device_module.current_device()
        self._tp_group = get_parallel().tp_group
        self._capture_stream = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        self._enable_torch_compile = getattr(
            cuda_graph_runner, "enable_torch_compile", False
        )
        # Reuse one device-bound worker for graph input updates.
        self._update_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="npu-graph-update",
            initializer=self._device_module.set_device,
            initargs=(self._device_id,),
        )
        # Event of the last replay; the next rebind waits on it so the device is
        # done reading the host-side seq_lens array before we rewrite it.
        self._rebind_fence = None

    @staticmethod
    def cuda_graph_disabled() -> bool:
        """Preserve the legacy multi-layer NPU graph initialization guard."""
        return cuda_graph_fully_disabled()

    @property
    def supports_shared_read_events(self) -> bool:
        # Preserve the legacy NPUGraphRunner replay behavior.
        return False

    @property
    def cache_loc_dtype(self):
        return torch.int32

    def get_update_attr_name(self):
        """Preserve each runner's existing captured attention update protocol."""
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )
        from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
            EAGLEDraftCudaGraphRunner,
        )
        from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
            EAGLEDraftExtendCudaGraphRunner,
        )
        from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
            MultiLayerEagleDraftExtendCudaGraphRunner,
        )

        # FIA V1 uses actual_seq_lengths_kv; FIA V2 uses actual_seq_kvlen.
        # Check specialized runners before their common decode base class.
        runner_update_attrs = (
            (MultiLayerEagleDraftExtendCudaGraphRunner, "actual_seq_kvlen"),
            (EAGLEDraftExtendCudaGraphRunner, "actual_seq_lengths_kv"),
            (EAGLEDraftCudaGraphRunner, "actual_seq_lengths_kv"),
        )
        for runner_type, attr_name in runner_update_attrs:
            if isinstance(self._decode_runner, runner_type):
                return attr_name
        if isinstance(self._decode_runner, DecodeCudaGraphRunner):
            return self._decode_update_attr_name()
        raise TypeError(
            f"Unsupported NPU graph update runner: {type(self._decode_runner).__name__}"
        )

    def _decode_update_attr_name(self):
        runner = self._decode_runner
        architectures = runner.model_runner.model_config.hf_config.architectures or []
        use_v2 = any(
            arch
            in ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM", "Step3p5ForCausalLM")
            for arch in architectures
        ) or (
            envs.SGLANG_NPU_USE_FIAS_V2_BSND.get()
            and runner.model_runner.spec_algorithm.is_dspark()
            and runner.capture_forward_mode.is_target_verify()
        )
        return "actual_seq_kvlen" if use_v2 else "actual_seq_lengths_kv"

    def _decode_update_lengths(self, forward_batch):
        runner = self._decode_runner
        if forward_batch.forward_mode.is_target_verify():
            # Only DFlash refreshes forward_metadata.seq_lens_cpu_list at
            # replay; other algorithms keep the capture-time list, so
            # recompute from the live batch as before — reading the
            # backend list there would replay stale capture-time lengths.
            if runner.model_runner.spec_algorithm.is_dflash():
                _attn = runner._replay_attn_backend()
                _meta = getattr(_attn, "forward_metadata", None)
                _meta_list = getattr(_meta, "seq_lens_cpu_list", None)
            else:
                _meta_list = None
            if _meta_list is not None:
                # graph.update must carry the exact KV length already
                # computed in forward_metadata.seq_lens_cpu_list (it
                # already includes the draft block for DFlash); do not
                # recompute and double-add here.
                seq_lens = list(_meta_list)
            else:
                # Hybrid attention wrappers may not expose replay metadata.
                # Prefer the live CPU mirror to avoid a blocking D2H copy;
                # fall back to device lengths when the mirror is unavailable.
                if forward_batch.seq_lens_cpu is not None:
                    seq_lens_cpu = (
                        forward_batch.seq_lens_cpu[: runner.raw_bs]
                        + runner.captured_req_width
                    )
                else:
                    seq_lens_cpu = (
                        forward_batch.seq_lens.cpu() + runner.captured_req_width
                    )
                seq_lens = seq_lens_cpu.tolist() + [0] * (runner.bs - runner.raw_bs)
        else:
            # Reuse the live CPU mirror when available to avoid
            # a blocking D2H copy.
            if forward_batch.seq_lens_cpu is not None:
                seq_lens = forward_batch.seq_lens_cpu[: runner.raw_bs].tolist() + [
                    0
                ] * (runner.bs - runner.raw_bs)
            else:
                seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (
                    runner.bs - runner.raw_bs
                )
        return seq_lens

    def prepare_replay(self, forward_batch):
        runner = self._decode_runner
        if runner is None or forward_batch.needs_forward_metadata_init():
            return False
        if not runner.model_runner.spec_algorithm.is_dflash():
            # Preserve the NPU pre-planned path without refreshing metadata
            # or running the common load_batch preparation.
            runner.buffers.input_ids[: runner.raw_num_token].copy_(
                forward_batch.input_ids
            )
            runner.buffers.positions[: runner.raw_num_token].copy_(
                forward_batch.positions
            )
            if (
                envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get()
                and forward_batch.mrope_positions is not None
            ):
                runner.buffers.mrope_positions[:, : runner.raw_num_token].copy_(
                    forward_batch.mrope_positions
                )
        else:
            from sglang.srt.model_executor.forward_batch_info import (
                compute_local_num_token_non_padded_cpu,
                enable_num_token_non_padded,
            )
            from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
                build_replay_fb_view,
            )

            # NPU skips the DFLASH verify pre-planning, so load_batch may
            # never have recorded the padded batch shapes; recompute them
            # on every batch (the verify batch size varies with
            # concurrency).
            raw_bs = forward_batch.batch_size
            if runner.require_mlp_tp_gather:
                bs = runner._pad_to_bucket(
                    runner._max_dp_batch_size(forward_batch), runner.capture_bs
                )
            else:
                bs = runner._pad_to_bucket(raw_bs, runner.capture_bs)
            runner.raw_bs = raw_bs
            runner.raw_num_token = raw_bs * runner.captured_req_width
            runner.bs = bs
            # Restore the DeepEP dispatch mode recorded at capture time
            # (mirrors load_batch); an interleaved eager extend may have
            # switched it.
            runner.deepep_adapter.replay()
            # Refresh the static DP token buffers bound by the captured
            # graph (stale values misalign dp-gather segments across
            # ranks); mirror the capture-side uniform
            # [padded_num_tokens] * num_dp_ranks.
            if runner.require_mlp_tp_gather:
                _padded_num_tokens = bs * runner.captured_req_width
                runner.buffers.global_num_tokens_gpu.fill_(_padded_num_tokens)
                runner.buffers.global_num_tokens_for_logprob_gpu.fill_(
                    _padded_num_tokens
                )
            if (
                enable_num_token_non_padded()
                and runner.require_gathered_buffer
                and not runner.enable_prefill_cp
            ):
                runner.buffers.num_token_non_padded.fill_(
                    compute_local_num_token_non_padded_cpu(
                        global_num_token_non_padded=(
                            forward_batch.global_num_token_non_padded_cpu
                        ),
                        num_tokens_per_dp=bs * runner.captured_req_width,
                        sharded=runner.model_runner.attn_tp_sequence_sharded(
                            bs * runner.captured_req_width
                        ),
                    )
                )
            runner.buffers.input_ids[: runner.raw_num_token].copy_(
                forward_batch.input_ids
            )
            runner.buffers.positions[: runner.raw_num_token].copy_(
                forward_batch.positions
            )
            if (
                runner.model_runner.is_draft_worker
                and forward_batch.input_embeds is not None
            ):
                runner.buffers.input_embeds[: runner.raw_num_token].copy_(
                    forward_batch.input_embeds
                )
            if (
                envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get()
                and forward_batch.mrope_positions is not None
            ):
                runner.buffers.mrope_positions[:, : runner.raw_num_token].copy_(
                    forward_batch.mrope_positions
                )

            # The pre-planned path skipped init_forward_metadata_out_graph;
            # refresh attention metadata so replay reads correct KV pages.
            runner.buffers.seq_lens[: runner.raw_bs].copy_(
                forward_batch.seq_lens_cpu[: runner.raw_bs]
            )
            runner.buffers.seq_lens[runner.raw_bs : runner.bs].fill_(
                runner.seq_len_fill_value
            )
            runner.buffers.seq_lens_cpu[: runner.raw_bs].copy_(
                forward_batch.seq_lens_cpu[: runner.raw_bs]
            )
            runner.buffers.seq_lens_cpu[runner.raw_bs : runner.bs].fill_(
                runner.seq_len_fill_value
            )
            runner.buffers.req_pool_indices[: runner.raw_bs].copy_(
                forward_batch.req_pool_indices[: runner.raw_bs]
            )
            runner.buffers.req_pool_indices[runner.raw_bs : runner.bs].fill_(0)
            # Refresh the static out_cache_loc bound by the captured graph
            # for full-pool KV writes in save_kv_cache (replay would
            # otherwise write verify KV to stale capture-time slots).
            if forward_batch.out_cache_loc is not None:
                _padded_num_token = runner.bs * runner.captured_req_width
                _n = min(runner.raw_num_token, forward_batch.out_cache_loc.shape[0])
                runner.buffers.out_cache_loc[:_n].copy_(
                    forward_batch.out_cache_loc[:_n]
                )
                runner.buffers.out_cache_loc[_n:_padded_num_token].zero_()
            fb_view = build_replay_fb_view(
                forward_batch=forward_batch,
                buffers=runner.buffers,
                bs=runner.bs,
                raw_bs=runner.raw_bs,
                num_tokens=runner.bs * runner.captured_req_width,
                global_num_tokens_cpu=runner._global_num_tokens_for_graph(
                    runner.bs * runner.captured_req_width
                ),
                seq_len_fill_value=runner.seq_len_fill_value,
                capture_forward_mode=runner.capture_forward_mode,
                is_encoder_decoder=runner.is_encoder_decoder,
            )
            runner._replay_attn_backend().init_forward_metadata_out_graph(fb_view)
        # Preserve the legacy NPU pre-planned replay key.
        runner._replay_graph_key = runner._make_graph_key(runner.bs)
        return True

    @contextmanager
    def prepare_capture_forward(self, model, enable_compile, **kwargs):
        """Yield the capture forward callable; restore temporary state on exit."""
        if enable_compile:
            yield torch.compile(
                torch.no_grad()(model.forward),
                fullgraph=True,
                dynamic=False,
                backend=get_compiler_backend("npugraph_ex"),
            )
        else:
            yield model.forward

    def finish_capture_profile(self, options, profile_context):
        # NPU exports traces through its profiler handler.
        pass

    def capture_profile_context(self, options):
        import torch_npu
        from torch_npu.profiler import ProfilerActivity, profile

        output_dir = os.path.join(
            os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp"), "graph_capture_profile"
        )
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Profiling starts for graph capture for NPU. Traces will be saved to: {output_dir}"
        )
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        )
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                output_dir, async_mode=True
            ),
            experimental_config=experimental_config,
        )
        return profile_context

    @contextmanager
    def capture_session(self, stream):
        if self._pool is None:
            self._pool = self._device_module.graph_pool_handle()
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        import torch_npu  # noqa: F401  (verifies NPU availability)

        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = torch.npu.NPUGraph()

        if self._enable_torch_compile:
            skip_guard_context = torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        else:
            skip_guard_context = empty_context()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = torch.npu.graph

        with (
            skip_guard_context,
            graph_ctx(
                graph,
                pool=self._pool,
                stream=self._capture_stream,
                auto_dispatch_capture=True,
            ),
        ):
            out = forward_fn()

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        runner = self._decode_runner
        if runner is not None:
            from sglang.srt.speculative import (
                multi_layer_eagle_draft_extend_cuda_graph_runner as multi_layer,
            )

            is_multi_layer_draft_extend = isinstance(
                runner, multi_layer.MultiLayerEagleDraftExtendCudaGraphRunner
            )
            if (
                is_multi_layer_draft_extend
                and static_forward_batch.seq_lens_cpu is None
            ):
                # The shared runner propagates a missing live mirror as None.
                # Its device buffer may include stale rows beyond this bucket.
                raise ValueError(
                    "Multi-layer draft-extend NPU graph replay requires current "
                    "CPU sequence lengths."
                )
            hf_config = runner.model_runner.model_config.hf_config
            # The old multi-layer runner always updated Host lengths. Preserve
            # that contract independently of the generic DSA/V4 opt-out.
            if is_multi_layer_draft_extend or not (
                is_deepseek_dsa(hf_config) or is_deepseek_v4(hf_config)
            ):
                # The EAGLE draft check must precede the decode fallback:
                # EAGLEDraftCudaGraphRunner also subclasses
                # DecodeCudaGraphRunner.
                from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
                    EAGLEDraftCudaGraphRunner,
                )
                from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
                    EAGLEDraftExtendCudaGraphRunner,
                )

                if isinstance(runner, EAGLEDraftExtendCudaGraphRunner):
                    cpu_update_input = self._eagle_draft_extend_update_lengths(
                        runner, static_forward_batch
                    )
                    return self.replay_with_input_update(
                        shape_key, seq_lens=None, cpu_update_input=cpu_update_input
                    )
                if isinstance(runner, EAGLEDraftCudaGraphRunner):
                    cpu_update_input = self._build_eagle_draft_update_payload(
                        static_forward_batch
                    )
                    return self.replay_with_input_update(
                        shape_key, seq_lens=None, cpu_update_input=cpu_update_input
                    )
                return self.replay_with_input_update(
                    shape_key,
                    seq_lens=self._decode_update_lengths(static_forward_batch),
                    attr_name=self.get_update_attr_name(),
                    attr_type=[],
                )
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def _eagle_draft_extend_update_lengths(self, runner, forward_batch):
        seq_lens = forward_batch.seq_lens_cpu.tolist() + [0] * (
            runner.bs - runner.raw_bs
        )
        return [{self.get_update_attr_name(): seq_lens}]

    def _build_eagle_draft_update_payload(self, forward_batch):
        runner = self._decode_runner
        # One {attr_name: seq_lens} dict per speculative draft step: the
        # kth draft forward reads KV lengths seq_len + 1 + k (one draft token
        # is written to KV per step); padding rows carry 0 so attention
        # skips them.
        attr_name = self.get_update_attr_name()
        cpu_update_input = []
        for step in range(runner.speculative_num_steps - 1):
            seq_lens_cpu = forward_batch.seq_lens_cpu[: runner.raw_bs] + step + 1
            seq_lens = seq_lens_cpu.tolist() + [0] * (runner.bs - runner.raw_bs)
            cpu_update_input.append({attr_name: seq_lens})
        return cpu_update_input

    def replay_with_input_update(
        self,
        shape_key: ShapeKey,
        seq_lens: Any,
        attr_name: str = None,
        attr_type: Any = None,
        cpu_update_input: list = None,
    ) -> Any:
        """Rebind seq_lens on the recorded NPU graph, then replay.

        NPUGraph.update must complete before replay can consume the updated
        KV lengths. Used when the model is not deepseek-nsa.

        Two calling conventions:
        1. (legacy) seq_lens + attr_name + attr_type:
           Constructs cpu_update_input=[{attr_name: seq_lens}] internally.
        2. cpu_update_input: A list of {attr_name: seq_lens} dicts,
           one per speculative step.  Used by EAGLE draft runners.
        """
        if cpu_update_input is None:
            if isinstance(attr_type, torch.Tensor):
                seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
            cpu_update_input = [{attr_name: seq_lens}]

        graph = self._graphs[shape_key]

        if self._rebind_fence is not None:
            self._rebind_fence.synchronize()

        update_future = self._update_executor.submit(
            graph.update, cpu_update_input=cpu_update_input
        )
        graph.replay()
        update_future.result()

        fence = self._device_module.Event()
        fence.record()
        self._rebind_fence = fence
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._update_executor.shutdown(wait=True, cancel_futures=True)
        if self._rebind_fence is not None:
            self._rebind_fence.synchronize()
            self._rebind_fence = None
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
        self._decode_runner = None

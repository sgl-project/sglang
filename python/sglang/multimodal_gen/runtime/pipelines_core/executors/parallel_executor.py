# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List

import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_world_group,
    get_world_rank,
)
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import maybe_nvtx_range

logger = init_logger(__name__)


class ParallelExecutor(PipelineExecutor):
    """
    The correctness of the execution relies on the parallelism_type declared by stages

    """

    # side CUDA streams for multi-stage execution levels, created once and
    # reused so allocator pools stay bounded; entry i serves level member i+1
    _level_streams: list[torch.cuda.Stream] | None = None
    _level_thread_pool: ThreadPoolExecutor | None = None
    _overlap_logged: bool = False

    def _execute_stages(
        self,
        stages: List[PipelineStage],
        batch: Any,
        server_args: ServerArgs,
        run_stage: Callable[[PipelineStage, Any], Any],
        allow_concurrency: bool = False,
    ) -> Any:
        """Execute declared stage levels, respecting stage parallelism types.

        A level with one member (every stage outside add_parallel_stages
        groups) keeps the exact per-paradigm behavior below. A multi-member
        level of replicated stages on a single-request payload runs
        concurrently; any other multi-member level degrades to declaration
        order, which is always a valid schedule.
        """
        if server_args.enable_cfg_parallel:
            rank = get_classifier_free_guidance_rank()
        else:
            rank = get_world_rank()
        cfg_group = get_cfg_group()
        group = get_world_group()

        use_nvtx = self._should_use_stage_nvtx(batch, server_args)
        levels = self.group_stages_into_execution_levels(stages)
        # component offload moves modules between stages under shared
        # residency state, which concurrent members would race; resident
        # deployments keep the overlap
        allow_concurrency = allow_concurrency and not (
            server_args.dit_cpu_offload
            or server_args.dit_layerwise_offload
            or server_args.text_encoder_cpu_offload
            or server_args.image_encoder_cpu_offload
            or server_args.vae_cpu_offload
            or server_args.layerwise_offload_components
            or self._runtime_offload_active()
        )

        with self._component_residency_request(stages, batch, server_args):
            stage_index = 0
            for level in levels:
                if (
                    len(level) > 1
                    and allow_concurrency
                    and isinstance(batch, Req)
                    and all(
                        member.parallelism_type == StageParallelismType.REPLICATED
                        and member.concurrency_safe
                        for member in level
                    )
                ):
                    batch = self._run_parallel_level(
                        level, stage_index, batch, server_args, run_stage, use_nvtx
                    )
                else:
                    for member_index, member in enumerate(level):
                        batch = self._run_stage_by_paradigm(
                            member,
                            stage_index + member_index,
                            batch,
                            server_args,
                            run_stage,
                            use_nvtx,
                            rank=rank,
                            cfg_group=cfg_group,
                            group=group,
                        )
                stage_index += len(level)
        return batch

    def _runtime_offload_active(self) -> bool:
        """Whether any pipeline module is currently layerwise-offloaded.

        The memory-aware loader and compile-scoped offload can wrap modules
        in layerwise offload without any server_args flag reflecting it, so
        the config flags alone are not a complete signal; module state is.
        """
        from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
            is_layerwise_offloaded_module,
        )

        pipeline = getattr(self.component_residency_manager, "pipeline", None)
        modules = getattr(pipeline, "modules", None)
        if not modules:
            return False
        return any(
            isinstance(module, torch.nn.Module)
            and is_layerwise_offloaded_module(module)
            for module in modules.values()
        )

    def _run_parallel_level(
        self,
        level: List[PipelineStage],
        first_stage_index: int,
        payload: Any,
        server_args: ServerArgs,
        run_stage: Callable[[PipelineStage, Any], Any],
        use_nvtx: bool,
    ) -> Any:
        """Run one multi-stage level concurrently.

        Member 0 runs on the calling thread and the current stream; the rest
        run on pool threads, each pinned to a reused side stream that waits
        on the current stream before starting and is waited on after joining,
        so every cross-level ordering the serial schedule provided is kept.
        Members must return the payload they received: with concurrent
        members there is no defined chaining order, so in-place mutation of
        disjoint request state is the only composable contract.
        """
        if not ParallelExecutor._overlap_logged:
            ParallelExecutor._overlap_logged = True
            logger.info(
                "Overlapping declared-parallel stages: %s",
                [stage._component_stage_name() for stage in level],
            )
        cuda_ready = torch.cuda.is_available()
        side_count = len(level) - 1
        if cuda_ready:
            if self._level_streams is None:
                self._level_streams = []
            while len(self._level_streams) < side_count:
                self._level_streams.append(torch.cuda.Stream())
        if self._level_thread_pool is None:
            self._level_thread_pool = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="sgl-stage-level"
            )
        current_stream = torch.cuda.current_stream() if cuda_ready else None
        # grad mode is thread-local: a fresh pool thread runs grad-enabled
        # even when the caller executes the pipeline under inference_mode /
        # no_grad, and a no-grad forward run with gradients on retains its
        # entire autograd graph in GPU memory
        caller_inference_mode = torch.is_inference_mode_enabled()
        caller_grad_enabled = torch.is_grad_enabled()

        def caller_grad_context():
            if caller_inference_mode:
                return torch.inference_mode()
            return torch.set_grad_enabled(caller_grad_enabled)

        # residency bookkeeping mutates shared per-stage state, so it runs
        # serially before any member starts; the concurrent portion is the
        # stage forwards themselves
        for member_index, stage in enumerate(level):
            self.before_stage(
                stage, first_stage_index + member_index, payload, server_args
            )

        def run_member(member_index: int, stage: PipelineStage) -> Any:
            stage_name = stage._component_stage_name()

            def call() -> Any:
                with maybe_nvtx_range(f"stage_{stage_name}", use_nvtx):
                    return self.run_stage_with_context(
                        stage, payload, server_args, run_stage
                    )

            if member_index == 0:
                return call()
            with caller_grad_context():
                if not cuda_ready:
                    return call()
                stream = self._level_streams[member_index - 1]
                stream.wait_stream(current_stream)
                with torch.cuda.stream(stream):
                    return call()

        futures = [
            self._level_thread_pool.submit(run_member, index, stage)
            for index, stage in enumerate(level[1:], start=1)
        ]
        first_error: BaseException | None = None
        results = [None] * len(level)
        try:
            results[0] = run_member(0, level[0])
        except BaseException as error:  # noqa: BLE001 - re-raised after join
            first_error = error
        for index, future in enumerate(futures, start=1):
            try:
                results[index] = future.result()
            except BaseException as error:  # noqa: BLE001 - keep first error
                if first_error is None:
                    first_error = error
        if cuda_ready:
            for stream in self._level_streams[:side_count]:
                current_stream.wait_stream(stream)
        if first_error is not None:
            raise first_error
        for stage, result in zip(level, results):
            if result is not payload:
                raise RuntimeError(
                    f"parallel stage {stage._component_stage_name()} must "
                    "return the payload it received; concurrent members "
                    "compose through in-place updates of disjoint state"
                )
        return payload

    def _run_stage_by_paradigm(
        self,
        stage: PipelineStage,
        stage_index: int,
        batch: Any,
        server_args: ServerArgs,
        run_stage: Callable[[PipelineStage, Any], Any],
        use_nvtx: bool,
        *,
        rank: int,
        cfg_group,
        group,
    ) -> Any:
        """Run one stage under its declared parallelism type."""
        # TODO: decide when to gather on main when CFG_PARALLEL -> MAIN_RANK_ONLY
        paradigm = stage.parallelism_type

        if paradigm == StageParallelismType.MAIN_RANK_ONLY:
            if rank == 0:
                # Only main rank executes, others just wait
                batch = self._run_stage_with_executor_hooks(
                    stage,
                    stage_index,
                    batch,
                    server_args,
                    run_stage,
                    use_nvtx,
                )
            torch.distributed.barrier()

        elif paradigm == StageParallelismType.CFG_PARALLEL:
            local_batch = batch
            local_batch_fields = stage.cfg_parallel_local_batch_fields(
                batch, server_args
            )
            # filter local batch fields from batch
            if rank == 0 and local_batch_fields:
                local_field_values = {
                    name: getattr(batch, name) for name in local_batch_fields
                }
                for name in local_batch_fields:
                    setattr(batch, name, None)
            else:
                local_field_values = {}

            obj_list = [batch] if rank == 0 else []
            try:
                # `dist.broadcast(src=...)` expects a global rank for process groups.
                broadcasted_list = broadcast_pyobj(
                    obj_list,
                    rank=get_world_rank(),
                    dist_group=cfg_group.cpu_group,
                    src=cfg_group.ranks[0],
                )
            finally:
                if rank == 0:
                    # resume local batch fields on rank 0
                    for name, value in local_field_values.items():
                        setattr(batch, name, value)
            if rank != 0:
                batch = broadcasted_list[0]
                for name in local_batch_fields:
                    setattr(batch, name, getattr(local_batch, name))
            batch = self._run_stage_with_executor_hooks(
                stage,
                stage_index,
                batch,
                server_args,
                run_stage,
                use_nvtx,
            )

            torch.distributed.barrier()

        elif paradigm == StageParallelismType.REPLICATED:
            batch = self._run_stage_with_executor_hooks(
                stage,
                stage_index,
                batch,
                server_args,
                run_stage,
                use_nvtx,
            )
        elif paradigm == StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS:
            obj_list = []
            if rank == 0:
                # Only main rank executes, others just wait
                try:
                    batch = self._run_stage_with_executor_hooks(
                        stage,
                        stage_index,
                        batch,
                        server_args,
                        run_stage,
                        use_nvtx,
                    )
                    obj_list = [True, batch]
                except Exception as e:
                    obj_list = [False, e]

            # Send batch to other ranks
            broadcasted_list = broadcast_pyobj(
                obj_list, rank=rank, dist_group=group.cpu_group, src=0
            )
            success, broadcasted_batch = broadcasted_list

            if not success:
                if isinstance(broadcasted_batch, BaseException):
                    raise RuntimeError("Error on rank 0") from broadcasted_batch
                raise RuntimeError(f"Error on rank 0: {broadcasted_batch}")

            if rank != 0:
                batch = broadcasted_batch

            torch.distributed.barrier()
        return batch

    def execute(
        self,
        stages: List[PipelineStage],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        return self._execute_stages(
            stages,
            batch,
            server_args,
            lambda stage, current: stage(current, server_args),
            allow_concurrency=True,
        )

    def execute_group(
        self,
        stages: List[PipelineStage],
        batches: list[Req],
        server_args: ServerArgs,
    ):
        return self._execute_stages(
            stages,
            batches,
            server_args,
            lambda stage, current: stage.run_grouped_requests(current, server_args),
        )

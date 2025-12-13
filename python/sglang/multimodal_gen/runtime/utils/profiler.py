import os

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SGLDiffusionProfiler:
    """
    A wrapper around torch.profiler to simplify usage in pipelines.
    Supports both full profiling and scheduled profiling.


    1. if profile_all_stages is on: profile all stages, including all denoising steps
    2. otherwise, if num_profiled_timesteps is specified: profile {num_profiled_timesteps} denoising steps. profile all steps if num_profiled_timesteps==-1
    """

    _instance = None

    def __init__(
        self,
        request_id: str | None = None,
        rank: int = 0,
        full_profile: bool = False,
        num_steps: int | None = None,
        num_inference_steps: int | None = None,
        log_dir: str = "./logs",
    ):
        self.request_id = request_id or "profile_trace"
        self.rank = rank
        self.full_profile = full_profile
        self.log_dir = log_dir

        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except OSError:
            pass

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        common_torch_profiler_args = dict(
            activities=activities,
            record_shapes=True,
            with_stack=True,
            on_trace_ready=None,
        )
        if self.full_profile:
            # profile all stages
            self.profiler = torch.profiler.profile(**common_torch_profiler_args)
            self.profile_mode_id = "full stages"
        else:
            # profile denoising stage only
            warmup = 1
            num_actual_steps = num_inference_steps if num_steps == -1 else num_steps
            self.num_active_steps = num_actual_steps + warmup
            self.profiler = torch.profiler.profile(
                **common_torch_profiler_args,
                schedule=torch.profiler.schedule(
                    skip_first=0,
                    wait=0,
                    warmup=warmup,
                    active=self.num_active_steps,
                    repeat=1,
                ),
            )
            self.profile_mode_id = f"{num_actual_steps} steps"

        logger.info(f"Profiling request: {request_id} for {self.profile_mode_id}...")

        self.has_stopped = False

        SGLDiffusionProfiler._instance = self
        self.start()

    def start(self):
        logger.info("Starting Profiler...")
        self.profiler.start()

    def _step(self):
        self.profiler.step()

    def step_stage(self):
        if self.full_profile:
            self._step()

    def step_denoising_step(self):
        if not self.full_profile:
            if self.num_active_steps >= 0:
                self._step()
                self.num_active_steps -= 1
            else:
                # early exit when enough steps are captured, to reduce the trace file size
                self.stop(dump_rank=0)

    @classmethod
    def get_instance(cls) -> "SGLDiffusionProfiler":
        return cls._instance

    def stop(self, export_trace: bool = True, dump_rank: int | None = None):
        if self.has_stopped:
            return
        self.has_stopped = True
        logger.info("Stopping Profiler...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.profiler.stop()

        if export_trace:
            self._export_trace(dump_rank)

        SGLDiffusionProfiler._instance = None

    def _export_trace(self, dump_rank: int | None = None):
        if dump_rank is None:
            dump_rank = self.rank

        try:
            os.makedirs(self.log_dir, exist_ok=True)
            sanitized_profile_mode_id = self.profile_mode_id.replace(" ", "_")
            trace_path = os.path.abspath(
                os.path.join(
                    self.log_dir,
                    f"{self.request_id}-{sanitized_profile_mode_id}-global-rank{dump_rank}.trace.json.gz",
                )
            )
            logger.info(f"Saving profiler traces to: {trace_path}")
            self.profiler.export_chrome_trace(trace_path)
        except Exception as e:
            logger.error(f"Failed to export trace: {e}")

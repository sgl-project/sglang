"""VPP startup resources and stage execution, without shared static inputs."""

from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.model_executor.runner.vpp_stage import execute_vpp_stage


class VPPStageRunner(BaseRunner):
    def __init__(self, model_runner):
        super().__init__(model_runner)
        with self.model_context():
            self.prepare_kernel_resources()

    def warmup(self):
        # Whole-model dummy forwards cannot follow the VPP activation protocol.
        # Device workspaces are prepared at construction; model stage kernels
        # initialize lazily on real inputs.
        return

    def can_run_graph(self, forward_batch):
        return False

    def load_batch(self, forward_batch, **kwargs):
        return forward_batch

    def execute(self, forward_batch, pp_proxy_tensors=None, *, vpp_stage_id=None):
        if vpp_stage_id is None:
            raise ValueError("VPP execution requires an explicit vpp_stage_id")
        mr = self.model_runner
        return execute_vpp_stage(
            model=mr.model,
            attn_backend=mr.attn_backend,
            forward_batch=forward_batch,
            stage_id=vpp_stage_id,
            pp_proxy_tensors=pp_proxy_tensors,
            device_timer=mr.device_timer,
        )

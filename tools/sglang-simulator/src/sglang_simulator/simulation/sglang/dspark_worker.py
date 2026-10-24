"""Route native DSPark worker calls to an explicit model-free simulation adapter.

There is deliberately no default cost or acceptance model. The adapter must
implement actual prefill/injection, decode, allocation and result publication;
until it is installed, a DSPark forward fails with an attributable data gap.
"""

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.sglang.dspark import (
    DSparkStepResult,
    dspark_result_from_native,
)
from sglang_simulator.time_predictor.base import PredictorError


def _runtime():
    from sglang_simulator.simulation.sglang.scheduler import C_SchedulerHook

    predictor = C_SchedulerHook.INFERENCE_PREDICTOR
    runtime = getattr(predictor, "dspark_runtime", None)
    if runtime is None or not getattr(predictor, "supports_dspark", False):
        raise PredictorError(
            "data_unavailable",
            "DSPark simulation requires complete worker costs and an independent progress model",
        )
    return runtime


class C_DSparkDraftInputHook(BaseHook):
    HOOK_CLASS_NAME = "DFlashDraftInputV2"
    HOOK_MODULE_NAME = "sglang.srt.speculative.dflash_info_v2"
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        original_prepare = target.prepare_for_decode

        def prepare(self, batch):
            from sglang_simulator.simulation.sglang.dspark_runtime import (
                INPUT_ATTRIBUTE,
                prepare_cpu_decode,
            )

            if getattr(self, INPUT_ATTRIBUTE, None) is not None:
                return prepare_cpu_decode(self, batch)
            return original_prepare(self, batch)

        target.prepare_for_decode = prepare


class C_DSparkWorkerHook(BaseHook):
    HOOK_CLASS_NAME = "DSparkWorkerV2"
    HOOK_MODULE_NAME = "sglang.srt.speculative.dspark_components.dspark_worker_v2"
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        def initialize(
            self, server_args, gpu_id, nccl_port, target_worker, draft_worker_cls=None
        ):
            # Reuse the native base lifecycle and target allocator. No model
            # weights, fake draft runner or device acceptance kernel is created.
            from sglang.srt.arg_groups.overrides import resolving_view
            from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

            # Native resolution preserves raw CLI inputs, including an omitted
            # target width. Read its declared Q6 through the owning view.
            cfg = resolving_view(server_args)
            BaseSpecWorker.__init__(self)
            if getattr(cfg, "speculative_dspark_block_size", None) != 5 or (
                getattr(cfg, "speculative_num_draft_tokens", None) != 6
            ):
                raise PredictorError(
                    "unsupported_execution_profile",
                    "DSPark simulator requires resolved draft Q5 / target Q6",
                )
            self.server_args = server_args
            self.gpu_id = gpu_id
            self.nccl_port = nccl_port
            self._target_worker = target_worker
            self._draft_worker = None
            self._hosts_draft = True
            self.model_runner = target_worker.model_runner
            self.device = target_worker.device
            self.gamma = 5
            self.query_token_num = 5
            self.verify_num_draft_tokens = 6
            self.speculative_num_draft_tokens = 6
            # These native base methods operate on the shared target pool when
            # no physical draft model is present.
            for name in (
                "alloc_memory_pool",
                "init_attention_backends",
                "init_cuda_graphs",
                "weight_update_runners",
            ):
                setattr(self, name, getattr(BaseSpecWorker, name).__get__(self, target))

        def forward(
            self, batch, on_publish=None, grammar_barrier=None, *, pp_proxy_tensors=None
        ):
            # Native Scheduler supplies this callback for every DSPark forward.
            # The native worker only consumes it when batch.has_grammar; an
            # explicitly grammar-free batch needs no callback or grammar model.
            if (
                grammar_barrier is not None
                and getattr(batch, "has_grammar", None) is not False
            ) or pp_proxy_tensors is not None:
                raise PredictorError(
                    "unsupported_execution_profile",
                    "DSPark simulator has no grammar-overlap or pipeline adapter",
                )
            result = _runtime().forward_batch_generation(
                worker=self, batch=batch, on_publish=on_publish
            )
            if type(dspark_result_from_native(result)) is not DSparkStepResult:
                raise PredictorError(
                    "invalid_dspark_step",
                    "DSPark adapter omitted its causal step result",
                )
            return result

        def finished(self, *, rid, natural_stop):
            _runtime().note_request_finished(rid=rid, natural_stop=natural_stop)

        def clear(self):
            _runtime().clear_cache_pool()

        def reject_budget_override(self, frac):
            raise PredictorError(
                "unsupported_execution_profile",
                "DSPark simulator has no dynamic verify-budget adapter",
            )

        target.__init__ = initialize
        target.forward_batch_generation = forward
        target.note_request_finished = finished
        target.clear_cache_pool = clear
        target.set_dspark_forced_budget_frac = reject_budget_override
        # The native observers depend on physical logits/acceptance buffers.
        # Simulator/progress evidence is exposed by its predictor instead.
        target.dump_info_records = lambda self: None
        target.clear_info_records = lambda self: None
        target.block_accept_estimate_log_suffix = lambda self: None
        target.spec_v2_attn_backends = property(
            lambda self: (self.target_worker.model_runner.attn_backend,)
        )

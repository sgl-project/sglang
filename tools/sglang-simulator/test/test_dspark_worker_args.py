"""Worker initialization consumes native declarations without rewriting inputs."""

import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest
from sglang_simulator.simulation.sglang.dspark_worker import C_DSparkWorkerHook
from sglang_simulator.time_predictor.base import PredictorError


@pytest.fixture
def native_worker(monkeypatch):
    overrides = ModuleType("sglang.srt.arg_groups.overrides")
    views = []

    def resolving_view(args):
        views.append(args)
        return SimpleNamespace(**{**vars(args), **args._declared})

    overrides.resolving_view = resolving_view
    base = ModuleType("sglang.srt.speculative.base_spec_worker")

    class Base:
        def __init__(self):
            self.native_base_initialized = True

        def alloc_memory_pool(self, **kwargs):
            return self._target_worker

        init_attention_backends = alloc_memory_pool
        init_cuda_graphs = alloc_memory_pool
        weight_update_runners = alloc_memory_pool

        @property
        def target_worker(self):
            return self._target_worker

    base.BaseSpecWorker = Base
    monkeypatch.setitem(sys.modules, overrides.__name__, overrides)
    monkeypatch.setitem(sys.modules, base.__name__, base)

    class Worker(Base):
        pass

    C_DSparkWorkerHook.hook(Worker)
    target = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=object()), device="cpu"
    )

    def create(args):
        return Worker(args, 0, 1234, target)

    return SimpleNamespace(
        create=create, target=target, views=views, overrides=overrides
    )


@pytest.mark.parametrize("raw_width", [None, 6])
def test_worker_accepts_native_declared_width_and_preserves_raw_record(
    native_worker, raw_width
):
    args = SimpleNamespace(
        speculative_dspark_block_size=5,
        speculative_num_draft_tokens=raw_width,
        _declared={"speculative_num_draft_tokens": 6},
    )
    before = copy.deepcopy(vars(args))
    worker = native_worker.create(args)
    assert native_worker.views == [args]
    assert vars(args) == before
    assert worker.server_args is args and worker.native_base_initialized
    assert worker.query_token_num == 5 and worker.verify_num_draft_tokens == 6
    assert worker.target_worker is native_worker.target
    assert worker.model_runner is native_worker.target.model_runner
    assert worker.alloc_memory_pool() is native_worker.target


@pytest.mark.parametrize(
    "declared",
    [
        {"speculative_dspark_block_size": 4},
        {"speculative_num_draft_tokens": 5},
        {"speculative_num_draft_tokens": None},
    ],
)
def test_worker_rejects_unsupported_declaration_despite_supported_raw_widths(
    native_worker, declared
):
    args = SimpleNamespace(
        speculative_dspark_block_size=5,
        speculative_num_draft_tokens=6,
        _declared=declared,
    )
    before = copy.deepcopy(vars(args))
    with pytest.raises(PredictorError, match="resolved draft Q5 / target Q6"):
        native_worker.create(args)
    assert native_worker.views == [args] and vars(args) == before


def test_worker_preserves_native_view_failure_without_raw_fallback(native_worker):
    failure = ValueError("native declaration failure")

    def reject(_args):
        raise failure

    native_worker.overrides.resolving_view = reject
    args = SimpleNamespace(
        speculative_dspark_block_size=5, speculative_num_draft_tokens=6
    )
    with pytest.raises(ValueError) as error:
        native_worker.create(args)
    assert error.value is failure

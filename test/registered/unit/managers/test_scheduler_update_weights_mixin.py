from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class _ModelRunner:
    def __init__(self):
        self.remote_calls = []
        self.sharded_calls = []

    def save_remote_model(self, url):
        self.remote_calls.append(url)

    def save_sharded_model(self, path, pattern=None, max_size=None):
        self.sharded_calls.append((path, pattern, max_size))


class _Worker:
    def __init__(self):
        self.model_runner = _ModelRunner()


class _Scheduler(SchedulerUpdateWeightsMixin):
    def __init__(self, draft_worker=None):
        self.tp_worker = _Worker()
        self.draft_worker = draft_worker


def test_save_sharded_model_accepts_rpc_kwargs():
    scheduler = _Scheduler()

    scheduler.save_sharded_model(
        path="/tmp/model", pattern="model-*.safetensors", max_size=1024
    )

    assert scheduler.tp_worker.model_runner.sharded_calls == [
        ("/tmp/model", "model-*.safetensors", 1024)
    ]


def test_save_remote_model_accepts_rpc_kwargs_with_draft_worker():
    draft_worker = _Worker()
    scheduler = _Scheduler(draft_worker=draft_worker)

    scheduler.save_remote_model(url="s3://main", draft_url="s3://draft")

    assert scheduler.tp_worker.model_runner.remote_calls == ["s3://main"]
    assert draft_worker.model_runner.remote_calls == ["s3://draft"]

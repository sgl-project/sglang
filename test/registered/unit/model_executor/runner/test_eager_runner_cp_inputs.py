import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.runner import eager_runner
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Model:
    def __init__(self, remapped_ids):
        self.remapped_ids = remapped_ids
        self.pp_group = SimpleNamespace(is_last_rank=True)
        self.lm_head = None
        self.model_kwargs = None
        self.logits_ids = None

    def prepare_cp_inputs(self, forward_batch, **kwargs):
        model_kwargs = dict(kwargs)
        if self.remapped_ids is not None:
            model_kwargs["input_ids"] = self.remapped_ids
        return torch.zeros(4, 8), forward_batch.positions, model_kwargs

    def model(self, input_ids, positions, forward_batch, **kwargs):
        self.model_kwargs = kwargs
        return torch.zeros(4, 8)

    def logits_processor(self, input_ids, hidden_states, lm_head, forward_batch, aux):
        self.logits_ids = input_ids
        return hidden_states


def _run(model, forward_batch):
    runner = EagerRunner.__new__(EagerRunner)
    runner.model_runner = SimpleNamespace(model=model)
    sharded_ids = []

    @contextlib.contextmanager
    def fake_shard(embeds, positions, forward_batch, input_ids):
        sharded_ids.append(input_ids)
        yield embeds, positions, input_ids

    with (
        mock.patch.object(eager_runner, "cp_shard_model_inputs", fake_shard),
        mock.patch.object(
            eager_runner, "cp_gather_after_forward", lambda h, fb, stream: h
        ),
        mock.patch.object(torch.cuda, "current_stream", lambda: None),
    ):
        runner._execute_extend_cp(forward_batch, {})
    return sharded_ids[0]


class TestEagerRunnerCPInputs(CustomTestCase):
    def test_ids_remapped_by_prepare_cp_inputs_reach_shard_and_logits(self):
        """DeepSeek-V4.1 maps image pad hashes to image_token_id while preparing CP
        inputs; the runner sharded and scored the raw ids, so Engram / hash / vision
        routing saw unmapped image tokens."""
        forward_batch = SimpleNamespace(
            input_ids=torch.tensor([7, 900001, 900002, 8]),
            positions=torch.arange(4),
        )
        remapped = torch.tensor([7, 5, 5, 8])
        model = _Model(remapped)
        self.assertIs(_run(model, forward_batch), remapped)
        self.assertIs(model.logits_ids, remapped)
        self.assertNotIn("input_ids", model.model_kwargs)

        plain = _Model(remapped_ids=None)
        self.assertIs(_run(plain, forward_batch), forward_batch.input_ids)
        self.assertIs(plain.logits_ids, forward_batch.input_ids)


if __name__ == "__main__":
    unittest.main()

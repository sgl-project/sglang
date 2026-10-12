"""A weight-update session must leave every parameter that peers write by address where it was published."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.model_executor.model_runner_components.weight_updater as weight_updater_mod
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4), requires_grad=False)


def _published(model):
    return {
        name: (param.data_ptr(), param.numel(), param.element_size())
        for name, param in model.named_parameters()
    }


def _weight_updater(model, published_weight_info):
    model_runner = SimpleNamespace(
        remote_instance_weight_transporter=SimpleNamespace(
            weight_info=published_weight_info
        )
    )
    return WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=None,
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=None,
        recapture_cuda_graph=None,
        get_model_runner=lambda: model_runner,
    )


@contextmanager
def _postprocess(postprocess_weights):
    loader = weight_updater_mod.DefaultModelLoader
    with (
        patch.object(
            loader, "restore_weights_before_loading", lambda model, device: None
        ),
        patch.object(loader, "postprocess_weights", postprocess_weights),
    ):
        yield


def _rebind_weight(model, device):
    model.weight.data = torch.zeros(4)


class TestPublishedStorage(CustomTestCase):
    def test_a_session_that_keeps_published_storage_passes(self):
        model = _Model()
        weight_updater = _weight_updater(model, _published(model))

        with _postprocess(lambda model, device: None):
            weight_updater.begin_weight_update()
            weight_updater.end_weight_update(run_post_load=False)

    def test_a_postprocess_that_moves_a_published_parameter_fails_the_session(self):
        """Peer writes into the old address would land in freed memory."""
        model = _Model()
        weight_updater = _weight_updater(model, _published(model))

        with _postprocess(_rebind_weight):
            weight_updater.begin_weight_update()
            with self.assertRaisesRegex(
                AssertionError, "moved published parameters.*weight published at"
            ):
                weight_updater.end_weight_update(run_post_load=False)

    def test_a_rank_that_published_nothing_is_not_checked(self):
        """Without published addresses no peer writes by address, so storage may move."""
        model = _Model()
        weight_updater = _weight_updater(model, None)

        with _postprocess(_rebind_weight):
            weight_updater.begin_weight_update()
            weight_updater.end_weight_update(run_post_load=False)


if __name__ == "__main__":
    unittest.main()

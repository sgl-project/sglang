"""PP replicas preserve native vocabulary sharding and coordinated startup."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.speculative.dspark_components.dspark_shared_modules import (
    resolve_dspark_shared_modules,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dspark_shared_modules_utils import launch_shared_module_test
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=50, suite="base-a-test-cpu")


class TestDSparkSharedModules(CustomTestCase):
    def test_pp1_borrows_custom_modules_without_copy(self):
        embedding = torch.nn.Embedding(11, 4)
        head = torch.nn.Linear(4, 11)
        model = SimpleNamespace(get_input_embeddings=lambda: embedding, lm_head=head)
        actual = resolve_dspark_shared_modules(
            target_model=model,
            pp_group=SimpleNamespace(world_size=1),
            tp_group=None,
            device="cpu",
        )
        self.assertIs(actual[0], embedding)
        self.assertIs(actual[1], head)
        del model.get_input_embeddings
        model.model = SimpleNamespace(get_input_embeddings=lambda: embedding)
        self.assertIs(
            resolve_dspark_shared_modules(
                target_model=model,
                pp_group=SimpleNamespace(world_size=1),
                tp_group=None,
                device="cpu",
            )[0],
            embedding,
        )

    def test_tp2_pp2_native_modules(self):
        launch_shared_module_test(self, tp_size=2, pp_size=2)

    def test_pp4_intermediate_stages(self):
        launch_shared_module_test(self, tp_size=1, pp_size=4)


if __name__ == "__main__":
    unittest.main()

import gc
import json
import os
import unittest

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

import sglang as sgl
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    LocalSerializedTensor,
)
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_cuda_ci(est_time=90, stage="extra-a", runner_config="1-gpu-large")

PARAMS = (
    [
        f"model.language_model.layers.{layer}.mlp.{proj}_proj.weight"
        for layer in range(4)
        for proj in ("gate", "up", "down")
    ]
    + ["mtp.fc.weight"]
    + [f"mtp.layers.0.mlp.{proj}_proj.weight" for proj in ("gate", "up", "down")]
)
NUM_ROUNDS = 3


def _load_checkpoint_tensors(model, names):
    path = model if os.path.isdir(model) else snapshot_download(model)
    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    tensors = {}
    for name in names:
        with safe_open(os.path.join(path, weight_map[name]), framework="pt") as f:
            tensors[name] = f.get_tensor(name)
    return tensors


class TestUpdateWeightsFromTensorWithDraft(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST
        cls.weights = _load_checkpoint_tensors(cls.model, PARAMS)
        cls.engine = sgl.Engine(
            model_path=cls.model,
            speculative_algorithm="NEXTN",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
            mem_fraction_static=0.6,
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "engine", None) is not None:
            cls.engine.shutdown()

    def test_repeated_updates_release_every_producer_tensor(self):
        monkey_patch_torch_reductions()
        for round_idx in range(NUM_ROUNDS):
            with self.subTest(round=round_idx):
                gc.collect()
                torch.cuda.ipc_collect()
                memory_before = torch.cuda.memory_allocated()
                named_tensors = [(name, self.weights[name].cuda()) for name in PARAMS]
                payload = [
                    (
                        name,
                        LocalSerializedTensor(
                            values=[MultiprocessingSerializer.serialize(tensor)]
                        ),
                    )
                    for name, tensor in named_tensors
                ]
                self.engine.begin_weight_update()
                success, message = self.engine.update_weights_from_tensor(payload)
                self.engine.end_weight_update()
                self.assertTrue(success, message)
                del named_tensors, payload
                gc.collect()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                leaked = torch.cuda.memory_allocated() - memory_before
                self.assertLessEqual(leaked, 1024, f"{leaked} bytes still held")


if __name__ == "__main__":
    unittest.main()

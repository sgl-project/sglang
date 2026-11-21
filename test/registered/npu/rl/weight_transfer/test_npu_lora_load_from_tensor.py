"""Phase 3 NPU tests: LoRA adapters delivered as serialized tensors.

An RL trainer holds adapter weights in memory, so the engine must accept them
without a checkpoint. Both payload shapes the trainer can send are covered:
one serialized state dict per rank, and one flattened buffer per rank. The
oracle is the *same* adapter loaded from disk by ``/load_lora_adapter``: the
two paths must produce identical tokens and token logprobs.
"""

import hashlib
import json
import os
import tempfile
import unittest

import torch

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_rl_utils import (
    assert_logprob_signature_equal,
    load_lora_adapter,
    load_lora_adapter_from_tensors,
    logprob_signature,
    npu_rl_env,
    npu_server,
    unload_lora_adapter,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=600, suite="extra-b-test-1-npu-a3")
register_npu_ci(est_time=600, suite="nightly-test-1-npu-a3", nightly=True)

MODEL = QWEN3_0_6B_WEIGHTS_PATH
HAVE_MODEL = os.path.isfile(os.path.join(MODEL, "config.json"))
LORA_RANK = 8
LORA_ALPHA = 16
# Adapter weight scale: large enough that the adapted model must answer
# differently, small enough to stay in a sane numeric range.
LORA_A_STD = 0.1
LORA_B_STD = 0.1
ADAPTER_SEED = 20261021
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
)
SERVER_ARGS = (
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
    "--disable-overlap-schedule",
    "--disable-radix-cache",
    "--context-length",
    "2048",
    "--max-running-requests",
    "8",
    "--mem-fraction-static",
    "0.5",
    "--enable-lora",
    "--max-lora-rank",
    str(LORA_RANK),
    "--lora-target-modules",
    "all",
    # The default csgmv backend is Triton-only; Ascend needs its own kernels.
    "--lora-backend",
    "ascend",
)


def _adapter_seed(name: str) -> int:
    digest = hashlib.sha256(f"{ADAPTER_SEED}:{name}".encode()).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def _target_linears(model_path: str):
    """``(module_name, in_features, out_features)`` for the adapter targets.

    Building the model on the meta device keeps this cheap: only the config is
    needed to know which projections exist and how wide they are.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in TARGET_MODULES:
            continue
        targets.append((name, module.in_features, module.out_features))
    if not targets:
        raise AssertionError(f"{model_path}: no LoRA target projections found")
    return targets


def adapter_tensors(model_path: str):
    """Deterministic PEFT adapter for ``model_path``, in the same key layout a
    ``save_pretrained`` adapter directory uses."""
    tensors = {}
    for name, in_features, out_features in _target_linears(model_path):
        key = f"base_model.model.{name}"
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_adapter_seed(key + ".lora_A.weight"))
        a = (
            torch.empty(LORA_RANK, in_features, dtype=torch.float32)
            .normal_(0.0, LORA_A_STD, generator=generator)
            .to(torch.bfloat16)
        )
        generator.manual_seed(_adapter_seed(key + ".lora_B.weight"))
        b = (
            torch.empty(out_features, LORA_RANK, dtype=torch.float32)
            .normal_(0.0, LORA_B_STD, generator=generator)
            .to(torch.bfloat16)
        )
        tensors[key + ".lora_A.weight"] = a
        tensors[key + ".lora_B.weight"] = b
    return tensors


def adapter_config() -> dict:
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": MODEL,
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "target_modules": list(TARGET_MODULES),
        "use_dora": False,
    }


def write_adapter(destination: str, tensors, config: dict) -> str:
    from safetensors.torch import save_file

    os.makedirs(destination, exist_ok=True)
    with open(
        os.path.join(destination, "adapter_config.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        os.path.join(destination, "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )
    return destination


@unittest.skipUnless(HAVE_MODEL, f"model not found: {MODEL}")
class TestNpuLoraLoadFromTensor(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tensors = adapter_tensors(MODEL)
        cls.config = adapter_config()
        cls._adapter_dir = tempfile.TemporaryDirectory(prefix="npu-lora-adapter-")
        cls.adapter_dir = write_adapter(
            os.path.join(cls._adapter_dir.name, "adapter"), cls.tensors, cls.config
        )
        cls._server = npu_server(MODEL, extra_args=SERVER_ARGS, env=npu_rl_env())
        cls.base_url = cls._server.__enter__()
        cls.baseline = logprob_signature(cls.base_url, PROMPTS)

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)
        cls._adapter_dir.cleanup()

    def tearDown(self):
        for lora_name in (
            "disk-adapter",
            "tensor-adapter",
            "flat-adapter",
        ):
            try:
                unload_lora_adapter(self.base_url, lora_name)
            except Exception:
                pass

    def _load_disk_adapter(self, lora_name: str) -> None:
        load_lora_adapter(self.base_url, lora_name, self.adapter_dir)

    def _signature(self, lora_name: str):
        return logprob_signature(self.base_url, PROMPTS, lora_path=lora_name)

    def test_01_tensor_load_matches_disk_adapter(self):
        self._load_disk_adapter("disk-adapter")
        from_disk = self._signature("disk-adapter")
        self.assertNotEqual(
            [item[0] for item in from_disk],
            [item[0] for item in self.baseline],
            "the adapter had no effect -- the payload is a no-op",
        )

        load_lora_adapter_from_tensors(
            self.base_url, "tensor-adapter", self.tensors, self.config
        )
        assert_logprob_signature_equal(
            self._signature("tensor-adapter"),
            from_disk,
            "adapter loaded from tensors does not match the same adapter on disk",
        )

        # The adapter just loaded must not be aliased to the disk one: with the
        # disk adapter gone, the tensor-loaded copy still has to produce it.
        unload_lora_adapter(self.base_url, "disk-adapter")
        assert_logprob_signature_equal(
            self._signature("tensor-adapter"),
            from_disk,
            "unloading the disk adapter changed the tensor-loaded adapter",
        )
        assert_logprob_signature_equal(
            logprob_signature(self.base_url, PROMPTS),
            self.baseline,
            "requests without lora_path changed while an adapter was loaded",
        )

    def test_02_flattened_bucket_matches_per_tensor(self):
        self._load_disk_adapter("disk-adapter")
        from_disk = self._signature("disk-adapter")

        load_lora_adapter_from_tensors(
            self.base_url, "tensor-adapter", self.tensors, self.config
        )
        per_tensor = self._signature("tensor-adapter")

        load_lora_adapter_from_tensors(
            self.base_url,
            "flat-adapter",
            self.tensors,
            self.config,
            load_format="flattened_bucket",
        )
        assert_logprob_signature_equal(
            self._signature("flat-adapter"),
            per_tensor,
            "flattened_bucket adapter does not match the per-tensor payload",
        )
        assert_logprob_signature_equal(
            self._signature("flat-adapter"),
            from_disk,
            "flattened_bucket adapter does not match the same adapter on disk",
        )


if __name__ == "__main__":
    unittest.main()

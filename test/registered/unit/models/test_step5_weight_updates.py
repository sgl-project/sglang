"""Step5 load_weights: a strict checkpoint load, then weight updates streamed like any other model's."""

import json
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from sglang.srt.environ import envs
from sglang.srt.models.step5 import Step5ForCausalLM, _step5_bf16_trtllm_supported
from sglang.srt.models.step5_mtp import Step5MTP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_NUM_EXPERTS = 4
_EXPERTS = "model.layers.3.moe.experts."
_PACKED = "model.layers.3.moe."


class _RecordingParam:
    """A parameter that records what its weight loader was handed."""

    device = torch.device("cpu")

    def __init__(self, dtype: torch.dtype, calls: list):
        self.dtype = dtype
        self._calls = calls

    def weight_loader(self, param, weight, *args, shard_id=None, expert_id=None):
        self._calls.append((self, weight, shard_id, expert_id))


class _FakeStep5:
    """The attributes Step5ForCausalLM.load_weights reads, around one FP8 MoE layer and a norm."""

    def __init__(self):
        self.calls = []
        fp8, fp32 = torch.float8_e4m3fn, torch.float32
        self.params = {
            _EXPERTS + "w13_weight": _RecordingParam(fp8, self.calls),
            _EXPERTS + "w2_weight": _RecordingParam(fp8, self.calls),
            _EXPERTS + "w13_weight_scale_inv": _RecordingParam(fp32, self.calls),
            _EXPERTS + "w2_weight_scale_inv": _RecordingParam(fp32, self.calls),
            "model.norm.weight": _RecordingParam(torch.bfloat16, self.calls),
        }
        self.config = SimpleNamespace(
            moe_num_experts=_NUM_EXPERTS,
            num_hidden_layers=92,
            num_nextn_predict_layers=0,
        )
        self.nvfp4_expert_layers = frozenset()
        self.nvfp4_expert_layouts = {}
        self.model = SimpleNamespace(layers=[])
        self.experts = SimpleNamespace(
            quant_method=SimpleNamespace(weight_block_size=[2, 2]),
            num_local_experts=_NUM_EXPERTS,
            _map_global_expert_id_to_local_expert_id=lambda expert_id: expert_id,
        )
        self.checkpoint_path = None
        self._checkpoint_loaded = False
        self._checkpoint_duplicate_names = frozenset()
        self.duplicate_scans = 0

    def named_parameters(self):
        return list(self.params.items())

    def get_submodule(self, name):
        assert name == _EXPERTS.rstrip(".")
        return self.experts

    def _checkpoint_duplicate_owners(self, expected, model_path):
        return Step5ForCausalLM._checkpoint_duplicate_owners(expected, model_path)

    def _indexed_duplicate_weights(self, weights, duplicates, model_path):
        self.duplicate_scans += 1
        return Step5ForCausalLM._indexed_duplicate_weights(
            weights, duplicates, model_path
        )

    def load_weights(self, weights):
        return Step5ForCausalLM.load_weights(self, weights)

    def expert_calls(self, param_name: str) -> list[tuple[str, int]]:
        param = self.params[_EXPERTS + param_name]
        return [(shard, expert) for p, _, shard, expert in self.calls if p is param]


def _checkpoint() -> list[tuple[str, torch.Tensor]]:
    """Every target weight, experts packed as the released checkpoint stores them."""
    weights = [("model.norm.weight", torch.ones(8, dtype=torch.bfloat16))]
    for projection in ("gate_proj", "up_proj", "down_proj"):
        weights.append(
            (
                f"{_PACKED}{projection}.weight",
                torch.zeros(_NUM_EXPERTS, 4, 4, dtype=torch.float8_e4m3fn),
            )
        )
        weights.append(
            (f"{_PACKED}{projection}.weight_scale_inv", torch.ones(_NUM_EXPERTS, 1, 1))
        )
    return weights


def _save_with_a_duplicated_norm(root: str) -> None:
    """Save _checkpoint() as two shards that both hold model.norm.weight."""
    weights = dict(_checkpoint())
    save_file(weights, f"{root}/model-00001.safetensors")
    save_file(
        {"model.norm.weight": weights["model.norm.weight"]},
        f"{root}/model-00002.safetensors",
    )
    owners = {name: "model-00001.safetensors" for name in weights}
    owners["model.norm.weight"] = "model-00002.safetensors"
    with open(f"{root}/model.safetensors.index.json", "w") as index:
        json.dump({"weight_map": owners}, index)


def _loaded() -> _FakeStep5:
    model = _FakeStep5()
    model.load_weights(_checkpoint())
    model.calls.clear()
    return model


class TestStep5CheckpointLoad(CustomTestCase):
    def test_loads_every_expert_of_a_packed_tensor(self):
        model = _FakeStep5()
        model.load_weights(_checkpoint())

        self.assertEqual(
            model.expert_calls("w13_weight"),
            [(s, e) for s in ("w1", "w3") for e in range(4)],
        )
        self.assertEqual(model.duplicate_scans, 1)

    def test_names_what_it_is_missing(self):
        model = _FakeStep5()
        partial = [
            (name, weight) for name, weight in _checkpoint() if "down_proj" not in name
        ]

        with self.assertRaisesRegex(ValueError, "missing 2 weights"):
            model.load_weights(partial)

    def test_refuses_undeclared_bf16_experts_without_the_opt_in(self):
        """Without SGLANG_STEP5_QUANTIZE_BF16_MOE, a BF16 checkpoint tensor for an FP8
        expert is refused instead of quantized."""
        model = _FakeStep5()
        checkpoint = [
            (
                name,
                weight.to(torch.bfloat16)
                if name.endswith("gate_proj.weight")
                else weight,
            )
            for name, weight in _checkpoint()
        ]

        with (
            envs.SGLANG_STEP5_QUANTIZE_BF16_MOE.override(False),
            self.assertRaisesRegex(ValueError, "Refusing implicit BF16-to-FP8"),
        ):
            model.load_weights(checkpoint)

    def test_refuses_per_expert_names(self):
        model = _FakeStep5()
        per_expert = _EXPERTS + "1.gate_proj.weight"

        with self.assertRaisesRegex(ValueError, "Unknown Step5 target weight"):
            model.load_weights(
                [
                    *_checkpoint(),
                    (per_expert, torch.zeros(4, 4, dtype=torch.float8_e4m3fn)),
                ]
            )


class TestStep5WeightUpdates(CustomTestCase):
    def test_accept_one_expert_at_a_time(self):
        model = _loaded()
        weight = torch.full((4, 4), 2.0).to(torch.float8_e4m3fn)

        model.load_weights(
            [
                (_EXPERTS + "2.gate_proj.weight", weight),
                (_EXPERTS + "2.gate_proj.weight_scale_inv", torch.ones(1, 1)),
            ]
        )

        self.assertEqual(model.expert_calls("w13_weight"), [("w1", 2)])
        self.assertEqual(model.expert_calls("w13_weight_scale_inv"), [("w1", 2)])
        loaded = next(
            w
            for p, w, _, _ in model.calls
            if p is model.params[_EXPERTS + "w13_weight"]
        )
        self.assertTrue(torch.equal(loaded.float(), weight.float()))

    def test_requantize_bf16_experts_without_the_opt_in(self):
        """A BF16 update into an FP8 expert is block-quantized with its scales written,
        whether or not checkpoint-BF16 experts are served as FP8."""
        bf16 = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
        for opt_in in (False, True):
            model = _loaded()
            with (
                self.subTest(opt_in=opt_in),
                envs.SGLANG_STEP5_QUANTIZE_BF16_MOE.override(opt_in),
            ):
                model.load_weights([(_EXPERTS + "2.gate_proj.weight", bf16)])

                [(_, fp8, _, _)] = [
                    c
                    for c in model.calls
                    if c[0] is model.params[_EXPERTS + "w13_weight"]
                ]
                [(_, scale, shard, expert)] = [
                    c
                    for c in model.calls
                    if c[0] is model.params[_EXPERTS + "w13_weight_scale_inv"]
                ]
                self.assertEqual(fp8.dtype, torch.float8_e4m3fn)
                self.assertEqual((shard, expert), ("w1", 2))
                dequantized = fp8.float() * scale.repeat_interleave(
                    2, 0
                ).repeat_interleave(2, 1)
                torch.testing.assert_close(dequantized, bf16.float(), rtol=0.07, atol=0)

    def test_may_send_any_subset_and_skip_the_disk_index(self):
        model = _loaded()

        model.load_weights([("model.norm.weight", torch.ones(8, dtype=torch.bfloat16))])

        self.assertEqual(model.duplicate_scans, 1)
        self.assertEqual(len(model.calls), 1)

    def test_still_refuse_unknown_names(self):
        model = _loaded()
        for name in (
            _EXPERTS + f"{_NUM_EXPERTS}.gate_proj.weight",
            _EXPERTS + "1.gate_proj.bias",
            "model.nope",
        ):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "Unknown Step5 target weight"),
            ):
                model.load_weights([(name, torch.zeros(4, 4))])

    def test_refuse_the_same_expert_twice_in_one_call(self):
        model = _loaded()
        name = _EXPERTS + "1.down_proj.weight"
        weight = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)

        with self.assertRaisesRegex(ValueError, "Duplicate Step5 target weight"):
            model.load_weights([(name, weight), (name, weight)])

    def test_a_disk_reload_may_resend_what_the_checkpoint_duplicates(self):
        """A tensor the checkpoint stores in two shards arrives twice on a disk reload;
        identical copies load, and differing copies are refused."""
        norm = torch.ones(8, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as root:
            _save_with_a_duplicated_norm(root)
            model = _FakeStep5()
            model.checkpoint_path = root
            model.load_weights(_checkpoint())

            model.load_weights(
                [("model.norm.weight", norm), ("model.norm.weight", norm)]
            )
            with self.assertRaisesRegex(ValueError, "duplicate copies differ"):
                model.load_weights(
                    [("model.norm.weight", norm), ("model.norm.weight", 2 * norm)]
                )


class _FakeStep5MTP:
    """The attributes Step5MTP.load_weights reads, for draft layer 4 of a 4-layer target."""

    def __init__(self, checkpoint_path: str):
        self.calls = []
        self.params = {
            name: _RecordingParam(torch.bfloat16, self.calls)
            for name in (
                "model.embed_tokens.weight",
                "model.enorm.weight",
                "model.hnorm.weight",
                "model.eh_proj.weight",
                "model.norm.weight",
                "lm_head.weight",
                "model.mtp_block.input_layernorm.weight",
            )
        }
        self.config = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=1)
        self.layer_selection = SimpleNamespace(start_layer=4)
        self.checkpoint_path = checkpoint_path

    def named_parameters(self):
        return list(self.params.items())


class TestStep5DraftCheckpointLoad(CustomTestCase):
    def test_loads_the_indexed_copy_of_a_duplicated_tensor(self):
        """The draft load reads a tensor the checkpoint stores in two shards from the
        shard the index names, once, instead of failing to start."""
        draft = {
            name: torch.ones(2, dtype=torch.bfloat16)
            for name in (
                "model.layers.4.enorm.weight",
                "model.layers.4.hnorm.weight",
                "model.layers.4.eh_proj.weight",
                "model.layers.4.transformer.shared_head.norm.weight",
                "model.layers.4.transformer.shared_head.output.weight",
                "model.layers.4.input_layernorm.weight",
            )
        }
        stale = torch.ones(2, dtype=torch.bfloat16)
        owned = torch.full((2,), 2.0, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as root:
            save_file(
                {**draft, "model.embed_tokens.weight": stale},
                f"{root}/model-00001.safetensors",
            )
            save_file(
                {"model.embed_tokens.weight": owned}, f"{root}/model-00002.safetensors"
            )
            owners = {name: "model-00001.safetensors" for name in draft}
            owners["model.embed_tokens.weight"] = "model-00002.safetensors"
            with open(f"{root}/model.safetensors.index.json", "w") as index:
                json.dump({"weight_map": owners}, index)
            model = _FakeStep5MTP(root)

            Step5MTP.load_weights(
                model,
                [
                    *draft.items(),
                    ("model.embed_tokens.weight", stale),
                    ("model.embed_tokens.weight", owned),
                ],
            )

        embed = model.params["model.embed_tokens.weight"]
        embed_loads = [w for p, w, _, _ in model.calls if p is embed]
        self.assertEqual(len(embed_loads), 1)
        self.assertTrue(torch.equal(embed_loads[0], owned))


class TestStep5Bf16ExpertRunner(CustomTestCase):
    def test_blackwell_uses_trtllm_only_with_the_swiglu_step_patch(self):
        """Blackwell without FlashInfer's SwigluStep patch serves checkpoint-BF16
        experts on the Triton runner instead of refusing to start."""
        for patched in (False, True):
            core = types.ModuleType("flashinfer.fused_moe.core")
            if patched:
                core.TRTLLM_BF16_SWIGLU_STEP_SUPPORTED = True
            fused_moe = types.ModuleType("flashinfer.fused_moe")
            fused_moe.core = core
            with (
                self.subTest(patched=patched),
                patch(
                    "sglang.srt.models.step5.get_platform",
                    return_value=SimpleNamespace(is_sm100=True),
                ),
                patch.dict(
                    sys.modules,
                    {
                        "flashinfer.fused_moe": fused_moe,
                        "flashinfer.fused_moe.core": core,
                    },
                ),
            ):
                self.assertEqual(_step5_bf16_trtllm_supported(), patched)


if __name__ == "__main__":
    unittest.main()

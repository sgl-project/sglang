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

from sglang.srt.model_loader.loader import post_load_weights
from sglang.srt.models.step3p7 import Step3p7ForConditionalGeneration
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

    def __init__(self, dtype: torch.dtype, calls: list):
        self.dtype = dtype
        self.device = torch.device("cpu")
        self._calls = calls

    def weight_loader(self, param, weight, *args, shard_id=None, expert_id=None):
        self._calls.append((self, weight, shard_id, expert_id))


class _FakeStep5(Step5ForCausalLM):
    """The attributes Step5ForCausalLM.load_weights reads, around one MoE layer and a norm.

    FP8 experts use 2x2 weight blocks so 4x4 test tensors span several scale blocks."""

    def __init__(self, *, bf16_experts: bool = False):
        torch.nn.Module.__init__(self)
        self.calls = []
        fp8, fp32 = torch.float8_e4m3fn, torch.float32
        self.params = {
            "model.norm.weight": _RecordingParam(torch.bfloat16, self.calls),
        }
        if bf16_experts:
            for name in ("w13_weight", "w2_weight"):
                self.params[_EXPERTS + name] = _RecordingParam(
                    torch.bfloat16, self.calls
                )
        else:
            for name in ("w13_weight", "w2_weight"):
                self.params[_EXPERTS + name] = _RecordingParam(fp8, self.calls)
                self.params[_EXPERTS + name + "_scale_inv"] = _RecordingParam(
                    fp32, self.calls
                )
        self.experts = SimpleNamespace(
            quant_method=SimpleNamespace(weight_block_size=[2, 2]),
            num_local_experts=_NUM_EXPERTS,
            _map_global_expert_id_to_local_expert_id=lambda expert_id: expert_id,
        )
        self.config = SimpleNamespace(
            moe_num_experts=_NUM_EXPERTS,
            num_hidden_layers=92,
            num_nextn_predict_layers=0,
        )
        self.nvfp4_expert_layers = frozenset()
        self.nvfp4_expert_layouts = {}
        self.model = SimpleNamespace(layers=[])
        self.checkpoint_path = None
        self._checkpoint_loaded = False
        self._checkpoint_duplicate_names = frozenset()
        self.duplicate_scans = 0

    def named_parameters(self):
        return list(self.params.items())

    def get_submodule(self, name):
        assert name == _EXPERTS.rstrip(".")
        return self.experts

    def _indexed_duplicate_weights(self, weights, duplicates, model_path):
        self.duplicate_scans += 1
        return Step5ForCausalLM._indexed_duplicate_weights(
            weights, duplicates, model_path
        )

    def expert_calls(self, param_name: str) -> list[tuple[str, int]]:
        param = self.params[_EXPERTS + param_name]
        return [(shard, expert) for p, _, shard, expert in self.calls if p is param]


def _checkpoint(*, bf16_experts: bool = False) -> list[tuple[str, torch.Tensor]]:
    """Every target weight, experts packed as the released checkpoint stores them."""
    weights = [("model.norm.weight", torch.ones(8, dtype=torch.bfloat16))]
    for projection in ("gate_proj", "up_proj", "down_proj"):
        if bf16_experts:
            weights.append(
                (
                    f"{_PACKED}{projection}.weight",
                    torch.zeros(_NUM_EXPERTS, 4, 4, dtype=torch.bfloat16),
                )
            )
            continue
        weights.append(
            (
                f"{_PACKED}{projection}.weight",
                torch.zeros(_NUM_EXPERTS, 4, 4, dtype=torch.float8_e4m3fn),
            )
        )
        weights.append(
            (f"{_PACKED}{projection}.weight_scale_inv", torch.ones(_NUM_EXPERTS, 2, 2))
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


def _loaded(*, bf16_experts: bool = False) -> _FakeStep5:
    model = _FakeStep5(bf16_experts=bf16_experts)
    model.load_weights(_checkpoint(bf16_experts=bf16_experts))
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

    def test_bf16_into_fp8_experts_is_requantized_without_the_opt_in(self):
        """A BF16 trainer update for FP8 experts is re-quantized to block FP8 with its
        scales, without SGLANG_STEP5_QUANTIZE_BF16_MOE, which only picks how checkpoint
        BF16 experts are served; a checkpoint load still refuses the implicit cast."""
        model = _loaded()
        weight = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4) - 8

        model.load_weights([(_EXPERTS + "2.gate_proj.weight", weight)])

        self.assertEqual(model.expert_calls("w13_weight"), [("w1", 2)])
        self.assertEqual(model.expert_calls("w13_weight_scale_inv"), [("w1", 2)])
        loaded = {p: w for p, w, _, _ in model.calls if p in model.params.values()}
        fp8 = loaded[model.params[_EXPERTS + "w13_weight"]]
        scale = loaded[model.params[_EXPERTS + "w13_weight_scale_inv"]]
        self.assertEqual((fp8.dtype, scale.shape), (torch.float8_e4m3fn, (2, 2)))
        dequantized = fp8.float() * scale.repeat_interleave(2, 0).repeat_interleave(
            2, 1
        )
        torch.testing.assert_close(dequantized, weight.float(), rtol=0.07, atol=0)

        bf16_checkpoint = [
            (name, w.to(torch.bfloat16) if name.endswith("proj.weight") else w)
            for name, w in _checkpoint()
            if not name.endswith("_scale_inv")
        ]
        with self.assertRaisesRegex(ValueError, "implicit BF16-to-FP8"):
            _FakeStep5().load_weights(bf16_checkpoint)

    def test_refuse_fp8_into_bf16_experts_before_writing(self):
        """Copying an FP8 update into checkpoint-BF16 experts drops its scales; it is
        refused before the expert weight is overwritten."""
        model = _loaded(bf16_experts=True)
        weight = torch.ones(4, 4).to(torch.float8_e4m3fn)

        with self.assertRaisesRegex(ValueError, "Refusing FP8 weight"):
            model.load_weights(
                [
                    (_EXPERTS + "2.gate_proj.weight", weight),
                    (_EXPERTS + "2.gate_proj.weight_scale_inv", torch.ones(2, 2)),
                ]
            )
        self.assertEqual(model.calls, [])

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


class TestStep5LoadersThatBypassLoadWeights(CustomTestCase):
    def test_first_update_after_a_dummy_start_may_stream_a_subset(self):
        """With --load-format dummy (or another loader that writes weights without
        load_weights), the first streamed RL update is an update, not the checkpoint."""
        model = _FakeStep5()
        post_load_weights(model)

        model.load_weights([("model.norm.weight", torch.ones(8, dtype=torch.bfloat16))])

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.duplicate_scans, 0)


_MTP = "model.layers.92."


class _FakeStep5MTP:
    """The attributes Step5MTP.load_weights reads, around one MTP block norm."""

    def __init__(self, checkpoint_path):
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
        self.config = SimpleNamespace(num_hidden_layers=92, num_nextn_predict_layers=3)
        self.layer_selection = SimpleNamespace(start_layer=92)
        self.checkpoint_path = checkpoint_path

    def named_parameters(self):
        return list(self.params.items())

    def load_weights(self, weights):
        return Step5MTP.load_weights(self, weights)


class TestStep5DraftCheckpointLoad(CustomTestCase):
    def test_loads_the_indexed_copy_of_a_duplicated_tensor(self):
        """The MTP draft loads through the target's duplicate-owner resolution: a tensor
        stored in two shards loads once, from the shard the index names."""
        names = (
            "model.embed_tokens.weight",
            *(
                _MTP + local
                for local in (
                    "hnorm.weight",
                    "eh_proj.weight",
                    "transformer.shared_head.norm.weight",
                    "transformer.shared_head.output.weight",
                    "input_layernorm.weight",
                )
            ),
        )
        weights = {name: torch.zeros(2, dtype=torch.bfloat16) for name in names}
        stale = {_MTP + "enorm.weight": torch.full((2,), 7.0, dtype=torch.bfloat16)}
        indexed = torch.full((2,), 3.0, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as root:
            save_file({**weights, **stale}, f"{root}/model-00001.safetensors")
            save_file(
                {_MTP + "enorm.weight": indexed}, f"{root}/model-00002.safetensors"
            )
            owners = {name: "model-00001.safetensors" for name in weights}
            owners[_MTP + "enorm.weight"] = "model-00002.safetensors"
            with open(f"{root}/model.safetensors.index.json", "w") as index:
                json.dump({"weight_map": owners}, index)
            model = _FakeStep5MTP(root)

            # Native iterators stream every shard, so both copies arrive.
            model.load_weights(
                [*weights.items(), *stale.items(), (_MTP + "enorm.weight", indexed)]
            )

        enorm = [
            w for p, w, _, _ in model.calls if p is model.params["model.enorm.weight"]
        ]
        self.assertEqual(len(enorm), 1)
        self.assertTrue(torch.equal(enorm[0], indexed))


class _FakeStep5VL:
    """The vision-language wrapper around a fake Step5 language model, with a two-weight vision tower."""

    def __init__(self):
        self.language_model = _FakeStep5()
        calls = self.language_model.calls
        self.vision = {
            "vision_model.conv1.weight": _RecordingParam(torch.bfloat16, calls),
            "vit_large_projector.weight": _RecordingParam(torch.bfloat16, calls),
        }

    def named_parameters(self, remove_duplicate=True):
        return [
            *self.vision.items(),
            *(
                ("language_model." + n, p)
                for n, p in self.language_model.params.items()
            ),
        ]

    def load_weights(self, weights):
        return Step3p7ForConditionalGeneration._load_step5_weights(self, weights)


def _vision_checkpoint() -> list[tuple[str, torch.Tensor]]:
    return [
        ("vision_model.conv1.weight", torch.ones(2, dtype=torch.bfloat16)),
        ("vit_large_projector.weight", torch.ones(2, dtype=torch.bfloat16)),
    ]


class TestStep5VisionLanguageWeightUpdates(CustomTestCase):
    def test_checkpoint_load_needs_the_whole_vision_tower(self):
        model = _FakeStep5VL()

        with self.assertRaisesRegex(ValueError, "vision is missing weights"):
            model.load_weights([*_checkpoint(), _vision_checkpoint()[0]])

    def test_updates_may_leave_the_vision_tower_out(self):
        """Text RL trains the decoder only, so its weight sync sends no vision weights."""
        model = _FakeStep5VL()
        model.load_weights([*_checkpoint(), *_vision_checkpoint()])
        model.language_model.calls.clear()

        model.load_weights([("model.norm.weight", torch.ones(8, dtype=torch.bfloat16))])

        self.assertEqual(len(model.language_model.calls), 1)

    def test_updates_load_the_vision_weights_they_carry(self):
        """A VL RL weight sync sends the trained tower under its checkpoint names."""
        model = _FakeStep5VL()
        model.load_weights([*_checkpoint(), *_vision_checkpoint()])
        model.language_model.calls.clear()
        projector = torch.full((2,), 3.0, dtype=torch.bfloat16)

        model.load_weights([("vit_large_projector.weight", projector)])

        ((param, weight, _, _),) = model.language_model.calls
        self.assertIs(param, model.vision["vit_large_projector.weight"])
        self.assertIs(weight, projector)


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

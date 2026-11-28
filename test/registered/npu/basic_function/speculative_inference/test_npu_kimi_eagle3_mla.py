"""NPU weight loading, kernel compatibility, and runtime coverage for MLA EAGLE3 drafts."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from safetensors.torch import save_file
from sgl_kernel_npu.operators.matmul import batch_matmul_transpose
from transformers import DeepseekV3Config, PretrainedConfig

from sglang.srt.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.hardware_backend.npu.utils import init_npu_backend
from sglang.srt.models.kimi_k25_eagle3 import Eagle3DeepseekV2ForCausalLM
from sglang.srt.runtime_context import get_context, get_device, get_parallel
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=300, suite="extra-b-test-1-npu-a3")
register_npu_ci(est_time=300, suite="nightly-test-1-npu-a3", nightly=True)


def projection_config(**overrides):
    fields = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        vocab_size=128,
        draft_vocab_size=128,
        tie_word_embeddings=False,
        hidden_act="silu",
        rope_parameters={
            "rope_type": "deepseek_yarn",
            "rope_theta": 10000,
            "factor": 2.0,
            "original_max_position_embeddings": 16,
        },
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        fc_norm=True,
        norm_output=True,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


def synthetic_weights(config):
    # Separate Q/KV and gate/up tensors mirror the published kimi-k2.6-eagle3.1-mla layout.
    h, q, kv = config.hidden_size, config.q_lora_rank, config.kv_lora_rank
    heads, nope = config.num_attention_heads, config.qk_nope_head_dim
    rope, value = config.qk_rope_head_dim, config.v_head_dim
    intermediate = config.intermediate_size
    shapes = {
        "embed_tokens.weight": (config.vocab_size, h),
        "lm_head.weight": (config.draft_vocab_size, h),
        "fc.weight": (h, 3 * h),
        "layers.0.self_attn.q_a_proj.weight": (q, 2 * h),
        "layers.0.self_attn.kv_a_proj_with_mqa.weight": (kv + rope, 2 * h),
        "layers.0.self_attn.q_b_proj.weight": (heads * (nope + rope), q),
        "layers.0.self_attn.kv_b_proj.weight": (heads * (nope + value), kv),
        "layers.0.self_attn.o_proj.weight": (h, heads * value),
        "layers.0.mlp.gate_proj.weight": (intermediate, h),
        "layers.0.mlp.up_proj.weight": (intermediate, h),
        "layers.0.mlp.down_proj.weight": (h, intermediate),
        "layers.0.self_attn.q_a_layernorm.weight": (q,),
        "layers.0.self_attn.kv_a_layernorm.weight": (kv,),
        "layers.0.input_layernorm.weight": (h,),
        "layers.0.hidden_norm.weight": (h,),
        "layers.0.post_attention_layernorm.weight": (h,),
        "norm.weight": (h,),
        **{f"fc_norm.{i}.weight": (h,) for i in range(3)},
    }
    generator = torch.Generator().manual_seed(20261010)
    return {
        name: (torch.randn(shape, generator=generator) * 0.02 + (len(shape) == 1)).to(
            torch.bfloat16
        )
        for name, shape in shapes.items()
    }


def assert_loaded_weights(model, weights):
    params = dict(model.named_parameters())
    attn_prefix = "model.midlayer.self_attn."
    fused = params[attn_prefix + "fused_qkv_a_proj_with_mqa.weight"].cpu()
    torch.testing.assert_close(
        fused,
        torch.cat(
            [
                weights["layers.0.self_attn.q_a_proj.weight"],
                weights["layers.0.self_attn.kv_a_proj_with_mqa.weight"],
            ]
        ),
        rtol=0,
        atol=0,
    )
    for name, weight in weights.items():
        if "q_a_proj" in name or "kv_a_proj_with_mqa" in name:
            continue
        mapped = name.replace("layers.0.", "midlayer.")
        mapped = mapped if mapped.startswith("lm_head.") else "model." + mapped
        offset = 0
        if ".gate_proj." in mapped or ".up_proj." in mapped:
            offset = weight.shape[0] if ".up_proj." in mapped else 0
            mapped = mapped.replace(".gate_proj.", ".gate_up_proj.").replace(
                ".up_proj.", ".gate_up_proj."
            )
        actual = params[mapped].cpu()
        if ".gate_up_proj." in mapped:
            actual = actual[offset : offset + weight.shape[0]]
        torch.testing.assert_close(actual, weight, rtol=0, atol=0)
    attn = model.model.midlayer.self_attn
    expanded = weights["layers.0.self_attn.kv_b_proj.weight"].unflatten(
        0, (-1, attn.qk_nope_head_dim + attn.v_head_dim)
    )
    torch.testing.assert_close(
        attn.w_kc.cpu(), expanded[:, : attn.qk_nope_head_dim], rtol=0, atol=0
    )
    assert attn.w_vc.is_contiguous(), "NPU MLA requires contiguous w_vc"
    torch.testing.assert_close(
        attn.w_vc.cpu(),
        expanded[:, attn.qk_nope_head_dim :].transpose(1, 2),
        rtol=0,
        atol=0,
    )


class TestKimiEagle3MLANPUWeightsAndOps(CustomTestCase):
    def test_weight_loading_fc_projection_and_ascendc_bmm(self):
        init_npu_backend()
        torch.npu.set_device(0)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        with (
            get_context().override_server_args(tp_size=1),
            get_device().override(device="npu"),
            get_parallel().override(tp_rank=0, attn_tp_rank=0),
        ):
            init_distributed_environment(
                1, 0, f"tcp://127.0.0.1:{port}", 0, backend="hccl", timeout=60
            )
            initialize_model_parallel(backend="hccl")
            try:
                config = projection_config()
                model = Eagle3DeepseekV2ForCausalLM(config).to(
                    device="npu:0", dtype=torch.bfloat16
                )
                weights = synthetic_weights(config)
                model.load_weights(weights.items())
                assert_loaded_weights(model, weights)
                self.assertIsNone(model.model._fc_gatherer)

                weight = weights["fc.weight"].npu()
                for tokens in (0, 1, 5, 17, 129):
                    inputs = (
                        torch.randn(
                            tokens, 192, generator=torch.Generator().manual_seed(42)
                        )
                        .bfloat16()
                        .npu()
                    )
                    actual = model.model.fc(inputs)[0]
                    expected = torch.nn.functional.linear(inputs, weight)
                    torch.testing.assert_close(
                        actual.cpu(), expected.cpu(), rtol=0.01, atol=0.01
                    )

                # Exercise device-agnostic embed/head sharing on NPU.
                new_embed = torch.nn.Parameter(weights["embed_tokens.weight"].npu())
                new_head = torch.nn.Parameter(weights["lm_head.weight"].npu())
                model.set_embed(new_embed)
                self.assertIs(model.model.embed_tokens.weight, new_embed)
                model.set_embed_and_head(new_embed, new_head)
                self.assertIs(model.lm_head.weight, new_head)

                # B * K = 128 * 512 = 65536 triggers batch_matmul_transpose's
                # ascendc dispatch on arch22 (A2/A3), which requires contiguous w_vc.
                ascendc_cfg = projection_config(
                    hidden_size=256,
                    intermediate_size=256,
                    num_attention_heads=128,
                    q_lora_rank=128,
                    kv_lora_rank=512,
                    qk_nope_head_dim=128,
                    qk_rope_head_dim=64,
                    v_head_dim=128,
                )
                ascendc_model = Eagle3DeepseekV2ForCausalLM(ascendc_cfg).to(
                    device="npu:0", dtype=torch.bfloat16
                )
                kv_b = (
                    torch.randn(
                        128 * (128 + 128),
                        512,
                        generator=torch.Generator().manual_seed(7),
                    )
                    * 0.02
                ).to(torch.bfloat16)
                ascendc_model.load_weights(
                    [("layers.0.self_attn.kv_b_proj.weight", kv_b)]
                )
                w_vc = ascendc_model.model.midlayer.self_attn.w_vc
                self.assertTrue(w_vc.is_contiguous())
                attn_out = (
                    torch.randn(
                        4,
                        128,
                        512,
                        generator=torch.Generator().manual_seed(9),
                    )
                    .to(device="npu:0", dtype=torch.bfloat16)
                    .contiguous()
                )
                bmm_out = batch_matmul_transpose(attn_out, w_vc)
                ref = torch.bmm(
                    attn_out.float().transpose(0, 1), w_vc.float()
                ).transpose(0, 1)
                torch.testing.assert_close(
                    bmm_out.float().cpu(), ref.cpu(), rtol=0.02, atol=0.02
                )
            finally:
                cleanup_dist_env_and_memory()


def write_tiny_models(directory, vocab_size=128):
    config = DeepseekV3Config(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=2,
        n_shared_experts=1,
        first_k_dense_replace=4,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        max_position_embeddings=256,
        rope_parameters={
            "rope_type": "deepseek_yarn",
            "rope_theta": 50000.0,
            "factor": 2.0,
            "original_max_position_embeddings": 128,
        },
        architectures=["DeepseekV3ForCausalLM"],
    ).to_dict()
    config.update(moe_layer_freq=1, first_k_dense_replace=4)
    target, draft = Path(directory) / "target", Path(directory) / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "config.json").write_text(json.dumps(config))
    config.update(
        architectures=["Eagle3DeepseekV2ForCausalLM"],
        num_hidden_layers=1,
        draft_vocab_size=vocab_size,
        fc_norm=True,
        norm_output=True,
        eagle_config={"eagle_aux_hidden_state_layer_ids": [0, 1, 2]},
    )
    (draft / "config.json").write_text(json.dumps(config))
    return target, draft


def write_acceptance_weights(target, draft):
    # The target maps even prompt IDs to 1 and odd IDs to 0, then alternates.
    # Zero draft logits propose 0, so every request accepts and rejects drafts.
    config = PretrainedConfig.from_dict(json.loads((draft / "config.json").read_text()))
    draft_weights = {
        name: torch.zeros_like(weight)
        for name, weight in synthetic_weights(config).items()
    }
    for name, weight in draft_weights.items():
        if weight.ndim == 1:
            weight.fill_(1)
    save_file(draft_weights, draft / "model.safetensors")
    target_config = json.loads((target / "config.json").read_text())
    target_weights = {
        "model.embed_tokens.weight": draft_weights["embed_tokens.weight"].clone(),
        "model.norm.weight": draft_weights["norm.weight"].clone(),
        "lm_head.weight": draft_weights["lm_head.weight"].clone(),
    }
    for layer in range(target_config["num_hidden_layers"]):
        for name, weight in draft_weights.items():
            if not name.startswith("layers.0.") or ".hidden_norm." in name:
                continue
            if "q_a_proj" in name or "kv_a_proj_with_mqa" in name:
                weight = weight[:, : config.hidden_size]
            target_weights["model." + name.replace("layers.0.", f"layers.{layer}.")] = (
                weight.clone()
            )
    for token in range(config.vocab_size):
        target_weights["model.embed_tokens.weight"][token, token % 2] = 1
    target_weights["lm_head.weight"][0, 1] = 1
    target_weights["lm_head.weight"][1, 0] = 1
    save_file(target_weights, target / "model.safetensors")


def tiny_engine_options(target, draft, graph, disable_radix_cache=True):
    options = dict(
        model_path=str(target),
        tp_size=1,
        load_format="safetensors"
        if (target / "model.safetensors").exists()
        else "dummy",
        skip_tokenizer_init=True,
        device="npu",
        attention_backend="ascend",
        dtype="bfloat16",
        disable_cuda_graph=not graph,
        disable_radix_cache=disable_radix_cache,
        context_length=256,
        max_total_tokens=1024,
        chunked_prefill_size=128,
        mem_fraction_static=0.25,
        max_running_requests=4,
        random_seed=123,
        log_level="warning",
    )
    if graph:
        options.update(cuda_graph_bs_decode=[1, 2], cuda_graph_max_bs_decode=2)
    if draft is not None:
        options.update(
            speculative_algorithm="EAGLE3",
            speculative_draft_load_format="safetensors"
            if (draft / "model.safetensors").exists()
            else "dummy",
            speculative_draft_model_path=str(draft),
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
        )
    return options


def generate_tiny(
    target,
    draft=None,
    *,
    graph=False,
    disable_radix_cache=True,
    batches=(((3, 5, 7, 9),),),
    max_new_tokens=12,
    env_overrides=None,
):
    if env_overrides:
        # Subprocess isolates cached NPU env flags (MLAPO / FIA / FIA_NZ).
        payload = {
            "target": str(target),
            "draft": str(draft) if draft is not None else None,
            "graph": graph,
            "disable_radix_cache": disable_radix_cache,
            "batches": [[list(p) for p in batch] for batch in batches],
            "max_new_tokens": max_new_tokens,
        }
        helper = (
            "import json, sys; from pathlib import Path; "
            "from sglang.srt.entrypoints.engine import Engine; "
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r}); "
            f"import {Path(__file__).stem} as mod; "
            "p = json.loads(sys.argv[1]); "
            "opts = mod.tiny_engine_options(Path(p['target']), "
            "Path(p['draft']) if p['draft'] else None, "
            "p['graph'], p['disable_radix_cache']); "
            "eng = Engine(**opts)\n"
            "try:\n"
            "    out = [eng.generate(input_ids=b, sampling_params="
            "{'temperature': 0.0, 'max_new_tokens': p['max_new_tokens'], "
            "'ignore_eos': True}) for b in p['batches']]\n"
            "    print('TINY_OUT:' + json.dumps(out))\n"
            "finally:\n"
            "    eng.shutdown()\n"
        )
        env = os.environ.copy()
        env.update(env_overrides)
        proc = subprocess.run(
            [sys.executable, "-c", helper, json.dumps(payload)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"Subprocess failed (rc={proc.returncode}, env={env_overrides}):\n"
                f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
            )
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("TINY_OUT:"):
                return json.loads(line[len("TINY_OUT:") :])
        raise AssertionError(
            f"Missing TINY_OUT marker in stdout:\n{proc.stdout[-2000:]}"
        )

    from sglang.srt.entrypoints.engine import Engine

    engine = Engine(**tiny_engine_options(target, draft, graph, disable_radix_cache))
    try:
        return [
            engine.generate(
                input_ids=[list(prompt) for prompt in batch],
                sampling_params={
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            )
            for batch in batches
        ]
    finally:
        engine.shutdown()


class TestKimiEagle3MLARejectFallback(CustomTestCase):
    def test_dummy_draft_preserves_greedy_target_tokens(self):
        # Dummy weights exercise rejection and bonus-token cache advancement.
        with tempfile.TemporaryDirectory() as directory:
            target, draft = write_tiny_models(directory)
            baseline = generate_tiny(target)[0][0]
            speculative = generate_tiny(target, draft)[0][0]
            self.assertEqual(len(baseline["output_ids"]), 12)
            self.assertEqual(speculative["output_ids"], baseline["output_ids"])
            self.assertGreater(speculative["meta_info"]["spec_verify_ct"], 1)
            self.assertGreater(speculative["meta_info"]["spec_num_proposed_drafts"], 0)


class TestKimiEagle3MLAAcceptance(CustomTestCase):
    def test_draft_acceptance_eager_graph_radix_and_mlapo(self):
        # Constructed checkpoints guarantee both acceptance and rejection.
        # The long prompt crosses a 128-token KV page.
        batches = (((1, 2, 3, 4),), ((4, 3, 2, 1, 0), tuple(range(8)) * 14 + (1,)))
        with tempfile.TemporaryDirectory() as directory:
            target, draft = write_tiny_models(directory, vocab_size=8)
            write_acceptance_weights(target, draft)
            baseline = generate_tiny(target, batches=batches, max_new_tokens=24)
            cases = [
                ("eager", False, True, None),
                ("aclgraph", True, True, None),
                ("aclgraph_radix", True, False, None),
                ("aclgraph_mlapo", True, True, {"SGLANG_NPU_USE_MLAPO": "1"}),
                (
                    "aclgraph_mlapo_fia_nz",
                    True,
                    True,
                    {
                        "SGLANG_NPU_USE_MLAPO": "1",
                        "ASCEND_USE_FIA": "1",
                        "SGLANG_USE_FIA_NZ": "1",
                    },
                ),
            ]
            for label, graph, disable_radix, env_overrides in cases:
                with self.subTest(mode=label):
                    speculative = generate_tiny(
                        target,
                        draft,
                        graph=graph,
                        disable_radix_cache=disable_radix,
                        batches=batches,
                        max_new_tokens=24,
                        env_overrides=env_overrides,
                    )
                    self.assert_tiny_outputs(label, baseline, speculative)

    def assert_tiny_outputs(self, label, baseline, speculative):
        self.assertEqual(len(speculative), len(baseline))
        num_correct_drafts = num_proposed_drafts = 0
        for expected_batch, actual_batch in zip(baseline, speculative):
            self.assertEqual(len(actual_batch), len(expected_batch))
            for expected, actual in zip(expected_batch, actual_batch):
                self.assertEqual(len(expected["output_ids"]), 24)
                self.assertEqual(set(expected["output_ids"]), {0, 1})
                self.assertTrue(
                    all(
                        left != right
                        for left, right in zip(
                            expected["output_ids"], expected["output_ids"][1:]
                        )
                    )
                )
                self.assertEqual(actual["output_ids"], expected["output_ids"])
                metrics = actual["meta_info"]
                self.assertGreater(metrics["spec_verify_ct"], 1)
                num_correct_drafts += metrics["spec_num_correct_drafts"]
                num_proposed_drafts += metrics["spec_num_proposed_drafts"]
        self.assertGreater(num_correct_drafts, 0)
        self.assertLess(num_correct_drafts, num_proposed_drafts)
        print(
            json.dumps(
                {
                    "mode": label,
                    "fixture": "alternating-target",
                    "correct_drafts": num_correct_drafts,
                    "proposed_drafts": num_proposed_drafts,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()

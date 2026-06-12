"""
MTP/GDN speculative decoding smoke test on CPU.

Runs ``python -m sglang.bench_one_batch`` so the speculative path
(_SpecBenchRunner: TpModelWorker target + NEXTN/EAGLE draft worker) is driven
end-to-end without a server. The model is a tiny dummy-weight Qwen3-Next-style
hybrid config (layer 0: GDN linear attention, layer 1: full attention) written
to a tempdir together with a minimal local tokenizer, so no model weights or
tokenizer files are downloaded. The NEXTN draft model reuses the target config
(Qwen3NextForCausalLMMTP) and is also dummy-initialized.

This exercises the CPU GDN spec wiring: the multi-step draft backend
(build_draft_decode_metadata_cpu), the fused GDN verify path, and the conv
intermediate-state window used for speculative verification.

The speculative args mirror the CUDA suite
test/registered/models_e2e/test_qwen3_next_models_mtp.py (TestQwen3NextMTPV2);
the dummy-weight bench_one_batch shape mirrors
test/registered/models_e2e/test_dummy_grok_models.py.

Usage:
SGLANG_USE_CPU_ENGINE=1 python3 -m unittest test_spec_bench_mtp_cpu.TestSpecBenchMTPCPU
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, is_in_ci

register_cpu_ci(est_time=300, suite="base-b-test-cpu")

# Tiny Qwen3-Next-style hybrid config. num_hidden_layers=2 with
# full_attention_interval=2 yields the minimal hybrid pattern
# (layer 0: linear_attention/GDN, layer 1: full attention). The dims follow the
# real Qwen/Qwen3-Next-80B-A3B-Instruct config.json, scaled down to sizes the
# CPU (intel_amx / GDN / fused MoE) kernels are already unit-tested with.
TINY_QWEN3_NEXT_CONFIG = {
    "architectures": ["Qwen3NextForCausalLM"],
    "model_type": "qwen3_next",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "decoder_sparse_step": 1,
    "full_attention_interval": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 512,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 64,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_value_head_dim": 64,
    "max_position_embeddings": 4096,
    "mlp_only_layers": [],
    "moe_intermediate_size": 64,
    "norm_topk_prob": True,
    "num_attention_heads": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "output_router_logits": False,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 10000000,
    "router_aux_loss_coef": 0.001,
    "shared_expert_intermediate_size": 64,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    # bench_one_batch synthesizes input ids in [0, 10000), so keep vocab above
    # that while staying small.
    "vocab_size": 16384,
}

# With --load-format dummy neither the target nor the NEXTN draft should read
# any weight file. If the draft path nevertheless requires a real checkpoint,
# the subprocess fails with one of these weight-file errors; skip (not fail)
# in that case since it is a known limitation, not a CPU spec regression.
# Keep the markers weight-file specific: every run of this tempdir model logs
# "Failed to load generation config ...: ... does not appear to have a file
# named generation_config.json" (no generation_config.json is written), so a
# generic "does not appear to have a file named" marker would turn ANY
# failure into a skip.
_DRAFT_NEEDS_REAL_CKPT_MARKERS = (
    "Error no file named",  # transformers: missing pytorch_model.bin/model.safetensors
    "does not appear to have a file named model.safetensors",
    "does not appear to have a file named pytorch_model.bin",
    "no safetensors weight found",
    "No safetensors weights found",
    "Cannot find any model weights",  # sglang model_loader
)

_BENCH_TIMEOUT_S = 900


def _write_tiny_qwen3_next_model(model_dir: str) -> None:
    """Write the tiny hf config plus a minimal local tokenizer.

    The tokenizer is only used by bench_one_batch for optional custom prompts
    (unused here), so a tiny offline WordLevel tokenizer is enough and avoids
    any download.
    """
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(TINY_QWEN3_NEXT_CONFIG, f, indent=2)

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    vocab.update({f"tok{i}": i for i in range(3, 256)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(os.path.join(model_dir, "tokenizer.json"))

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "model_max_length": TINY_QWEN3_NEXT_CONFIG["max_position_embeddings"],
    }
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)


class TestSpecBenchMTPCPU(CustomTestCase):

    def test_mtp_dummy_qwen3_next(self):
        with tempfile.TemporaryDirectory() as model_dir:
            _write_tiny_qwen3_next_model(model_dir)
            result_filename = os.path.join(model_dir, "result.jsonl")

            command = [
                sys.executable,
                "-m",
                "sglang.bench_one_batch",
                "--model-path",
                model_dir,
                "--device",
                "cpu",
                "--batch-size",
                "2",
                "--input-len",
                "128",
                "--output-len",
                "8",
                "--result-filename",
                result_filename,
                "--load-format",
                "dummy",
                "--dtype",
                "bfloat16",
                "--attention-backend",
                "intel_amx",
                # Mirror TestQwen3NextMTPV2 (CUDA) spec args.
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--max-total-tokens",
                "4096",
                "--max-running-requests",
                "4",
                "--disable-radix-cache",
            ]
            env = {**os.environ, "SGLANG_USE_CPU_ENGINE": "1"}
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            try:
                stdout, _ = process.communicate(timeout=_BENCH_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.fail(
                    f"bench_one_batch speculative run timed out after "
                    f"{_BENCH_TIMEOUT_S} s."
                )
            finally:
                kill_process_tree(process.pid)
            output = stdout.decode(errors="backslashreplace")
            print(f"Output: {output}", flush=True)

            if process.returncode != 0:
                if any(m in output for m in _DRAFT_NEEDS_REAL_CKPT_MARKERS):
                    self.skipTest(
                        "Dummy-weight NEXTN draft requires a real draft "
                        "checkpoint on this build; see bench output above."
                    )
                self.fail(
                    f"bench_one_batch speculative run failed with exit code "
                    f"{process.returncode}; see output above."
                )

            # The SD decode log line is only printed by the _SpecBenchRunner
            # path, so this asserts the speculative worker actually ran.
            self.assertIn("Decode(SD)", output)

            self.assertTrue(
                os.path.exists(result_filename),
                "bench_one_batch did not write the result file "
                "(the latency run was skipped).",
            )
            with open(result_filename) as f:
                results = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(results), 1)
            result = results[0]

            # SD bookkeeping: at least one verify step ran, and every decode
            # step commits at least the bonus token (accept_length >= 1).
            self.assertGreaterEqual(result["num_spec_steps"], 1)
            self.assertGreaterEqual(result["avg_accept_length"], 1.0)

            if is_in_ci():
                self.assertGreater(result["overall_throughput"], 0)


if __name__ == "__main__":
    unittest.main()

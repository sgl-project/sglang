"""The step-12 debt on the supplied-instance surface may only shrink.

A callee that takes ``server_args`` keeps the supplied-instance contract: the
caller chose the object, so no global-read ratchet counts it. Step 12 changes
what that object *carries* — the instance stays at the user's raw input — so a
callee reading a field **resolution fills in** would start seeing the CLI default
instead of the effective value.

This pins that intersection. Each entry is one (file, field) pair where a
parameter named ``server_args`` is read for a field resolution writes; the plan
doc carries the proposed disposition per field
(``global_context/12-raw-input-config.md``, "the supplied-instance conversion
list"). New pairs fail: a new one is new step-12 work, and the moment to decide
where the value should come from is when the read is written, not during the
flip. Pairs that disappear also fail, with the entry to delete — the list is the
measurement, not a memory of one.

The written-field set is derived here rather than hardcoded: eight
representative configs are resolved and compared against the dataclass defaults,
the same matrix the context repo's audit tool uses.
"""

import ast
import dataclasses
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import sglang
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_PACKAGE_ROOT = Path(next(iter(sglang.__path__))) / "srt"

# The config the resolution pipeline owns; reading the in-flight record is their
# job, not a supplied-instance read.
_OWNERS = ("server_args.py", "runtime_context.py", "arg_groups/")

_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}

# One config resolves only its own decisions, so the written set is a union.
_MATRIX = (
    {},
    {
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    },
    {"dp_size": 2, "tp_size": 2, "enable_dp_attention": True},
    {"enable_hierarchical_cache": True, "hicache_ratio": 2.0},
    {"disaggregation_mode": "prefill"},
    {"tp_size": 2, "attn_cp_size": 2},
    {"enable_lora": True, "max_lora_rank": 16},
    {"kv_cache_dtype": "fp8_e4m3", "page_size": 64},
)

_PASSED = frozenset(
    {"model_path", "device", "random_seed", "tokenizer_path", "served_model_name"}
)

_EXPOSED = {
    ("configs/model_config.py", "_speculative_draft_quantization_explicitly_set"),
    ("constrained/base_grammar_backend.py", "grammar_backend"),
    ("disaggregation/encode_receiver.py", "encoder_transfer_backend"),
    ("disaggregation/encode_server.py", "mm_process_config"),
    (
        "distributed/device_communicators/mooncake_transfer_engine.py",
        "encoder_transfer_backend",
    ),
    ("dllm/config.py", "max_running_requests"),
    ("kv_canary/capacities.py", "chunked_prefill_size"),
    ("kv_canary/capacities.py", "cuda_graph_config"),
    ("kv_canary/token_oracle/install.py", "sampling_backend"),
    ("layers/moe/kt_ep_wrapper.py", "chunked_prefill_size"),
    ("layers/moe/utils.py", "speculative_moe_runner_backend"),
    ("managers/data_parallel_controller.py", "load_balance_method"),
    ("managers/load_snapshot.py", "load_balance_method"),
    (
        "managers/scheduler_components/new_token_ratio_tracker.py",
        "schedule_conservativeness",
    ),
    ("managers/tokenizer_manager.py", "encoder_transfer_backend"),
    ("mem_cache/allocation_sizing.py", "page_size"),
    (
        "model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py",
        "cuda_graph_config",
    ),
    ("models/sarvam_moe.py", "attention_backend"),
    ("utils/common.py", "page_size"),
    ("utils/cuda_vmm_transport_utils.py", "mm_feature_transport"),
}


class TestSuppliedInstanceExposure(CustomTestCase):
    def setUp(self):
        # Resolving the matrix writes process state on the way through (the
        # multimodal transport handler sets SGLANG_USE_CUDA_IPC_TRANSPORT, and
        # `EnvField.set()` flips a descriptor flag `os.environ` does not carry).
        # Leaking it makes *later* files in the same worker fail, which is how
        # this was found -- so the case restores what it touched.
        super().setUp()
        state = (dict(os.environ), self._env_field_flags())
        self.addCleanup(self._restore_process_state, state)

    @staticmethod
    def _env_field_flags() -> dict:
        from sglang.srt.environ import EnvField, envs

        flags = {}
        for klass in reversed(type(envs).__mro__):
            for name, field in vars(klass).items():
                if isinstance(field, EnvField):
                    flags[name] = field._set_to_none
        return flags

    @staticmethod
    def _restore_process_state(state) -> None:
        from sglang.srt.environ import envs

        saved_environ, saved_flags = state
        os.environ.clear()
        os.environ.update(saved_environ)
        for name, was_none in saved_flags.items():
            getattr(type(envs), name)._set_to_none = was_none

    def _config_dir(self) -> str:
        config_dir = tempfile.mkdtemp(prefix="supplied_instance_")
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        with open(os.path.join(config_dir, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        return config_dir

    def _resolution_written_fields(self) -> set:
        model_path = self._config_dir()
        written = set()
        for extra in _MATRIX:
            try:
                resolved = ServerArgs(
                    model_path=model_path, device="cuda", random_seed=42, **extra
                )
            except Exception:
                continue  # a combination this environment cannot support
            for field in dataclasses.fields(resolved):
                if field.name in _PASSED or field.name in extra:
                    continue
                if field.default is dataclasses.MISSING:
                    continue
                if getattr(resolved, field.name) != field.default:
                    written.add(field.name)
        return written

    def _supplied_instance_reads(self) -> set:
        pairs = set()
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT).as_posix()
            if rel.startswith(_OWNERS):
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                params = {a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)}
                if "server_args" not in params:
                    continue
                for node in ast.walk(fn):
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "server_args"
                        and isinstance(node.ctx, ast.Load)
                    ):
                        pairs.add((rel, node.attr))
        return pairs

    def test_the_exposed_set_matches_the_pinned_list(self):
        written = self._resolution_written_fields()
        found = {pair for pair in self._supplied_instance_reads() if pair[1] in written}
        new = sorted(found - _EXPOSED)
        gone = sorted(_EXPOSED - found)
        self.assertEqual(
            ([], []),
            (new, gone),
            "the supplied-instance step-12 surface drifted.\n"
            f"  new (decide where the resolved value comes from): {new}\n"
            f"  gone (delete from _EXPOSED): {gone}",
        )


if __name__ == "__main__":
    unittest.main()

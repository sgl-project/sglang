"""Resolution writes are recorded, not just applied.

The projection that replaces field materialization reads the declaration stash,
so a resolution write that only assigns the field is invisible to it. The
converted fields are therefore pinned two ways: no bare assignment survives in
the source, and after resolution every one of them agrees with what the stash
says. The second check is the one that keeps the transition honest -- while
`_declare` still writes the field immediately, a stash entry and a field can
only disagree if some other code assigned the field behind the stash's back.
"""

import ast
import json
import os
import pathlib
import shutil
import tempfile
import unittest

import sglang
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).resolve().parent / "srt"

# Fields whose resolution writes go through `_declare`. The set grows one batch
# at a time; a field joins it in the commit that converts its writes.
_DECLARED_FIELDS = frozenset(
    {
        "_speculative_draft_quantization_explicitly_set",
        "custom_weight_loader",
        "debug_cuda_graph",
        "detokenizer_worker_num",
        "enable_dp_attention_local_control_broadcast",
        "enable_dynamic_batch_tokenizer",
        "enable_flashinfer_allreduce_fusion",
        "enable_flexkv",
        "enable_lmcache",
        "enable_lora",
        "enable_prefill_delayer",
        "enable_return_hidden_states",
        "enable_tokenizer_batch_encode",
        "enable_torch_symm_mem",
        "enforce_disable_flashinfer_allreduce_fusion",
        "enforce_shared_experts_fusion",
        "expert_distribution_recorder_buffer_size",
        "grammar_backend",
        "hicache_ratio",
        "image_processor_backend",
        "keep_mm_feature_on_device",
        "mm_feature_transport",
        "mm_process_config",
        "optimistic_prefill_attempts",
        "pre_warm_nccl",
        "prefill_delayer_max_delay_passes",
        "prefill_delayer_token_usage_low_watermark",
        "random_seed",
        "remote_instance_weight_loader_start_seed_via_transfer_engine",
        "return_hidden_states_mode",
        "schedule_conservativeness",
        "served_model_name",
        "skip_server_warmup",
        "soft_watchdog_timeout",
        "speculative_draft_load_format",
        "tool_call_parser",
        "triton_attention_num_kv_splits",
        "uses_mamba_radix_cache",
    }
)

# Shapes the agreement check runs on. Each needs a real config.json:
# `model_path="dummy"` takes the pipeline's early return.
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

_SHAPES = (
    {"tp_size": 2, "dwdp_size": 2},
    {"random_seed": None},
    {"enable_deterministic_inference": True},
    {"enable_return_hidden_states": True},
    {
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    },
    {"dp_size": 2, "tp_size": 2, "enable_dp_attention": True},
    {"enable_hierarchical_cache": True},
    {"disaggregation_mode": "prefill"},
    {"enable_lora": True, "max_lora_rank": 16},
    {"kv_cache_dtype": "fp8_e4m3", "page_size": 64},
)

# Which converted fields the shapes above reach; the rest need a device or an
# architecture no CPU fixture has, and the source scan covers those. Pinned so
# a shape that stops reaching a field fails here. Add to it when adding a shape.
_REACHED_BY_SHAPES = frozenset(
    {
        "_speculative_draft_quantization_explicitly_set",
        "custom_weight_loader",
        "enable_dp_attention_local_control_broadcast",
        "enable_flashinfer_allreduce_fusion",
        "enforce_disable_flashinfer_allreduce_fusion",
        "expert_distribution_recorder_buffer_size",
        "grammar_backend",
        "hicache_ratio",
        "keep_mm_feature_on_device",
        "mm_feature_transport",
        "mm_process_config",
        "random_seed",
        "return_hidden_states_mode",
        "schedule_conservativeness",
        "served_model_name",
        "uses_mamba_radix_cache",
    }
)


def _server_args_writers(tree, path):
    """Assignment targets that land on a ServerArgs instance.

    Two mechanisms reach the same instance during resolution: a handler writing
    `self.<field>`, and a helper elsewhere in the tree writing through a
    `ServerArgs`-annotated parameter -- `set_default_server_args(args)` is
    called from the pipeline and writes `args.<field>`. Both bypass the
    declaration stash, so both have to be scanned; scanning only the handlers
    would let a field look converted while a second writer still assigns it.
    """
    names = {"self"} if path.name == "server_args.py" else set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            annotation = arg.annotation
            if isinstance(annotation, ast.Constant):
                text = annotation.value
            elif isinstance(annotation, ast.Name):
                text = annotation.id
            elif isinstance(annotation, ast.Attribute):
                text = annotation.attr
            else:
                continue
            if text == "ServerArgs":
                names.add(arg.arg)
    return names


def _bare_assignments():
    """Assignments to a converted field that never reach the stash."""
    found = []
    for path in sorted(_SRT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        names = _server_args_writers(tree, path)
        if not names:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in names
                    and target.attr in _DECLARED_FIELDS
                ):
                    found.append(
                        f"{path.relative_to(_SRT)}:{node.lineno} "
                        f"{target.value.id}.{target.attr}"
                    )
    return sorted(found)


def _stash_overlay(server_args):
    """What the declarations say, last writer wins -- the projection's input."""
    overlay = {}
    for _source, declared in getattr(server_args, "_resolved_overrides", None) or ():
        overlay.update(declared)
    return overlay


class TestResolutionDeclarations(CustomTestCase):
    def setUp(self):
        # Resolution writes environment variables, and those outlive the
        # record that set them.
        super().setUp()
        environment = dict(os.environ)

        def restore():
            os.environ.clear()
            os.environ.update(environment)

        self.addCleanup(restore)

    def _resolve(self, extra):
        """A fully-resolved config: a real config.json, so the pipeline runs
        past its dummy-model early return."""
        path = tempfile.mkdtemp(prefix="declarations_")
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        with open(os.path.join(path, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        fields = {"random_seed": 42}
        fields.update(extra)
        return ServerArgs(model_path=path, device="cuda", **fields)

    def test_converted_fields_are_not_assigned_bare(self):
        bare = _bare_assignments()
        self.assertEqual(
            bare,
            [],
            "a converted field is assigned directly, so the projection would "
            "not see this write:\n  " + "\n  ".join(bare),
        )

    def test_the_stash_agrees_with_the_fields_it_declared(self):
        mismatches = []
        for shape in _SHAPES:
            server_args = self._resolve(shape)
            overlay = _stash_overlay(server_args)
            for field, declared in overlay.items():
                if field not in _DECLARED_FIELDS:
                    continue
                actual = getattr(server_args, field)
                if actual != declared:
                    mismatches.append(
                        f"{shape} -> {field}: field={actual!r} stash={declared!r}"
                    )
        self.assertEqual(
            mismatches,
            [],
            "a declared field and its stash entry disagree, so something "
            "assigned the field behind the declaration:\n  " + "\n  ".join(mismatches),
        )

    def test_the_shapes_reach_the_fields_they_are_meant_to(self):
        """A green agreement check over an empty stash would prove nothing."""
        declared = set()
        for shape in _SHAPES:
            declared |= set(_stash_overlay(self._resolve(shape))) & _DECLARED_FIELDS
        missing = sorted(_REACHED_BY_SHAPES - declared)
        self.assertEqual(
            missing,
            [],
            "the shapes no longer reach these converted fields, so the "
            "agreement check silently stopped covering them:\n  "
            + "\n  ".join(missing),
        )


if __name__ == "__main__":
    unittest.main()

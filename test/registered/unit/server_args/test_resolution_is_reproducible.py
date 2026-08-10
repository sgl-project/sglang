"""Resolution is a pure function of the raw input plus this node's environment.

The end state for the configuration tier keeps ``ServerArgs`` at the user's raw
input and lets every process that publishes derive the resolved values itself
(bags do not cross a process boundary — a child projects its own from the record
it is handed). That is only sound if resolving the same raw input twice gives
the same answer, so this pins it:

- twice in this process, from equal raw inputs, every field agrees;
- the resolution is not order-dependent on a shared registry (a second config
  resolved after the first does not inherit its declarations);
- the raw record the two started from is itself unchanged by resolving a
  sibling.

A failure here means some resolution step reads state it also writes, and the
"re-derive in the child" contract would silently diverge between the launcher
and its schedulers.

Scope: this pins reproducibility *within one process*, which is the fork case
(the child inherits the parent's environment and its module-level caches). A
spawn child starts with cold module state instead -- ``functools`` memos in
``runtime_context`` among them -- so a divergence that needs a cold cache to
show up is outside what these cases can see.
"""

import copy
import dataclasses
import json
import os
import shutil
import tempfile
import unittest

import torch

from sglang.srt.environ import EnvField, envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
# Also on a GPU runner: the resolution branches that matter most (backend
# defaults, DeepSeek handlers, capability gates) go through `is_cuda()` /
# `is_hip()` / device capability, which inspect the actual hardware -- passing
# device="cuda" on a CPU box does not reach them, so a leak confined to a GPU
# handler would never fail the CPU registration alone.
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
# ROCm too: `is_hip()` gates its own set of backend and DeepSeek handlers, which
# neither the CPU suite nor a CUDA runner reaches.
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

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

_DEEPSEEK_MINI_CONFIG = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "model_type": "deepseek_v3",
    "hidden_size": 16,
    "intermediate_size": 32,
    "moe_intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "first_k_dense_replace": 1,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
    "kv_lora_rank": 8,
    "q_lora_rank": 8,
    "qk_nope_head_dim": 8,
    "qk_rope_head_dim": 8,
    "v_head_dim": 8,
    "topk_method": "greedy",
    "scoring_func": "softmax",
    # index_topk puts this config on the DSA path, whose handlers are the ones
    # that fan out the most (and write process state on the way through).
    "index_topk": 4,
    "index_head_dim": 8,
    "index_n_heads": 2,
}

_MULTIMODAL_MINI_CONFIG = {
    "architectures": ["Qwen2VLForConditionalGeneration"],
    "model_type": "qwen2_vl",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
    "vision_config": {
        "depth": 2,
        "hidden_size": 16,
        "num_heads": 2,
        "in_chans": 3,
        "patch_size": 14,
        "spatial_merge_size": 2,
    },
}

# The shapes the step-12 audit calls out as the ones whose resolution branches
# touch process state: a plain text model, a speculative launch, and a MoE/MLA
# architecture whose handlers fan out the most.
_SHAPES = (
    ("plain", _MINI_CONFIG, {}),
    (
        "speculative",
        _MINI_CONFIG,
        dict(
            speculative_algorithm="EAGLE",
            speculative_num_steps=2,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=3,
        ),
    ),
    # The multimodal transport handler is the one that writes
    # SGLANG_USE_CUDA_IPC_TRANSPORT and reads `is_set()` on the way in, so the
    # shape that exercises it belongs in the dual-resolve matrix.
    ("multimodal", _MULTIMODAL_MINI_CONFIG, {}),
    # device="cpu" is the one device every host can resolve, and it is what
    # reaches `_handle_cpu_backends` — the golden device="cuda" default never
    # does, so without this shape a leak confined to the CPU handlers would
    # pass the whole matrix.
    ("plain_cpu_device", _MINI_CONFIG, dict(device="cpu")),
)

# Resolving the DSA shape needs a physical device: the DeepSeek-DSA arm of
# `_handle_model_specific_adjustments` probes `torch.cuda.get_device_capability()`
# to pick the KV dtype and split backends, and that raises on a driverless
# host. The GPU registrations are what exercise this shape; a GPU-less runner
# resolves the other shapes only. (Whether a CPU runner even *reaches* the DSA
# arm depends on the installed transformers surfacing `index_topk` from the
# mini config, so without this gate the crash appears runner-dependently.)
if torch.cuda.is_available():
    _SHAPES = _SHAPES + (("deepseek_dsa", _DEEPSEEK_MINI_CONFIG, {}),)

# The one field a previous resolution genuinely dictates for the next one in
# this process: `_handle_multimodal_feature_transport` writes
# SGLANG_USE_CUDA_IPC_TRANSPORT so tokenizer workers inherit the decision, and
# the next resolution reads `is_set()` and adopts it -- even for a text-only
# model, and even across Engines. That is main's behaviour (reproduced on the
# stack's base commit), it is inert for a text model, and pinning it here is
# deliberate: the assertion below states the exception explicitly so a *new*
# sticky field fails this case instead of hiding behind it.
_STICKY_ACROSS_RESOLUTIONS = frozenset({"mm_feature_transport"})

# `random_seed` is pinned by `_resolved` so it would compare equal anyway; it
# stays listed because a case that stops pinning it must not silently start
# comparing a value resolution randomizes.
_NOT_COMPARABLE = frozenset({"random_seed"})


class TestResolutionIsReproducible(CustomTestCase):
    def _config_dir(self, config: dict = None) -> str:
        config_dir = tempfile.mkdtemp(prefix="resolution_repro_")
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        with open(os.path.join(config_dir, "config.json"), "w") as handle:
            json.dump(config or _MINI_CONFIG, handle)
        return config_dir

    def _process_state(self):
        """What a resolution may leave behind: the environment and the
        descriptor-level flag `EnvField.set()` flips, which `os.environ` does
        not carry."""
        # Walk the MRO: `vars(type(envs))` alone would miss fields declared on
        # a base class.
        fields = {}
        for klass in reversed(type(envs).__mro__):
            for name, field in vars(klass).items():
                if isinstance(field, EnvField):
                    fields[name] = field
        return (
            dict(os.environ),
            {name: field._set_to_none for name, field in fields.items()},
        )

    def _restore_process_state(self, state):
        saved_environ, saved_none_flags = state
        os.environ.clear()
        os.environ.update(saved_environ)
        for name, was_none in saved_none_flags.items():
            getattr(type(envs), name)._set_to_none = was_none

    def setUp(self):
        # Resolution writes process state on the way through --
        # `_handle_multimodal_feature_transport` sets SGLANG_USE_CUDA_IPC_TRANSPORT
        # so tokenizer workers inherit the decision, and the same handler reads
        # `is_set()` on the way in. One resolution is therefore visible to the
        # next one in this process. These cases restore what they touched, and
        # it is a standing caveat on the determinism pinned here: the guarantee
        # holds per raw input *and* the process state a previous resolution left.
        self._pristine_state = self._process_state()
        self.addCleanup(self._restore_process_state, self._pristine_state)

    def _callTestMethod(self, method):
        # No retry here. CustomTestCase retries once in CI, but `addCleanup`
        # runs after the last attempt, so a second attempt would start from the
        # state the first one leaked -- exactly the regression these cases exist
        # to catch, turned into a pass.
        unittest.TestCase._callTestMethod(self, method)

    def _resolved(self, model_path: str, **kwargs) -> ServerArgs:
        # device="cuda" keeps the golden path host-independent: an
        # accelerator-less runner resolves only the base platform, where
        # get_device() raises.
        kwargs.setdefault("device", "cuda")
        kwargs.setdefault("random_seed", 42)
        return ServerArgs(model_path=model_path, **kwargs)

    def _comparable(self, server_args: ServerArgs) -> dict:
        """The dataclass fields, and only those.

        Non-field artifacts a resolution leaves on the instance (the
        `_resolved_overrides` provenance, a cached `model_config`) are not in
        here; `test_the_declaration_provenance_is_reproducible` is what covers
        the one of those that a shared mutable could corrupt.
        """
        out = {}
        for field in dataclasses.fields(server_args):
            if field.name in _NOT_COMPARABLE:
                continue
            value = getattr(server_args, field.name)
            # Nested dataclasses (cuda_graph_config) compare structurally, and
            # everything else is deep-copied: a snapshot that stored the live
            # list/dict would follow an in-place mutation, which is exactly the
            # regression `test_resolving_a_sibling_leaves_the_first_alone` looks
            # for.
            out[field.name] = (
                dataclasses.asdict(value)
                if dataclasses.is_dataclass(value)
                else copy.deepcopy(value)
            )
        return out

    def test_two_resolutions_of_the_same_input_agree(self):
        for label, config, kwargs in _SHAPES:
            with self.subTest(shape=label):
                # Each shape starts from the state the test method started in,
                # not from what the previous shape's resolution left behind.
                self._restore_process_state(self._pristine_state)
                model_path = self._config_dir(config)
                first = self._resolved(model_path, **kwargs)
                second = self._resolved(model_path, **kwargs)
                self.assertEqual(self._comparable(first), self._comparable(second))

    def test_a_resolution_does_not_leak_into_the_next(self):
        # A config resolved with an explicit, non-default backend must not shift
        # what the next one picks. Residual process state (env, caches) is the
        # hazard, not the declaration registry, whose providers are import-time.
        model_path = self._config_dir()
        # The control has to be taken *before* the explicit resolution: if that
        # one contaminated the process, a control read afterwards would inherit
        # the same contamination and the assertion would pass vacuously.
        default_before = self._resolved(model_path)
        # torch_native is not the default on either CPU or CUDA, so the probe
        # really diverges on the suite that runs this.
        explicit = self._resolved(model_path, attention_backend="torch_native")
        self.assertEqual(explicit.attention_backend, "torch_native")
        self.assertNotEqual(
            self._comparable(explicit),
            self._comparable(default_before),
            "the probe resolved to the same config as the default, so this case "
            "would pass without exercising order dependence",
        )
        default_after = self._resolved(model_path)
        # Every field, not just the backend: a declaration registry takes
        # arbitrary field dicts, so a leak can land anywhere.
        self.assertEqual(
            self._comparable(default_after), self._comparable(default_before)
        )

        # A backend probe only diverges on the kernel fields. The shapes whose
        # handlers write process state on the way through -- the multimodal
        # transport one sets SGLANG_USE_CUDA_IPC_TRANSPORT and reads `is_set()`
        # on the way in, DSA fans out the furthest -- are the ones that can
        # leave something the *next* default reads, so each gets its own turn as
        # the intermediate. Whether those writes actually fire is
        # device-dependent, which is why this case is registered on the GPU
        # runners as well as CPU.
        intermediates = (
            ("multimodal", _MULTIMODAL_MINI_CONFIG, {}),
            ("torch_compile", _MINI_CONFIG, dict(enable_torch_compile=True)),
        )
        if torch.cuda.is_available():
            # Same device gate as _SHAPES: the DSA arm probes the device
            # capability during resolution.
            intermediates += (("deepseek_dsa", _DEEPSEEK_MINI_CONFIG, {}),)
        for label, config, kwargs in intermediates:
            with self.subTest(intermediate=label):
                # Each intermediate starts from the pristine process, for two
                # reasons: it must not inherit what the previous iteration left,
                # and the handlers under test branch on *unset* state -- the
                # multimodal one auto-selects the transport only when
                # SGLANG_USE_CUDA_IPC_TRANSPORT is not set, and any earlier
                # resolution in this process has already set it. `default_before`
                # is the control precisely because it was taken on this state.
                self._restore_process_state(self._pristine_state)
                # The auto-selection branch under test requires the legacy
                # variable UNSET; a runner that exports it (a supported
                # deployment setting) would otherwise pin every resolution to
                # its value and this subtest would assert the environment
                # rather than the handler. `_STICKY_ACROSS_RESOLUTIONS`
                # already excludes the affected field from the equality
                # against `default_before`, so clearing it here does not skew
                # that comparison.
                envs.SGLANG_USE_CUDA_IPC_TRANSPORT.clear()
                self._resolved(self._config_dir(config), **kwargs)
                after = self._resolved(model_path)
                without_sticky = lambda snapshot: {
                    k: v
                    for k, v in snapshot.items()
                    if k not in _STICKY_ACROSS_RESOLUTIONS
                }
                self.assertEqual(
                    without_sticky(self._comparable(after)),
                    without_sticky(self._comparable(default_before)),
                )
                if label == "multimodal":
                    # And the documented exception, asserted rather than
                    # assumed: the multimodal handler's env write does reach
                    # the next resolution. What it carries is the
                    # intermediate's own device-dependent selection — cuda_ipc
                    # on single-node CUDA, cpu on the CPU/ROCm runners (the
                    # same `is_cuda()` gate the handler branches on).
                    expected = "cuda_ipc" if is_cuda() else "cpu"
                    self.assertEqual(after.mm_feature_transport, expected)

    def test_resolving_a_sibling_leaves_the_first_alone(self):
        for label, config, kwargs in _SHAPES:
            with self.subTest(shape=label):
                self._restore_process_state(self._pristine_state)
                model_path = self._config_dir(config)
                first = self._resolved(model_path, **kwargs)
                snapshot = self._comparable(first)
                self._resolved(
                    model_path, tp_size=2, chunked_prefill_size=1024, **kwargs
                )
                self.assertEqual(self._comparable(first), snapshot)

    def test_the_declaration_provenance_is_reproducible(self):
        model_path = self._config_dir()
        first = self._resolved(model_path)
        # Snapshot before the second resolution: if a regression had the
        # registry hand out a shared mutable list, resolving `second` would
        # mutate what `first` still points at and the two would compare equal.
        first_provenance = copy.deepcopy(getattr(first, "_resolved_overrides", None))
        second = self._resolved(model_path)
        self.assertEqual(
            first_provenance,
            getattr(second, "_resolved_overrides", None),
        )
        # And the first record's own list is untouched by the second
        # resolution -- a shared mutable would show up here.
        self.assertEqual(getattr(first, "_resolved_overrides", None), first_provenance)


if __name__ == "__main__":
    unittest.main()

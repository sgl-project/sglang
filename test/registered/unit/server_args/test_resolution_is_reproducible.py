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
import pathlib
import shutil
import tempfile
import unittest
import unittest.mock

import torch

import sglang
from sglang.srt.arg_groups.overrides import model_config_of, resolution_result
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

# Since #34662 made cuda_ipc opt-in, no auto-resolution reaches the handler's
# cuda_ipc arm any more -- an explicit request is the only way in, so without
# this shape that arm (two raises and the pool-budget logging) is resolved by
# nothing in this file. It is also the only shape whose env write differs from
# what the *next* resolution would pick on its own, which is what keeps the
# sticky-carry assertion in `test_a_resolution_does_not_leak_into_the_next`
# from being vacuous. Gated on `is_cuda()` rather than
# `torch.cuda.is_available()`: the handler raises for cuda_ipc off NVIDIA CUDA,
# ROCm included.
_CUDA_IPC_SHAPES = ()
if is_cuda():
    _CUDA_IPC_SHAPES = (
        (
            "multimodal_cuda_ipc",
            _MULTIMODAL_MINI_CONFIG,
            dict(mm_feature_transport="cuda_ipc"),
        ),
    )
    _SHAPES = _SHAPES + _CUDA_IPC_SHAPES

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


class _RestoresProcessState:
    """Resolution leaves process state behind, so a case that resolves has to
    put it back. `_handle_multimodal_feature_transport` sets
    `SGLANG_USE_CUDA_IPC_TRANSPORT` and the same handler reads `is_set()` on the
    way in, so one resolution is visible to the next one in this process -- and
    `TestMultimodalFeatureTransport` is the case that notices.
    """

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


class TestResolutionIsReproducible(_RestoresProcessState, CustomTestCase):
    def _config_dir(self, config: dict = None) -> str:
        config_dir = tempfile.mkdtemp(prefix="resolution_repro_")
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        with open(os.path.join(config_dir, "config.json"), "w") as handle:
            json.dump(config or _MINI_CONFIG, handle)
        return config_dir

    def _resolved(self, model_path: str, **kwargs) -> ServerArgs:
        # device="cuda" keeps the golden path host-independent: an
        # accelerator-less runner resolves only the base platform, where
        # get_device() raises.
        kwargs.setdefault("device", "cuda")
        kwargs.setdefault("random_seed", 42)
        server_args = ServerArgs(model_path=model_path, **kwargs)
        server_args.resolve_once()
        return server_args

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
            # The resolution result, not the field: a declaration-only resolver
            # never writes the field, so comparing fields would miss exactly
            # the decisions a leak would shift.
            value = resolution_result(server_args, field.name)
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
        # Auto-resolution picks cpu on every runner this case runs on now that
        # cuda_ipc is opt-in, so the auto shape above writes what the next
        # resolution would have picked anyway; the explicit shape is what makes
        # the carry observable at all.
        intermediates += _CUDA_IPC_SHAPES
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
                intermediate = self._resolved(self._config_dir(config), **kwargs)
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
                # And the documented exception, asserted rather than assumed,
                # for every intermediate: each one runs the transport handler,
                # so each one writes the variable the next resolution reads.
                # What carries is the legacy *boolean*, not the tri-state field
                # -- the handler writes 1 only for cuda_ipc -- so every other
                # selection comes back as cpu, which is what keeps this honest
                # if a cuda_vmm shape is ever added (its carry is cpu, not
                # cuda_vmm). The `cuda_ipc` shape is the one whose carry
                # differs from the cpu that `default_before` resolved to.
                expected = (
                    "cuda_ipc"
                    if resolution_result(intermediate, "mm_feature_transport")
                    == "cuda_ipc"
                    else "cpu"
                )
                self.assertEqual(
                    resolution_result(after, "mm_feature_transport"), expected
                )

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

    def test_the_gate_refuses_a_second_resolution(self):
        """A record that has been resolved is left exactly as it was.

        Every publishing process calls the gate, and in a child the record
        arrived already resolved -- so this is the property that keeps the
        child agreeing with the parent. The handlers are not written to survive
        a second pass over their own output (the DP-attention step derives the
        chunked prefill size *from* the chunked prefill size), which is why the
        gate refuses rather than re-deriving.
        """
        for label, config, kwargs in _SHAPES:
            with self.subTest(shape=label):
                self._restore_process_state(self._pristine_state)
                model_path = self._config_dir(config)
                resolved = self._resolved(model_path, **kwargs)
                snapshot = self._comparable(resolved)
                declarations = list(getattr(resolved, "_resolved_overrides", []))
                resolved.resolve_once()
                self.assertEqual(self._comparable(resolved), snapshot)
                self.assertEqual(
                    list(getattr(resolved, "_resolved_overrides", [])), declarations
                )

    def test_the_gate_closes_on_the_dummy_path_too(self):
        """The dummy model leaves the pipeline early, and the gate still shuts.

        That exit is above the materialization the gate reads, so a dummy
        record answered "not resolved yet" forever and every publish of one ran
        the handlers again. Nothing about the early exit makes a second pass
        safe -- the handlers above it declare and apply like any other -- and
        the four that do run happening to be idempotent today is what the gate
        exists to stop depending on. So this counts entries rather than
        comparing values: the values agree either way.
        """
        self._restore_process_state(self._pristine_state)
        record = ServerArgs(model_path="dummy")
        record.resolve_once()

        entries = []
        from sglang.srt.arg_groups import pipeline as pipeline_module

        original = pipeline_module.run_resolution_pipeline

        def counted(server_args):
            entries.append(1)
            return original(server_args)

        with unittest.mock.patch.object(
            pipeline_module, "run_resolution_pipeline", counted
        ):
            record.resolve_once()
        self.assertEqual(
            entries,
            [],
            "a resolved dummy record entered the pipeline again, so every "
            "publish of one re-runs the handlers",
        )

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


class TestProgramsResolveBeforeReadingResolution(CustomTestCase):
    """A program that builds its own record resolves it before reading what
    resolution decides.

    Construction is inert, so a program that builds a record and then reads a
    resolution-written field reads the CLI default. Two of these shipped past
    the earlier censuses because those are rooted at the `sglang` package: the
    model gateway's launcher sized its worker plan from a raw `dp_size`
    (`--dwdp-size 4` launched one server instead of four) and a speculative
    benchmark forwarded `--mem-fraction-static None` to the server it spawns.
    So the universe here is the *repository*, not the package.
    """

    # Entries that hand the record on instead of reading it. Reason required.
    _EXEMPT: dict = {}

    def _repo_root(self):
        # <repo>/python/sglang/__init__.py -> <repo>
        root = pathlib.Path(next(iter(sglang.__path__))).resolve().parents[1]
        if root.name == "python":
            root = root.parent
        return root

    def _written_fields(self):
        """Fields resolution declares, read out of the pipeline's own source.

        Deliberately local: the chain ratchet has a wider derivation (it also
        walks the model-override registries), but it arrives later in this
        series, and a check that imports it would fail at this PR's boundary.
        Coarser is fine here -- what this needs is the fields the entries below
        actually read -- and the floor keeps it from drifting narrower.
        """
        import ast
        import dataclasses as _dataclasses

        from sglang.srt.server_args import ServerArgs as _ServerArgs

        srt = pathlib.Path(next(iter(sglang.__path__))).resolve() / "srt"
        declarers = {"declare_resolution", "declare_late_resolution"}
        fields = set()
        field_names = {field.name for field in _dataclasses.fields(_ServerArgs)}
        # The record plus every module under `arg_groups/`: a handler declares
        # from whichever of the two it lives in.
        sources = [srt / "server_args.py", *sorted((srt / "arg_groups").rglob("*.py"))]
        for source in sources:
            tree = ast.parse(source.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                # Registry data: provider dict keys are field names as
                # *data*, invisible to the keyword scan below. Filtered
                # against the real field set.
                if isinstance(node, ast.Dict):
                    fields |= {
                        key.value
                        for key in node.keys
                        if isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and key.value in field_names
                    }
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                called = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else getattr(func, "id", "")
                )
                if called in declarers or called == "update":
                    fields |= {
                        kw.arg
                        for kw in node.keywords
                        if kw.arg and (called != "update" or kw.arg in field_names)
                    }
        return fields

    def _candidates(self, root):
        """Source files that build a record, with the names they bind it to."""
        import ast

        skip = {".git", "build", "dist", "node_modules", ".venv", "target"}
        found = {}
        for path in sorted(root.rglob("*.py")):
            parts = set(path.relative_to(root).parts)
            if parts & skip:
                continue
            rel = path.relative_to(root).as_posix()
            # Tests build raw records on purpose.
            if rel.startswith("test/") or "/test/" in rel or "/tests/" in rel:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            # Which local names are *the srt record*, by import source: the
            # diffusion runtime has a same-spelled `ServerArgs` with no
            # resolution, so the spelling alone is not enough.
            record_classes, record_helpers = set(), set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                for alias in node.names:
                    bound = alias.asname or alias.name
                    if node.module == "sglang" and alias.name == "ServerArgs":
                        record_classes.add(bound)
                    if node.module == "sglang.srt.server_args":
                        if alias.name == "ServerArgs":
                            record_classes.add(bound)
                        if alias.name == "prepare_server_args":
                            record_helpers.add(bound)
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                    # `x: ServerArgs = ...` is an AnnAssign, not an Assign.
                    targets = [node.target]
                else:
                    continue
                call = getattr(node, "value", None)
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                # `prepare_server_args(argv)` is the CLI launcher's way.
                builds = (
                    isinstance(func, ast.Name)
                    and func.id in (record_classes | record_helpers)
                ) or (
                    isinstance(func, ast.Attribute)
                    and func.attr == "from_cli_args"
                    and isinstance(func.value, ast.Name)
                    and func.value.id in record_classes
                )
                if builds:
                    names |= {t.id for t in targets if isinstance(t, ast.Name)}
            if names:
                found[rel] = (tree, names, path)
        return found

    def test_every_program_that_builds_a_record_resolves_it(self):
        import ast

        root = self._repo_root()
        candidates = self._candidates(root)
        self.assertGreater(
            len(candidates),
            10,
            f"only {len(candidates)} files build a record under {root}; either "
            "this is not a source checkout or the scan broke",
        )
        written = self._written_fields()
        self.assertGreater(len(written), 50, "the written-field set collapsed")
        # What the escaped entries actually read: a narrower derivation goes
        # quiet on exactly those.
        for field in ("dp_size", "mem_fraction_static"):
            self.assertIn(field, written)

        offenders = []
        for rel, (tree, names, path) in sorted(candidates.items()):
            source = path.read_text(encoding="utf-8-sig")
            if "resolve_once(" in source or "publish(" in source:
                continue
            reads = sorted(
                {
                    f"{node.attr}:{node.lineno}"
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in names
                    and node.attr in written
                }
            )
            if reads and rel not in self._EXEMPT:
                offenders.append(f"{rel} reads {', '.join(reads[:4])}")
        self.assertEqual(
            offenders,
            [],
            "a program builds its own record and reads what resolution decides "
            "without resolving it, so it reads the CLI default:\n  "
            + "\n  ".join(offenders),
        )
        self.assertEqual(
            sorted(set(self._EXEMPT) - set(candidates)),
            [],
            "an exemption names a file that no longer builds a record",
        )


class TestForksResolveFirst(CustomTestCase):
    """A process that forks a child to run the record resolves it first.

    The pipeline probes the device (the default attention backend reads the CUDA
    capability), and a forked child cannot initialize CUDA once its parent has.
    Construction used to resolve, so the probe always happened in whoever built
    the record; now it happens at the gate, and the gate must not be reached for
    the first time inside a fork.
    """

    # Sites inside the launcher: `_launch_subprocesses` resolves at its top, so
    # every fork below it already has a resolved record.
    _AFTER_LAUNCHER_RESOLVE = {
        "srt/entrypoints/engine.py",
        "srt/managers/data_parallel_controller.py",
        "srt/disaggregation/encoder/grpc_server.py",
        "srt/disaggregation/encoder/runtime.py",
        "srt/elastic_ep/expert_backup_manager.py",
    }

    def test_every_fork_of_a_record_has_a_resolved_one(self):
        import ast

        package_root = pathlib.Path(next(iter(sglang.__path__))).resolve()
        offenders, examined = [], 0
        for path in sorted(package_root.rglob("*.py")):
            rel = path.relative_to(package_root).as_posix()
            if rel.startswith("test/") or "/test/" in rel:
                continue
            # The diffusion runtime has its own record with no gate.
            if rel.startswith("multimodal_gen/"):
                continue
            try:
                source = path.read_text(encoding="utf-8-sig")
                if "Process" not in source:
                    continue
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                forks = [
                    call
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call)
                    and (
                        (
                            isinstance(call.func, ast.Attribute)
                            and call.func.attr == "Process"
                        )
                        or (
                            isinstance(call.func, ast.Name)
                            and call.func.id == "Process"
                        )
                    )
                    and "server_args" in (ast.get_source_segment(source, call) or "")
                ]
                if not forks:
                    continue
                body = ast.get_source_segment(source, node) or ""
                examined += 1
                # `spawn` starts a fresh interpreter, so the child may probe.
                if 'get_context("spawn")' in body or "'spawn'" in body:
                    continue
                if "resolve_once(" in body or "publish(" in body:
                    continue
                if rel in self._AFTER_LAUNCHER_RESOLVE:
                    continue
                offenders.append(f"{rel}:{forks[0].lineno} {node.name}")
        self.assertGreater(
            examined, 5, f"only {examined} fork sites found; the scan broke"
        )
        self.assertEqual(
            offenders,
            [],
            "these fork a child that will resolve the record, without resolving "
            "it first -- the child cannot initialize CUDA if this process "
            f"already has:\n  " + "\n  ".join(offenders),
        )


class TestACopyStaysResolved(_RestoresProcessState, CustomTestCase):
    """A resolved record copied with `dataclasses.replace` loses what makes it
    resolved, and the next publish resolves it a second time -- over values it
    already decided. The Ray paths copy a resolved record to set
    `dist_init_addr`, which is how they reach this.
    """

    def _resolved(self):
        config_dir = tempfile.mkdtemp(prefix="replace_resolved_")
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        with open(os.path.join(config_dir, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        # Two steps that are not repeatable on their own output.
        server_args = ServerArgs(
            model_path=config_dir,
            device="cuda",
            dp_size=2,
            tp_size=2,
            enable_dp_attention=True,
            random_seed=42,
        )
        server_args.resolve_once()
        return server_args

    def test_a_bare_replace_resolves_again_and_lands_in_the_same_place(self):
        """A bare copy resolves to the same place: the fields are the raw input.

        `dataclasses.replace` copies the fields, so a bare copy re-runs
        resolution over the *same input* the parent got -- the DP-attention
        halving and the conservativeness scaling apply once. `replace_resolved`
        buys something else: it carries the parent's declarations and its
        `model_config`, so the copy answers without resolving at all.
        """
        parent = self._resolved()
        bare = dataclasses.replace(parent, dist_init_addr="1.2.3.4:5000")
        self.assertFalse(
            getattr(bare, "_resolution_finished", False),
            "a bare replace carried the flag; then this test proves nothing",
        )
        bare.resolve_once()
        drifted = {
            field.name: (
                resolution_result(parent, field.name),
                resolution_result(bare, field.name),
            )
            for field in dataclasses.fields(parent)
            if field.name not in ("dist_init_addr", "random_seed")
            and repr(resolution_result(parent, field.name))
            != repr(resolution_result(bare, field.name))
        }
        self.assertEqual(
            drifted,
            {},
            "resolving a bare copy landed somewhere else, so the pipeline is "
            "reading its own output again",
        )

    def test_replace_resolved_keeps_the_parents_resolution(self):
        parent = self._resolved()
        copy_ = parent.replace_resolved("ray.test", dist_init_addr="1.2.3.4:5000")
        self.assertTrue(getattr(copy_, "_resolution_finished", False))
        drifted = {
            field.name: (getattr(parent, field.name), getattr(copy_, field.name))
            for field in dataclasses.fields(parent)
            if field.name != "dist_init_addr"
            and getattr(parent, field.name) != getattr(copy_, field.name)
        }
        self.assertEqual(
            drifted,
            {},
            f"the copy differs from its parent beyond the change: {drifted}",
        )
        self.assertEqual(copy_.dist_init_addr, "1.2.3.4:5000")

    def test_the_copy_carries_what_resolution_left_on_the_record(self):
        """Not just the stash and the flag.

        `model_config_of()` memoizes on the record, and that cache is filled
        during resolution. A copy that is marked resolved but arrives without it
        cannot fill it -- the read-only guard refuses the cache write -- so the
        first `model_config_of()` raises. That is what killed the Ray
        schedulers, and it is why the carry is enumerated from the instance
        rather than from a list of names.
        """
        parent = self._resolved()
        copy_ = parent.replace_resolved("ray.test", dist_init_addr="1.2.3.4:5000")
        fields = {field.name for field in dataclasses.fields(parent)}
        missing = sorted(
            name
            for name in vars(parent)
            if name not in fields and name not in vars(copy_)
        )
        self.assertEqual(
            missing,
            [],
            f"the copy did not carry what resolution left on the record: {missing}",
        )
        self.assertIsNotNone(model_config_of(copy_))
        # Containers are copied, so the copy's declaration stays with it.
        self.assertEqual(
            len(parent._resolved_overrides) + 1, len(copy_._resolved_overrides)
        )

    def test_the_change_reaches_the_bags(self):
        """The projection reads the raw snapshot plus the declarations, so a
        change the copy only wrote to the field would publish the parent's raw
        value."""
        from sglang.srt.runtime_context import (
            get_parallel,
            get_schedule,
            publish,
            reset_context,
        )

        parent = self._resolved()
        copy_ = parent.replace_resolved("ray.test", dist_init_addr="1.2.3.4:5000")
        self.addCleanup(reset_context)
        reset_context()
        publish(copy_, role="scheduler")
        self.assertEqual(get_parallel().dist_init_addr, "1.2.3.4:5000")
        self.assertEqual(
            get_schedule().chunked_prefill_size,
            resolution_result(parent, "chunked_prefill_size"),
            "publishing the copy re-ran resolution; the bag disagrees with what "
            "the parent's resolution decided",
        )

    def test_no_bare_replace_of_a_record_outside_the_helper(self):
        """`dataclasses.replace` on a record is the helper's job now.

        Derived, not listed: any `dataclasses.replace` whose first argument is
        named for a record. The helper's own call is the positive control -- if
        the scan stops seeing it, the scan broke rather than the tree.
        """
        import ast

        # The repository, not the package: the gateway is outside `sglang/`.
        package_root = pathlib.Path(next(iter(sglang.__path__))).resolve().parents[1]
        if package_root.name == "python":
            package_root = package_root.parent
        helper = "python/sglang/srt/server_args.py"
        bare, inside_helper = [], 0

        def replaces_a_record(node, record_names):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "replace"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "dataclasses"
                and node.args
            ):
                return False
            first = node.args[0]
            name = (
                first.id if isinstance(first, ast.Name) else getattr(first, "attr", "")
            )
            return name in record_names or "server_args" in name

        for path in sorted(package_root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            except SyntaxError:
                continue
            rel = path.relative_to(package_root).as_posix()
            # `self` is a record only inside the record's own class body.
            in_record_class = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
            ]
            for scope, record_names in [(tree, set())] + [
                (klass, {"self"}) for klass in in_record_class
            ]:
                for node in ast.walk(scope):
                    if not replaces_a_record(node, record_names):
                        continue
                    if rel == helper and record_names:
                        inside_helper += 1
                    elif not record_names:
                        bare.append(f"{rel}:{node.lineno}")
        self.assertEqual(
            inside_helper,
            1,
            "the scan no longer finds `replace_resolved`'s own call; it broke",
        )
        self.assertEqual(
            bare,
            [],
            "a record is copied with a bare `dataclasses.replace`, so the copy "
            "loses the parent's resolution and the next publish resolves it "
            "again: " + ", ".join(bare),
        )


class TestTheResolutionSeamHasOneCaller(CustomTestCase):
    """The pipeline is entered from exactly one place, and that place decides
    whether it runs at all.

    ``resolve_once`` is the gate: the handlers are not written to survive a
    second pass over their own output, so a record must go through the pipeline
    at most once. Keeping the pipeline itself down to a single caller is what
    makes that gate impossible to bypass -- and what keeps the remaining move
    (construction time to publish time) a matter of who calls the gate.
    """

    def test_only_the_gate_runs_the_pipeline(self):
        import ast
        from pathlib import Path

        import sglang

        package_root = Path(next(iter(sglang.__path__)))
        callers = []
        for path in sorted(package_root.rglob("*.py")):
            try:
                source = path.read_text()
                if "run_resolution_pipeline" not in source:
                    continue
                tree = ast.parse(source)
            except SyntaxError:
                continue
            # The full (class, function, ...) scope chain, so the assertion can
            # say "the one caller is ServerArgs.__post_init__" -- not merely
            # that nothing outside a function named __post_init__ calls it.
            scopes = {}
            for node in ast.walk(tree):
                own = scopes.get(id(node), ())
                if isinstance(
                    node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    own = own + (node.name,)
                for child in ast.iter_child_nodes(node):
                    scopes[id(child)] = own
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "run_resolution_pipeline"
                ):
                    rel = path.relative_to(package_root).as_posix()
                    callers.append((rel, ".".join(scopes.get(id(node), ()))))
        # Every call, compared whole: a removed call, a duplicate inside
        # __post_init__, or another class growing a same-named __post_init__
        # all show up here.
        self.assertEqual(
            [("srt/server_args.py", "ServerArgs.resolve_once")],
            callers,
            "the resolution pipeline must be entered exactly once, from "
            f"ServerArgs.resolve_once; found: {callers}",
        )

    def test_the_gate_is_reached_from_the_launcher_and_from_publish(self):
        """Both entries go through the gate, so neither can resolve twice.

        The launcher resolves the engine's record before reading any resolved
        value from it; every publishing process asks the gate on the way in and
        finds nothing left to do when the record arrived resolved.
        """
        import ast
        from pathlib import Path

        import sglang

        package_root = Path(next(iter(sglang.__path__)))
        callers = []
        for path in sorted(package_root.rglob("*.py")):
            try:
                source = path.read_text()
                if "resolve_once" not in source:
                    continue
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                # `self.resolve_once()` at construction; publish looks the
                # attribute up first, so it appears as a bare name call.
                called = (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "resolve_once"
                ) or (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "resolve_once"
                )
                if called:
                    callers.append(path.relative_to(package_root).as_posix())
        machinery = {"srt/entrypoints/engine.py", "srt/runtime_context.py"}
        self.assertEqual(
            [
                # Program entries: each builds a record from its own
                # arguments and then reads effective configuration, or hands it
                # to a fork that must not be the first to probe the device.
                "benchmark/endpoint.py",
                "benchmark/offline_throughput.py",
                "benchmark/one_batch.py",
                "benchmark/one_batch_server.py",
                "compile_deep_gemm.py",
                "lang/backend/runtime_endpoint.py",
                "launch_server.py",
                # The mechanism.
                "srt/entrypoints/engine.py",
                "srt/entrypoints/http_server_engine.py",
                "srt/runtime_context.py",
            ],
            sorted(set(callers)),
            f"the resolution gate grew or lost a caller: {sorted(set(callers))}",
        )
        # The rule the list stands for: a caller that is not the mechanism
        # resolves a record it built itself from argv. Anything else was handed
        # one someone already resolved, or should publish.
        for caller in sorted(set(callers) - machinery):
            source = (package_root / caller).read_text()
            # Either the module turned argv into the record -- the dataclass,
            # the CLI classmethod, or the argv helper `launch_server.py` uses
            # -- or it hands the record to a fork, which has to resolve first:
            # the pipeline probes the device and a forked child cannot
            # re-initialize CUDA. A worker handed a resolved record is neither.
            builds_its_own = any(
                spelling in source
                for spelling in (
                    "ServerArgs(",
                    ".from_cli_args(",
                    "prepare_server_args(",
                )
            ) or ("Process(" in source and "server_args" in source)
            # `assertTrue`, not `assertIn`: the container is a whole module.
            self.assertTrue(
                builds_its_own,
                f"{caller} calls the resolution gate but does not build the "
                "record it resolves; a record it was handed is already "
                "resolved by whoever built it, and publish resolves what it "
                "is handed",
            )


class TestResolutionStaysLazy(CustomTestCase):
    """Resolving a dummy model must not load the families it never reaches.

    The forwarding slots imported their hook only when the step ran, so a
    `ServerArgs(model_path="dummy")` resolution touched four hook modules. With
    the slots gone the imports are function-local for the same reason, and a
    module-level one costs every caller of the dummy boundary -- which is every
    `override_server_args` in the test suite.
    """

    def test_no_hook_module_imports_another_at_module_scope(self):
        import ast

        import sglang

        srt = pathlib.Path(next(iter(sglang.__path__))).resolve() / "srt"
        offenders = []
        for path in sorted((srt / "arg_groups").glob("*.py")):
            for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.startswith("sglang.srt.arg_groups")
                    and node.module.endswith("_hook")
                ):
                    offenders.append(f"{path.name}:{node.lineno} -> {node.module}")
        self.assertEqual(
            offenders,
            [],
            "a hook module imports another at module scope, so loading one "
            "family drags in a family it may never call. Import it inside the "
            "function that calls it:\n  " + "\n  ".join(offenders),
        )

    def test_no_family_is_imported_before_the_step_that_calls_it(self):
        """Source-level, so it holds whatever else the process has imported.

        Every hook import inside the dispatcher must come after the imports of
        the families reached earlier and before its own first call -- what an
        eager block at the top of the function breaks, and what a `sys.modules`
        diff cannot see once another test has loaded those modules.
        """
        import ast

        import sglang

        srt = pathlib.Path(next(iter(sglang.__path__))).resolve() / "srt"
        tree = ast.parse(
            (srt / "arg_groups" / "pipeline.py").read_text(encoding="utf-8-sig")
        )
        dispatch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_resolution_pipeline"
        )
        early_return = min(
            (
                n.lineno
                for n in ast.walk(dispatch)
                if isinstance(n, ast.Return) and n.value is None
            ),
            default=None,
        )
        self.assertIsNotNone(early_return, "the dummy short circuit is gone")

        imported_early, called_early = set(), set()
        for node in ast.walk(dispatch):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.endswith("_hook")
                and node.lineno < early_return
            ):
                imported_early.add(node.module.rsplit(".", 1)[-1])
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.lineno < early_return
            ):
                called_early.add(node.func.id)

        hooks = {}
        for path in sorted((srt / "arg_groups").glob("*_hook.py")):
            for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
                if isinstance(node, ast.FunctionDef):
                    hooks[node.name] = path.stem
        needed_early = {hooks[name] for name in called_early if name in hooks}
        self.assertEqual(
            imported_early - needed_early,
            set(),
            "the dispatcher imports a hook family before the dummy short "
            "circuit without calling it there, so every dummy resolution pays "
            "for a family it never reaches",
        )

    def test_a_dummy_resolution_loads_only_what_it_reaches(self):
        """The same claim measured, in an interpreter of its own.

        In-process this would be vacuous: another test that resolved a real
        model has already imported the late families, and the `sys.modules`
        diff comes back empty.
        """
        import subprocess
        import sys

        import sglang

        probe = (
            "import sys\n"
            "from sglang.srt.server_args import ServerArgs\n"
            "before = set(sys.modules)\n"
            "ServerArgs(model_path='dummy').resolve_once()\n"
            "print(','.join(sorted(m.rsplit('.', 1)[-1] for m in set(sys.modules) - before"
            " if '.arg_groups.' in m and m.endswith('_hook'))))\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(
            pathlib.Path(next(iter(sglang.__path__))).resolve().parent
        )
        out = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        loaded = [name for name in out.stdout.strip().split(",") if name]
        self.assertTrue(loaded, f"the probe reported nothing:\n{out.stdout}")
        for late in ("model_hook", "cuda_graph_hook", "attention_hook", "lora_hook"):
            self.assertNotIn(late, loaded)


if __name__ == "__main__":
    unittest.main()

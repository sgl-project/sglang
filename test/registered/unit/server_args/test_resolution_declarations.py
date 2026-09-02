"""Resolution writes are recorded, not just applied.

The projection that replaces field materialization reads the declaration stash,
so a resolution write that only assigns the field is invisible to it. Every
resolver declares now -- the record's handlers through `self._declare`, the
hooks and hardware defaults through `declare_resolution` -- and that is pinned
two ways: no bare assignment to a field survives anywhere a ServerArgs instance
is in reach, and after resolution `resolution_result` answers for every declared
field with what the stash holds. The second check is what the stash is measured
against: the two can disagree only if something wrote behind the stash's back. A
third check runs the other way -- every field resolution moved has to be
explained by the stash, which covers the spellings a source scan cannot see.
"""

import ast
import copy
import dataclasses
import json
import os
import pathlib
import shutil
import tempfile
import unittest
import unittest.mock

import sglang
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).resolve().parent / "srt"

# Every field of the record: resolution has no bare-assignment writer left, so
# the scan states that as a whole rather than a converted-so-far list.
_RESOLVED_FIELDS = frozenset(field.name for field in dataclasses.fields(ServerArgs))

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
    # A pass and a handler both decide this one: waterfill forces `deepep`
    # and the ascend handler wants `none`. Without this shape nothing in
    # the set reaches a field two writers disagree about.
    {"enable_waterfill": True, "moe_a2a_backend": "ascend_tp"},
)

# Which converted fields the shapes above reach; the rest need a device or an
# architecture no CPU fixture has, and the source scan covers those. Pinned so
# a shape that stops reaching a field fails here. Add to it when adding a shape.
_REACHED_BY_SHAPES = frozenset(
    {
        "_speculative_draft_quantization_explicitly_set",
        "allowed_media_domains",
        "attention_backend",
        "chunked_prefill_size",
        "cuda_graph_config",
        "custom_weight_loader",
        "device",
        "disable_cuda_graph",
        "disaggregation_ib_device",
        "dp_size",
        "enable_dp_attention",
        "enable_dp_attention_local_control_broadcast",
        "enable_dp_lm_head",
        "enable_flashinfer_allreduce_fusion",
        "encoder_transfer_backend",
        "enforce_disable_flashinfer_allreduce_fusion",
        "ep_size",
        "expert_distribution_recorder_buffer_size",
        "flashinfer_allreduce_fusion_backend",
        "grammar_backend",
        "hicache_ratio",
        "keep_mm_feature_on_device",
        "load_balance_method",
        "max_running_requests",
        "mem_fraction_static",
        "mm_feature_transport",
        "mm_process_config",
        "moe_a2a_backend",
        "moe_dense_tp_size",
        "moe_dp_size",
        "page_size",
        "random_seed",
        "return_hidden_states_mode",
        "sampling_backend",
        "schedule_conservativeness",
        "served_model_name",
        "speculative_algorithm",
        "speculative_draft_model_quantization",
        "tokenizer_path",
        "uses_mamba_radix_cache",
    }
)


def _late_resolvers():
    """Callables that reach `declare_late_resolution`, derived per module."""
    found = set()
    for relative in ("server_args.py", "parser/template_detection.py"):
        tree = ast.parse((_SRT / relative).read_text(encoding="utf-8-sig"))
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        def reaches(name, seen=None):
            seen = seen if seen is not None else set()
            if name in seen or name not in functions:
                return False
            seen.add(name)
            for node in ast.walk(functions[name]):
                if not isinstance(node, ast.Call):
                    continue
                called = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else getattr(node.func, "id", None)
                )
                if called == "declare_late_resolution":
                    return True
                if called and reaches(called, seen):
                    return True
            return False

        found |= {name for name in functions if reaches(name)}
    return found


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
    # A parameter *named* `server_args` counts with or without the annotation.
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
        names |= {
            arg.arg for arg in args.posonlyargs + args.args if arg.arg == "server_args"
        }
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
            # Destructured targets count: `(sa.a, sa.b) = f()` writes two
            # fields and is not an `ast.Attribute` at the top level.
            flat = []
            for target in targets:
                if isinstance(target, (ast.Tuple, ast.List)):
                    flat.extend(target.elts)
                else:
                    flat.append(target)
            for target in flat:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in names
                    and target.attr in _RESOLVED_FIELDS
                ):
                    found.append(
                        f"{path.relative_to(_SRT)}:{node.lineno} "
                        f"{target.value.id}.{target.attr}"
                    )
    return sorted(found)


def shape_key(shape):
    """A shape rendered short enough for a failure message."""
    return ",".join(f"{k}={v}" for k, v in sorted(shape.items())) or "defaults"


def _stash_overlay(server_args):
    """What the declarations say, last writer wins -- the projection's input."""
    overlay = {}
    for _source, declared in getattr(server_args, "_resolved_overrides", None) or ():
        overlay.update(declared)
    return overlay


def _live_topology_leaves():
    """Names `ParallelContext` serves from the live topology, not the config.

    Read out of the class: each shadowed name arrives as `self._v("<name>",
    <getter>)`. Inferring them from "did the read raise" is wrong -- it only
    raises while the process groups are missing, so in a process where an
    earlier test built them the property answers the *live* size and a leaf
    check reads it as a config mismatch (`parallel.tp_size: bag=1
    resolution=2`). Whether they are shadowed is a property of the class, not
    of the process.
    """
    tree = ast.parse((_SRT / "runtime_context.py").read_text(encoding="utf-8-sig"))
    parallel = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "ParallelContext"
    )
    names = set()
    for node in ast.walk(parallel):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_v"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            names.add(node.args[0].value)
    return frozenset(names)


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
        server_args = ServerArgs(model_path=path, device="cuda", **fields)
        server_args.resolve_once()
        return server_args

    def test_converted_fields_are_not_assigned_bare(self):
        bare = _bare_assignments()
        self.assertEqual(
            bare,
            [],
            "a converted field is assigned directly, so the projection would "
            "not see this write:\n  " + "\n  ".join(bare),
        )

    def test_the_stash_accounts_for_every_change_resolution_made(self):
        """The other direction: a field resolution moved is in the stash.

        The source scan states that no *assignment* escapes, which leaves the
        spellings a source scan cannot see -- a computed name, a write through
        a helper the scan does not recognize as holding the record. This
        compares the resolved value against what the caller supplied (or the
        field's default) and asks the stash to explain every difference, which
        is what the projection has to be able to do.
        """
        unexplained = []
        for shape in _SHAPES:
            supplied = {"random_seed": 42, **shape}
            server_args = self._resolve(shape)
            overlay = _stash_overlay(server_args)
            for field in dataclasses.fields(server_args):
                if field.name in ("model_path", "device") or field.name in overlay:
                    continue
                if field.name in supplied:
                    before = supplied[field.name]
                elif field.default is not dataclasses.MISSING:
                    before = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    before = field.default_factory()
                else:
                    continue
                after = getattr(server_args, field.name, None)
                if after != before:
                    unexplained.append(
                        f"{shape} -> {field.name}: {before!r} -> {after!r}"
                    )
        self.assertEqual(
            unexplained,
            [],
            "resolution moved these fields without declaring them, so the "
            "projection would answer with the unresolved value:\n  "
            + "\n  ".join(unexplained),
        )

    def test_a_declaration_only_resolver_leaves_the_field_alone(self):
        """The direction of travel: resolution decides, the record does not move.

        A resolver that only declares -- a model-specific override, a registry
        entry -- writes nothing onto the record. The projection carries its
        answer and the field still holds what the caller passed.
        """
        from sglang.srt.arg_groups.arg_utils import namespace_of
        from sglang.srt.arg_groups.overrides import resolution_result

        found = []
        for shape in _SHAPES:
            server_args = self._resolve(shape)
            raw = getattr(server_args, "_raw_input", None) or {}
            for field in namespace_of(type(server_args)):
                if field not in raw:
                    continue
                decided = resolution_result(server_args, field)
                on_record = getattr(server_args, field)
                if decided == on_record:
                    continue
                # It moved away from the record's value, so the record must
                # still hold exactly what the caller passed.
                self.assertEqual(
                    on_record,
                    raw[field],
                    f"{shape} -> {field}: the record holds {on_record!r}, which "
                    f"is neither the raw input {raw[field]!r} nor what "
                    f"resolution decided ({decided!r})",
                )
                found.append((shape_key(shape), field))
        self.assertNotEqual(
            found,
            [],
            "no field is resolved by declaration alone any more, so this check "
            "no longer covers anything -- either the shapes stopped reaching "
            "one or the declarations are writing the fields again",
        )

    def test_the_whole_object_readback_carries_only_fields(self):
        """`/server_info` and its gRPC and in-process twins report
        `ServerArgs.resolved_dict()`.

        The dump is exactly the field names, carrying the resolution result
        for each. It holds none of the resolution bookkeeping (`_raw_input`, the
        declaration stash, the finished flag) and no `ModelConfig` memo: none of
        that is configuration, and all of it would cross IPC with the
        readback.
        """
        server_args = self._resolve({"tp_size": 2})
        dump = server_args.resolved_dict()
        self.assertEqual(
            sorted(dump),
            sorted(field.name for field in dataclasses.fields(server_args)),
            "the readback dump is no longer exactly the fields",
        )
        leaked = sorted(
            name
            for name in vars(server_args)
            if name not in dump and not name.startswith("__")
        )
        self.assertNotEqual(
            leaked, [], "nothing to leak any more -- this check is now vacuous"
        )

    def test_every_published_leaf_is_what_resolution_decided(self):
        """One hop further than the check above: the leaf a reader reads.

        The projection's *input* agreeing with the record says nothing about
        the last hop: whether the leaf is reachable through the path the
        metadata declares, and whether it carries the resolved value once it
        is. Both sides here come from that metadata, so this cannot tell that
        a field is assigned to the *wrong* group -- the readers are the
        independent source for that, and
        `test_server_args_namespaces.py::test_the_readers_agree_with_the_namespace_metadata`
        is where the two are compared.
        """
        import sglang.srt.runtime_context as runtime_context
        from sglang.srt.arg_groups.arg_utils import namespace_of
        from sglang.srt.arg_groups.overrides import resolution_result
        from sglang.srt.runtime_context import publish, reset_context

        mapping = namespace_of(ServerArgs)
        self.assertGreater(len(mapping), 400, "the namespace mapping collapsed")

        self.assertEqual(
            set(),
            _live_topology_leaves() & set(mapping),
            "a parallel leaf gained a live member of the same name, so the "
            "comparison below reads the group rather than the published leaf",
        )

        compared = 0
        unreachable, mismatched = [], []
        for shape in _SHAPES:
            self.addCleanup(reset_context)
            server_args = self._resolve(shape)
            publish(server_args, role="scheduler")
            for field, path in mapping.items():
                groups = path.split(".")
                accessor = getattr(runtime_context, f"get_{groups[0]}", None)
                if accessor is None:
                    unreachable.append(f"no get_{groups[0]}() for {path}.{field}")
                    continue
                node = accessor()
                try:
                    for group in groups[1:]:
                        node = getattr(node, group)
                    leaf = getattr(node, field)
                except Exception as exc:
                    unreachable.append(f"{path}.{field}: {type(exc).__name__}: {exc}")
                    continue
                decided = resolution_result(server_args, field)
                compared += 1
                if leaf is not decided and leaf != decided:
                    mismatched.append(
                        f"{shape} -> {path}.{field}: bag={leaf!r} resolution={decided!r}"
                    )
            reset_context()
        self.assertEqual(
            unreachable,
            [],
            "these leaves are mapped to a namespace that cannot serve them, so "
            "a reader following the mapping raises:\n  " + "\n  ".join(unreachable),
        )
        self.assertEqual(
            mismatched,
            [],
            "the published leaf and the resolution result disagree:\n  "
            + "\n  ".join(mismatched),
        )
        self.assertGreater(
            compared, 2000, f"only {compared} leaves were compared; the walk broke"
        )

    def test_a_child_that_received_the_record_publishes_the_same_bags(self):
        """A forked worker gets the record by pickle, and re-projects from it.

        Every process publishes, so a child's bags are only right if the
        declarations travelled with the object -- and the gate has to hold on
        the far side, or the child re-runs handlers over their own output. The
        parent's bags are the reference: this is the multi-process half of the
        projection, and nothing else exercises it.
        """
        import pickle

        import sglang.srt.runtime_context as runtime_context
        from sglang.srt.arg_groups.arg_utils import namespace_of
        from sglang.srt.runtime_context import publish, reset_context

        mapping = namespace_of(ServerArgs)

        def leaves():
            out = {}
            for field, path in mapping.items():
                groups = path.split(".")
                accessor = getattr(runtime_context, f"get_{groups[0]}", None)
                if accessor is None:
                    continue
                node = accessor()
                try:
                    for group in groups[1:]:
                        node = getattr(node, group)
                    out[f"{path}.{field}"] = repr(getattr(node, field))
                except Exception:
                    continue
            return out

        for shape in _SHAPES:
            self.addCleanup(reset_context)
            parent = self._resolve(shape)
            publish(parent, role="scheduler")
            expected = leaves()

            blob = pickle.dumps(parent)
            reset_context()
            child = pickle.loads(blob)
            entered = []
            from sglang.srt.arg_groups import pipeline as pipeline_module

            original = pipeline_module.run_resolution_pipeline

            def counted(server_args, _original=original):
                entered.append(1)
                return _original(server_args)

            with unittest.mock.patch.object(
                pipeline_module, "run_resolution_pipeline", counted
            ):
                publish(child, role="scheduler")
            self.assertEqual(
                entered,
                [],
                f"{shape}: the child resolved again, so its handlers ran over "
                "the parent's output",
            )
            differences = {
                key: (expected[key], value)
                for key, value in leaves().items()
                if expected.get(key) != value
            }
            self.assertEqual(
                differences,
                {},
                f"{shape}: the child published different values than the "
                f"parent: {differences}",
            )
            reset_context()

    def test_late_resolution_reaches_the_projection(self):
        """Resolution staged after `__post_init__` is still resolution.

        The parser detection and the LoRA normalization run at launcher stage --
        they need a tokenizer, a chat template, an adapter directory -- and they
        declare through `declare_late_resolution`. The declaration is the only
        home for what they decide: the record keeps `--reasoning-parser auto`,
        and the bags a process publishes carry the detected parser.

        A real model path, not the dummy one: a dummy record never materializes,
        so its `resolve_once` re-runs and re-snapshots the raw input from
        already-late-resolved fields, which hides exactly this.
        """
        from sglang.srt.arg_groups.overrides import declare_late_resolution
        from sglang.srt.runtime_context import get_serving, publish, reset_context

        server_args = self._resolve({"reasoning_parser": "auto"})
        self.addCleanup(reset_context)
        declare_late_resolution(
            server_args, "template-detection", reasoning_parser="qwen3"
        )
        self.assertEqual(
            resolution_result(server_args, "reasoning_parser"),
            "qwen3",
            "the projection still reports what the caller asked for, so the "
            "bags would publish an unresolved parser",
        )
        publish(server_args, role="tokenizer")
        self.assertEqual(get_serving().reasoning_parser, "qwen3")
        self.assertEqual(
            server_args.reasoning_parser,
            "auto",
            "the record is the operator's input; late resolution declares, it "
            "does not write back",
        )

    def test_pre_engine_late_resolution_reaches_the_projection(self):
        """A launcher declaration survives the engine's first resolution pass."""
        from sglang.srt.arg_groups.overrides import declare_late_resolution

        server_args = ServerArgs(model_path="dummy")
        declare_late_resolution(
            server_args,
            "launcher",
            enable_forward_pass_metrics=True,
        )

        server_args.resolve_once()

        self.assertTrue(resolution_result(server_args, "enable_forward_pass_metrics"))
        self.assertFalse(server_args.enable_forward_pass_metrics)

    def test_validation_can_still_resolve_before_the_record_is_published(self):
        """The LoRA checks resolve, so they must precede publish.

        `check_server_args` is not read-only: it infers `enable_lora`, parses
        adapter paths and normalizes target modules through late resolution,
        which a published record refuses. The launcher order is what keeps this
        legal, and this is the assertion that notices if it moves. What those
        declarations decide reaches the bags; the record keeps the raw form the
        operator passed.
        """
        from sglang.srt.runtime_context import get_lora, publish, reset_context

        server_args = self._resolve(
            {
                "enable_lora": True,
                "max_lora_rank": 16,
                "lora_target_modules": ["q_proj"],
            }
        )
        self.addCleanup(reset_context)
        server_args.check_server_args()
        publish(server_args, role="tokenizer")
        self.assertEqual(
            get_lora().enable_lora, resolution_result(server_args, "enable_lora")
        )
        self.assertEqual(
            get_lora().lora_target_modules,
            resolution_result(server_args, "lora_target_modules"),
        )
        self.assertEqual(
            server_args.lora_target_modules,
            ["q_proj"],
            "normalization is a declaration; the record keeps what was passed",
        )

    def test_the_launcher_finishes_resolving_before_it_publishes(self):
        """Every late resolver runs above the publish, in the source.

        A published record refuses to be written, so a late resolver below the
        publish raises at startup rather than at test time -- and only for the
        configuration that reaches it, which is why the LoRA path can break
        while every other launch stays green. Both sides are derived: which
        callables reach `declare_late_resolution`, and where the launcher calls
        them.
        """
        launcher = _SRT / "entrypoints/engine.py"
        late = {"check_server_args", "resolve_auto_parsers"} | _late_resolvers()
        tree = ast.parse(launcher.read_text(encoding="utf-8-sig"))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_launch_subprocesses"
        )
        published_at = [
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "publish"
        ]
        self.assertEqual(len(published_at), 1, "the launcher publishes once")
        too_late = sorted(
            f"{name}() at line {node.lineno}"
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            for name in [
                (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else getattr(node.func, "id", None)
                )
            ]
            if name in late and node.lineno > published_at[0]
        )
        self.assertEqual(
            too_late,
            [],
            f"these resolve after the launcher publishes at line "
            f"{published_at[0]}, and a published record refuses to be "
            f"written:\n  " + "\n  ".join(too_late),
        )

    def test_an_undeclared_field_still_holds_the_raw_input(self):
        """Nothing writes a field behind the stash's back.

        Comparing the stash against `resolution_result` would agree by
        construction -- both are the same last-writer-wins walk over
        `_resolved_overrides`, spelled forwards and backwards. The independent
        source is the record's own `_raw_input` snapshot: a field with no
        declaration has to still equal what the caller passed, because the only
        sanctioned way to move one is to declare it.
        """
        moved = []
        for shape in _SHAPES:
            server_args = self._resolve(shape)
            overlay = _stash_overlay(server_args)
            raw_input = getattr(server_args, "_raw_input", None)
            self.assertTrue(raw_input, f"{shape}: the record kept no raw snapshot")
            for field in dataclasses.fields(server_args):
                name = field.name
                if name in overlay or name not in raw_input:
                    continue
                current = getattr(server_args, name, None)
                if current != raw_input[name]:
                    moved.append(
                        f"{shape} -> {name}: raw={raw_input[name]!r} "
                        f"field={current!r}"
                    )
        self.assertEqual(
            moved,
            [],
            "these fields moved without a declaration, so the bags publish one "
            "value while the record shows another:\n  " + "\n  ".join(moved),
        )

    def test_no_immediate_writer_overrides_a_deferred_one(self):
        """A handler must not declare over a value a pass already decided.

        Routing a bare write into the stash changed which writer wins: the
        appended entry is replayed last, so a handler now beats a pass or a
        registry entry that ran earlier -- where before, the pass's declaration
        was applied on top of the handler's bare write. One handler was found
        that way (it gated on the raw field while its neighbours read the
        resolving view, so `--enable-waterfill --moe-a2a-backend ascend_tp`
        silently stopped forcing `deepep`).

        This is the invariant rather than that instance: walk the stash in
        order and fail when an immediate writer declares a field whose previous
        entry came from a deferred writer with a *different* value. The
        deferred sources are derived from the live registries and the constant
        override table, so a new pass is covered without being listed.
        """
        from sglang.srt.arg_groups import overrides

        deferred = {
            getattr(fn, "__qualname__", getattr(fn, "__name__", ""))
            for fn in overrides.POST_PROCESS_PASSES
        }
        deferred |= {
            getattr(fn, "__qualname__", getattr(fn, "__name__", ""))
            for fns in overrides._MODEL_OVERRIDE_FNS.values()
            for fn in fns
        }
        deferred |= {
            getattr(fn, "__qualname__", getattr(fn, "__name__", ""))
            for _predicate, fn in overrides._PREDICATE_OVERRIDE_FNS
        }
        # The constant arch -> {field: value} table is a deferred writer too --
        # it has no callable, and its stash source is spelled by the collector
        # (`MODEL_OVERRIDES[<arch>]`).
        deferred |= {f"MODEL_OVERRIDES[{arch!r}]" for arch in overrides.MODEL_OVERRIDES}
        self.assertGreater(
            len(deferred), 40, "the deferred-writer set collapsed; nothing to compare"
        )

        inversions = []
        for shape in _SHAPES:
            server_args = self._resolve(shape)
            decided_by = {}
            for source, declared in getattr(server_args, "_resolved_overrides", []):
                for field, value in declared.items():
                    previous = decided_by.get(field)
                    if (
                        previous is not None
                        and previous[0] in deferred
                        and source not in deferred
                        and previous[1] != value
                    ):
                        inversions.append(
                            f"{shape} -> {field}: {previous[0]} decided "
                            f"{previous[1]!r}, then {source} declared {value!r}"
                        )
                    decided_by[field] = (source, value)
        self.assertEqual(
            inversions,
            [],
            "a handler declared over a value a pass or a registry entry had "
            "already decided; if the handler is meant to win, say so, and if "
            "it is gating on the field, it has to read the resolving view:\n  "
            + "\n  ".join(inversions),
        )

    def test_a_nested_resolution_decision_reaches_the_bags(self):
        """Resolution also decides *inside* a declared object.

        The graph sizing writes `cuda_graph_config.decode.max_bs` through the
        object the parse step declared -- no field is assigned, so nothing
        records it. It reaches the bags because the stash holds that same
        object; a copy taken when it was declared would publish the `None` the
        parse step declared while the process runs with a real batch size.
        """
        from sglang.srt.runtime_context import get_exec, publish, reset_context

        server_args = self._resolve({"disaggregation_mode": "prefill"})
        self.addCleanup(reset_context)
        # Snapshot before publishing: the bag serves the very object the record
        # holds, so comparing them after the fact compares an object with
        # itself and passes however the projection behaves.
        expected = copy.deepcopy(resolution_result(server_args, "cuda_graph_config"))
        publish(server_args, role="scheduler")
        published = get_exec().graph.cuda_graph_config
        resolved = expected
        self.assertIsNotNone(
            published.decode.max_bs,
            "the published graph config carries the batch size the parse step "
            "declared, not the one the sizing handler decided",
        )
        self.assertEqual(
            (
                published.decode.max_bs,
                published.decode.backend,
                published.prefill.max_bs,
                published.prefill.backend,
            ),
            (
                resolved.decode.max_bs,
                resolved.decode.backend,
                resolved.prefill.max_bs,
                resolved.prefill.backend,
            ),
            "the bags and the record disagree about the graph configuration, "
            "so a decision made inside the declared object was dropped",
        )

    def test_every_platform_hook_that_takes_the_record_is_captured(self):
        """A second out-of-tree config hook must not arrive uncaptured.

        `apply_server_args_defaults` is the one method on the platform
        interface that is handed the record, and its implementations live in
        other distributions -- no source scan of this tree can see what they
        write, so the pipeline diffs the record across the call instead. A new
        hook of the same shape would be invisible again, and this is what
        notices. Derived from the interface rather than listed: a rename keeps
        working, an addition fails.
        """
        interface = _SRT / "platforms" / "interface.py"
        tree = ast.parse(interface.read_text(encoding="utf-8-sig"))
        taking_the_record = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            arguments = node.args
            names = [
                arg.arg
                for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs
            ]
            if any(name == "server_args" or name.endswith("_args") for name in names):
                taking_the_record.add(node.name)
        self.assertEqual(
            taking_the_record,
            {"apply_server_args_defaults"},
            "the platform interface hands the startup record to a method this "
            "test does not know about; either it only reads, or its writes need "
            "capturing like apply_server_args_defaults",
        )

        pipeline = (_SRT / "arg_groups" / "pipeline.py").read_text(encoding="utf-8-sig")
        for hook in sorted(taking_the_record):
            self.assertIn(
                f"current_platform.{hook},",
                pipeline,
                f"{hook} is called directly instead of through the write "
                "capture, so an out-of-tree plugin's defaults would be dropped "
                "by the projection",
            )

    def test_the_shapes_reach_the_fields_they_are_meant_to(self):
        """A green agreement check over an empty stash would prove nothing."""
        declared = set()
        for shape in _SHAPES:
            declared |= set(_stash_overlay(self._resolve(shape))) & _RESOLVED_FIELDS
        missing = sorted(_REACHED_BY_SHAPES - declared)
        self.assertEqual(
            missing,
            [],
            "the shapes no longer reach these converted fields, so the "
            "agreement check silently stopped covering them:\n  "
            + "\n  ".join(missing),
        )

    def test_a_platform_plugin_default_reaches_the_projection(self):
        """An out-of-tree platform writes the fields; the diff declares them.

        The plugin interface is not ours to convert -- implementations live in
        other distributions -- so its writes are captured rather than declared.
        Without the capture the projection falls through to the raw snapshot,
        which was taken before the plugin ran, and publishes the value the
        plugin overrode.
        """

        # The pipeline asks the platform other questions on the way through
        # (whether it is out of tree, whether it supports piecewise capture),
        # and which of those it reaches depends on the host.
        from sglang.srt.platforms import current_platform

        class _Plugin(type(current_platform)):
            device_name = "oot"

            def apply_server_args_defaults(self, server_args):
                server_args.attention_backend = "triton"
                server_args.schedule_conservativeness = 0.5

        from sglang.srt.arg_groups import pipeline as pipeline_module

        # The write capture runs in the dispatcher, so that is the namespace the
        # plugin has to be installed in.
        with unittest.mock.patch.object(pipeline_module, "current_platform", _Plugin()):
            server_args = self._resolve({})
        self.assertEqual(
            (
                resolution_result(server_args, "attention_backend"),
                resolution_result(server_args, "schedule_conservativeness"),
            ),
            ("triton", 0.5),
            "the platform plugin's defaults did not reach the resolution "
            "result, so the projection publishes what the operator passed "
            "instead of what the platform decided",
        )


class TestDeclaredValuesAreNotEditedLater(CustomTestCase):
    """A declaration records a value, not a handle on one.

    The stash keeps whatever object the declaring handler passed, so a handler
    that declares a mutable and then edits it in place rewrites an entry that
    already went into the log. The projection still answers with the end state,
    which is why nothing else notices: what is lost is *which* handler decided
    what, and `validate_declarations` never sees the later change at all.
    """

    def setUp(self):
        super().setUp()
        environment = dict(os.environ)

        def restore():
            os.environ.clear()
            os.environ.update(environment)

        self.addCleanup(restore)

    def _resolve_recording_each_entry(self, **supplied):
        """Resolve, deep-copying every stash entry the moment it is appended.

        The property is about the stash, so the seam is the stash: a list that
        snapshots on append. Every declaration path -- `declare_resolution`,
        `declare_late_resolution`, `declare_direct_writes` and the passes --
        reaches it through `.append`, whatever it was imported as.
        """
        recorded = []

        class _SnapshotOnAppend(list):
            def append(self, entry):
                super().append(entry)
                recorded.append((len(self) - 1, copy.deepcopy(entry)))

        class _WatchedArgs(ServerArgs):
            """Whatever list the pipeline installs, snapshot what lands in it.

            The pipeline resets the stash at the start of a resolution, so the
            seam has to survive that assignment rather than precede it.
            """

            def __setattr__(self, name, value):
                if name == "_resolved_overrides" and not isinstance(
                    value, _SnapshotOnAppend
                ):
                    value = _SnapshotOnAppend(value)
                super().__setattr__(name, value)

        path = tempfile.mkdtemp(prefix="declared_values_")
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        with open(os.path.join(path, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        server_args = _WatchedArgs(
            model_path=path, device="cuda", random_seed=42, **supplied
        )
        server_args.resolve_once()
        return server_args, recorded

    def test_no_entry_changes_after_it_is_recorded(self):
        # One shape per family of handlers that decides a graph setting.
        for label, supplied in (
            ("plain", {}),
            ("cuda_graph_knobs", {"cuda_graph_max_bs_decode": 16}),
            ("chunked_prefill", {"chunked_prefill_size": 1024}),
            ("explicit_json", {"cuda_graph_config": {"decode": {"max_bs": 12}}}),
            ("disaggregation", {"disaggregation_mode": "prefill"}),
            ("deterministic", {"enable_deterministic_inference": True}),
            ("speculative", {"speculative_algorithm": "EAGLE"}),
            ("dp_attention", {"tp_size": 2, "dp_size": 2, "enable_dp_attention": True}),
        ):
            with self.subTest(shape=label):
                server_args, recorded = self._resolve_recording_each_entry(**supplied)
                stash = server_args._resolved_overrides
                self.assertGreater(
                    len(recorded),
                    0,
                    "nothing was recorded, so this case is not watching the "
                    "declaration paths it thinks it is",
                )
                drifted = [
                    (index, was, stash[index])
                    for index, was in recorded
                    if stash[index] != was
                ]
                self.assertEqual(
                    [],
                    drifted,
                    "these entries changed after they were declared, so the log "
                    f"credits the wrong handler for the end state: {drifted}",
                )


if __name__ == "__main__":
    unittest.main()

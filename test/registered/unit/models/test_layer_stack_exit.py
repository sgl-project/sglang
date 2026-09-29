"""A decoder layer may leave its FFN all-reduce to the next layer's input norm.
The last layer on a rank has no next layer there, so a model built from such
layers must pass their output through ``finish_layer_stack`` before the final
norm or the send to the next pipeline rank. Between layers, an in-place write to
a layer's output needs the completed sum as well. Every set below is derived
from the model sources."""

import ast
import unittest
from pathlib import Path

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODELS_DIR = Path(sglang.__file__).resolve().parent / "srt" / "models"
UNREDUCED = "UnreducedOutput"
EXIT = "finish_layer_stack"
COMPLETE = "reduce_output"
FINAL_NORMS = {"norm", "norm_f", "final_layernorm"}
IN_PLACE = {"add_", "sub_", "mul_", "copy_"}


def base_names(node):
    return [
        base.id if isinstance(base, ast.Name) else base.attr
        for base in node.bases
        if isinstance(base, (ast.Name, ast.Attribute))
    ]


def method(node, name):
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    return None


def called_name(call):
    func = call.func
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)


def calls(node, name):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and called_name(sub) == name:
            yield sub


def delegates_to_super(forward):
    return any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Call)
        and getattr(call.func.value.func, "id", None) == "super"
        for call in calls(forward, "forward")
    )


def declares_no_deferral(node):
    """Builds its communicator with allow_deferred_ffn_reduction=False: its FFN
    completes its own reduction."""
    return any(
        keyword.arg == "allow_deferred_ffn_reduction"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        for keyword in call.keywords
    )


def defers(node):
    """Asks whether to leave the reduction to the next layer (directly or
    through ffn_exit), or wraps its output as unreduced itself."""
    if declares_no_deferral(node):
        return False
    return any(
        any(calls(node, name))
        for name in ("ffn_exit", "should_fuse_mlp_allreduce_with_next_layer", UNREDUCED)
    )


class Census:
    def __init__(self, include_complete=False):
        self.include_complete = include_complete
        self.trees = {}
        self.classes = {}
        for path in sorted(MODELS_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text())
            self.trees[path] = tree
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self.classes.setdefault(node.name, []).append((path, node))
        self.deferring = self._deferring_layers()
        self.family = self._layer_family()
        self.stacks = self._layer_stacks()

    def _deferring_layers(self):
        # Module-level helpers that defer on a layer's behalf.
        helpers = {
            path: {
                st.name
                for st in tree.body
                if isinstance(st, ast.FunctionDef) and defers(st)
            }
            for path, tree in self.trees.items()
        }
        deferring = {
            name
            for name, defs in self.classes.items()
            for path, node in defs
            if defers(node)
            or any(any(calls(node, h)) for h in helpers[path])
            or (
                self.include_complete
                and any(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id in {"LayerCommunicator", "MHCLayerCommunicator"}
                    for call in ast.walk(node)
                )
            )
        }
        # A subclass defers through its base unless it replaces forward
        # without delegating to it.
        while True:
            inherited = {
                name
                for name, defs in self.classes.items()
                if name not in deferring
                for _, node in defs
                if set(base_names(node)) & deferring
                and (self.include_complete or not declares_no_deferral(node))
                and (
                    method(node, "forward") is None
                    or delegates_to_super(method(node, "forward"))
                )
            }
            if not inherited:
                return deferring
            deferring |= inherited

    def _layer_family(self):
        """Classes derived from a deferring layer, whether or not they still
        defer (a subclass that replaces forward may not)."""
        family = set(self.deferring)
        while True:
            derived = {
                name
                for name, defs in self.classes.items()
                if name not in family
                for _, node in defs
                if set(base_names(node)) & family
            }
            if not derived:
                return family
            family |= derived

    def registries(self, path, names):
        """Module-level names bound to a value that references ``names``, such
        as a table of decoder layer types."""
        out = {}
        for st in self.trees[path].body:
            if isinstance(st, (ast.Assign, ast.AnnAssign)) and st.value is not None:
                members = {
                    sub.id
                    for sub in ast.walk(st.value)
                    if isinstance(sub, ast.Name) and sub.id in names
                }
                targets = st.targets if isinstance(st, ast.Assign) else [st.target]
                for target in targets:
                    if members and isinstance(target, ast.Name):
                        out[target.id] = members
        return out

    def built_layers(self, path, node, names):
        """The classes in ``names`` this class references other than as a base
        or an isinstance argument, looking through module-level tables."""
        skip = {id(base) for base in node.bases}
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and called_name(call) in ("isinstance", "issubclass")
                and len(call.args) > 1
            ):
                skip |= {id(sub) for sub in ast.walk(call.args[1])}
        tables = self.registries(path, names)
        found = set()
        for sub in ast.walk(node):
            if id(sub) in skip:
                continue
            name = sub.id if isinstance(sub, ast.Name) else getattr(sub, "attr", None)
            if name in names:
                found.add(name)
            elif name in tables:
                found |= tables[name]
        return found

    def _layer_stacks(self):
        """Classes whose instances run deferring layers: they build them, hold a
        forward-less container that does, or subclass such a class."""
        stacks = {}
        runners = set(self.deferring)
        while True:
            grown = False
            for name, defs in self.classes.items():
                if name in self.deferring:
                    continue
                for path, node in defs:
                    key = (path, node.name)
                    if key in stacks:
                        continue
                    layers = self.built_layers(path, node, runners)
                    parents = set(base_names(node)) & {n for _, n in stacks}
                    if not layers and not parents:
                        continue
                    if not layers and self.built_layers(
                        path, node, self.family - self.deferring
                    ):
                        continue  # a subclass that builds layers which never defer
                    stacks[key] = (node, layers & self.deferring)
                    if self.used_forward(path, node) is None:
                        runners.add(name)
                    grown = True
            if not grown:
                return stacks

    def used_forward(self, path, node, seen=()):
        forward = method(node, "forward")
        if forward is not None:
            return path, node, forward
        for base in base_names(node):
            for base_path, base_node in self.classes.get(base, ()):
                if (base_path, base) not in seen:
                    found = self.used_forward(
                        base_path, base_node, seen + ((base_path, base),)
                    )
                    if found:
                        return found
        return None

    def subjects(self):
        """forward methods the layer stacks run, with the deferring layers
        each stack builds."""
        out = {}
        for (path, _), (node, layers) in self.stacks.items():
            found = self.used_forward(path, node)
            if found is None:
                continue  # a container, run by the class that holds it
            fpath, fnode, forward = found
            key = f"{fpath.relative_to(MODELS_DIR)}:{fnode.name}"
            entry = out.setdefault(key, (forward, set()))
            entry[1].update(layers)
        return out

    def layer_calls_exit(self, name):
        return any(
            method(node, "forward") is not None
            and any(calls(method(node, "forward"), EXIT))
            for _, node in self.classes.get(name, ())
        )


def inside_loop(tree, target):
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)) and node is not tree:
            if any(sub is target for sub in ast.walk(node)):
                return True
    return False


def is_final_norm(call):
    func, chain = call.func, []
    while isinstance(func, ast.Attribute):
        chain.append(func.attr)
        func = func.value
    return bool(FINAL_NORMS & set(chain))


class TestLayerStackExit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.census = Census()
        cls.subjects = cls.census.subjects()
        cls.complete_census = Census(include_complete=True)

    def test_derivation_finds_each_kind_of_stack(self):
        for name in (
            "glm4_moe_lite.py:Glm4MoeLiteModel",  # builds its layers
            "qwen2_moe.py:Qwen2MoeModel",  # run by Qwen3-MoE, which passes them in
            "qwen3_next.py:Qwen3NextModel",  # layers defer through a helper
            "nemotron_h.py:NemotronHModel",  # layers come from a type table
            "dots3_common/nextn.py:Dot3NoteModelNextN",  # holds an MTP head
            "mistral_large_3_eagle.py:MistralLarge3EagleModel",  # subclass
        ):
            self.assertIn(name, self.subjects)

    def test_each_stack_ends_at_the_exit(self):
        problems = []
        for name, (forward, layers) in sorted(self.complete_census.subjects().items()):
            exits = list(calls(forward, EXIT)) + list(calls(forward, "take_output"))
            if not exits:
                if delegates_to_super(forward):
                    continue
                if layers and all(
                    self.complete_census.layer_calls_exit(n) for n in layers
                ):
                    continue  # the last layer ends the stack itself
                problems.append(f"{name}: no {EXIT} call")
                continue
            if len(exits) > 1 or inside_loop(forward, exits[0]):
                problems.append(f"{name}: {EXIT} must be called once, after the loop")
                continue
            line = exits[0].lineno
            for call in ast.walk(forward):
                if not isinstance(call, ast.Call) or call.lineno >= line:
                    continue
                if called_name(call) == "PPProxyTensors" or is_final_norm(call):
                    problems.append(
                        f"{name} line {call.lineno}: {called_name(call)} before {EXIT}"
                    )
        self.assertEqual(problems, [])

    def test_pipeline_residual_is_received_through_the_boundary(self):
        problems = []
        for name, (forward, _) in self.complete_census.subjects().items():
            for node in ast.walk(forward):
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "pp_proxy_tensors"
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value == "residual"
                ):
                    problems.append(f"{name}:{node.lineno}")
        self.assertEqual(problems, [])

    def test_models_use_boundary_access_for_output_values(self):
        problems = []
        forbidden = {
            "UnreducedOutput",
            "HandoffOutput",
            "OwedOutput",
            "Contribution",
            "ResidualStream",
            "reduce_output",
            "layer_input_buffer",
        }
        for path, tree in self.complete_census.trees.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in forbidden:
                    problems.append(f"{path.name}:{node.lineno}: {node.id}")
        for name, (forward, _) in self.complete_census.subjects().items():
            for node in ast.walk(forward):
                if (
                    isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.Add)
                    and isinstance(node.left, ast.Name)
                    and isinstance(node.right, ast.Name)
                    and {node.left.id, node.right.id} == {"hidden_states", "residual"}
                ):
                    problems.append(f"{name}:{node.lineno}: residual add")
        self.assertEqual(problems, [])

    def test_split_prefill_completes_only_at_the_final_segment(self):
        census = self.complete_census
        containers = {name for _, name in census.stacks}
        while True:
            grown = {
                name
                for name, definitions in census.classes.items()
                for path, node in definitions
                if set(base_names(node)) & containers
                or census.built_layers(path, node, containers)
            } - containers
            if not grown:
                break
            containers |= grown
        checked = 0
        for name in sorted(containers):
            for path, node in census.classes[name]:
                forward = method(node, "forward_split_prefill")
                if forward is None:
                    continue
                if any(calls(forward, "forward_split_prefill")):
                    continue  # delegates to the inherited implementation
                with self.subTest(model=name, file=path.name):
                    exits = list(calls(forward, EXIT)) + list(
                        calls(forward, "take_output")
                    )
                    self.assertEqual(len(exits), 1)
                    exit_call = exits[0]
                    self.assertFalse(inside_loop(forward, exit_call))
                    final_segments = [
                        item
                        for item in ast.walk(forward)
                        if isinstance(item, ast.If)
                        and isinstance(item.test, ast.Compare)
                        and isinstance(item.test.left, ast.Name)
                        and item.test.left.id == "end"
                        and any(part is exit_call for part in ast.walk(item))
                    ]
                    early_returns = [
                        item
                        for item in forward.body
                        if isinstance(item, ast.If)
                        and isinstance(item.test, ast.Compare)
                        and isinstance(item.test.left, ast.Name)
                        and item.test.left.id == "end"
                        and any(isinstance(op, ast.NotEq) for op in item.test.ops)
                        and any(isinstance(part, ast.Return) for part in item.body)
                        and item.end_lineno < exit_call.lineno
                    ]
                    self.assertEqual(len(final_segments) + len(early_returns), 1)
                    for call in ast.walk(forward):
                        if isinstance(call, ast.Call) and is_final_norm(call):
                            self.assertLess(exit_call.lineno, call.lineno)
                    checked += 1
        self.assertGreater(checked, 0)

    def test_in_place_writes_between_layers_see_the_reduced_sum(self):
        problems = []
        for name, (forward, _) in sorted(self.subjects.items()):
            completes = [call.lineno for call in calls(forward, COMPLETE)]
            for call in ast.walk(forward):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr in IN_PLACE
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "hidden_states"
                    and not any(line < call.lineno for line in completes)
                ):
                    problems.append(f"{name} line {call.lineno}")
        self.assertEqual(
            problems, [], f"call {COMPLETE}(hidden_states) before writing in place"
        )


if __name__ == "__main__":
    unittest.main()

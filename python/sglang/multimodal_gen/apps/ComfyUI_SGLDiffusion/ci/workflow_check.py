"""Static validation of ComfyUI API-format workflows against node definitions.

Definitions are dumped from a real ComfyUI checkout (plus this plugin) in a
subprocess, so importing ComfyUI never pollutes the caller's interpreter:

    python workflow_check.py --comfyui-dir /path/to/ComfyUI [--workflows DIR]

The validation itself is pure Python over a JSON-able definitions dict:
    {class_type: {"input": INPUT_TYPES(), "returns": [...]}}
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = PLUGIN_DIR / "workflows"

# Node packs that are not part of ComfyUI core or this plugin. They are
# reported as unverified, not as failures.
DEFAULT_EXTERNAL_CLASSES = frozenset(
    {"easy prompt", "easy showAnything", "MinimaxH3LatentUpscaler3D"}
)
# Combo inputs whose options come from the host's files, not from the node.
DEFAULT_SKIP_COMBO_INPUTS = frozenset({"image"})

# Appended by the dump script to combos that list host files.
FILES_MARKER = "__host_files__"
WARNING_KINDS = frozenset({"unknown_input"})
_AUTOGROW = "COMFY_AUTOGROW_V3"
_DYNCOMBO = "COMFY_DYNAMICCOMBO_V3"


@dataclass(frozen=True)
class Problem:
    workflow: str
    node_id: str
    class_type: str
    kind: str
    message: str

    @property
    def severity(self) -> str:
        # ComfyUI drops inputs a node does not declare at execution time, so
        # these are stale-widget warnings, not failures.
        return "warning" if self.kind in WARNING_KINDS else "error"

    def __str__(self) -> str:
        return f"{self.workflow}: node {self.node_id} [{self.class_type}] {self.kind}: {self.message}"


@dataclass
class Report:
    workflow: str
    problems: list
    unverified: list  # (node_id, class_type) of external classes

    @property
    def errors(self) -> list:
        return [p for p in self.problems if p.severity == "error"]

    @property
    def warnings(self) -> list:
        return [p for p in self.problems if p.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def _sections(defn):
    inp = defn.get("input", {})
    return {**inp.get("required", {}), **inp.get("optional", {})}


def _spec_type(spec):
    return spec[0] if spec else None


def _spec_opts(spec):
    return spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}


def _dyn_children(spec):
    """Names of all inputs nested under any option of a dynamic combo."""
    names = {}
    for opt in _spec_opts(spec).get("options", []):
        inputs = opt.get("inputs", {})
        for sec in ("required", "optional"):
            for k, v in inputs.get(sec, {}).items():
                names[k] = v
                if _spec_type(v) == _DYNCOMBO:
                    names.update(_dyn_children(v))
    return names


def _expand_inputs(defn):
    """Return (allowed {name: spec}, required set, autogrow {prefix: type})."""
    allowed, autogrow = {}, {}
    for name, spec in _sections(defn).items():
        t = _spec_type(spec)
        if t == _AUTOGROW:
            tmpl = _spec_opts(spec).get("template", {}).get("input", {})
            prefix = _spec_opts(spec).get("template", {}).get("prefix", "")
            for sec in ("required", "optional"):
                for _, sub in tmpl.get(sec, {}).items():
                    autogrow[prefix] = _spec_type(sub)
            allowed[name] = spec
        elif t == _DYNCOMBO:
            allowed[name] = spec
            for child, cspec in _dyn_children(spec).items():
                allowed.setdefault(child, cspec)
                allowed.setdefault(f"{name}.{child}", cspec)
        else:
            allowed[name] = spec
    required = set(defn.get("input", {}).get("required", {}))
    return allowed, required, autogrow


def _combo_options(spec):
    t = _spec_type(spec)
    opts = (
        t
        if isinstance(t, list)
        else _spec_opts(spec).get("options")
        if t == "COMBO"
        else None
    )
    if opts and FILES_MARKER in opts:
        return None  # options come from files on the host
    return opts


def _types_compatible(want, got):
    if want in ("*", got):
        return True
    if isinstance(want, str) and "," in want:
        return got in want.split(",")
    if isinstance(got, str) and "," in got:
        return want in got.split(",")
    return got == "*"


def validate_workflow(
    name,
    wf,
    defs,
    external_classes=DEFAULT_EXTERNAL_CLASSES,
    skip_combo_inputs=DEFAULT_SKIP_COMBO_INPUTS,
) -> Report:
    problems, unverified = [], []

    def add(nid, cls, kind, msg):
        problems.append(Problem(name, nid, cls, kind, msg))

    for nid, node in wf.items():
        cls = node.get("class_type")
        if cls is None:
            add(nid, "?", "no_class_type", "node has no class_type")
            continue
        defn = defs.get(cls)
        if defn is None:
            if cls in external_classes:
                unverified.append((nid, cls))
            else:
                add(nid, cls, "unknown_class", "not in core, plugin or external list")
            continue
        allowed, required, autogrow = _expand_inputs(defn)
        inputs = node.get("inputs", {})
        for r in sorted(required):
            if r not in inputs:
                add(nid, cls, "missing_input", f"required input '{r}' is absent")
        for key, val in inputs.items():
            spec = allowed.get(key)
            if spec is None:
                for prefix, atype in autogrow.items():
                    if key.startswith(prefix) or (
                        "." in key and key.split(".", 1)[1].startswith(prefix)
                    ):
                        spec = [atype, {}]
                        break
            if spec is None:
                add(nid, cls, "unknown_input", f"'{key}' is not an input of this node")
                continue
            if isinstance(val, list) and len(val) == 2 and isinstance(val[1], int):
                src = wf.get(str(val[0]))
                if src is None:
                    add(nid, cls, "dangling_link", f"'{key}' -> missing node {val[0]}")
                    continue
                sdef = defs.get(src.get("class_type"))
                if sdef is None:
                    continue  # external source node, type unknown
                rets = sdef.get("returns", [])
                if not 0 <= val[1] < len(rets):
                    add(
                        nid,
                        cls,
                        "bad_output_slot",
                        f"'{key}' -> node {val[0]} [{src['class_type']}] slot {val[1]}, it has {len(rets)} outputs",
                    )
                    continue
                want = _spec_type(spec)
                if (
                    isinstance(want, str)
                    and not want.startswith("COMFY_")
                    and not _types_compatible(want, rets[val[1]])
                ):
                    add(
                        nid,
                        cls,
                        "type_mismatch",
                        f"'{key}' wants {want}, node {val[0]} slot {val[1]} gives {rets[val[1]]}",
                    )
            else:
                opts = _combo_options(spec)
                if opts and key not in skip_combo_inputs and val not in opts:
                    add(
                        nid,
                        cls,
                        "bad_combo_value",
                        f"'{key}'={val!r} not in {list(opts)[:8]}",
                    )
    return Report(name, problems, unverified)


def validate_dir(workflow_dir, defs, **kw):
    reports = []
    for path in sorted(Path(workflow_dir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            reports.append(validate_workflow(path.name, json.load(f), defs, **kw))
    return reports


_DUMP_SCRIPT = r"""
import asyncio, importlib.util, json, os, sys
comfy, plugin, out = sys.argv[1:4]
FILES_MARKER = "__host_files__"
sys.path.insert(0, comfy)
os.chdir(comfy)
import nodes
asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False))
mapping = dict(nodes.NODE_CLASS_MAPPINGS)
spec = importlib.util.spec_from_file_location(
    "sgld_plugin", os.path.join(plugin, "__init__.py"), submodule_search_locations=[plugin])
mod = importlib.util.module_from_spec(spec)
sys.modules["sgld_plugin"] = mod
spec.loader.exec_module(mod)
plugin_classes = dict(mod.NODE_CLASS_MAPPINGS)
if not plugin_classes:
    raise SystemExit("plugin registered no nodes (import failed inside ComfyUI env)")
mapping.update(plugin_classes)
import folder_paths
_orig = folder_paths.get_filename_list
def _marked(folder):
    return list(_orig(folder)) + [FILES_MARKER]
folder_paths.get_filename_list = _marked
defs = {}
for name, cls in mapping.items():
    try:
        defs[name] = {"input": cls.INPUT_TYPES(), "returns": list(cls.RETURN_TYPES)}
    except Exception as e:
        defs[name] = {"input": {}, "returns": list(getattr(cls, "RETURN_TYPES", ())), "error": repr(e)}
with open(out, "w") as f:
    json.dump({"defs": defs, "plugin_classes": sorted(plugin_classes)}, f, default=list)
"""


def dump_definitions(comfyui_dir, plugin_dir=PLUGIN_DIR, python=sys.executable):
    """Return (defs, plugin_class_names) from a real ComfyUI checkout."""
    env = dict(os.environ)
    # The plugin imports sglang; make the in-tree package win.
    src_root = str(Path(__file__).resolve().parents[5])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src_root, env.get("PYTHONPATH")]))
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "defs.json")
        proc = subprocess.run(
            [python, "-c", _DUMP_SCRIPT, str(comfyui_dir), str(plugin_dir), out],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"definition dump failed:\n{proc.stderr[-2000:]}")
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
    return data["defs"], data["plugin_classes"]


def format_reports(reports):
    lines = []
    for r in reports:
        lines.append(f"{'OK  ' if r.ok else 'FAIL'} {r.workflow}")
        lines += [
            f"  {p.severity} {p.kind}: node {p.node_id} [{p.class_type}] {p.message}"
            for p in r.problems
        ]
        lines += [f"  unverified external node {n} [{c}]" for n, c in r.unverified]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--comfyui-dir", default=os.environ.get("COMFYUI_DIR"))
    ap.add_argument("--workflows", default=str(WORKFLOW_DIR))
    ap.add_argument("--external", nargs="*", default=sorted(DEFAULT_EXTERNAL_CLASSES))
    args = ap.parse_args(argv)
    if not args.comfyui_dir:
        ap.error("--comfyui-dir or COMFYUI_DIR is required")
    defs, _ = dump_definitions(args.comfyui_dir)
    reports = validate_dir(
        args.workflows, defs, external_classes=frozenset(args.external)
    )
    print(format_reports(reports))
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
"""Convert a ComfyUI workflow (UI or API JSON) to run its DiT on SGLD.

Changes, all listed in the report:
- a stock diffusion-model loader with an H3 / Qwen-Image / Z-Image / FLUX.1 file
  becomes ``SGLDUNETLoader`` (also inside subgraphs);
- ``LoraLoader`` / ``LoraLoaderModelOnly`` on that model become ``SGLDLoraLoader``.
  ``LoraLoader``'s CLIP consumers are wired to its CLIP input, so the LoRA's
  text-encoder half is no longer applied.
Everything else is only reported: other MODEL -> MODEL nodes on the SGLD model
(speed patches, caches, compile) and LoRA stack nodes. Nothing depends on what is
installed on the machine that runs the converter.
"""

import copy
import json

from . import rules

MUTED, BYPASS = 2, 4
SG_INPUT, SG_OUTPUT = -10, -20  # LiteGraph ids of a subgraph's input / output node
_LINK_FIELDS = ("id", "origin_id", "origin_slot", "target_id", "target_slot", "type")
_README = "README section 'Converting a workflow'"


class Report:
    def __init__(self):
        self.changes, self.reminders = [], []

    def markdown(self) -> str:
        lines = ["# SGLD conversion report", "", "## Changes", ""]
        lines += [f"- {c}" for c in self.changes] or ["- none"]
        if self.reminders:
            lines += ["", "## Reminders: not converted, check by hand", ""]
            lines += [f"- {r}" for r in self.reminders]
        return "\n".join(lines) + "\n"


def _label(where, node):
    title = node.get("title") or (node.get("_meta") or {}).get("title")
    kind = node.get("type") or node.get("class_type")
    return f"`{where}` {kind}" + (f" ({title})" if title and title != kind else "")


def _retype(node, kind):
    node["type"] = kind
    props = node.setdefault("properties", {})
    props["Node name for S&R"] = kind
    # The frontend offers to install the pack named here when a node is missing;
    # the old core / pack id would point at the wrong one.
    for key in ("cnr_id", "ver", "aux_id"):
        props.pop(key, None)


def _active(node) -> bool:
    return node.get("mode") not in (MUTED, BYPASS)


# ---------------------------------------------------------------- UI workflow


def _lget(link, key):
    return link[_LINK_FIELDS.index(key)] if isinstance(link, list) else link.get(key)


def _lset(link, key, value):
    if isinstance(link, list):
        link[_LINK_FIELDS.index(key)] = value
    else:
        link[key] = value


class _Graph:
    """The top-level workflow or one subgraph definition."""

    def __init__(self, data, where):
        self.data, self.where = data, where
        self.nodes = {n["id"]: n for n in data.get("nodes") or []}
        self.links = {_lget(l, "id"): l for l in data.get("links") or [] if l}

    def out_links(self, node_id, slot) -> list:
        if node_id == SG_INPUT:
            return self.data["inputs"][slot].setdefault("linkIds", [])
        out = self.nodes[node_id]["outputs"][slot]
        if out.get("links") is None:
            out["links"] = []
        return out["links"]

    def drop_link(self, link_id):
        """Remove a link and its origin's reference; the caller clears the target."""
        link = self.links.pop(link_id, None)
        if link is None:
            return
        self.data["links"].remove(link)
        origin = self.out_links(_lget(link, "origin_id"), _lget(link, "origin_slot"))
        if link_id in origin:
            origin.remove(link_id)

    def clear_target(self, link):
        target, slot = _lget(link, "target_id"), _lget(link, "target_slot")
        if target == SG_OUTPUT:
            ids = self.data["outputs"][slot].get("linkIds") or []
            if _lget(link, "id") in ids:
                ids.remove(_lget(link, "id"))
        elif target in self.nodes:
            self.nodes[target]["inputs"][slot]["link"] = None


def _model_input(node):
    """Index of the input that carries the node's MODEL; None if it has none, or
    -1 if several MODEL inputs make it ambiguous (e.g. model merging)."""
    inputs = node.get("inputs") or []
    for i, inp in enumerate(inputs):
        if inp.get("name") == "model":
            return i
    if node.get("type") in ("Reroute", rules.SET_NODE) and inputs:
        return 0
    typed = [i for i, inp in enumerate(inputs) if inp.get("type") == "MODEL"]
    return typed[0] if len(typed) == 1 else (-1 if typed else None)


class _UIConverter:
    def __init__(self, wf):
        self.report = Report()
        self.root = _Graph(wf, "")
        self.subgraphs = {
            sg["id"]: _Graph(sg, f"{sg.get('name') or sg['id']}/")
            for sg in (wf.get("definitions") or {}).get("subgraphs") or []
        }
        self.graphs = [self.root, *self.subgraphs.values()]

    def instances(self, sg):
        nodes = [(g, n) for g in self.graphs for n in g.nodes.values()]
        return [(g, n) for g, n in nodes if n.get("type") == sg.data["id"]]

    def trace_input(self, g, node, index, seen):
        link = g.links.get((node.get("inputs") or [])[index].get("link"))
        if link is None:  # nothing connected: not an SGLD model
            return ("other", node)
        return self.trace(g, _lget(link, "origin_id"), _lget(link, "origin_slot"), seen)

    def trace(self, g, node_id, slot, seen=None):
        """Where the MODEL leaving ``node_id``/``slot`` is loaded: ("sgld", node),
        ("other", node) for any other root, or None when it cannot be told."""
        seen = set() if seen is None else seen
        if (id(g), node_id, slot) in seen:
            return None
        seen.add((id(g), node_id, slot))
        if node_id == SG_INPUT:  # follow every instance of this subgraph outward
            name = g.data["inputs"][slot]["name"]
            found = []
            for pg, inst in self.instances(g):
                names = [i.get("name") for i in inst.get("inputs") or []]
                if name not in names:
                    return None
                # Instances may share an upstream loader; trace each one afresh.
                found.append(self.trace_input(pg, inst, names.index(name), set(seen)))
            kinds = {f and f[0] for f in found}
            return found[0] if found and len(kinds) == 1 else None
        node = g.nodes.get(node_id)
        if node is None:
            return None
        if node.get("type") in rules.SGLD_LOADERS:
            return ("sgld", node)
        if node.get("type") == rules.GET_NODE:
            source = self.set_node(g, node)
            return None if source is None else self.trace_input(*source, 0, seen)
        if node.get("type") in self.subgraphs:  # follow the subgraph output inward
            sg = self.subgraphs[node["type"]]
            for link in sg.links.values():
                target = (_lget(link, "target_id"), _lget(link, "target_slot"))
                if target == (SG_OUTPUT, slot):
                    origin = (_lget(link, "origin_id"), _lget(link, "origin_slot"))
                    return self.trace(sg, *origin, seen)
            return None
        index = _model_input(node)
        if index is None:
            return ("other", node)
        return None if index < 0 else self.trace_input(g, node, index, seen)

    def set_node(self, g, get_node):
        """The SetNode a GetNode reads (same graph first, then the top level)."""
        name = (get_node.get("widgets_values") or [None])[0]
        for graph in (g, self.root):
            for node in graph.nodes.values():
                if (
                    node.get("type") == rules.SET_NODE
                    and (node.get("widgets_values") or [None])[0] == name
                    and node.get("inputs")
                ):
                    return graph, node
        return None

    def model_names(self, g, node):
        """The file a loader loads, once per subgraph instance when the name is a
        promoted widget (the inner widget then holds a stale value)."""
        values = node.get("widgets_values")
        own = values[0] if isinstance(values, list) and values else None
        if g is self.root:
            return [own]
        inp = next(
            (i for i in node.get("inputs") or [] if i.get("name") == "unet_name"), None
        )
        link = g.links.get(inp.get("link")) if inp else None
        promoted = (
            g.data["inputs"][_lget(link, "origin_slot")]["name"]
            if link and _lget(link, "origin_id") == SG_INPUT
            else None
        )
        names = []
        for _, inst in self.instances(g):
            values = inst.get("widgets_values") or []
            proxies = (inst.get("properties") or {}).get("proxyWidgets") or []
            widgets = [i["name"] for i in inst.get("inputs") or [] if i.get("widget")]
            if [str(node["id"]), "unet_name"] in proxies:
                idx = proxies.index([str(node["id"]), "unet_name"])
            elif promoted in widgets:
                idx = widgets.index(promoted)
            else:
                idx = None
            names.append(values[idx] if idx is not None and idx < len(values) else own)
        return names or [own]

    def swap_loaders(self):
        for g in self.graphs:
            for node in g.nodes.values():
                kind = node.get("type")
                if kind not in rules.STOCK_LOADERS:
                    continue
                names = self.model_names(g, node)
                sgld = [rules.is_sgld_model(n) for n in names]
                where = _label(f"{g.where}{node['id']}", node)
                if any(sgld) and not all(sgld):
                    self.report.reminders.append(
                        f"{where}: its subgraph is used with SGLD and non-SGLD models "
                        f"({', '.join(map(str, names))}); left unchanged."
                    )
                if not all(sgld):
                    continue
                values = list(node.get("widgets_values") or [names[0]])
                # SGLDUNETLoader has UNETLoader's widgets (the GGUF loader lacks
                # weight_dtype); the SGLD worker serves only weight_dtype 'default'.
                _note_weight_dtype(self.report, where, values[1:2])
                node["widgets_values"] = [values[0], "default"]
                _retype(node, rules.SGLD_LOADER)
                inputs = node.setdefault("inputs", [])
                if not any(i.get("name") == "sgld_options" for i in inputs):
                    inputs.append(
                        {"name": "sgld_options", "type": "SGLD_OPTIONS", "link": None}
                    )
                self.report.changes.append(
                    f"{where} -> SGLDUNETLoader (`{rules.file_name(names[0])}`)"
                )

    def convert_lora(self, g, node):
        kind, where = node["type"], _label(f"{g.where}{node['id']}", node)
        values = node.get("widgets_values") or []
        inputs = node.get("inputs") or []
        clip = next((i for i in inputs if i.get("name") == "clip"), None)
        clip_link = g.links.get(clip.get("link")) if clip else None
        source = clip_link and (
            _lget(clip_link, "origin_id"),
            _lget(clip_link, "origin_slot"),
        )
        kept = []
        for inp in inputs:
            if inp.get("name") in ("model", "lora_name", "strength_model"):
                kept.append(inp)
            elif inp.get("link") is not None:
                g.drop_link(inp["link"])
        for slot, inp in enumerate(kept):
            if inp.get("link") in g.links:
                _lset(g.links[inp["link"]], "target_slot", slot)
        node["inputs"] = kept
        outputs = node.get("outputs") or []
        for out in outputs[1:]:  # LoraLoader's CLIP output
            for link_id in list(out.get("links") or []):
                link = g.links[link_id]
                if source:
                    _lset(link, "origin_id", source[0])
                    _lset(link, "origin_slot", source[1])
                    g.out_links(*source).append(link_id)
                else:
                    g.clear_target(link)
                    g.drop_link(link_id)
        node["outputs"] = outputs[:1]
        node["widgets_values"] = [values[0], values[1], "", "all"]
        _retype(node, "SGLDLoraLoader")
        self.report.changes.append(
            f"{where} -> SGLDLoraLoader (`{values[0]}` @ {values[1]})"
        )
        if kind == "LoraLoader":
            self.report.reminders.append(
                f"{where}: the LoRA's CLIP half (strength_clip {values[2]}) is not applied; "
                "its CLIP consumers now take the CLIP that fed the LoRA node"
                + ("" if source else " (none was connected, so they are disconnected)")
                + ". SGLDLoraLoader applies the diffusion-model half only."
            )

    def run(self):
        self.swap_loaders()
        kinds = {n.get("type") for g in self.graphs for n in g.nodes.values()}
        if not kinds & rules.SGLD_LOADERS:
            return
        for g in self.graphs:
            for node in list(g.nodes.values()):
                kind, index = node.get("type"), _model_input(node)
                if (
                    not _active(node)
                    or index is None
                    or rules.is_sgld_node(kind)
                    # Nodes inside a subgraph are checked where they are defined.
                    or kind in self.subgraphs
                    or kind in ("Reroute", rules.SET_NODE)
                ):
                    continue
                found = None if index < 0 else self.trace_input(g, node, index, set())
                where = _label(f"{g.where}{node['id']}", node)
                if kind in rules.LORA_NODES:
                    if found and found[0] == "sgld":
                        self.convert_lora(g, node)
                    elif found is None:
                        self.report.reminders.append(
                            f"{where}: could not tell whether its MODEL comes from an "
                            "SGLD loader; left unchanged. On an SGLD model, replace it "
                            "with SGLDLoraLoader."
                        )
                elif found and found[0] == "sgld":
                    outputs = [o.get("type") for o in node.get("outputs") or []]
                    _remind(self.report, where, kind, "MODEL" in outputs)


def _note_weight_dtype(report, where, dtype):
    if dtype and dtype[0] not in ("default", None):
        report.reminders.append(
            f"{where}: weight_dtype `{dtype[0]}` reset to `default`; the SGLD worker "
            "loads the file's own dtype. For lower memory, use an FP8 or INT8 file "
            "or SGLDOptions quantization."
        )


def _remind(report, where, kind, outputs_model):
    """Reminder for a node that takes the SGLD model and is not converted."""
    if kind in rules.STACK_NODES:
        report.reminders.append(
            f"{where}: LoRA stack nodes are not converted. Replace it with chained "
            "SGLDLoraLoader nodes, one per LoRA (diffusion-model half only)."
        )
    elif (
        outputs_model
        and kind not in rules.SAMPLING_ONLY
        and not str(kind).startswith(rules.SAMPLING_ONLY_PREFIX)
    ):
        report.reminders.append(
            f"{where}: patches the SGLD model, which runs in the SGLD worker. If it is "
            "a speed patch (attention backend, cache, compile), remove or bypass it "
            "and set the equivalent in SGLDOptions (attention_backend, cache_mode, "
            f"execution_mode); see the {_README}. Other DiT patches (ControlNet, "
            "block patches) need the native ComfyUI loader."
        )


def convert_workflow(wf):
    """Return (converted copy of a UI workflow, Report)."""
    wf = copy.deepcopy(wf)
    converter = _UIConverter(wf)
    converter.run()
    return wf, converter.report


# ---------------------------------------------------------------- API prompt


def _is_link(value) -> bool:
    return isinstance(value, list) and len(value) == 2


def convert_api(prompt):
    """Same conversion for an API prompt (``{id: {class_type, inputs}}``)."""
    prompt = copy.deepcopy(prompt)
    report = Report()
    for nid, node in prompt.items():
        inputs = node.get("inputs") or {}
        kind, name = node.get("class_type"), inputs.get("unet_name")
        if kind in rules.STOCK_LOADERS and rules.is_sgld_model(name):
            report.changes.append(
                f"{_label(nid, node)} -> SGLDUNETLoader (`{rules.file_name(name)}`)"
            )
            node["class_type"] = rules.SGLD_LOADER
            _note_weight_dtype(report, _label(nid, node), [inputs.get("weight_dtype")])
            inputs["weight_dtype"] = "default"
    if not any(n.get("class_type") in rules.SGLD_LOADERS for n in prompt.values()):
        return prompt, report

    def trace(nid, seen):
        node = prompt.get(str(nid))
        if node is None or nid in seen:
            return None
        seen.add(nid)
        if node.get("class_type") in rules.SGLD_LOADERS:
            return "sgld"
        model = (node.get("inputs") or {}).get("model")
        return trace(str(model[0]), seen) if _is_link(model) else "other"

    # In API JSON a node outputs a MODEL when another node takes it as its model.
    model_sources = {
        str(n["inputs"]["model"][0])
        for n in prompt.values()
        if _is_link((n.get("inputs") or {}).get("model"))
    }
    for nid, node in prompt.items():
        kind, inputs = node.get("class_type"), node.get("inputs") or {}
        model = inputs.get("model")
        if not _is_link(model) or rules.is_sgld_node(kind):
            continue
        found, where = trace(str(model[0]), set()), _label(nid, node)
        if kind not in rules.LORA_NODES:
            if found == "sgld":
                _remind(report, where, kind, nid in model_sources)
            continue
        if found is None:
            report.reminders.append(
                f"{where}: could not trace its MODEL; left unchanged."
            )
        if found != "sgld":
            continue
        node["class_type"] = "SGLDLoraLoader"
        node["inputs"] = {
            "model": model,
            "lora_name": inputs.get("lora_name"),
            "strength_model": inputs.get("strength_model", 1.0),
            "nickname": "",
            "target": "all",
        }
        report.changes.append(
            f"{where} -> SGLDLoraLoader (`{inputs.get('lora_name')}`)"
        )
        if kind == "LoraLoader":
            for other in prompt.values():
                for key, value in (other.get("inputs") or {}).items():
                    if _is_link(value) and str(value[0]) == nid and value[1] == 1:
                        other["inputs"][key] = copy.deepcopy(inputs.get("clip"))
            report.reminders.append(
                f"{where}: the LoRA's CLIP half (strength_clip "
                f"{inputs.get('strength_clip')}) is not applied; its CLIP consumers "
                "now take the CLIP that fed the LoRA node."
            )
    return prompt, report


def is_api_prompt(data) -> bool:
    return (
        isinstance(data, dict)
        and "nodes" not in data
        and all(isinstance(v, dict) and "class_type" in v for v in data.values())
    )


def convert(data):
    return (convert_api if is_api_prompt(data) else convert_workflow)(data)


def dumps(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)

# SPDX-License-Identifier: Apache-2.0
"""Offline workflow converter: stock ComfyUI workflows -> SGLD integrated mode."""

import copy
import glob
import json
import os

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.convert import (
    convert,
    convert_api,
    convert_workflow,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.convert.__main__ import main

H3 = "minimax_h3_t2v_int8.safetensors"
LTX = "ltx-2.3-22b-dev.safetensors"


class _Builder:
    """A UI (LiteGraph) export: nodes with typed inputs / outputs and a links table."""

    def __init__(self):
        self.wf = {"last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [],
                   "groups": [], "config": {}, "extra": {}, "version": 0.4}  # fmt: skip

    def node(self, kind, widgets=(), inputs=(), outputs=(), title=None):
        nid = self.wf["last_node_id"] = self.wf["last_node_id"] + 1
        node = {
            "id": nid, "type": kind, "pos": [nid * 300, 0], "size": [270, 82],
            "flags": {}, "order": nid, "mode": 0,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in inputs],
            "outputs": [{"name": t, "type": t, "slot_index": i, "links": []} for i, t in enumerate(outputs)],
            "properties": {"cnr_id": "comfy-core", "ver": "0.3.66", "Node name for S&R": kind},
            "widgets_values": list(widgets),
        }  # fmt: skip
        if title:
            node["title"] = title
        self.wf["nodes"].append(node)
        return node

    def link(self, a, a_slot, b, b_slot):
        lid = self.wf["last_link_id"] = self.wf["last_link_id"] + 1
        kind = a["outputs"][a_slot]["type"]
        self.wf["links"].append([lid, a["id"], a_slot, b["id"], b_slot, kind])
        a["outputs"][a_slot]["links"].append(lid)
        b["inputs"][b_slot]["link"] = lid
        return lid


def _model_chain(b, loader, *patches):
    """loader -> patches... -> KSampler; returns the KSampler."""
    prev = loader
    for kind, widgets in patches:
        node = b.node(kind, widgets, [("model", "MODEL")], ["MODEL"])
        b.link(prev, 0, node, 0)
        prev = node
    sampler = b.node("KSampler", [0, "fixed", 8], [("model", "MODEL")], ["LATENT"])
    b.link(prev, 0, sampler, 0)
    return sampler


def _nodes(wf):
    return {n["id"]: n for n in wf["nodes"]}


def _link(wf, lid):
    return next(l for l in wf["links"] if l[0] == lid)


def test_loader_swap_and_lora_model_only() -> None:
    b = _Builder()
    loader = b.node("UNETLoader", [H3, "fp8_e4m3fn"], outputs=["MODEL"])
    _model_chain(b, loader, ("LoraLoaderModelOnly", ["turbo.safetensors", 0.8]))
    wf = b.wf
    out, report = convert_workflow(wf)
    nodes = _nodes(out)
    assert nodes[1]["type"] == "SGLDUNETLoader"
    # The SGLD worker serves only weight_dtype 'default'; the change is reported.
    assert nodes[1]["widgets_values"] == [H3, "default"]
    assert nodes[1]["inputs"] == [{"name": "sgld_options", "type": "SGLD_OPTIONS", "link": None}]  # fmt: skip
    assert "cnr_id" not in nodes[1]["properties"]
    assert nodes[2]["type"] == "SGLDLoraLoader"
    assert nodes[2]["widgets_values"] == ["turbo.safetensors", 0.8, "", "all"]
    assert out["links"] == wf["links"] and wf["nodes"][0]["type"] == "UNETLoader"
    assert len(report.changes) == 2
    assert len(report.reminders) == 1 and "fp8_e4m3fn" in report.reminders[0]


def test_lora_loader_drops_its_clip_half_and_rewires_clip_consumers() -> None:
    b = _Builder()
    loader = b.node("UNETLoader", [H3, "default"], outputs=["MODEL"])
    clip = b.node("CLIPLoader", ["umt5.safetensors", "wan"], outputs=["CLIP"])
    lora = b.node("LoraLoader", ["style.safetensors", 0.7, 0.5],
                  [("model", "MODEL"), ("clip", "CLIP")], ["MODEL", "CLIP"], title="Style")  # fmt: skip
    pos = b.node("CLIPTextEncode", ["a cat"], [("clip", "CLIP")], ["CONDITIONING"])
    neg = b.node("CLIPTextEncode", [""], [("clip", "CLIP")], ["CONDITIONING"])
    sampler = b.node("KSampler", [], [("model", "MODEL")], ["LATENT"])
    b.link(loader, 0, lora, 0)
    clip_in = b.link(clip, 0, lora, 1)
    to_pos, to_neg = b.link(lora, 1, pos, 0), b.link(lora, 1, neg, 0)
    b.link(lora, 0, sampler, 0)
    last_link_id = b.wf["last_link_id"]

    out, report = convert_workflow(b.wf)
    nodes = _nodes(out)
    lora = nodes[lora["id"]]
    assert lora["type"] == "SGLDLoraLoader"
    assert lora["widgets_values"] == ["style.safetensors", 0.7, "", "all"]
    assert [i["name"] for i in lora["inputs"]] == ["model"]
    assert [o["name"] for o in lora["outputs"]] == ["MODEL"]
    # Both text encoders now read the CLIP loader directly.
    assert _link(out, to_pos)[1:3] == [clip["id"], 0]
    assert _link(out, to_neg)[1:3] == [clip["id"], 0]
    assert nodes[clip["id"]]["outputs"][0]["links"] == [to_pos, to_neg]
    assert all(l[0] != clip_in for l in out["links"])
    assert out["last_link_id"] == last_link_id
    (reminder,) = report.reminders
    assert "Style" in reminder and "CLIP half" in reminder and "0.5" in reminder


def test_lora_on_a_model_comfyui_keeps_is_left_alone() -> None:
    """A KJ loader (LTX) next to H3: its LoRA must stay stock, not become a no-op."""
    b = _Builder()
    _model_chain(b, b.node("UNETLoader", [H3, "default"], outputs=["MODEL"]))
    ltx = b.node("DiffusionModelLoaderKJ", [LTX, "bf16"], outputs=["MODEL"])
    _model_chain(b, ltx, ("LoraLoaderModelOnly", ["ltx_detail.safetensors", 1.0]),
                 ("PathchSageAttentionKJ", ["auto"]))  # fmt: skip
    out, report = convert_workflow(b.wf)
    kinds = [n["type"] for n in out["nodes"]]
    assert "LoraLoaderModelOnly" in kinds and "SGLDLoraLoader" not in kinds
    assert not report.reminders and len(report.changes) == 1


def test_untraceable_lora_is_reported_not_rewritten() -> None:
    b = _Builder()
    a = b.node("UNETLoader", [H3, "default"], outputs=["MODEL"])
    c = b.node("UNETLoader", [H3, "default"], outputs=["MODEL"])
    merge = b.node("ModelMergeSimple", [0.5], [("model1", "MODEL"), ("model2", "MODEL")], ["MODEL"])  # fmt: skip
    b.link(a, 0, merge, 0)
    b.link(c, 0, merge, 1)
    _model_chain(b, merge, ("LoraLoaderModelOnly", ["x.safetensors", 1.0]))
    out, report = convert_workflow(b.wf)
    assert any(n["type"] == "LoraLoaderModelOnly" for n in out["nodes"])
    assert any("could not tell" in r for r in report.reminders)


def test_model_passed_through_kjnodes_set_get_is_traced() -> None:
    """Community workflows pass the MODEL around with KJNodes Set/Get pairs; a LoRA
    behind a GetNode is on the SGLD model, and Set/Get themselves are not patches."""
    b = _Builder()
    loader = b.node("UNETLoader", [H3, "default"], outputs=["MODEL"])
    setter = b.node("SetNode", ["MODEL fl2va"], [("MODEL", "MODEL")], ["*"])
    b.link(loader, 0, setter, 0)
    getter = b.node("GetNode", ["MODEL fl2va"], outputs=["MODEL"])
    shift = b.node("MiniMaxH3SigmaShift", [5.0], [("model", "MODEL")], ["MODEL"])
    b.link(getter, 0, shift, 0)
    _model_chain(b, shift, ("LoraLoaderModelOnly", ["turbo.safetensors", 1.0]))
    out, report = convert_workflow(b.wf)
    assert _nodes(out)[5]["type"] == "SGLDLoraLoader"
    assert not report.reminders  # SetNode and a sigma-shift node are not flagged


def test_speed_nodes_and_lora_stacks_are_reported_not_changed() -> None:
    b = _Builder()
    loader = b.node("UNETLoader", [H3, "default"], outputs=["MODEL"])
    _model_chain(b, loader, ("PathchSageAttentionKJ", ["auto", False]),
                 ("ModelSamplingSD3", [5.0]),
                 ("Power Lora Loader (rgthree)", [{}, {"on": True, "lora": "a.safetensors", "strength": 1}]),
                 ("TeaCache", [0.2]))  # fmt: skip
    bypassed = b.node("TorchCompileModel", ["inductor"], [("model", "MODEL")], ["MODEL"])  # fmt: skip
    bypassed["mode"] = 4
    b.link(loader, 0, bypassed, 0)
    out, report = convert_workflow(b.wf)
    nodes = _nodes(out)
    assert [nodes[i]["type"] for i in (2, 3, 4, 5)] == ["PathchSageAttentionKJ", "ModelSamplingSD3",
            "Power Lora Loader (rgthree)", "TeaCache"]  # fmt: skip
    assert all(nodes[i]["mode"] == 0 for i in (2, 3, 4, 5))
    assert nodes[4]["widgets_values"] == b.wf["nodes"][3]["widgets_values"]
    sage, stack, tea = report.reminders
    assert "PathchSageAttentionKJ" in sage and "SGLDOptions" in sage
    assert "chained SGLDLoraLoader" in stack and "TeaCache" in tea


def _subgraph_workflow():
    """A v1 export: a loader subgraph whose model file is a promoted widget."""
    inner = {
        "id": 3, "type": "UNETLoader", "pos": [0, 0], "size": [270, 82], "mode": 0,
        "inputs": [{"name": "unet_name", "type": "COMBO", "widget": {"name": "unet_name"}, "link": 1}],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": [2]}],
        "properties": {"Node name for S&R": "UNETLoader"}, "widgets_values": [LTX, "default"],
    }  # fmt: skip
    lora = {
        "id": 4, "type": "LoraLoaderModelOnly", "pos": [300, 0], "size": [270, 82], "mode": 0,
        "inputs": [{"name": "model", "type": "MODEL", "link": 2}],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": [3]}],
        "properties": {}, "widgets_values": ["turbo.safetensors", 1.0],
    }  # fmt: skip
    sg = {
        "id": "a1b2c3d4-0000-4000-8000-000000000001", "version": 1, "name": "Load H3",
        "state": {"lastGroupId": 0, "lastNodeId": 4, "lastLinkId": 3, "lastRerouteId": 0},
        "revision": 0, "config": {},
        "inputNode": {"id": -10, "bounding": [0, 0, 120, 60]},
        "outputNode": {"id": -20, "bounding": [900, 0, 120, 60]},
        "inputs": [{"id": "in-1", "name": "unet_name", "type": "COMBO", "linkIds": [1]}],
        "outputs": [{"id": "out-1", "name": "MODEL", "type": "MODEL", "linkIds": [3]}],
        "widgets": [], "nodes": [inner, lora], "groups": [],
        "links": [
            {"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 3, "target_slot": 0, "type": "COMBO"},
            {"id": 2, "origin_id": 3, "origin_slot": 0, "target_id": 4, "target_slot": 0, "type": "MODEL"},
            {"id": 3, "origin_id": 4, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "MODEL"},
        ],
        "extra": {},
    }  # fmt: skip
    instance = {
        "id": 5, "type": sg["id"], "pos": [0, 0], "size": [270, 60], "mode": 0,
        "inputs": [{"name": "unet_name", "type": "COMBO", "widget": {"name": "unet_name"}, "link": None}],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": [10]}],
        "properties": {}, "widgets_values": [H3],
    }  # fmt: skip
    speed = {
        "id": 6, "type": "PathchSageAttentionKJ", "pos": [300, 0], "size": [270, 60], "mode": 0,
        "inputs": [{"name": "model", "type": "MODEL", "link": 10}],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": [11]}],
        "properties": {}, "widgets_values": ["auto"],
    }  # fmt: skip
    sampler = {
        "id": 7, "type": "KSampler", "pos": [600, 0], "size": [270, 260], "mode": 0,
        "inputs": [{"name": "model", "type": "MODEL", "link": 11}],
        "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}],
        "properties": {}, "widgets_values": [],
    }  # fmt: skip
    return {
        "id": "f00d", "revision": 0, "version": 1,
        "state": {"lastGroupId": 0, "lastNodeId": 7, "lastLinkId": 11, "lastRerouteId": 0},
        "nodes": [instance, speed, sampler],
        "links": [
            {"id": 10, "origin_id": 5, "origin_slot": 0, "target_id": 6, "target_slot": 0, "type": "MODEL"},
            {"id": 11, "origin_id": 6, "origin_slot": 0, "target_id": 7, "target_slot": 0, "type": "MODEL"},
        ],
        "groups": [], "config": {}, "extra": {},
        "definitions": {"subgraphs": [sg]},
    }  # fmt: skip


def test_loader_in_subgraph_uses_the_promoted_model_name() -> None:
    wf = _subgraph_workflow()
    out, report = convert_workflow(wf)
    sg = out["definitions"]["subgraphs"][0]
    inner = _nodes(sg)
    assert inner[3]["type"] == "SGLDUNETLoader"
    # The model input of the subgraph's LoRA traces to the loader inside it, and the
    # speed node outside traces into the subgraph through its output.
    assert inner[4]["type"] == "SGLDLoraLoader"
    assert inner[4]["widgets_values"] == ["turbo.safetensors", 1.0, "", "all"]
    (reminder,) = report.reminders
    assert "PathchSageAttentionKJ" in reminder
    # No ids were added, so every id counter still covers what is in the file.
    assert out["state"] == wf["state"] and sg["state"]["lastLinkId"] >= max(
        l["id"] for l in sg["links"]
    )
    assert out["state"]["lastNodeId"] >= max(n["id"] for n in out["nodes"])


def test_weight_dtype_from_a_primitive_is_disconnected() -> None:
    """A linked weight_dtype overrides the saved widget value at run time."""
    b = _Builder()
    loader = b.node(
        "UNETLoader", [H3, "fp8_e4m3fn"], [("weight_dtype", "COMBO")], ["MODEL"]
    )
    loader["inputs"][0]["widget"] = {"name": "weight_dtype"}
    prim = b.node("PrimitiveNode", ["fp8_e4m3fn"], outputs=["COMBO"])
    other = b.node(
        "UNETLoader", [LTX, "fp8_e4m3fn"], [("weight_dtype", "COMBO")], ["MODEL"]
    )
    to_loader, to_other = b.link(prim, 0, loader, 0), b.link(prim, 0, other, 0)
    _model_chain(b, loader)
    out, report = convert_workflow(b.wf)
    nodes = _nodes(out)
    assert nodes[loader["id"]]["type"] == "SGLDUNETLoader"
    assert nodes[loader["id"]]["widgets_values"] == [H3, "default"]
    assert nodes[loader["id"]]["inputs"][0]["link"] is None
    assert all(l[0] != to_loader for l in out["links"])
    # The LTX loader stays stock and keeps its connection.
    assert nodes[prim["id"]]["outputs"][0]["links"] == [to_other]
    assert nodes[other["id"]]["inputs"][0]["link"] == to_other
    assert any("input link was removed" in r for r in report.reminders)


def _promoted_dtype_workflow(proxy):
    """The loader subgraph with weight_dtype promoted as well, used twice."""
    wf = _subgraph_workflow()
    sg = wf["definitions"]["subgraphs"][0]
    inner = sg["nodes"][0]
    inner["inputs"].append({"name": "weight_dtype", "type": "COMBO",
                            "widget": {"name": "weight_dtype"}, "link": None})  # fmt: skip
    first = wf["nodes"][0]
    if proxy:  # newer frontend: proxyWidgets name the inner widgets directly
        first["inputs"] = []
        first["properties"]["proxyWidgets"] = [
            ["3", "unet_name"],
            ["3", "weight_dtype"],
        ]
    else:  # v1: a subgraph input linked to the inner widget
        inner["inputs"][1]["link"] = 4
        sg["inputs"].append(
            {"id": "in-2", "name": "weight_dtype", "type": "COMBO", "linkIds": [4]}
        )
        sg["links"].append({"id": 4, "origin_id": -10, "origin_slot": 1, "target_id": 3,
                            "target_slot": 1, "type": "COMBO"})  # fmt: skip
        first["inputs"].append({"name": "weight_dtype", "type": "COMBO",
                                "widget": {"name": "weight_dtype"}, "link": None})  # fmt: skip
    first["widgets_values"] = [H3, "fp8_e4m3fn"]
    second = copy.deepcopy(first)
    second.update(id=8, outputs=[{"name": "MODEL", "type": "MODEL", "links": []}])
    second["widgets_values"] = [H3, "fp8_e5m2"]
    wf["nodes"].append(second)
    wf["state"]["lastNodeId"] = 8
    return wf


def test_promoted_weight_dtype_is_reset_on_every_instance() -> None:
    for proxy in (True, False):
        out, report = convert_workflow(_promoted_dtype_workflow(proxy))
        nodes = _nodes(out)
        assert nodes[5]["widgets_values"] == [H3, "default"]
        assert nodes[8]["widgets_values"] == [H3, "default"]
        inner = _nodes(out["definitions"]["subgraphs"][0])[3]
        assert inner["type"] == "SGLDUNETLoader"
        notes = [r for r in report.reminders if "weight_dtype" in r]
        assert any("fp8_e4m3fn" in r for r in notes) and any(
            "fp8_e5m2" in r for r in notes
        )


def test_promoted_weight_dtype_fed_by_a_link_outside_is_disconnected() -> None:
    wf = _promoted_dtype_workflow(proxy=False)
    prim = {"id": 9, "type": "PrimitiveNode", "mode": 0, "inputs": [],
            "outputs": [{"name": "COMBO", "type": "COMBO", "links": [13]}],
            "properties": {}, "widgets_values": ["fp8_e4m3fn"]}  # fmt: skip
    wf["nodes"].append(prim)
    wf["nodes"][0]["inputs"][1]["link"] = 13
    wf["links"].append({"id": 13, "origin_id": 9, "origin_slot": 0, "target_id": 5, "target_slot": 1, "type": "COMBO"})  # fmt: skip
    wf["state"].update(lastNodeId=9, lastLinkId=13)
    out, report = convert_workflow(wf)
    nodes = _nodes(out)
    assert (
        nodes[5]["inputs"][1]["link"] is None and nodes[9]["outputs"][0]["links"] == []
    )
    assert all(l["id"] != 13 for l in out["links"])
    assert nodes[5]["widgets_values"] == [H3, "default"]
    assert any("input link on subgraph instance" in r for r in report.reminders)


def test_lora_inside_subgraph_traces_its_model_through_the_instance() -> None:
    """Subgraph that takes MODEL in: its LoRA is on an SGLD model only via the parent."""
    wf = _subgraph_workflow()
    sg = wf["definitions"]["subgraphs"][0]
    sg["inputs"] = [{"id": "in-1", "name": "model", "type": "MODEL", "linkIds": [2]}]
    sg["nodes"] = [n for n in sg["nodes"] if n["id"] == 4]
    sg["links"] = [l for l in sg["links"] if l["id"] != 1]
    sg["links"][0].update(origin_id=-10, origin_slot=0)
    loader = {"id": 8, "type": "UNETLoader", "mode": 0, "inputs": [],
              "outputs": [{"name": "MODEL", "type": "MODEL", "links": [12]}],
              "properties": {}, "widgets_values": [H3, "default"]}  # fmt: skip
    instance = wf["nodes"][0]
    instance["inputs"] = [{"name": "model", "type": "MODEL", "link": 12}]
    instance["widgets_values"] = []
    wf["nodes"].append(loader)
    wf["links"].append({"id": 12, "origin_id": 8, "origin_slot": 0, "target_id": 5, "target_slot": 0, "type": "MODEL"})  # fmt: skip
    wf["state"].update(lastNodeId=8, lastLinkId=12)
    out, _ = convert_workflow(wf)
    assert _nodes(out)[8]["type"] == "SGLDUNETLoader"
    assert out["definitions"]["subgraphs"][0]["nodes"][0]["type"] == "SGLDLoraLoader"


def test_non_sgld_workflow_and_converted_workflow_are_unchanged() -> None:
    b = _Builder()
    _model_chain(b, b.node("UNETLoader", [LTX, "default"], outputs=["MODEL"]),
                 ("LoraLoaderModelOnly", ["x.safetensors", 1.0]))  # fmt: skip
    out, report = convert_workflow(b.wf)
    assert out == b.wf and not report.changes and not report.reminders

    for wf in (b.wf, _subgraph_workflow()):
        wf["nodes"][0]["widgets_values"][0] = H3
        once, _ = convert(wf)
        twice, report = convert(once)
        assert twice == once and not report.changes


def test_api_prompt() -> None:
    prompt = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": H3, "weight_dtype": "fp8_e5m2"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t5.safetensors", "type": "wan"}},
        "3": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["2", 0],
              "lora_name": "s.safetensors", "strength_model": 0.6, "strength_clip": 1.0}},
        "4": {"class_type": "PathchSageAttentionKJ", "inputs": {"model": ["3", 0], "sage_attention": "auto"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 1], "text": "a cat"}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0]}},
        "7": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "ltx-2-q8.gguf"}},
        "8": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["7", 0], "lora_name": "l.safetensors", "strength_model": 1.0}},
    }  # fmt: skip
    out, report = convert_api(prompt)
    assert out["1"]["class_type"] == "SGLDUNETLoader"
    assert out["1"]["inputs"]["weight_dtype"] == "default"
    assert out["3"] == {"class_type": "SGLDLoraLoader", "inputs": {"model": ["1", 0],
                        "lora_name": "s.safetensors", "strength_model": 0.6, "nickname": "", "target": "all"}}  # fmt: skip
    assert out["5"]["inputs"]["clip"] == ["2", 0]
    assert (
        out["4"] == prompt["4"] and out["7"] == prompt["7"] and out["8"] == prompt["8"]
    )
    dtype, clip, sage = report.reminders
    assert "fp8_e5m2" in dtype and "CLIP half" in clip
    assert "PathchSageAttentionKJ" in sage
    assert convert_api(out)[0] == out

    linked = {"9": {"class_type": "PrimitiveNode", "inputs": {}},
              "1": {"class_type": "UNETLoader", "inputs": {"unet_name": H3, "weight_dtype": ["9", 0]}}}  # fmt: skip
    out, report = convert_api(linked)
    assert out["1"]["inputs"]["weight_dtype"] == "default"
    (note,) = report.reminders
    assert "connected to node 9" in note and "['9'" not in note


def test_bundled_sgld_workflows_need_no_changes() -> None:
    root = os.path.join(os.path.dirname(__file__), "../../apps/ComfyUI_SGLDiffusion/workflows")  # fmt: skip
    paths = sorted(glob.glob(os.path.join(root, "*.json")))
    assert paths
    for path in paths:
        with open(path, encoding="utf-8") as f:
            wf = json.load(f)
        out, report = convert(copy.deepcopy(wf))
        assert out == wf and not report.changes and not report.reminders, path


def test_cli_writes_workflow_and_report(tmp_path, capsys) -> None:
    b = _Builder()
    _model_chain(b, b.node("UNETLoader", [H3, "default"], outputs=["MODEL"]))
    src = tmp_path / "wf.json"
    src.write_text(json.dumps(b.wf))
    main([str(src)])
    out = json.loads((tmp_path / "wf_sgld.json").read_text())
    assert out["nodes"][0]["type"] == "SGLDUNETLoader"
    assert "UNETLoader -> SGLDUNETLoader" in (tmp_path / "wf_sgld.md").read_text()
    assert "wrote" in capsys.readouterr().out

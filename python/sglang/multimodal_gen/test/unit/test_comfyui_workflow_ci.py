# SPDX-License-Identifier: Apache-2.0
"""Tests for the ComfyUI workflow CI helpers (static check, native converter,
output comparison, runner helpers).

Tests that need real node definitions read a ComfyUI checkout from the
COMFYUI_DIR environment variable and are skipped without it.
"""

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

CI_DIR = Path(__file__).resolve().parents[2] / "apps" / "ComfyUI_SGLDiffusion" / "ci"
WORKFLOWS = sorted((CI_DIR.parent / "workflows").glob("*.json"))


def _load(name):
    sys.path.insert(0, str(CI_DIR))
    try:
        spec = importlib.util.spec_from_file_location(name, CI_DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(CI_DIR))


wc = _load("workflow_check")
nc = _load("native_convert")
cmp_ = _load("compare")
rw = _load("run_workflows")

# Expected mode of every shipped workflow; a new workflow must be added here.
EXPECTED_MODE = {
    "flux_sgld_sp.json": "integrated",
    "minimax_h3_r2v_sgld.json": "integrated",
    "minimax_h3_t2v_sgld.json": "integrated",
    "minimax_h3_t2v_sgld_upscaler.json": "integrated",
    "qwen_image_sgld.json": "integrated",
    "z-image_sgld.json": "integrated",
    "sgld_image2video.json": "server",
    "sgld_text2img.json": "server",
}

# Real defects in the shipped workflows (see test_shipped_workflow_is_valid).
KNOWN_BROKEN = {
    "qwen_image_sgld.json": "SaveImage node 60 has no 'images' link",
    "sgld_text2img.json": "node 11 uses class SGLDiffusionSetLora; the node is "
    "named SGLDiffusionServerSetLora",
}


def _wf(name):
    return json.loads((CI_DIR.parent / "workflows" / name).read_text())


@pytest.fixture(scope="module")
def real_defs():
    comfy = os.environ.get("COMFYUI_DIR")
    if not comfy or not Path(comfy, "nodes.py").is_file():
        pytest.skip(
            "set COMFYUI_DIR to a ComfyUI checkout to use real node definitions"
        )
    defs, plugin_classes = wc.dump_definitions(comfy)
    assert "SGLDUNETLoader" in plugin_classes
    return defs


# Small hand-written definitions for the negative tests.
DEFS = {
    "Loader": {
        "input": {"required": {"name": [["a", "b"]]}, "optional": {"opt": ["INT", {}]}},
        "returns": ["MODEL", "CLIP"],
    },
    "Sampler": {
        "input": {"required": {"model": ["MODEL", {}], "steps": ["INT", {}]}},
        "returns": ["LATENT"],
    },
}


def _good():
    return {
        "1": {"class_type": "Loader", "inputs": {"name": "a"}},
        "2": {"class_type": "Sampler", "inputs": {"model": ["1", 0], "steps": 4}},
    }


def _kinds(wf, **kw):
    return sorted(p.kind for p in wc.validate_workflow("t", wf, DEFS, **kw).problems)


def test_validator_accepts_good_workflow():
    assert _kinds(_good()) == []


@pytest.mark.parametrize(
    "mutate, kind",
    [
        (lambda w: w["2"]["inputs"].pop("steps"), "missing_input"),
        (lambda w: w["2"]["inputs"].update(model=["1", 5]), "bad_output_slot"),
        (lambda w: w["2"]["inputs"].update(model=["1", 1]), "type_mismatch"),
        (lambda w: w["2"]["inputs"].update(model=["9", 0]), "dangling_link"),
        (lambda w: w["1"]["inputs"].update(name="zzz"), "bad_combo_value"),
        (lambda w: w["2"]["inputs"].update(bogus=1), "unknown_input"),
        (lambda w: w["2"].update(class_type="Nope"), "unknown_class"),
    ],
)
def test_validator_catches_broken_workflow(mutate, kind):
    wf = _good()
    mutate(wf)
    assert kind in _kinds(wf)


def test_validator_reports_all_problems_and_severity():
    wf = _good()
    wf["2"]["inputs"].pop("steps")
    wf["2"]["inputs"]["bogus"] = 1
    rep = wc.validate_workflow("t", wf, DEFS)
    assert {p.kind for p in rep.problems} == {"missing_input", "unknown_input"}
    assert [p.kind for p in rep.errors] == ["missing_input"]
    assert not rep.ok
    only_warn = _good()
    only_warn["2"]["inputs"]["bogus"] = 1
    assert wc.validate_workflow("t", only_warn, DEFS).ok


def test_external_class_is_unverified_not_failed():
    wf = _good()
    wf["3"] = {"class_type": "Ext", "inputs": {}}
    rep = wc.validate_workflow("t", wf, DEFS, external_classes=frozenset({"Ext"}))
    assert rep.ok and rep.unverified == [("3", "Ext")]


def test_host_file_combos_are_not_checked():
    defs = {
        "Loader": {
            "input": {"required": {"name": [["x", wc.FILES_MARKER]]}},
            "returns": [],
        }
    }
    wf = {"1": {"class_type": "Loader", "inputs": {"name": "anything.safetensors"}}}
    assert wc.validate_workflow("t", wf, defs).ok


def test_dynamic_combo_children_and_autogrow_inputs_are_accepted():
    sub = [
        "COMFY_DYNAMICCOMBO_V3",
        {
            "options": [
                {
                    "key": "auto",
                    "inputs": {"required": {"codec": ["COMBO", {"options": ["auto"]}]}},
                }
            ]
        },
    ]
    grow = [
        "COMFY_AUTOGROW_V3",
        {"template": {"input": {"required": {"img": ["IMAGE", {}]}}, "prefix": "img_"}},
    ]
    defs = {
        "N": {
            "input": {"required": {"fmt": sub}, "optional": {"imgs": grow}},
            "returns": [],
        },
        "Img": {"input": {}, "returns": ["IMAGE"]},
    }
    wf = {
        "1": {"class_type": "Img", "inputs": {}},
        "2": {
            "class_type": "N",
            "inputs": {"fmt": "auto", "codec": "auto", "imgs.img_0": ["1", 0]},
        },
    }
    assert wc.validate_workflow("t", wf, defs).ok


def test_shipped_workflows_are_all_classified():
    assert {p.name for p in WORKFLOWS} == set(EXPECTED_MODE)
    for p in WORKFLOWS:
        assert nc.classify(json.loads(p.read_text())) == EXPECTED_MODE[p.name]


def _maybe_xfail(name):
    # strict: fixing a workflow makes the test XPASS and fails until removed here
    if name in KNOWN_BROKEN:
        return [pytest.mark.xfail(strict=True, reason=KNOWN_BROKEN[name])]
    return []


@pytest.mark.parametrize(
    "path",
    [pytest.param(p, id=p.name, marks=_maybe_xfail(p.name)) for p in WORKFLOWS],
)
def test_shipped_workflow_is_valid(path, real_defs):
    rep = wc.validate_workflow(path.name, json.loads(path.read_text()), real_defs)
    assert rep.ok, "\n" + "\n".join(map(str, rep.errors))


@pytest.mark.parametrize("name", [n for n, m in EXPECTED_MODE.items() if m == "server"])
def test_server_mode_workflows_are_refused(name):
    with pytest.raises(nc.NotConvertible):
        nc.to_native(_wf(name))


def test_converter_repoints_links_and_maps_lora():
    wf = {
        "1": {"class_type": "SGLDOptions", "inputs": {"num_gpus": 2}},
        "2": {
            "class_type": "SGLDUNETLoader",
            "inputs": {
                "unet_name": "m.safetensors",
                "weight_dtype": "default",
                "sgld_options": ["1", 0],
            },
        },
        "3": {
            "class_type": "SGLDLoraLoader",
            "inputs": {
                "model": ["2", 0],
                "lora_name": "l.safetensors",
                "strength_model": 0.5,
                "nickname": "x",
                "target": "all",
            },
        },
        "4": {"class_type": "KSampler", "inputs": {"model": ["3", 0]}},
    }
    orig = copy.deepcopy(wf)
    out, _ = nc.to_native(wf, {"l.safetensors": "native_l.safetensors"})
    assert wf == orig
    assert "1" not in out
    assert out["2"] == {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "m.safetensors", "weight_dtype": "default"},
    }
    assert out["3"] == {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["2", 0],
            "lora_name": "native_l.safetensors",
            "strength_model": 0.5,
        },
    }
    assert out["4"]["inputs"]["model"] == ["3", 0]


def test_converter_rejects_dangling_link_to_removed_node():
    wf = {
        "1": {"class_type": "SGLDOptions", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
    }
    with pytest.raises(nc.NotConvertible):
        nc.to_native(wf)


@pytest.mark.parametrize(
    "name", [n for n, m in EXPECTED_MODE.items() if m == "integrated"]
)
def test_converter_on_shipped_integrated_workflows(name):
    wf = _wf(name)
    out, _ = nc.to_native(wf)
    classes = {n["class_type"] for n in out.values()}
    assert not classes & nc.INTEGRATED_CLASSES
    n_options = sum(n["class_type"] == "SGLDOptions" for n in wf.values())
    assert len(out) == len(wf) - n_options
    for node in out.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
                assert v[0] in out
    assert nc.classify(out) == "native"


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=_maybe_xfail(n))
        for n, m in EXPECTED_MODE.items()
        if m == "integrated"
    ],
)
def test_converted_workflows_pass_native_validation(name, real_defs):
    out, _ = nc.to_native(_wf(name))
    rep = wc.validate_workflow(name, out, real_defs)
    assert rep.ok, "\n" + "\n".join(map(str, rep.errors))


def test_psnr_values():
    a = np.full((8, 8, 3), 100, dtype=np.uint8)
    assert cmp_.psnr(a, a) == float("inf")
    assert cmp_.psnr(a, a + 10) == pytest.approx(10 * np.log10(255**2 / 100))


def test_compare_threshold_behaviour():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 256, (32, 32, 3)).astype(np.uint8)
    close = np.clip(ref.astype(int) + rng.integers(-3, 4, ref.shape), 0, 255)
    other = rng.integers(0, 256, ref.shape).astype(np.uint8)
    assert cmp_.compare_arrays(close, ref).passed
    bad = cmp_.compare_arrays(other, ref)
    assert not bad.passed and "psnr" in bad.reason
    # A strict threshold turns the near-identical pair into a failure.
    assert not cmp_.compare_arrays(close, ref, min_psnr_db=60).passed
    assert not cmp_.compare_arrays(ref[:16], ref).passed


def test_compare_files_images(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, (16, 16, 3)).astype(np.uint8)
    Image.fromarray(arr).save(tmp_path / "a.png")
    Image.fromarray(arr).save(tmp_path / "b.png")
    Image.fromarray(255 - arr).save(tmp_path / "c.png")
    assert cmp_.compare_files(tmp_path / "a.png", tmp_path / "b.png").passed
    assert not cmp_.compare_files(tmp_path / "a.png", tmp_path / "c.png").passed


def test_runner_model_helpers(tmp_path):
    (tmp_path / "diffusion_models").mkdir()
    (tmp_path / "diffusion_models" / "have.safetensors").write_bytes(b"x")
    wf = {
        "1": {
            "class_type": "SGLDUNETLoader",
            "inputs": {"unet_name": "have.safetensors"},
        },
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
    }
    assert rw.missing_models(wf, tmp_path) == ["vae/ae.safetensors"]
    mapped = rw.apply_model_map(wf, {"ae.safetensors": "tiny.safetensors"})
    assert mapped["2"]["inputs"]["vae_name"] == "tiny.safetensors"
    assert wf["2"]["inputs"]["vae_name"] == "ae.safetensors"


def test_runner_output_files_ignores_temp_outputs():
    entry = {
        "outputs": {
            "9": {
                "images": [
                    {"filename": "a.png", "subfolder": "", "type": "output"},
                    {"filename": "t.png", "subfolder": "", "type": "temp"},
                ]
            },
            "10": {
                "images": [
                    {"filename": "v.mp4", "subfolder": "video", "type": "output"}
                ]
            },
        }
    }
    assert rw.output_files(entry) == [
        ("output", "", "a.png"),
        ("output", "video", "v.mp4"),
    ]

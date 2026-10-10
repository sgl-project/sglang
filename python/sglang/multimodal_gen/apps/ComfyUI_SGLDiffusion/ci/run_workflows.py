"""Run the shipped workflows headless in ComfyUI and compare with native.

NOT RUN ON GPU YET. Needs a GPU, a ComfyUI checkout and real model weights.

For each workflows/*.json:
  1. static check (workflow_check) as a preflight;
  2. skip if a referenced model file is missing under --models-dir;
  3. integrated mode: run the workflow with the SGLD plugin, run the
     native-converted twin with the same seeds, compare outputs;
  4. server mode: needs --server-url of a running `sglang serve`; only checks
     that the run succeeds and the output is not blank (no native twin).
Writes results.json and summary.txt into --out-dir; exit code 1 on any failure.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare  # noqa: E402
import native_convert  # noqa: E402
import workflow_check  # noqa: E402

# class_type -> (input key, models subfolder)
MODEL_INPUTS = {
    "SGLDUNETLoader": [("unet_name", "diffusion_models")],
    "UNETLoader": [("unet_name", "diffusion_models")],
    "CLIPLoader": [("clip_name", "text_encoders")],
    "DualCLIPLoader": [
        ("clip_name1", "text_encoders"),
        ("clip_name2", "text_encoders"),
    ],
    "VAELoader": [("vae_name", "vae")],
    "SGLDLoraLoader": [("lora_name", "loras")],
    "LoraLoaderModelOnly": [("lora_name", "loras")],
}


def apply_model_map(wf, model_map):
    """Replace model filenames using {original: replacement}."""
    out = copy.deepcopy(wf)
    for node in out.values():
        for key, _ in MODEL_INPUTS.get(node["class_type"], []):
            v = node["inputs"].get(key)
            if v in model_map:
                node["inputs"][key] = model_map[v]
    return out


def missing_models(wf, models_dir):
    """Return 'folder/file' entries the workflow needs but models_dir lacks."""
    missing = []
    for node in wf.values():
        for key, folder in MODEL_INPUTS.get(node["class_type"], []):
            name = node["inputs"].get(key)
            if (
                isinstance(name, str)
                and not (Path(models_dir) / folder / name).is_file()
            ):
                missing.append(f"{folder}/{name}")
    return sorted(set(missing))


def output_files(history_entry):
    """List (type, subfolder, filename) of saved outputs in a /history entry."""
    files = []
    for out in history_entry.get("outputs", {}).values():
        for item in out.get("images", []) + out.get("gifs", []):
            if item.get("type") == "output":
                files.append(
                    (item["type"], item.get("subfolder", ""), item["filename"])
                )
    return files


class ComfyServer:
    def __init__(self, comfyui_dir, port, extra_args=()):
        self.dir = Path(comfyui_dir)
        self.port = port
        self.extra_args = list(extra_args)
        self.proc = None
        self.link = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, with_plugin=True, timeout=300):
        custom = self.dir / "custom_nodes"
        link = custom / "ComfyUI_SGLDiffusion"
        if with_plugin and not link.exists():
            link.symlink_to(workflow_check.PLUGIN_DIR, target_is_directory=True)
            self.link = link
        cmd = [
            sys.executable,
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--disable-auto-launch",
            *self.extra_args,
        ]
        self.proc = subprocess.Popen(cmd, cwd=self.dir)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("ComfyUI exited during startup")
            try:
                urllib.request.urlopen(f"{self.url}/system_stats", timeout=2).read()
                return
            except Exception:
                time.sleep(2)
        raise TimeoutError("ComfyUI did not come up")

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.link and self.link.is_symlink():
            self.link.unlink()

    def _json(self, path, data=None):
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def run_prompt(self, wf, timeout):
        pid = self._json("/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})[
            "prompt_id"
        ]
        deadline = time.time() + timeout
        while time.time() < deadline:
            hist = self._json(f"/history/{pid}")
            if pid in hist:
                entry = hist[pid]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(
                        f"execution error: {json.dumps(status.get('messages'))[:1500]}"
                    )
                return entry
            time.sleep(3)
        raise TimeoutError(f"prompt {pid} did not finish in {timeout}s")

    def fetch(self, item, dest):
        _, sub, name = item
        q = urllib.parse.urlencode(
            {"filename": name, "subfolder": sub, "type": "output"}
        )
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{self.url}/view?{q}", timeout=120) as r:
            dest.write_bytes(r.read())
        return dest


def run_variant(comfyui_dir, wf, out_dir, port, timeout, with_plugin):
    srv = ComfyServer(comfyui_dir, port)
    try:
        srv.start(with_plugin=with_plugin)
        files = output_files(srv.run_prompt(wf, timeout))
        return [srv.fetch(f, Path(out_dir) / f[2]) for f in files]
    finally:
        srv.stop()


def process_workflow(name, wf, args, defs, model_map):
    res = {"workflow": name, "mode": native_convert.classify(wf)}
    rep = workflow_check.validate_workflow(name, wf, defs)
    if not rep.ok:
        return {
            **res,
            "status": "fail",
            "reason": "static check: " + "; ".join(map(str, rep.errors)),
        }
    wf = apply_model_map(wf, model_map)
    miss = missing_models(wf, args.models_dir)
    if miss:
        return {**res, "status": "skipped", "reason": "missing models", "missing": miss}
    out = Path(args.out_dir) / name.removesuffix(".json")
    try:
        if res["mode"] == "server":
            if not args.server_url:
                return {**res, "status": "skipped", "reason": "no --server-url"}
            for node in wf.values():
                if node["class_type"] == "SGLDiffusionServerModel":
                    node["inputs"]["base_url"] = args.server_url
            files = run_variant(
                args.comfyui_dir, wf, out / "sgld", args.port, args.timeout, True
            )
            blank = [f.name for f in files if f.stat().st_size == 0]
            if not files or blank:
                return {**res, "status": "fail", "reason": f"no/empty outputs {blank}"}
            return {
                **res,
                "status": "pass",
                "outputs": [str(f) for f in files],
                "compared": False,
            }
        native_wf, _ = native_convert.to_native(wf)
        sgld = run_variant(
            args.comfyui_dir, wf, out / "sgld", args.port, args.timeout, True
        )
        nat = run_variant(
            args.comfyui_dir, native_wf, out / "native", args.port, args.timeout, False
        )
        if not sgld or len(sgld) != len(nat):
            return {
                **res,
                "status": "fail",
                "reason": f"output count sgld={len(sgld)} native={len(nat)}",
            }
        cmps = [
            compare.compare_files(
                a, b, min_psnr_db=args.min_psnr, min_corr=args.min_corr
            )
            for a, b in zip(sorted(sgld), sorted(nat))
        ]
        bad = [c.reason for c in cmps if not c.passed]
        return {
            **res,
            "status": "fail" if bad else "pass",
            "reason": "; ".join(bad),
            "compared": True,
            "psnr_db": [c.psnr_db for c in cmps],
            "corr": [c.corr for c in cmps],
        }
    except Exception as e:  # keep going so one broken workflow does not hide others
        return {**res, "status": "fail", "reason": f"{type(e).__name__}: {e}"[:2000]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--comfyui-dir", required=True)
    ap.add_argument("--models-dir", help="defaults to <comfyui-dir>/models")
    ap.add_argument("--workflows", default=str(workflow_check.WORKFLOW_DIR))
    ap.add_argument("--only", nargs="*", help="workflow file names to run")
    ap.add_argument(
        "--model-map", help="JSON {original filename: replacement filename}"
    )
    ap.add_argument(
        "--server-url", help="running sglang serve, enables server-mode workflows"
    )
    ap.add_argument("--out-dir", default="workflow_ci_out")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per prompt")
    ap.add_argument("--min-psnr", type=float, default=compare.DEFAULT_MIN_PSNR_DB)
    ap.add_argument("--min-corr", type=float, default=compare.DEFAULT_MIN_CORR)
    ap.add_argument("--fail-on-skip", action="store_true")
    args = ap.parse_args(argv)
    args.models_dir = args.models_dir or os.path.join(args.comfyui_dir, "models")
    model_map = json.load(open(args.model_map)) if args.model_map else {}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    defs, _ = workflow_check.dump_definitions(args.comfyui_dir)
    results = []
    for path in sorted(Path(args.workflows).glob("*.json")):
        if args.only and path.name not in args.only:
            continue
        results.append(
            process_workflow(path.name, json.load(open(path)), args, defs, model_map)
        )

    Path(args.out_dir, "results.json").write_text(json.dumps(results, indent=2))
    lines = []
    for r in results:
        extra = r.get("reason") or ""
        if r.get("missing"):
            extra += " " + ", ".join(r["missing"])
        lines.append(
            f"{r['status'].upper():8} {r['workflow']} [{r['mode']}] {extra}".rstrip()
        )
    summary = "\n".join(lines)
    Path(args.out_dir, "summary.txt").write_text(summary + "\n")
    print(summary)
    failed = any(r["status"] == "fail" for r in results)
    skipped = any(r["status"] == "skipped" for r in results)
    return 1 if failed or (args.fail_on_skip and skipped) else 0


if __name__ == "__main__":
    sys.exit(main())

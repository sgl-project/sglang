# SPDX-License-Identifier: Apache-2.0
"""python -m sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.convert IN.json [-o OUT.json]

Converts a ComfyUI workflow (UI or API JSON) to run its H3 / Qwen-Image / Z-Image /
FLUX.1 model on SGLD. Writes IN_sgld.json and IN_sgld.md (the report) next to the
input by default and prints the report. With ``-o -`` the workflow goes to stdout
and the report to stderr.
"""

import argparse
import json
import os
import sys

from .workflow import convert, dumps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workflow", help="ComfyUI workflow, UI or API JSON")
    ap.add_argument("-o", "--output", help="converted workflow (default IN_sgld.json)")
    args = ap.parse_args(argv)
    with open(args.workflow, encoding="utf-8") as f:
        out, report = convert(json.load(f))
    text = report.markdown()
    if args.output == "-":
        sys.stdout.write(dumps(out) + "\n")
        sys.stderr.write(text)
        return
    path = args.output or os.path.splitext(args.workflow)[0] + "_sgld.json"
    report_path = os.path.splitext(path)[0] + ".md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps(out) + "\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    sys.stdout.write(f"{text}\nwrote {path} and {report_path}\n")


if __name__ == "__main__":
    main()

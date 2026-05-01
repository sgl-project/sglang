"""Bundle one or more triage text reports into a single markdown document."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

FRAMEWORK_LABELS = {
    "sglang": "SGLang",
    "vllm": "vLLM",
    "trtllm": "TensorRT-LLM",
}

FRAMEWORK_ORDER = {"sglang": 0, "vllm": 1, "trtllm": 2}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render multiple profiler triage text outputs into one markdown file. "
            "Input files are expected to be the existing analysis_*.txt outputs "
            "already emitted by analyze_llm_torch_profile.py."
        )
    )
    parser.add_argument(
        "--analysis-root",
        type=str,
        default=None,
        help=(
            "Root directory to scan recursively for analysis_*.txt files. "
            "Parent directory names are used as model section ids."
        ),
    )
    parser.add_argument(
        "--analysis-file",
        action="append",
        default=[],
        help=(
            "Explicit analysis file entry. Use either PATH or LABEL=PATH. "
            "When LABEL is omitted, the parent directory name is used."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Unified LLM Torch Profiler Triage Bundle",
        help="Top-level markdown title.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write the bundled markdown to this file. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--include-toc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a simple table of contents.",
    )
    args = parser.parse_args(argv)
    if not args.analysis_root and not args.analysis_file:
        parser.error("Provide at least one of --analysis-root or --analysis-file.")
    return args


def framework_key_from_path(path: Path) -> str:
    lowered = path.name.lower()
    if "sglang" in lowered:
        return "sglang"
    if "vllm" in lowered:
        return "vllm"
    if "trtllm" in lowered or "tensorrt" in lowered:
        return "trtllm"
    return "other"


def framework_label(framework_key: str) -> str:
    return FRAMEWORK_LABELS.get(framework_key, framework_key)


def discover_analysis_files(root: Path) -> List[Tuple[str, Path]]:
    entries: List[Tuple[str, Path]] = []
    for path in sorted(root.rglob("analysis*.txt")):
        entries.append((path.parent.name, path))
    return entries


def parse_explicit_entry(raw: str) -> Tuple[str, Path]:
    if "=" in raw:
        label, path_text = raw.split("=", 1)
        path = Path(path_text).expanduser().resolve()
        return label.strip(), path
    path = Path(raw).expanduser().resolve()
    return path.parent.name, path


def slugify(text: str) -> str:
    chars = []
    last_dash = False
    for char in text.lower():
        if char.isalnum():
            chars.append(char)
            last_dash = False
        elif not last_dash:
            chars.append("-")
            last_dash = True
    return "".join(chars).strip("-")


def extract_model_name(report_text: str) -> Optional[str]:
    for line in report_text.splitlines():
        if line.startswith("Model: "):
            return line.split("Model: ", 1)[1].strip()
    return None


def choose_model_display_name(
    current: Optional[str],
    candidate: Optional[str],
    *,
    label: str,
) -> str:
    if candidate and candidate != label:
        if not current or current == label:
            return candidate
        if len(candidate) > len(current):
            return candidate
        return current
    if current:
        return current
    return label


def normalize_report_text(report_text: str) -> str:
    text = report_text.replace("\r\n", "\n").strip()
    if not text:
        return "_Empty analysis output._"
    heading_map = {
        "Triage View": "#### Triage View",
        "Kernel Table": "#### Kernel Table",
        "Overlap Opportunity Table": "#### Overlap Opportunity Table",
        "Fuse Opportunity Table": "#### Fuse Opportunity Table",
    }
    normalized_lines = []
    for line in text.splitlines():
        normalized_lines.append(heading_map.get(line, line))
    return "\n".join(normalized_lines)


def build_bundle_markdown(
    *,
    title: str,
    labeled_paths: Sequence[Tuple[str, Path]],
    include_toc: bool,
) -> str:
    grouped: Dict[str, List[Tuple[str, Path, str]]] = defaultdict(list)
    model_display: Dict[str, str] = {}

    for label, path in labeled_paths:
        raw_text = path.read_text(encoding="utf-8")
        report_text = normalize_report_text(raw_text)
        model_name = extract_model_name(report_text)
        grouped[label].append((framework_key_from_path(path), path, report_text))
        model_display[label] = choose_model_display_name(
            model_display.get(label),
            model_name,
            label=label,
        )

    ordered_labels = sorted(
        grouped,
        key=lambda item: (model_display[item].lower(), item.lower()),
    )

    lines: List[str] = [f"# {title}", ""]
    lines.append(
        f"_Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_"
    )
    lines.append("")

    if include_toc:
        lines.append("## Contents")
        lines.append("")
        for label in ordered_labels:
            lines.append(
                f"- [{model_display[label]}](#{slugify(model_display[label])})"
            )
        lines.append("")

    for label in ordered_labels:
        display_name = model_display[label]
        lines.append(f"## {display_name}")
        lines.append("")
        lines.append(f"Model id: `{label}`")
        lines.append("")

        records = sorted(
            grouped[label],
            key=lambda item: (
                FRAMEWORK_ORDER.get(item[0], 99),
                item[1].name.lower(),
            ),
        )

        for framework_key, path, report_text in records:
            lines.append(f"### {framework_label(framework_key)}")
            lines.append("")
            lines.append(f"Source: `{path}`")
            lines.append("")
            lines.append(report_text)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    labeled_paths: List[Tuple[str, Path]] = []
    if args.analysis_root:
        labeled_paths.extend(
            discover_analysis_files(Path(args.analysis_root).expanduser().resolve())
        )
    for raw_entry in args.analysis_file:
        labeled_paths.append(parse_explicit_entry(raw_entry))

    existing = []
    missing = []
    for label, path in labeled_paths:
        if path.is_file():
            existing.append((label, path))
        else:
            missing.append(str(path))
    if missing:
        raise SystemExit("Missing analysis files:\n" + "\n".join(missing))
    if not existing:
        raise SystemExit("No analysis files found.")

    markdown = build_bundle_markdown(
        title=args.title,
        labeled_paths=existing,
        include_toc=args.include_toc,
    )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

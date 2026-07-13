#!/usr/bin/env python3
"""Generate a PD state-machine full-chain latency SVG from raw experiment data."""

import csv
import html
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    if seconds >= 1:
        return f"{seconds:.3f} s"
    return f"{seconds * 1000:.2f} ms"


def load_controller_result(run_dir: Path) -> Dict[str, Any]:
    text = (run_dir / "controller.log").read_text(encoding="utf-8", errors="replace")
    start = text.find("{")
    if start < 0:
        raise ValueError(f"{run_dir / 'controller.log'} does not contain JSON")
    obj = json.loads(text[start:])
    if obj.get("actions") and isinstance(obj["actions"], list):
        first = obj["actions"][0]
        if isinstance(first, dict) and "actions" in first:
            return first
    return obj


def pick_run_dir(suite: Path, run_name: Optional[str]) -> Path:
    if run_name:
        run_dir = suite / run_name
        if not (run_dir / "controller.log").exists():
            raise FileNotFoundError(run_dir / "controller.log")
        return run_dir

    candidates = [
        path
        for path in sorted(suite.iterdir())
        if path.is_dir() and (path / "controller.log").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"no controller.log found under {suite}")

    def score(path: Path) -> int:
        text = (path / "controller.log").read_text(encoding="utf-8", errors="replace")
        return int("waiting_manifest_count" in text) + int("manifest_count" in text)

    return max(candidates, key=score)


def status_items(action: Dict[str, Any]) -> List[Dict[str, Any]]:
    response = action.get("response")
    if isinstance(response, list):
        items = response
    elif isinstance(response, dict):
        items = [response]
    else:
        items = []
    out = []
    for item in items:
        status = item.get("status") if isinstance(item, dict) else None
        if isinstance(status, dict):
            out.append(status)
    return out


def entry_values(
    entries: Iterable[Dict[str, Any]],
    fn: Callable[[Dict[str, Any]], Optional[float]],
) -> List[float]:
    values = []
    for entry in entries:
        timing = entry.get("timing") or {}
        value = fn(timing)
        if value is not None:
            values.append(value)
    return values


def max_value(values: List[float]) -> Optional[float]:
    return max(values) if values else None


def first_elapsed(actions: List[Dict[str, Any]], step: str) -> Optional[float]:
    for action in actions:
        if action.get("step") == step:
            return float(action.get("elapsed_seconds", 0.0))
    return None


def first_action(actions: List[Dict[str, Any]], step: str) -> Optional[Dict[str, Any]]:
    for action in actions:
        if action.get("step") == step:
            return action
    return None


def latest_status_by_role(actions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for action in actions:
        for status in status_items(action):
            timing_debug = status.get("timing_debug") or {}
            if timing_debug.get("entries"):
                latest[str(status.get("role"))] = status
    return latest


def build_stages(suite: Path, run_name: Optional[str]) -> List[Dict[str, Any]]:
    run_dir = pick_run_dir(suite, run_name)
    result = load_controller_result(run_dir)
    actions = result["actions"]
    latest = latest_status_by_role(actions)

    source_status = latest.get("source", {})
    target_status = latest.get("target", {})
    source_debug = source_status.get("timing_debug") or {}
    target_debug = target_status.get("timing_debug") or {}
    source_session = source_debug.get("session") or {}
    target_session = target_debug.get("session") or {}
    source_entries = source_debug.get("entries") or []
    target_entries = target_debug.get("entries") or []
    waiting_source_entries = [
        entry for entry in source_entries if entry.get("source_queue") == "waiting"
    ]
    waiting_target_entries = [
        entry for entry in target_entries if entry.get("source_queue") == "waiting"
    ]

    kv_target = max_value(
        entry_values(
            target_entries,
            lambda t: t["target_transfer_success_mono"] - t["target_transferring_mono"]
            if "target_transfer_success_mono" in t and "target_transferring_mono" in t
            else None,
        )
    )
    kv_source = max_value(
        entry_values(
            source_entries,
            lambda t: t["source_transferred_mono"] - t["source_sent_mono"]
            if "source_transferred_mono" in t and "source_sent_mono" in t
            else None,
        )
    )
    held_to_adopt = max_value(
        entry_values(
            target_entries,
            lambda t: t["target_adopted_mono"] - t["target_held_mono"]
            if "target_adopted_mono" in t and "target_held_mono" in t
            else None,
        )
    )
    waiting_held_to_adopt = max_value(
        entry_values(
            waiting_target_entries,
            lambda t: t["target_adopted_mono"] - t["target_held_mono"]
            if "target_adopted_mono" in t and "target_held_mono" in t
            else None,
        )
    )
    source_to_finish = max_value(
        entry_values(
            source_entries,
            lambda t: (
                t.get("source_finish_migrated_mono")
                or t.get("source_finish_released_mono")
                or t.get("source_waiting_released_mono")
            )
            - t["source_transferred_mono"]
            if "source_transferred_mono" in t
            and (
                "source_finish_migrated_mono" in t
                or "source_finish_released_mono" in t
                or "source_waiting_released_mono" in t
            )
            else None,
        )
    )
    waiting_source_to_finish = max_value(
        entry_values(
            waiting_source_entries,
            lambda t: t["source_waiting_released_mono"] - t["source_transferred_mono"]
            if "source_waiting_released_mono" in t and "source_transferred_mono" in t
            else None,
        )
    )

    observe_action = first_action(actions, "observe_source_quiesce")
    observe_response = (
        observe_action.get("response")
        if isinstance(observe_action, dict) and isinstance(observe_action.get("response"), dict)
        else {}
    )
    pre = (
        (first_elapsed(actions, "router_drain_source") or 0.0)
        + (first_elapsed(actions, "pause_source_admission") or 0.0)
        + (first_elapsed(actions, "observe_source_quiesce") or 0.0)
    )
    commit_finish = (first_elapsed(actions, "commit_decode_migration_target") or 0.0) + (
        first_elapsed(actions, "finish_decode_migration_source") or 0.0
    )
    role_resume = sum(
        first_elapsed(actions, step) or 0.0
        for step in [
            "set_source_runtime_role",
            "refresh_router_source_role",
            "resume_source_admission",
            "router_undrain_source",
        ]
    )
    total_seconds = float(result.get("total_seconds", 0.0))
    migration_seconds = float(result.get("migration_seconds", 0.0))
    post_idle = first_elapsed(actions, "post_migration_idle_assertion")
    if post_idle is None:
        post_idle = first_elapsed(actions, "wait_source_idle")

    stages = [
        (
            "Router drain source",
            "router 标记 source draining",
            first_elapsed(actions, "router_drain_source"),
            "router /worker/drain",
        ),
        (
            "Pause source admission",
            "暂停 source admission",
            first_elapsed(actions, "pause_source_admission"),
            "source /runtime_role/admission",
        ),
        (
            "Observe source quiesce",
            "观察 + 源端静默",
            first_elapsed(actions, "observe_source_quiesce"),
            "samples=%s; residual=%s; waiting=%s"
            % (
                observe_response.get("sample_count", 0),
                observe_response.get("source_total_residual_reqs", "n/a"),
                observe_response.get("source_waiting_queue_reqs", "n/a"),
            ),
        ),
        (
            "Scan running + waiting",
            "扫描 running / waiting",
            (source_session.get("scan_running_reqs_s") or 0.0)
            + (source_session.get("scan_waiting_reqs_s") or 0.0),
            "running=%s; waiting=%s; skipped=%s"
            % (
                source_session.get("running_reqs", 0),
                source_session.get("waiting_reqs", 0),
                source_session.get("waiting_skipped_count", 0),
            ),
        ),
        (
            "Build manifests",
            "构造 manifest",
            source_session.get("build_manifests_s"),
            "running=%s; waiting=%s; total=%s"
            % (
                source_session.get("running_manifest_count", 0),
                source_session.get("waiting_manifest_count", 0),
                source_session.get("manifest_count", 0),
            ),
        ),
        (
            "Freeze waiting queue",
            "冻结源 waiting_queue",
            source_session.get("freeze_waiting_reqs_s"),
            "frozen=%s" % source_session.get("waiting_frozen_count", 0),
        ),
        (
            "Target prepare receiver",
            "target/prepare receiver",
            first_elapsed(actions, "prepare_decode_migration_target"),
            "prepare=%s; receiver init max=%s"
            % (
                fmt_duration(target_session.get("prepare_target_entries_s")),
                fmt_duration(
                    max_value(
                        entry_values(
                            target_entries, lambda t: t.get("target_init_receiver_s")
                        )
                    )
                ),
            ),
        ),
        (
            "KV transfer",
            "KV 传输到 target held",
            kv_target,
            "target=%s; source=%s; controller=%s"
            % (fmt_duration(kv_target), fmt_duration(kv_source), fmt_duration(migration_seconds)),
        ),
        (
            "Target held queue",
            "请求进入 transferred_held",
            0.0 if target_entries else None,
            "held entries=%s; waiting-origin=%s"
            % (len(target_entries), len(waiting_target_entries)),
        ),
        (
            "Commit target",
            "target commit/adopt",
            first_elapsed(actions, "commit_decode_migration_target"),
            "held->adopt max=%s; waiting=%s"
            % (fmt_duration(held_to_adopt), fmt_duration(waiting_held_to_adopt)),
        ),
        (
            "Finish source",
            "source finish/release",
            first_elapsed(actions, "finish_decode_migration_source"),
            "xfer->finish max=%s; waiting release=%s"
            % (fmt_duration(source_to_finish), fmt_duration(waiting_source_to_finish)),
        ),
        (
            "Post-migration idle assertion",
            "迁移后 idle 断言",
            post_idle,
            "bounded check after source finish",
        ),
        (
            "Set runtime role",
            "decode -> prefill",
            first_elapsed(actions, "set_source_runtime_role"),
            "/runtime_role/set",
        ),
        (
            "Refresh router role",
            "router 刷新 role",
            first_elapsed(actions, "refresh_router_source_role"),
            "router role=prefill",
        ),
        (
            "Resume admission",
            "恢复 admission",
            first_elapsed(actions, "resume_source_admission"),
            "paused=false",
        ),
        (
            "Router undrain",
            "router undrain",
            first_elapsed(actions, "router_undrain_source"),
            "draining=false",
        ),
    ]

    return [
        {
            "idx": idx,
            "stage": stage,
            "cn": cn,
            "latency_s": "" if latency is None else latency,
            "label": fmt_duration(latency),
            "detail": detail,
            "run_dir": str(run_dir),
            "total_seconds": total_seconds,
            "migration_seconds": migration_seconds,
        }
        for idx, (stage, cn, latency, detail) in enumerate(stages, start=1)
    ]


def write_csv(path: Path, stages: List[Dict[str, Any]]) -> None:
    fields = [
        "idx",
        "stage",
        "cn",
        "latency_s",
        "label",
        "detail",
        "run_dir",
        "total_seconds",
        "migration_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(stages)


def svg_escape(value: Any) -> str:
    return html.escape(str(value), quote=False)


def wrap_text(text: str, limit: int) -> List[str]:
    if len(text) <= limit:
        return [text]
    lines = []
    current = ""
    for part in text.split("; "):
        candidate = part if not current else current + "; " + part
        if len(candidate) > limit and current:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[:2]


def write_svg(path: Path, stages: List[Dict[str, Any]]) -> None:
    width = 1900
    box_w, box_h = 232, 96
    gap, left, first_top, row_gap = 28, 70, 160, 200
    row_counts = []
    remaining = len(stages)
    while remaining > 0:
        count = min(6, remaining)
        row_counts.append(count)
        remaining -= count
    height = max(1040, first_top + max(1, len(row_counts)) * row_gap + 300)
    positions = []
    stage_index = 0
    for row, count in enumerate(row_counts):
        y = first_top + row * row_gap
        for j in range(count):
            if stage_index >= len(stages):
                break
            col = j if row % 2 == 0 else count - 1 - j
            positions.append((left + col * (box_w + gap), y))
            stage_index += 1

    colors = ["#f4f7fb", "#eef7f1", "#fff7e8", "#f5f1fb", "#eef6f7"]
    accents = ["#3569b7", "#2f7d58", "#b86b00", "#7655a6", "#237884"]
    total = stages[0].get("total_seconds", 0.0) if stages else 0.0
    migration = stages[0].get("migration_seconds", 0.0) if stages else 0.0
    run_dir = stages[0].get("run_dir", "") if stages else ""

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif;fill:#20242a}.title{font-size:32px;font-weight:700}.sub{font-size:17px;fill:#525866}.box-title{font-size:17px;font-weight:700}.lat{font-size:22px;font-weight:700}.detail{font-size:12px;fill:#525866}.small{font-size:14px;fill:#525866}.arrow{stroke:#6d7785;stroke-width:2.1;fill:none;marker-end:url(#arrow)}</style>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#6d7785"/></marker></defs>',
        '<text x="70" y="58" class="title">PD 状态机 D->P 全链路延迟</text>',
        f'<text x="70" y="92" class="sub">run={svg_escape(Path(run_dir).name)}; total={fmt_duration(float(total))}; controller migration={fmt_duration(float(migration))}</text>',
    ]

    for i in range(len(positions) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        if y1 == y2 and x2 > x1:
            svg.append(
                f'<path class="arrow" d="M{x1 + box_w + 5},{y1 + box_h / 2} L{x2 - 9},{y2 + box_h / 2}"/>'
            )
        elif y1 == y2:
            svg.append(
                f'<path class="arrow" d="M{x1 - 5},{y1 + box_h / 2} L{x2 + box_w + 9},{y2 + box_h / 2}"/>'
            )
        else:
            svg.append(
                f'<path class="arrow" d="M{x1 + box_w / 2},{y1 + box_h + 8} C{x1 + box_w / 2},{y1 + box_h + 70} {x2 + box_w / 2},{y2 - 70} {x2 + box_w / 2},{y2 - 10}"/>'
            )

    for i, (stage, (x, y)) in enumerate(zip(stages, positions)):
        color = colors[i % len(colors)]
        accent = accents[i % len(accents)]
        svg.extend(
            [
                f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="8" fill="{color}" stroke="{accent}" stroke-width="2"/>',
                f'<text x="{x + 14}" y="{y + 27}" class="box-title">{stage["idx"]}. {svg_escape(stage["cn"])}</text>',
                f'<text x="{x + 14}" y="{y + 58}" class="lat" fill="{accent}">{svg_escape(stage["label"])}</text>',
            ]
        )
        for line_idx, line in enumerate(wrap_text(str(stage["detail"]), 36)):
            svg.append(
                f'<text x="{x + 14}" y="{y + 80 + line_idx * 15}" class="detail">{svg_escape(line)}</text>'
            )

    summary_y = first_top + max(1, len(row_counts)) * row_gap + 40
    svg.extend(
        [
            f'<rect x="70" y="{summary_y}" width="1760" height="150" rx="8" fill="#f7f9fb" stroke="#c7d0db"/>',
            f'<text x="95" y="{summary_y + 34}" class="box-title">读图说明</text>',
        ]
    )
    notes = [
        "新增 waiting_queue 链路：扫描/筛选 -> 构造 waiting manifest -> 冻结源 waiting_queue -> target transferred_held -> commit 进入目标 waiting_queue -> source release。",
        "图上的 HTTP action 使用 controller elapsed_seconds；KV transfer、held->adopt、source release 使用 worker timing_debug 推导。",
        "waiting_manifest_count=0 时，图仍显示 waiting 阶段，但说明本次 trace 没制造出可迁移 waiting 请求。",
    ]
    for idx, line in enumerate(notes):
        svg.append(
            f'<text x="105" y="{summary_y + 66 + idx * 26}" class="small">- {svg_escape(line)}</text>'
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def main(argv: List[str]) -> int:
    if len(argv) not in (2, 3):
        print(
            "usage: make_pd_state_machine_latency_diagram.py <suite-dir> [run-name]",
            file=sys.stderr,
        )
        return 2
    suite = Path(argv[1])
    run_name = argv[2] if len(argv) == 3 else None
    stages = build_stages(suite, run_name)
    write_csv(suite / "pd_state_machine_full_chain_latency.csv", stages)
    write_svg(suite / "pd_state_machine_full_chain_latency.svg", stages)
    print(suite / "pd_state_machine_full_chain_latency.svg")
    print(suite / "pd_state_machine_full_chain_latency.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

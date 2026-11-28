"""Academic figures from frozen source evidence and native captures.

All visible text is English. The fixed canvas width is166mm; no tight crop or
data-time compression is used. This script performs no GPU or service action.
"""

import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.text import Text
from matplotlib.textpath import TextPath
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
WIDTH_MM = 166.0
MM = 1 / 25.4
BLUE = "#0072B2"
ORANGE = "#E69F00"
COMM = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#87919C"
INK = "#20272E"
GRID = "#DCE1E6"
PALE = "#F1F3F5"
SOURCES = {}
FIGURES = []

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#77818B",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "hatch.linewidth": 0.4,
    }
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    path = path.resolve()
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


BUNDLE = read(ROOT / "data/frozen_inputs.json")
for entry in BUNDLE["source_files"]:
    SOURCES[entry["path"]] = entry["sha256"]
EVIDENCE = BUNDLE["EVIDENCE"]
MODEL = BUNDLE["MODEL"]
ORIGINAL = BUNDLE["ORIGINAL"]
PAYLOAD = BUNDLE["PAYLOAD"]
GROUPING = BUNDLE["GROUPING"]
TIMELINE = BUNDLE["TIMELINE"]
AUDIT = BUNDLE["AUDIT"]
LEGACY = BUNDLE["LEGACY"]
NAMES = ["pro64k_micro4", "pro64k_nomicro"]
FIRST = BUNDLE["FIRST"]
TIMING = BUNDLE["TIMING"]


def canvas(height_mm):
    fig = plt.figure(figsize=(WIDTH_MM * MM, height_mm * MM))
    fig.set_dpi(160)
    return fig


def diagram(height_mm):
    fig = canvas(height_mm)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_MM)
    ax.set_ylim(0, height_mm)
    ax.axis("off")
    return fig, ax


def text(
    ax,
    x,
    y,
    s,
    *,
    size=9.5,
    weight="normal",
    color=INK,
    ha="left",
    va="center",
    **kwargs,
):
    return ax.text(
        x,
        y,
        s,
        fontsize=size,
        weight=weight,
        color=color,
        ha=ha,
        va=va,
        linespacing=1.18,
        **kwargs,
    )


def box(ax, x, y, w, h, s, *, face="white", edge="#A7B0B9", size=9.5, weight="normal"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=0.6))
    text(ax, x + w / 2, y + h / 2, s, size=size, weight=weight, ha="center")


def arrow(ax, start, end, *, label=None, label_xy=None, color="#59636E"):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="->", mutation_scale=7.5, color=color, linewidth=0.7
        )
    )
    if label:
        x, y = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        text(
            ax,
            x,
            y,
            label,
            ha="center",
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )


def panel(fig, x_mm, y_mm, label):
    fig.text(
        x_mm / WIDTH_MM,
        y_mm / (fig.get_size_inches()[1] / MM),
        label,
        fontsize=10.5,
        weight="bold",
        ha="left",
        va="center",
        color=INK,
    )


def clean_axis(ax, grid=True):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    if grid:
        ax.grid(axis="x", color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)


def embedded_fonts(path):
    reader = PdfReader(path)
    fonts = {}
    for page in reader.pages:
        for obj in (page.get("/Resources", {}).get("/Font", {}) or {}).values():
            f = obj.get_object()
            items = f.get("/DescendantFonts", [f])
            for item in items:
                child = item.get_object() if hasattr(item, "get_object") else item
                descriptor = child.get("/FontDescriptor")
                descriptor = descriptor.get_object() if descriptor else {}
                name = str(child.get("/BaseFont", f.get("/BaseFont", "unknown")))
                fonts[name] = any(
                    k in descriptor for k in ["/FontFile", "/FontFile2", "/FontFile3"]
                )
    assert fonts and all(fonts.values()), fonts
    return fonts


def save(fig, name, id_, caption, kind, data):
    # Bounds are checked on the same fixed canvas, with no cropping or rescale.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        s = artist.get_text()
        assert not re.search(r"[\u4e00-\u9fff]", s), s
        b = artist.get_window_extent(renderer)
        if (
            b.width
            and b.height
            and (
                b.x0 < -0.5
                or b.y0 < -0.5
                or b.x1 > fig.bbox.width + 0.5
                or b.y1 > fig.bbox.height + 0.5
            )
        ):
            bounds.append(dict(text=s, bbox=[b.x0, b.y0, b.x1, b.y1]))
    assert not bounds, (name, bounds)
    outputs = {}
    for ext in ["png", "svg", "pdf"]:
        path = ROOT / (name + "." + ext)
        kwargs = {
            "dpi": 600,
            "bbox_inches": None,
            "pad_inches": 0,
            "facecolor": "white",
        }
        if ext == "pdf":
            kwargs["metadata"] = {
                "Title": caption,
                "Author": "",
                "Subject": "Frozen trace and source analysis",
            }
        fig.savefig(path, **kwargs)
        if ext == "svg":
            # Matplotlib emits indentation-only trailing spaces. Keep generated
            # figures compatible with the repository whitespace hook.
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
        outputs[ext] = {
            "path": str(path.resolve()),
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
    fontproof = embedded_fonts(ROOT / (name + ".pdf"))
    svg = (ROOT / (name + ".svg")).read_text()
    assert "<text" in svg
    from PIL import Image

    png = Image.open(ROOT / (name + ".png"))
    assert abs(png.width / (600 * MM) - WIDTH_MM) < 0.05
    FIGURES.append(
        dict(
            name=name,
            id=id_,
            kind=kind,
            caption=caption,
            caption_en=caption,
            width_mm=WIDTH_MM,
            height_mm=float(fig.get_size_inches()[1] / MM),
            data=data,
            outputs=outputs,
            PDF_fonts_embedded=fontproof,
            editable_SVG_text_nodes=svg.count("<text"),
            PNG_pixels=list(png.size),
            canvas_text_bounds="passed",
        )
    )
    plt.close(fig)


def fig01():
    shape = EVIDENCE["shape_example"]
    cp = shape["CP"]
    ep = shape["MoE_EP"]
    n_mb = shape["row_microbatches"]
    partition = [
        int(n)
        for n in ORIGINAL["safe_restore_environment"][
            "SGLANG_PP_LAYER_PARTITION"
        ].split(",")
    ]
    assert sum(partition) == MODEL["num_hidden_layers"]
    fig, ax = diagram(166)
    ax.set_ylim(-24, 142)
    panel(fig, 5, 161, "(a) Parallel configuration")
    first = 0
    for stage, x in enumerate([5, 93]):
        box(ax, x, 77, 68, 53, "", face="#F6F7F8")
        last = first + partition[stage] - 1
        text(
            ax,
            x + 34,
            124,
            f"PP stage {stage}  (layers {first}–{last})",
            size=10.0,
            weight="bold",
            ha="center",
        )
        for r in range(cp):
            box(
                ax,
                x + 3,
                112 - r * 9,
                62,
                7.5,
                f"GPU {stage * cp + r} / CP {r} / EP {r}",
            )
        text(
            ax,
            x + 34,
            81.5,
            f"{shape['local_experts']} local experts per rank",
            ha="center",
        )
        first = last + 1
    arrow(ax, (76, 104), (90, 104))
    text(ax, 83, 111, "PP", ha="center")
    text(ax, 83, 97, "transfer", ha="center")
    panel(fig, 5, 93, "(b) Row partitioning within one stage")
    box(
        ax,
        5,
        44,
        40,
        17,
        f"{shape['global_chunk_tokens']:,} / CP{cp}\n{shape['CP_local_rows']:,} rows/rank",
        face="#EDF4FA",
    )
    arrow(ax, (47, 52.5), (56, 52.5))
    for i in range(n_mb):
        box(
            ax,
            58 + i * 27,
            44,
            24.5,
            17,
            f"MB{i}\n{shape['per_rank_rows_per_microbatch']:,} rows",
            face="#EDF4FA",
        )
    for r in range(cp):
        box(
            ax,
            5 + r * 26,
            14,
            23,
            16,
            f"CP{r}: MBi\n{shape['per_rank_rows_per_microbatch']:,} rows",
            face="#F1F6F4",
        )
    arrow(ax, (108, 22), (121, 22), color=COMM)
    text(ax, 114.5, 36, "AllGather", ha="center", color=COMM)
    box(
        ax,
        123,
        12,
        38,
        20,
        f"{shape['gathered_rows_per_microbatch']:,} rows\nper microbatch",
        face="#FBEFE9",
    )
    arrow(ax, (142, 11), (142, -2), color=BLUE)
    text(ax, 150, 4, "MoE", ha="center", color=BLUE)
    box(
        ax,
        123,
        -20,
        38,
        17,
        f"{shape['gathered_rows_per_microbatch']:,} partial\noutput rows",
        face="#EDF4FA",
    )
    arrow(ax, (121, -11.5), (50, -11.5), color=COMM)
    text(ax, 85, -3, "ReduceScatter", ha="center", color=COMM)
    box(
        ax,
        5,
        -20,
        43,
        17,
        f"{shape['per_rank_rows_per_microbatch']:,} output\nrows/rank",
        face="#F1F6F4",
    )
    caption = "Parallel configuration and row partitioning in CP4–EP4–PP2. A 65,536-token request gives 16,384 rows per CP rank, split into four 4,096-row microbatches. Each AllGather collects 16,384 rows; after local expert computation, ReduceScatter returns 4,096 output rows per rank. This is the large-row branch. Microbatching does not create additional PP stages, and QAHEAD does not cross the PP boundary."
    save(
        fig,
        "fig01_parallelism_and_rows",
        "fig01_parallelism_and_rows",
        caption,
        "source_schematic",
        dict(shape, PP_layer_partition=partition, MoE_TP=1),
    )


def fig02():
    fig, ax = diagram(142)
    panel(fig, 5, 137, "(a) Event dependencies for microbatch i")
    text(ax, 38, 127, "Compute stream (32)", ha="center", weight="bold", size=10.0)
    text(
        ax,
        128,
        127,
        "MoE communication stream (25)",
        ha="center",
        weight="bold",
        size=10.0,
    )
    ax.plot([38, 38], [21, 122], color="#D6E3EF", linewidth=1.2, zorder=0)
    ax.plot([128, 128], [21, 122], color="#F0D8CB", linewidth=1.2, zorder=0)
    box(
        ax,
        7,
        104,
        62,
        16,
        "P_i: prepare\nO-A/O-B, mHC, router + quant/pack",
        face="#EDF4FA",
    )
    box(ax, 97, 85, 62, 14, "G_i: AllGather", face="#FBEFE9")
    box(ax, 7, 64, 62, 16, "M_i: expert FFN\nFC1 → SwiGLU → FC2", face="#EDF4FA")
    box(ax, 7, 46, 62, 13, "S_i: local Shared", face="#F1F6F4")
    arrow(ax, (38, 63), (38, 60))
    arrow(ax, (38, 45), (38, 41))
    box(ax, 97, 45, 62, 14, "RS_i: ReduceScatter", face="#FBEFE9")
    box(
        ax,
        7,
        23,
        62,
        17,
        "Shared add + mHC post\nEligible next-layer Q-front",
        face="#F7EFF6",
    )
    arrow(ax, (70, 107), (96, 95), label="src_ready[i]", label_xy=(83, 106))
    arrow(ax, (96, 88), (70, 75), label="gather_done[i]", label_xy=(83, 87))
    arrow(ax, (70, 67), (96, 55), label="compute_done[i]", label_xy=(83, 66))
    arrow(ax, (96, 48), (70, 35), label="rs_done[i]", label_xy=(83, 47))
    text(ax, 128, 30, "AG and RS are serialized", ha="center", color=COMM)
    panel(fig, 5, 15, "(b) Submission order on the MoE communication stream")
    order = EVIDENCE["C007_logical_execution"]["MoE_collective_order"]
    for i, label in enumerate(order):
        x = 5 + i * 20
        box(ax, x, 2, 16, 8, label, face="#FBEFE9")
        if i < len(order) - 1:
            arrow(ax, (x + 16.5, 6), (x + 19, 6), color=COMM)
    caption = "Event dependencies and communication ordering in C007. Gathers and reduce-scatters are serialized on one communication stream while event dependencies permit overlap with other work on the compute stream. The closing mHC and eligible next-layer Q-front run during the drain phase, after all expert and local Shared work has been submitted. This dependency diagram is schematic, not drawn to scale."
    save(
        fig,
        "fig02_events_and_stream_order",
        "fig02_events_and_stream_order",
        caption,
        "source_schematic",
        dict(
            communication_order=order,
            event_dependencies=EVIDENCE["C007_logical_execution"]["event_dependencies"],
            main_stream=32,
            communication_stream=25,
            QAHEAD="drain phase; eligible next layer in same PP stage",
        ),
    )


def fig03():
    observed = [r for r in PAYLOAD["large_MoE_paths"] if r["stage"] == 0]
    packed = next(r for r in observed if r["scenario"] == NAMES[0])
    bf16 = next(r for r in observed if r["scenario"] == NAMES[1])
    row_count = EVIDENCE["shape_example"]["CP_local_rows"]
    pr = packed["allgather_input_shape_sample"][0][1]
    br = bf16["allgather_input_shape_sample"][0][1] * 2
    old = next(
        f["data"] for f in LEGACY["figures"] if f["name"] == "fig03_packed_payload"
    )
    parts = old["parts_bytes"]
    activation = parts["MXFP8激活"]
    meta = sum(v for k, v in parts.items() if k != "MXFP8激活")
    assert (
        activation + meta == pr
        and pr * row_count == packed["allgather_API_input_bytes_per_large_layer"]
    )
    per_layer = [
        packed["allgather_API_input_bytes_per_large_layer"] / 2**20,
        bf16["allgather_API_input_bytes_per_large_layer"] / 2**20,
    ]
    fig = canvas(151)
    panel(fig, 5, 146, "(a) Logical input per token row")
    ax = fig.add_axes([47 / WIDTH_MM, 105 / 151, 114 / WIDTH_MM, 33 / 151])
    ax.barh([1], [activation], height=0.46, color=BLUE, label="MXFP8 activation")
    ax.barh(
        [1],
        [meta],
        left=activation,
        height=0.46,
        color=ORANGE,
        hatch="///",
        label="Metadata",
    )
    ax.barh([0], [br], height=0.46, color=GRAY)
    ax.set_xlim(0, 12500)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([1, 0], ["C007 MXFP8", "Fallback BF16"])
    ax.set_xticks([0, 4000, 8000, 12000])
    ax.set_xlabel("Input bytes per row (B)")
    clean_axis(ax)
    ax.text(pr + 180, 1, f"{pr:,} B", va="center", fontsize=10.0)
    ax.text(br + 180, 0, f"{br:,} B", va="center", fontsize=10.0)
    fig.legend(
        handles=[
            Patch(facecolor=BLUE, label=f"Activation: {activation:,} B"),
            Patch(facecolor=ORANGE, hatch="///", label=f"Metadata: {meta:,} B"),
        ],
        loc="upper left",
        bbox_to_anchor=(47 / WIDTH_MM, 91 / 151),
        ncols=2,
        frameon=False,
        fontsize=9.5,
        handlelength=1.3,
        columnspacing=1.1,
    )
    panel(fig, 5, 77, f"(b) Metadata zoom: {meta} B per row")
    meta_ax = fig.add_axes([47 / WIDTH_MM, 60 / 151, 114 / WIDTH_MM, 7 / 151])
    meta_ax.set_xlim(0, meta)
    meta_ax.set_ylim(0, 1)
    meta_ax.axis("off")
    left = 0
    for value, label, color in [
        (parts["K32 scale"], "Scales", ORANGE),
        (parts["top-k权重"], "Weights", "#F3C76B"),
        (parts["expert IDs"], "IDs", "#F9E3B4"),
    ]:
        meta_ax.add_patch(
            Rectangle(
                (left, 0), value, 1, facecolor=color, edgecolor="white", linewidth=0.5
            )
        )
        meta_ax.text(
            left + value / 2, 0.5, f"{value} B", ha="center", va="center", fontsize=9.5
        )
        meta_ax.text(left + value / 2, -0.6, label, ha="center", va="top", fontsize=9.5)
        left += value
    # A thin link identifies the small original metadata segment without covering plot labels.
    start = ax.transData.transform((pr - meta / 2, 1.23))
    start = fig.transFigure.inverted().transform(start)
    link = [
        (start[0] * WIDTH_MM, start[1] * 151),
        (start[0] * WIDTH_MM, 139),
        (164, 139),
        (164, 68),
        (159, 68),
    ]
    fig.add_artist(
        Line2D(
            [x / WIDTH_MM for x, y in link],
            [y / 151 for x, y in link],
            transform=fig.transFigure,
            color="#8D969F",
            linewidth=0.55,
            linestyle=(0, (2, 2)),
        )
    )
    panel(fig, 5, 43, "(c) Logical input per GPU per large-token layer")
    ax = fig.add_axes([47 / WIDTH_MM, 12 / 151, 114 / WIDTH_MM, 24 / 151])
    ax.barh([1, 0], per_layer, height=0.46, color=[BLUE, GRAY])
    ax.set_xlim(0, 195)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([1, 0], ["4 × 4,096 rows", "16,384 rows"])
    ax.set_xticks([0, 40, 80, 120, 160])
    clean_axis(ax)
    ax.set_xlabel("AllGather API input (MiB)")
    for y, v in zip([1, 0], per_layer):
        ax.text(v + 3, y, f"{v:.2f}", va="center", fontsize=10.0)
    caption = "Logical AllGather input per token row, metadata layout, and input per GPU per large-token layer. Packed MXFP8 occupies 5,328 bytes per row, including 160 scale bytes and two 24-byte routing fields; BF16 occupies 10,240 bytes. Four microbatches submit 83.25 MiB per layer versus 160 MiB for fallback, a 1.9219-fold ratio. API input is not physical link traffic; 1 MiB equals 2^20 bytes."
    save(
        fig,
        "fig03_packed_payload",
        "fig03_packed_payload",
        caption,
        "observed_api_payload",
        dict(
            parts_bytes=parts,
            packed_bytes_per_row=pr,
            BF16_bytes_per_row=br,
            rows_per_GPU=row_count,
            packed_input_MiB_per_layer=per_layer[0],
            BF16_input_MiB_per_layer=per_layer[1],
            ratio=per_layer[1] / per_layer[0],
            observed_API_arguments=observed,
        ),
    )


COLORS = {
    "projection": ORANGE,
    "moe": BLUE,
    "epilogue": GREEN,
    "query": PURPLE,
    "ag": COMM,
    "rs": COMM,
}


def range_bar(ax, g, y, h, label=True):
    lo, hi = g["start_ms"], g["end_ms"]
    kind = g["kind"]
    lab = g["label"]
    hatch = "///" if kind == "ag" else None
    patch = Rectangle(
        (lo, y - h / 2),
        hi - lo,
        h,
        facecolor=COLORS[kind],
        edgecolor="white",
        linewidth=0.4,
        hatch=hatch,
        zorder=3,
    )
    ax.add_patch(patch)
    # Add labels only when the true interval is physically wide enough.
    if label and lab:
        if lab.startswith("AG"):
            lab = "G" + lab[2:]
        text_width = TextPath((0, 0), lab, size=9.5).get_extents().width
        interval_width = (
            (min(hi, ax.get_xlim()[1]) - max(lo, ax.get_xlim()[0]))
            / (ax.get_xlim()[1] - ax.get_xlim()[0])
            * ax.bbox.width
            * 72
            / ax.figure.dpi
        )
        visible_mid = (min(hi, ax.get_xlim()[1]) + max(lo, ax.get_xlim()[0])) / 2
        if text_width + 2 < interval_width:
            ax.text(
                visible_mid,
                y,
                lab,
                fontsize=9.5,
                ha="center",
                va="center",
                color="white",
                clip_on=True,
                zorder=5,
            )


def intersection(a, b):
    lo, hi = max(a["start_ms"], b["start_ms"]), min(a["end_ms"], b["end_ms"])
    assert hi > lo
    return lo, hi


def fig04():
    cases = {c["scenario"]: c for c in TIMELINE["scenarios"]}
    groups = GROUPING["groups"]
    main_groups = groups[NAMES[0]]
    fallback = groups[NAMES[1]]
    cc = [g for g in main_groups if g["stream"] == 25]
    assert all(a["end_ms"] <= b["start_ms"] for a, b in zip(cc, cc[1:]))
    fig = canvas(190)
    panel(fig, 5, 185, "(a) Observed GPU timeline")
    handles = [
        Patch(facecolor=COLORS[k], hatch="///" if k == "ag" else None, label=v)
        for k, v in [
            ("projection", "O: output projection"),
            ("moe", "M: FC1–FC2 interval"),
            ("epilogue", "Shared / helpers"),
            ("query", "Q: next-layer Q interval"),
            ("ag", "G: AllGather"),
            ("rs", "RS: ReduceScatter"),
        ]
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(34 / WIDTH_MM, 179 / 190),
        ncols=3,
        frameon=False,
        fontsize=9.5,
        handlelength=1.1,
        columnspacing=0.5,
    )
    ax = fig.add_axes([35 / WIDTH_MM, 104 / 190, 128 / WIDTH_MM, 59 / 190])
    ax.set_xlim(20, 84)
    ax.set_ylim(-0.15, 3.08)
    ax.set_xticks([20, 30, 40, 50, 60, 70, 80])
    ys = [2.45, 1.55, 0.35]
    ax.set_yticks(ys, ["C007\nCompute 32", "C007\nMoE comm 25", "Fallback\nMain 32"])
    clean_axis(ax)
    for g in main_groups:
        if g["end_ms"] < 20 or g["start_ms"] > 84:
            continue
        range_bar(
            ax,
            g,
            ys[0] if g["stream"] == 32 else ys[1],
            0.32,
            label=g["kind"] in ["moe", "ag", "rs"],
        )
    for g in fallback:
        if g["end_ms"] < 20 or g["start_ms"] > 84:
            continue
        range_bar(ax, g, ys[2], 0.32, label=g["kind"] in ["moe", "ag", "rs"])
    endq = max(g["end_ms"] for g in main_groups if g["kind"] == "query")
    ax.add_patch(
        Rectangle(
            (endq, ys[0] - 0.22),
            84 - endq,
            0.44,
            facecolor=PALE,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.text(
        (endq + 84) / 2,
        ys[0],
        "L4 continuation\nnot shown",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#69737D",
    )
    # Dashed windows identify the real-time zooms without changing any bar.
    for start, end, marker in [(24, 29, "b"), (49, 54, "c")]:
        ax.add_patch(
            Rectangle(
                (start, 1.23),
                end - start,
                1.54,
                facecolor="none",
                edgecolor="#7B858F",
                linewidth=0.7,
                linestyle=(0, (3, 2)),
                zorder=6,
            )
        )
        ax.text((start + end) / 2, 2.88, f"({marker})", ha="center", fontsize=9.5)
    ax.set_xlabel("Time from the first observed L3 QKV-A (ms)")
    panel(fig, 5, 83, "(b) AllGather–compute overlap")
    panel(fig, 91, 83, "(c) Tail RS–Q-front overlap")
    left = fig.add_axes([25 / WIDTH_MM, 28 / 190, 61 / WIDTH_MM, 44 / 190])
    right = fig.add_axes([103 / WIDTH_MM, 28 / 190, 61 / WIDTH_MM, 44 / 190])
    for a, lims in [(left, (24, 29)), (right, (49, 54))]:
        a.set_xlim(*lims)
        a.set_ylim(-0.10, 1.94)
        a.set_yticks([1.12, 0.28], ["C32", "M25"])
        a.set_xticks(range(lims[0], lims[1] + 1))
        clean_axis(a)
    events = cases[NAMES[0]]["events"]
    ag1 = next(
        e
        for e in events
        if e.get("layer") == 3
        and e.get("micro") == 1
        and e["label"] == "MoE packed-FP8 AllGather"
    )
    fc10 = next(
        e
        for e in events
        if e.get("layer") == 3
        and e.get("micro") == 0
        and e["label"] == "MoE expert FC1"
    )
    overlap_b = intersection(ag1, fc10)
    left.axvspan(*overlap_b, color="#F5DE9F", alpha=0.55, zorder=0)
    for e in events:
        if e["end_ms"] < 24 or e["start_ms"] > 29 or e.get("layer") != 3:
            continue
        if e["label"] not in [
            "MoE expert FC1",
            "MoE expert FC2 scatter",
            "MoE packed-FP8 AllGather",
            "MoE ReduceScatter",
        ]:
            continue
        kind = (
            "moe"
            if "expert" in e["label"]
            else "ag"
            if "AllGather" in e["label"]
            else "rs"
        )
        label = (
            ("FC1[" if e["label"] == "MoE expert FC1" else "FC2[")
            + str(e["micro"])
            + "]"
            if kind == "moe"
            else ("G" if kind == "ag" else "RS") + str(e["micro"])
        )
        range_bar(
            left,
            dict(e, kind=kind, label=label),
            1.12 if e["stream"] == 32 else 0.28,
            0.32,
        )
    left.text(
        sum(overlap_b) / 2,
        1.68,
        f"{overlap_b[1] - overlap_b[0]:.3f} ms",
        ha="center",
        fontsize=9.5,
    )
    overlaps_c = []
    for g in main_groups:
        if g["end_ms"] < 49 or g["start_ms"] > 54:
            continue
        if g["stream"] == 25:
            range_bar(right, g, 0.28, 0.32)
        elif g["kind"] == "query":
            # Render actual QKV-A and Q-B events separately, with a thin connector.
            right.plot(
                [g["start_ms"], g["end_ms"]],
                [1.12, 1.12],
                color=PURPLE,
                linewidth=0.6,
                zorder=1,
            )
            for e in g["events"]:
                right.add_patch(
                    Rectangle(
                        (e["start_ms"], 0.96),
                        e["end_ms"] - e["start_ms"],
                        0.32,
                        facecolor=PURPLE,
                        edgecolor="white",
                        linewidth=0.35,
                        zorder=3,
                    )
                )
            right.text(
                (g["start_ms"] + g["end_ms"]) / 2,
                1.40,
                g["label"],
                ha="center",
                fontsize=9.5,
                color=INK,
            )
    for qi, ri in [(0, 2), (2, 3)]:
        qb = next(
            e
            for e in events
            if e.get("layer") == 4
            and e.get("micro") == qi
            and e["label"] == "Attention Q-B"
        )
        rs = next(
            e
            for e in events
            if e.get("layer") == 3
            and e.get("micro") == ri
            and e["label"] == "MoE ReduceScatter"
        )
        lo, hi = intersection(qb, rs)
        right.axvspan(lo, hi, color="#F5DE9F", alpha=0.55, zorder=0)
        right.text((lo + hi) / 2, 1.75, f"{hi - lo:.3f} ms", ha="center", fontsize=9.5)
        overlaps_c.append(
            dict(
                query_micro=qi, RS_micro=ri, start_ms=lo, end_ms=hi, duration_ms=hi - lo
            )
        )
    fig.text(
        0.5,
        7 / 190,
        "Zoom axes use the same time reference as the overview (ms).",
        ha="center",
        fontsize=9.5,
    )
    caption = "Observed GPU events for trial 101, CP0/PP0. Each run is aligned to its own first observed L3 QKV-A launch, not a strict layer boundary. The overview preserves true O/M/Q interval widths, including internal gaps; zooms show native kernel events and measured intersections. AG and RS share one communication stream in C007. Unshown L4 activity is not idle time, and the excerpt is not a causal end-to-end latency decomposition."
    save(
        fig,
        "actual_timeline_report",
        "actual_timeline_report",
        caption,
        "native_trace_ranges_and_kernel_intersections",
        dict(
            xlim_ms=[20, 84],
            zooms_ms=[[24, 29], [49, 54]],
            representative="trial101 CP0/PP0",
            overview_source=GROUPING["source"],
            overview_source_sha256=GROUPING["source_sha256"],
            overlap_b=dict(
                start_ms=overlap_b[0],
                end_ms=overlap_b[1],
                duration_ms=overlap_b[1] - overlap_b[0],
            ),
            overlaps_c=overlaps_c,
            groups=groups,
            source_events=cases,
        ),
    )


def fig05():
    first = [FIRST[name] for name in NAMES]
    raw = [
        d["visible_transfer_summary"]["total"]["raw_full_event_sum_us"] / 1000
        for d in first
    ]
    exposed = [
        d["visible_transfer_summary"]["total"]["exposed_attributed_us"] / 1000
        for d in first
    ]
    windows = [d["window"]["duration_ms"] for d in first]
    assert all(
        abs(d["visible_transfer_summary"]["total"]["self_union_us"] / 1000 - r) < 1e-5
        for d, r in zip(first, raw)
    )
    variants = []
    for name in NAMES:
        s = TIMING[name]["service_prefill_seconds"]
        samples = np.array(s["samples"]) * 1000
        p50 = float(s["p50"] * 1000)
        p90 = float(s["p90"] * 1000)
        assert len(samples) == 10 and math.isclose(
            np.quantile(samples, 0.5, method="linear"), p50, rel_tol=1e-12
        )
        assert math.isclose(
            np.quantile(samples, 0.9, method="linear"), p90, rel_tol=1e-12
        )
        variants.append(
            dict(scenario=name, samples_ms=samples.tolist(), p50_ms=p50, p90_ms=p90)
        )
    fig = canvas(132)
    panel(fig, 5, 127, "(a) First 10 layers: communication residency")
    ax = fig.add_axes([35 / WIDTH_MM, 79 / 132, 128 / WIDTH_MM, 40 / 132])
    ys = [1, 0]
    ax.barh(ys, exposed, height=0.40, color=COMM, label="Exposed")
    ax.barh(
        ys,
        np.array(raw) - np.array(exposed),
        left=exposed,
        height=0.40,
        color="#D8E4EE",
        hatch="///",
        edgecolor="white",
        linewidth=0.3,
        label="Overlapped residency",
    )
    ax.set_xlim(0, 470)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_ylim(-0.45, 1.45)
    ax.set_yticks(ys, ["C007", "Fallback"])
    clean_axis(ax)
    ax.set_xlabel("Communication time (ms)")
    for y, r, e, w in zip(ys, raw, exposed, windows):
        ax.text(
            e / 2, y, f"{e:.3f}", ha="center", va="center", fontsize=10.0, color="white"
        )
        ax.text(r + 5, y, f"{r:.3f}", va="center", fontsize=9.5)
    fig.legend(
        handles=[
            Patch(facecolor=COMM, label="Exposed communication"),
            Patch(facecolor="#D8E4EE", hatch="///", label="Overlapped residency"),
        ],
        loc="upper left",
        bbox_to_anchor=(34 / WIDTH_MM, 67 / 132),
        frameon=False,
        ncols=2,
        fontsize=9.5,
        handlelength=1.2,
        columnspacing=1.0,
    )
    panel(fig, 5, 56, "(b) Unprofiled full-pipeline prefill latency")
    ax = fig.add_axes([35 / WIDTH_MM, 17 / 132, 128 / WIDTH_MM, 32 / 132])
    offsets = np.array([-0.09, 0.03, -0.03, 0.09, -0.06, 0.06, 0, -0.12, 0.12, 0.015])
    for y, v in zip(ys, variants):
        ax.scatter(
            v["samples_ms"],
            y + offsets,
            s=14,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=0.7,
            zorder=3,
        )
        ax.scatter(
            [v["p50_ms"]],
            [y],
            s=27,
            marker="D",
            facecolor=INK,
            edgecolor="white",
            linewidth=0.4,
            zorder=5,
        )
        ax.plot(
            [v["p90_ms"], v["p90_ms"]],
            [y - 0.16, y + 0.16],
            color=COMM,
            linewidth=1.2,
            zorder=4,
        )
        ax.text(
            v["p90_ms"] + 40,
            y + 0.15,
            f"p50 {v['p50_ms']:.3f}\np90 {v['p90_ms']:.3f}",
            fontsize=9.5,
            va="center",
        )
    ax.set_xlim(0, 2180)
    ax.set_ylim(-0.45, 1.45)
    ax.set_yticks(ys, ["C007", "Fallback"])
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    clean_axis(ax)
    ax.set_xlabel("Prefill latency (ms)")
    caption = "Communication exposure and unprofiled latency for two 64K execution paths. Panel (a) separates raw GPU communication residency into exposed and overlapped portions within the first 10 layers. Panel (b) shows 10 serial zero-cache requests per configuration, with paired input seeds across configurations; circles are individual requests, diamonds p50, and vertical ticks p90. The panels have different timing scopes. Both paths jointly change scheduling controls and gather format."
    save(
        fig,
        "fig04_exposed_communication_and_latency",
        "fig05_communication_and_latency",
        caption,
        "observed_timing_and_accounting",
        dict(
            raw_communication_ms=raw,
            exposed_communication_ms=exposed,
            windows_ms=windows,
            variants=variants,
            scatter_y_offsets=offsets.tolist(),
            sample_protocol="10 serial zero-cache requests with paired trial seeds across configurations; different inputs across trials",
            p90_method="linear interpolation; not a confidence interval",
            sources=[
                s
                for s in BUNDLE["source_files"]
                if any(n in s["path"] for n in NAMES)
                and (
                    s["path"].endswith("/first10.json")
                    or s["path"].endswith("_timing_requests.json")
                )
            ],
        ),
    )


def fig06():
    delta = AUDIT["delta_nomicro_minus_micro4"]
    components = dict(
        NCCL_exposed=delta["NCCL_exposed_ms"],
        noncommunication_exposed=delta["noncommunication_exposed_ms"],
        idle=delta["idle_ms"],
    )
    base = AUDIT["scenario_audits"][0]["window_ms"]
    end = AUDIT["scenario_audits"][1]["window_ms"]
    total = delta["window_ms"]
    assert math.isclose(sum(components.values()), total, abs_tol=1e-8) and math.isclose(
        end - base, total, abs_tol=1e-8
    )
    fig = canvas(96)
    panel(fig, 5, 91, "Primary profile difference: Fallback − C007")
    ax = fig.add_axes([23 / WIDTH_MM, 25 / 96, 138 / WIDTH_MM, 58 / 96])
    vals = list(components.values())
    colors = [COMM, BLUE, GRAY]
    cumulative = 0.0
    width = 0.60
    for i, (val, color) in enumerate(zip(vals, colors)):
        next_ = cumulative + val
        ax.bar(
            i,
            abs(val),
            bottom=min(cumulative, next_),
            width=width,
            color=color,
            zorder=3,
        )
        if i < 2:
            ax.plot(
                [i + width / 2, i + 1 - width / 2],
                [next_, next_],
                color="#7B858F",
                linewidth=0.65,
                linestyle=(0, (3, 2)),
                zorder=2,
            )
        label_y = max(cumulative, next_) + [20, 50, 82][i]
        ax.annotate(
            f"{val:+.3f}",
            xy=(i, max(cumulative, next_)),
            xytext=(i, label_y),
            ha="center",
            fontsize=10.0,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=INK) if i else None,
        )
        cumulative = next_
    ax.bar(3, total, width=width, color="#46525F", zorder=3)
    ax.text(3, total + 20, f"{total:.3f}", ha="center", fontsize=10.0)
    ax.set_xticks([0, 1, 2, 3], ["NCCL\nexposed", "Noncomm.\nexposed", "Idle", "Total"])
    ax.set_ylabel("Change in attributed time (ms)")
    ax.set_ylim(0, 680)
    ax.set_yticks([0, 200, 400, 600])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=5)
    caption = "Conservation of the primary-profile wall-window difference between fallback and C007, using the same attribution rule and representative CP0 ranks across both PP stages. Increased NCCL exposure, reduced noncommunication exposure, and increased idle time sum to 562.703 ms, matching 1783.987 minus 1221.284 ms. This is timeline accounting for primary trial 101, not a causal ablation or the difference between unprofiled p50 values."
    save(
        fig,
        "fig06_profile_delta",
        "fig06_profile_delta",
        caption,
        "observed_profile_accounting",
        dict(
            base_ms=base,
            components_ms=components,
            delta_total_ms=total,
            end_ms=end,
            source_pointer="/delta_nomicro_minus_micro4",
            primary_trial=101,
            unrelated_unprofiled_p50_delta_ms=delta["unprofiled_p50_ms"],
            source=next(
                s
                for s in BUNDLE["source_files"]
                if s["path"].endswith("/native_comparison_audit.json")
            ),
        ),
    )


def main():
    fig01()
    fig02()
    fig03()
    fig04()
    fig05()
    fig06()
    script = Path(__file__).resolve()
    manifest = dict(
        status="generated_pending_visual_QA",
        width_mm=WIDTH_MM,
        final_font_points=[9.5, 10.0, 10.5],
        font_family="DejaVu Sans",
        export=dict(
            PNG_dpi=600,
            SVG_editable_text=True,
            PDF_font_embedding=True,
            fixed_canvas=True,
        ),
        reproducibility=dict(
            input_bundle={
                "path": "data/frozen_inputs.json",
                "sha256": sha(ROOT / "data/frozen_inputs.json"),
            },
            python="3.12.14",
            matplotlib=matplotlib.__version__,
            numpy=np.__version__,
            requirements_file="requirements.txt",
        ),
        script={"path": script.name, "sha256": sha(script)},
        source_files=[{"path": p, "sha256": h} for p, h in sorted(SOURCES.items())],
        figures=FIGURES,
    )
    (ROOT / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {"figures": len(FIGURES), "manifest": str(ROOT / "figure_manifest.json")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

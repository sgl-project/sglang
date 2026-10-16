"""Plot only audited complete Beam OpenHands points; preserve failures in JSON."""

import argparse
import json
import os

from curve_config import inside_repo


def collect(root):
    rows = []
    for path in sorted(root.glob("*-c*/result.json")):
        r = json.loads(path.read_text())
        r["source"] = str(path.relative_to(root))
        rows.append(r)
    failures = []
    for path in sorted(root.glob("*-c*")):
        if path.is_dir() and not (path / "result.json").exists():
            failures.append(
                {
                    "case": path.name,
                    "status": "incomplete or failed; inspect raw evidence",
                }
            )
    return rows, failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", help="run directory, or reference-results.json")
    p.add_argument("--output", help="output directory (required for a JSON input)")
    p.add_argument("--layouts", nargs="+", choices=["tp4", "tep4", "tp8", "tep8"])
    p.add_argument("--concurrencies", nargs="+", type=int)
    p.add_argument("--percentile", choices=["p50", "p90", "both"], default="p50")
    a = p.parse_args()
    source = inside_repo(a.results)
    if source.is_file():
        if not a.output:
            p.error("JSON input requires --output so the source is preserved")
        rows, failures = json.loads(source.read_text())["points"], []
    else:
        rows, failures = collect(source)
    root = inside_repo(a.output) if a.output else source
    root.mkdir(parents=True, exist_ok=True)
    (root / "curve.json").write_text(
        json.dumps(
            {
                "points": rows,
                "unscored": failures,
                "note": "Simulated acceptance 4, draft8, real draft proposals; not a quality benchmark",
                "plot_selection": {
                    "layouts": a.layouts,
                    "concurrencies": a.concurrencies,
                    "percentile": a.percentile,
                },
            },
            indent=2,
        )
        + "\n"
    )
    good = [
        r
        for r in rows
        if r["status"] == "OK"
        and r["failed"] == 0
        and (not a.layouts or r["layout"] in a.layouts)
        and (not a.concurrencies or r["cc"] in a.concurrencies)
    ]
    if not good:
        p.error("No audited successful points match the selection")
    os.environ["MPLCONFIGDIR"] = str(root / "plot-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    percentiles = ("p50", "p90") if a.percentile == "both" else (a.percentile,)
    fig, axes = plt.subplots(
        1, len(percentiles), figsize=(6.5 * len(percentiles), 5.4), layout="constrained"
    )
    if len(percentiles) == 1:
        axes = [axes]
    for ax, percentile in zip(axes, percentiles):
        for layout in sorted({r["layout"] for r in good}):
            color = "#d95f02" if layout.startswith("tep") else "#1565c0"
            points = sorted(
                (r for r in good if r["layout"] == layout), key=lambda r: r["cc"]
            )
            ax.plot(
                [r[f"interactivity_{percentile}"] for r in points],
                [r["total_tok_s_per_gpu"] for r in points],
                "o-",
                color=color,
                label=layout.upper(),
            )
            for r in points:
                ax.annotate(
                    f"cc={r['cc']}",
                    (r[f"interactivity_{percentile}"], r["total_tok_s_per_gpu"]),
                    xytext=(-27, -12) if layout.startswith("tep") else (5, 7),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set_xlabel(f"Interactivity: 1000 / TPOT {percentile} (tokens/s/user)")
        ax.set_ylabel("Total input + output tokens/s/GPU (cached input included)")
        ax.grid(alpha=0.25)
        ax.legend()
    cache_sizes = (
        ", ".join(
            str(n)
            for n in sorted(
                {
                    r.get("hicache_gb_per_rank", 0)
                    for r in good
                    if r.get("hicache_gb_per_rank", 0)
                }
            )
        )
        or "off"
    )
    cache_note = (
        "HiCache off"
        if cache_sizes == "off"
        else f"HiCache {cache_sizes} GB/rank at cc ≥16; off below"
    )
    fig.suptitle(
        f"Beam NVFP4 OpenHands · {good[0]['n_gpus']} GB300\nDraft 8 / simulated acceptance 4 · {cache_note}"
    )
    for ext in ("svg", "png", "pdf"):
        fig.savefig(root / f"curve.{ext}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

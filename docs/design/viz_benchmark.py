#!/usr/bin/env python3
"""Visualize PIN benchmark: tell the story of WHY pinning works.

Parses sglang server logs and aiperf records to show:
  1. Cache fills up during flood traffic
  2. Without PIN: VIP blocks evicted, first request recomputes everything
  3. With PIN: VIP blocks survive, first request gets instant cache hit
  4. Per-turn TTFT comparison proving the speedup

Usage:
    python viz_benchmark.py /tmp/pin_benchmark_v4_fp8_v4/rep0
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

# -- Log parsing --

RE_PREFILL = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] "
    r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), "
    r"token usage: ([\d.]+)"
)
RE_RETRACT = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] "
    r"KV cache pool is full\. Retract requests\. #retracted_reqs: (\d+)"
)
RE_MAX_TOKENS = re.compile(r"max_total_num_tokens=(\d+)")


def parse_server_log(log_path: str) -> dict:
    prefills = []
    retractions = []
    max_tokens = None
    t0 = None

    with open(log_path) as f:
        for line in f:
            if max_tokens is None:
                m = RE_MAX_TOKENS.search(line)
                if m:
                    max_tokens = int(m.group(1))

            m = RE_PREFILL.search(line)
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                if t0 is None:
                    t0 = ts
                prefills.append({
                    "t": (ts - t0).total_seconds(),
                    "new_seq": int(m.group(2)),
                    "new_token": int(m.group(3)),
                    "cached_token": int(m.group(4)),
                    "token_usage": float(m.group(5)),
                })
                continue

            m = RE_RETRACT.search(line)
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                if t0 is None:
                    t0 = ts
                retractions.append({
                    "t": (ts - t0).total_seconds(),
                })

    return {"max_tokens": max_tokens, "prefills": prefills, "retractions": retractions}


def parse_aiperf_records(records_dir: str, prefix: str) -> list[dict]:
    """Parse per-turn TTFT from aiperf records JSONL."""
    turns = []
    for p in Path(records_dir).glob(f"{prefix}*.jsonl"):
        if "raw" in p.name:
            continue
        with open(p) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    meta = rec.get("metadata", {})
                    metrics = rec.get("metrics", {})
                    ttft_obj = metrics.get("time_to_first_token", {})
                    ttft = ttft_obj.get("value") if isinstance(ttft_obj, dict) else None
                    isl_obj = metrics.get("input_sequence_length", {})
                    isl = isl_obj.get("value") if isinstance(isl_obj, dict) else None
                    if ttft is not None and ttft > 0:
                        turns.append({
                            "turn": meta.get("turn_index", len(turns)),
                            "ttft_ms": ttft,
                            "isl": isl or 0,
                        })
                except (json.JSONDecodeError, KeyError):
                    continue
    turns.sort(key=lambda x: x["turn"])
    return turns


def detect_phases(prefills: list[dict]) -> dict:
    """Detect warmup/flood/replay boundaries from prefill patterns.

    Heuristic: warmup is the initial low-concurrency period (new_seq=1, sequential).
    Flood starts when we see sustained high new_seq or the gap between warmup and flood.
    Replay starts after the last retraction or when token_usage drops back down.
    """
    if not prefills:
        return {}

    # Find first retraction-level usage (>0.9) as flood start
    flood_start = None
    for i, p in enumerate(prefills):
        if p["token_usage"] > 0.85 and i > 5:
            flood_start = p["t"]
            break

    # Find where usage drops significantly after flood (replay start)
    replay_start = None
    if flood_start:
        peak_seen = False
        for p in prefills:
            if p["t"] < flood_start:
                continue
            if p["token_usage"] > 0.9:
                peak_seen = True
            if peak_seen and p["token_usage"] < 0.5:
                replay_start = p["t"]
                break

    return {
        "warmup_end": flood_start,
        "flood_start": flood_start,
        "replay_start": replay_start,
    }


def generate_html(bl: dict, pin: dict | None, bl_turns: list, pin_turns: list,
                   rep_json: dict | None) -> str:
    bl_phases = detect_phases(bl["prefills"])

    # Find the "money shot": first post-flood prefill with system-prompt-sized input
    def first_replay_prefill(data, phases):
        replay_t = phases.get("replay_start")
        if not replay_t:
            return None
        for p in data["prefills"]:
            if p["t"] >= replay_t and (p["cached_token"] + p["new_token"]) > 1000:
                return p
        return None

    bl_first = first_replay_prefill(bl, bl_phases)
    pin_phases = detect_phases(pin["prefills"]) if pin else {}
    pin_first = first_replay_prefill(pin, pin_phases) if pin else None

    data = {
        "baseline": bl,
        "pinned": pin,
        "bl_phases": bl_phases,
        "pin_phases": pin_phases,
        "bl_first_replay": bl_first,
        "pin_first_replay": pin_first,
        "bl_turns": bl_turns,
        "pin_turns": pin_turns,
        "rep": rep_json,
    }
    data_json = json.dumps(data, default=str)

    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Why PIN Works</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3"></script>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 20px 30px; background: #0f172a; color: #e2e8f0; }
  h1 { font-size: 1.5em; color: #f1f5f9; margin-bottom: 4px; }
  .subtitle { color: #94a3b8; font-size: 0.9em; margin-bottom: 24px; }
  h2 { font-size: 1em; color: #cbd5e1; margin: 0 0 12px 0; }

  .story { display: flex; flex-direction: column; gap: 24px; }

  /* The headline numbers */
  .headline { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; }
  .hbox { background: #1e293b; border-radius: 10px; padding: 16px 24px;
          flex: 1; min-width: 200px; text-align: center; }
  .hbox .label { font-size: 0.75em; color: #64748b; text-transform: uppercase;
                 letter-spacing: 0.05em; }
  .hbox .val { font-size: 2em; font-weight: 700; margin: 6px 0; }
  .hbox .detail { font-size: 0.8em; color: #94a3b8; }
  .green { color: #4ade80; }
  .red { color: #f87171; }
  .blue { color: #60a5fa; }
  .amber { color: #fbbf24; }

  /* Step cards */
  .step { background: #1e293b; border-radius: 10px; padding: 20px; }
  .step-num { display: inline-block; background: #334155; color: #94a3b8;
              border-radius: 50%; width: 24px; height: 24px; text-align: center;
              line-height: 24px; font-size: 0.75em; font-weight: 700;
              margin-right: 8px; vertical-align: middle; }
  .step h2 { display: inline; vertical-align: middle; }
  .step p { color: #94a3b8; font-size: 0.85em; margin: 8px 0 16px 0; }
  canvas { max-height: 280px; }

  .side-by-side { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 900px) { .side-by-side { grid-template-columns: 1fr; } }

  .money-shot { display: flex; gap: 16px; align-items: stretch; }
  .money-shot .scenario { flex: 1; background: #0f172a; border-radius: 8px;
                           padding: 16px; text-align: center; }
  .money-shot .scenario.bad { border: 2px solid #f87171; }
  .money-shot .scenario.good { border: 2px solid #4ade80; }
  .money-shot .tokens { font-size: 2.2em; font-weight: 700; }
  .money-shot .arrow { display: flex; align-items: center; justify-content: center;
                        font-size: 1.5em; color: #64748b; padding: 0 4px; }
</style>
</head>
<body>
<h1>Why PIN Works</h1>
<p class="subtitle">Block pinning prevents LRU eviction of high-value KV cache entries under multi-tenant load</p>

<div id="app"></div>

<script>
const D = """ + data_json + """;

const app = document.getElementById('app');
const bl = D.baseline, pin = D.pinned, rep = D.rep;
const blFirst = D.bl_first_replay, pinFirst = D.pin_first_replay;

// -- Headline numbers --
let headlineHTML = '<div class="headline">';

// Cache capacity
headlineHTML += `<div class="hbox">
  <div class="label">KV Cache Capacity</div>
  <div class="val blue">${bl.max_tokens ? bl.max_tokens.toLocaleString() : '?'}</div>
  <div class="detail">tokens (page_size=64)</div>
</div>`;

// Retractions
headlineHTML += `<div class="hbox">
  <div class="label">Baseline Retractions</div>
  <div class="val amber">${bl.retractions.length}</div>
  <div class="detail">times cache overflowed</div>
</div>`;

if (pin) {
  headlineHTML += `<div class="hbox">
    <div class="label">Pinned Retractions</div>
    <div class="val amber">${pin.retractions.length}</div>
    <div class="detail">times cache overflowed</div>
  </div>`;
}

// The key result: first replay cached tokens
if (blFirst) {
  headlineHTML += `<div class="hbox">
    <div class="label">Baseline: 1st VIP Prefill After Flood</div>
    <div class="val red">${blFirst.cached_token} cached</div>
    <div class="detail">${blFirst.new_token} new tokens (full recompute)</div>
  </div>`;
}
if (pinFirst) {
  headlineHTML += `<div class="hbox">
    <div class="label">Pinned: 1st VIP Prefill After Flood</div>
    <div class="val green">${pinFirst.cached_token} cached</div>
    <div class="detail">${pinFirst.new_token} new tokens (prefix hit!)</div>
  </div>`;
}
headlineHTML += '</div>';
app.innerHTML = headlineHTML;

// -- Story steps --
let storyHTML = '<div class="story">';

// Step 1: Cache fills up
storyHTML += `<div class="step">
  <span class="step-num">1</span>
  <h2>Flood traffic fills the KV cache</h2>
  <p>Multi-tenant traffic (8 concurrent sessions, 400 requests) fills the cache to capacity.
     The server starts retracting requests when usage hits 100%.</p>
  <div class="side-by-side">
    <div><canvas id="bl-fill"></canvas></div>
    <div><canvas id="pin-fill"></canvas></div>
  </div>
</div>`;

// Step 2: The money shot
if (blFirst && pinFirst) {
  storyHTML += `<div class="step">
    <span class="step-num">2</span>
    <h2>First VIP request after the flood</h2>
    <p>The VIP sends the same system prompt again. Without PIN, it was evicted -- every token
       must be recomputed. With PIN, it survived -- instant cache hit.</p>
    <div class="money-shot">
      <div class="scenario bad">
        <div class="label" style="color:#f87171">WITHOUT PIN</div>
        <div class="tokens red">${blFirst.cached_token}</div>
        <div class="detail">cached tokens</div>
        <div style="margin-top:8px;font-size:0.85em;color:#94a3b8">
          ${blFirst.new_token} tokens recomputed<br>
          Usage: ${(blFirst.token_usage * 100).toFixed(0)}%
        </div>
      </div>
      <div class="arrow">vs</div>
      <div class="scenario good">
        <div class="label" style="color:#4ade80">WITH PIN</div>
        <div class="tokens green">${pinFirst.cached_token}</div>
        <div class="detail">cached tokens</div>
        <div style="margin-top:8px;font-size:0.85em;color:#94a3b8">
          ${pinFirst.new_token} tokens recomputed<br>
          Usage: ${(pinFirst.token_usage * 100).toFixed(0)}%
        </div>
      </div>
    </div>
  </div>`;
}

// Step 3: Per-turn TTFT comparison
if (D.bl_turns.length > 0 && D.pin_turns.length > 0) {
  storyHTML += `<div class="step">
    <span class="step-num">3</span>
    <h2>Per-turn TTFT: Baseline vs Pinned</h2>
    <p>Time to first token for each conversation turn in the VIP replay.
       Turn 0 (system prompt) shows the biggest improvement from pinning.</p>
    <canvas id="ttft-compare"></canvas>
  </div>`;
}

// Step 4: Cache hit timeline during replay
storyHTML += `<div class="step">
  <span class="step-num">4</span>
  <h2>Cached tokens per prefill (full timeline)</h2>
  <p>Green dots = tokens served from cache. Red dots = tokens recomputed.
     During the VIP replay phase (right side), pinned shows consistent cache hits.</p>
  <div class="side-by-side">
    <div>
      <div style="text-align:center;color:#f87171;font-size:0.8em;margin-bottom:4px">BASELINE</div>
      <canvas id="bl-tokens"></canvas>
    </div>
    <div>
      <div style="text-align:center;color:#4ade80;font-size:0.8em;margin-bottom:4px">PINNED</div>
      <canvas id="pin-tokens"></canvas>
    </div>
  </div>
</div>`;

storyHTML += '</div>';
app.innerHTML += storyHTML;

// -- Chart rendering --
const darkGrid = { color: 'rgba(148,163,184,0.1)' };
const darkTick = { color: '#64748b' };
const defaultOpts = {
  animation: false,
  plugins: { legend: { labels: { color: '#94a3b8' } } },
};

// Cache fill chart (token usage over time with retraction markers)
function makeFillChart(id, data, label, phases) {
  const usagePts = data.prefills.map(p => ({x: p.t, y: p.token_usage * 100}));
  const retPts = data.retractions.map(r => ({x: r.t, y: 102}));

  const annotations = {};
  if (phases.flood_start) {
    annotations.flood = {
      type: 'line', xMin: phases.flood_start, xMax: phases.flood_start,
      borderColor: 'rgba(251,191,36,0.4)', borderWidth: 1, borderDash: [4,4],
      label: { display: true, content: 'Flood', color: '#fbbf24',
               backgroundColor: 'transparent', position: 'start', font: {size: 10} }
    };
  }
  if (phases.replay_start) {
    annotations.replay = {
      type: 'line', xMin: phases.replay_start, xMax: phases.replay_start,
      borderColor: 'rgba(96,165,250,0.4)', borderWidth: 1, borderDash: [4,4],
      label: { display: true, content: 'VIP Replay', color: '#60a5fa',
               backgroundColor: 'transparent', position: 'start', font: {size: 10} }
    };
  }

  new Chart(document.getElementById(id), {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'Token Usage', data: usagePts, showLine: true, borderWidth: 1.5,
          pointRadius: 0, borderColor: 'rgba(96,165,250,0.7)',
          backgroundColor: 'rgba(96,165,250,0.05)', fill: true },
        { label: 'Retraction', data: retPts, pointRadius: 4, pointStyle: 'triangle',
          backgroundColor: 'rgba(251,191,36,0.8)' },
      ]
    },
    options: {
      ...defaultOpts,
      scales: {
        x: { title: { display: true, text: 'Time (s)', color: '#64748b' },
             grid: darkGrid, ticks: darkTick },
        y: { title: { display: true, text: label, color: '#64748b' },
             min: 0, max: 110, grid: darkGrid, ticks: darkTick }
      },
      plugins: { ...defaultOpts.plugins, annotation: { annotations } },
    }
  });
}

makeFillChart('bl-fill', bl, 'Baseline Usage (%)', D.bl_phases);
if (pin) makeFillChart('pin-fill', pin, 'Pinned Usage (%)', D.pin_phases);

// Cached tokens scatter (full timeline)
function makeTokenChart(id, data) {
  new Chart(document.getElementById(id), {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'Cached', data: data.prefills.map(p => ({x: p.t, y: p.cached_token})),
          pointRadius: 3, backgroundColor: 'rgba(74,222,128,0.6)' },
        { label: 'New', data: data.prefills.map(p => ({x: p.t, y: p.new_token})),
          pointRadius: 2, backgroundColor: 'rgba(248,113,113,0.35)' },
      ]
    },
    options: {
      ...defaultOpts,
      scales: {
        x: { title: { display: true, text: 'Time (s)', color: '#64748b' },
             grid: darkGrid, ticks: darkTick },
        y: { title: { display: true, text: 'Tokens', color: '#64748b' },
             grid: darkGrid, ticks: darkTick }
      },
    }
  });
}
makeTokenChart('bl-tokens', bl);
if (pin) makeTokenChart('pin-tokens', pin);

// Per-turn TTFT bar chart
if (D.bl_turns.length > 0 && D.pin_turns.length > 0) {
  const maxTurns = Math.max(D.bl_turns.length, D.pin_turns.length);
  const labels = [];
  const blVals = [], pinVals = [];
  for (let i = 0; i < maxTurns; i++) {
    labels.push(i === 0 ? 'Turn 0 (sys prompt)' : `Turn ${i}`);
    blVals.push(D.bl_turns[i] ? D.bl_turns[i].ttft_ms : 0);
    pinVals.push(D.pin_turns[i] ? D.pin_turns[i].ttft_ms : 0);
  }

  new Chart(document.getElementById('ttft-compare'), {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        { label: 'Baseline TTFT (ms)', data: blVals,
          backgroundColor: 'rgba(248,113,113,0.6)', borderColor: 'rgba(248,113,113,0.8)',
          borderWidth: 1 },
        { label: 'Pinned TTFT (ms)', data: pinVals,
          backgroundColor: 'rgba(74,222,128,0.6)', borderColor: 'rgba(74,222,128,0.8)',
          borderWidth: 1 },
      ]
    },
    options: {
      ...defaultOpts,
      scales: {
        x: { grid: darkGrid, ticks: darkTick },
        y: { title: { display: true, text: 'TTFT (ms)', color: '#64748b' },
             grid: darkGrid, ticks: darkTick }
      },
    }
  });
}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Visualize PIN benchmark")
    parser.add_argument("rep_dir", help="Path to rep directory (e.g. /tmp/.../rep0)")
    parser.add_argument("--output", "-o", help="Output HTML file")
    args = parser.parse_args()

    rep_dir = Path(args.rep_dir)
    out_file = args.output or str(rep_dir / "report.html")

    bl_log = rep_dir / "baseline_server.log"
    pin_log = rep_dir / "pinned_server.log"
    if not bl_log.exists():
        print(f"ERROR: {bl_log} not found")
        return

    baseline_data = parse_server_log(str(bl_log))
    pinned_data = parse_server_log(str(pin_log)) if pin_log.exists() else None

    # Parse aiperf per-turn TTFT records
    bl_replay_dir = rep_dir / "baseline_vip_replay"
    pin_replay_dir = rep_dir / "pinned_vip_replay"
    # Fall back to old "probe" naming
    if not bl_replay_dir.exists():
        bl_replay_dir = rep_dir / "baseline_probe"
    if not pin_replay_dir.exists():
        pin_replay_dir = rep_dir / "pinned_probe"

    bl_turns = parse_aiperf_records(str(bl_replay_dir), "vip_replay") if bl_replay_dir.exists() else []
    if not bl_turns:
        bl_turns = parse_aiperf_records(str(bl_replay_dir), "probe") if bl_replay_dir.exists() else []
    pin_turns = parse_aiperf_records(str(pin_replay_dir), "vip_replay") if pin_replay_dir.exists() else []
    if not pin_turns:
        pin_turns = parse_aiperf_records(str(pin_replay_dir), "probe") if pin_replay_dir.exists() else []

    # Load rep summary JSON
    rep_json = None
    rep_num = rep_dir.name.replace("rep", "")
    rj = rep_dir.parent / f"rep{rep_num}.json"
    if rj.exists():
        with open(rj) as f:
            rep_json = json.load(f)

    html = generate_html(baseline_data, pinned_data, bl_turns, pin_turns, rep_json)
    with open(out_file, "w") as f:
        f.write(html)
    print(f"Report: {out_file}")


if __name__ == "__main__":
    main()

const DATA = __REAL_DATA__;
const $ = (s) => document.querySelector(s),
  NS = "http://www.w3.org/2000/svg",
  COLORS = ["var(--sampling)", "var(--tool)", "var(--s3)", "var(--s4)"];
let start = 0,
  span = DATA.summary.duration,
  selected = DATA.rows[0],
  drag = null;
const total = span;
const lanes = [
  ...new Map(
    DATA.rows.map((r) => [
      JSON.stringify([r.worker, r.conversation]),
      { worker: r.worker, conversation: r.conversation },
    ]),
  ).values(),
].sort(
  (a, b) =>
    (a.worker ?? Infinity) - (b.worker ?? Infinity) ||
    String(a.conversation).localeCompare(String(b.conversation), undefined, {
      numeric: true,
    }),
);

let metrics = Object.values(DATA.exporters)[0];
$("#subtitle").textContent =
  `${DATA.summary.model} · ${DATA.summary.requests} requests · ${DATA.summary.conversations} conversations × ${DATA.summary.turns} turns · concurrency ${DATA.summary.concurrency} · page ${DATA.summary.page_size} · ${total.toFixed(1)}s · ${DATA.summary.errors} errors`;
function elem(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
  parent.append(e);
  return e;
}
function text(parent, x, y, s, extra = {}) {
  const e = elem(
    "text",
    { x, y, fill: "var(--muted)", "font-size": 12, ...extra },
    parent,
  );
  e.textContent = s;
  return e;
}
function fmt(v) {
  if (v === null || v === undefined) return "Unavailable";
  return Math.abs(v) >= 1000
    ? (v / 1000).toFixed(1) + "k"
    : Math.abs(v) >= 10
      ? v.toFixed(1)
      : v.toFixed(2);
}
function details() {
  if (!selected) {
    $("#selected-title").textContent = "No request records";
    return;
  }
  $("#request").value = String(DATA.rows.indexOf(selected));
  $("#selected-title").textContent =
    `Conversation ${selected.conversation} · Turn ${selected.turn} · Worker ${selected.worker ?? "unknown"}`;
  $("#availability").textContent = [
    selected.error ? "Failed: " + selected.error : "",
    selected.unavailable.length
      ? "Phase timestamps unavailable: " + selected.unavailable.join(", ")
      : "",
  ]
    .filter(Boolean)
    .join(" · ");
  const fields = [
    ["Context", fmt(selected.context) + " tokens"],
    ["Output", fmt(selected.output) + " tokens"],
    [
      "Sampling",
      fmt(selected.latency == null ? null : selected.latency * 1000) + " ms",
    ],
    ["TTFT", fmt(selected.ttft == null ? null : selected.ttft * 1000) + " ms"],
    [
      "Client wait",
      fmt(selected.wait == null ? null : selected.wait * 1000) + " ms",
    ],
    [
      "GPU / CPU hits",
      `${selected.hits.device ?? "N/A"} / ${selected.hits.host ?? "N/A"} tokens`,
    ],
  ];
  $("#details").replaceChildren();
  for (const [label, value] of fields) {
    const field = document.createElement("div");
    const name = document.createElement("span");
    const content = document.createElement("strong");
    name.textContent = label;
    content.textContent = value;
    field.append(name, content);
    $("#details").append(field);
  }
}
function drawTimeline() {
  const svg = $("#timeline");
  if ($("#timeline-panel").hidden) return;
  let w = svg.parentElement.clientWidth,
    L = w < 500 ? 70 : 112,
    R = 12,
    h =
      102 +
      lanes.length * 33 +
      new Set(lanes.map((lane) => lane.worker)).size * 24;
  const x = (t) => L + ((t - start) / span) * (w - L - R);
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.style.width = "100%";
  svg.style.height = h + "px";
  let ticks = w < 500 ? 3 : 6;
  for (let i = 0; i <= ticks; i++) {
    let t = start + (span * i) / ticks,
      px = x(t);
    elem(
      "line",
      { x1: px, x2: px, y1: 25, y2: h - 30, stroke: "var(--line)" },
      svg,
    );
    text(svg, px, 15, t.toFixed(span < 3 ? 2 : 1) + "s", {
      "text-anchor": i === 0 ? "start" : i === ticks ? "end" : "middle",
    });
  }
  let groupIndex = -1;
  for (let lane = 0; lane < lanes.length; lane++) {
    let { conversation: c, worker } = lanes[lane];
    const startsGroup = lane === 0 || lanes[lane - 1].worker !== worker;
    if (startsGroup) groupIndex++;
    let y = 66 + lane * 33 + groupIndex * 24;
    if (startsGroup)
      text(
        svg,
        0,
        y - 5,
        worker === null ? "Worker unknown" : "Worker " + worker,
        { "font-size": 10 },
      );
    text(svg, 4, y + 14, "Convo " + c);
    for (const r of DATA.rows.filter(
      (v) => v.conversation === c && v.worker === worker,
    )) {
      for (const p of r.phases) {
        let a = Math.max(start, p.start),
          b = Math.min(start + span, p.end);
        if (b <= a) continue;
        let fill =
          p.type === "Sampling"
            ? "var(--sampling)"
            : p.type === "Tool call"
              ? "var(--tool)"
              : "var(--wait)";
        const g = elem(
          "g",
          {
            role: "button",
            "aria-label": `Conversation ${c} turn ${r.turn} ${p.type}, ${((p.end - p.start) * 1000).toFixed(2)} ms`,
            tabindex: 0,
          },
          svg,
        );
        const rect = elem(
          "rect",
          {
            x: x(a),
            y,
            width: Math.max(0.4, x(b) - x(a)),
            height: 22,
            fill,
            opacity: selected === r ? 1 : 0.66,
            rx: 2,
          },
          g,
        );
        g.style.cursor = "pointer";
        const choose = () => {
          selected = r;
          details();
          drawTimeline();
        };
        g.addEventListener("click", () => {
          if (!drag?.moved) choose();
        });
        g.addEventListener("keydown", (e) => {
          if (e.key === "Enter") choose();
        });
        if (p.type === "Sampling" && x(b) - x(a) > 24)
          text(g, (x(a) + x(b)) / 2, y + 15, "T" + r.turn, {
            "text-anchor": "middle",
            fill: "var(--bg)",
            "pointer-events": "none",
          });
        g.addEventListener("pointermove", (e) => {
          let tip = $("#tooltip"),
            bounds = svg.getBoundingClientRect();
          tip.hidden = false;
          tip.textContent = `Convo ${c} · Turn ${r.turn}\n${p.type}: ${((p.end - p.start) * 1000).toFixed(2)} ms\n${p.start.toFixed(3)}–${p.end.toFixed(3)}s`;
          tip.style.left =
            Math.max(0, Math.min(w - 270, e.clientX - bounds.left + 12)) + "px";
          tip.style.top = e.clientY - bounds.top + 24 + "px";
        });
        g.addEventListener("pointerleave", () => ($("#tooltip").hidden = true));
      }
    }
  }
  text(svg, (L + w - R) / 2, h - 7, "Elapsed time (seconds)", {
    "text-anchor": "middle",
  });
  svg.onpointerdown = (e) => {
    let px = e.clientX - svg.getBoundingClientRect().left;
    if (px < L) return;
    drag = { px, t: start + ((px - L) / (w - L - R)) * span, moved: false };
  };
  svg.onpointermove = (e) => {
    if (
      drag &&
      Math.abs(e.clientX - svg.getBoundingClientRect().left - drag.px) > 8
    )
      drag.moved = true;
  };
  svg.onpointerup = (e) => {
    if (drag?.moved) {
      let px = Math.max(
          L,
          Math.min(w - R, e.clientX - svg.getBoundingClientRect().left),
        ),
        end = start + ((px - L) / (w - L - R)) * span;
      start = Math.min(drag.t, end);
      span = Math.max(0.05, Math.abs(end - drag.t));
      $("#tooltip").hidden = true;
      let o = $("#window option[data-custom]");
      if (!o) {
        o = document.createElement("option");
        o.dataset.custom = "1";
        $("#window").append(o);
      }
      o.value = span;
      o.textContent = span.toFixed(2) + " seconds";
      $("#window").value = String(span);
      render();
    }
    setTimeout(() => (drag = null), 0);
  };
}
function drawMetrics() {
  if ($("#metrics-panel").hidden) return;
  const grid = $("#metric-grid");
  grid.replaceChildren();
  let group = null;
  for (const m of metrics) {
    if ((m.group || "Topline Metrics") !== group) {
      group = m.group || "Topline Metrics";
      let heading = document.createElement("h2");
      heading.textContent = group;
      heading.style.cssText = "grid-column:1/-1;font-size:16px;margin:6px 0 0";
      grid.append(heading);
    }
    drawMetricPanel(m, grid);
  }
}

function drawMetricPanel(m, grid) {
  const panel = document.createElement("div");
  panel.className = "panel";
  const heading = document.createElement("h3");
  heading.textContent = m.title;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", m.title);
  const legend = document.createElement("div");
  legend.className = "metric-legend";
  const source = document.createElement("p");
  source.className = "source";
  source.textContent = m.source;
  panel.append(heading, svg, legend, source);
  grid.append(panel);
  let w = panel.clientWidth,
    h = 160,
    L = 64,
    R = 12,
    T = 20,
    B = 35;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.style.width = "100%";
  svg.style.height = h + "px";
  let vs = m.series.flatMap((s) =>
    s.points.map((p) => p[1]).filter((v) => v !== null),
  );
  let max =
    m.unit === "%"
      ? 100
      : vs.length
        ? vs.reduce((maximum, value) => Math.max(maximum, value), 0) * 1.08
        : 1;
  if (max === 0) max = 1;
  const x = (t) => L + ((t - start) / span) * (w - L - R),
    y = (v) => T + (1 - v / max) * (h - T - B);
  elem(
    "rect",
    {
      x: L,
      y: T,
      width: w - L - R,
      height: h - T - B,
      fill: "none",
      stroke: "var(--line)",
    },
    svg,
  );
  for (let f of [0, 0.5, 1]) {
    let yy = y(max * f);
    text(svg, L - 6, yy + 4, fmt(max * f), { "text-anchor": "end" });
    if (f === 0.5)
      elem(
        "line",
        { x1: L, x2: w - R, y1: yy, y2: yy, stroke: "var(--line)" },
        svg,
      );
  }
  for (let f of [0, 0.5, 1])
    text(svg, x(start + span * f), h - 18, (start + span * f).toFixed(1), {
      "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle",
    });
  text(svg, L, T - 6, m.unit);
  text(svg, (L + w - R) / 2, h - 2, "Elapsed time (s)", {
    "text-anchor": "middle",
    "font-size": 11,
  });
  if (!vs.length)
    text(svg, (L + w - R) / 2, 76, "Unavailable", {
      "text-anchor": "middle",
    });
  const paths = [];
  m.series.forEach((s, j) => {
    let path = "",
      pen = false;
    let points = s.points.filter((p) => p[0] >= start && p[0] <= start + span);
    for (let [t, v] of points) {
      if (v === null) {
        pen = false;
        continue;
      }
      path += (pen ? "L" : "M") + x(t) + "," + y(v);
      pen = true;
    }
    let g = elem("g", {}, svg);
    elem(
      "path",
      {
        d: path,
        fill: "none",
        stroke: COLORS[s.color ?? j % 4],
        "stroke-width": 1.6,
        "stroke-dasharray": s.dashed ? "5 4" : "",
      },
      g,
    );
    for (let [t, v] of points)
      if (v !== null)
        elem(
          "circle",
          { cx: x(t), cy: y(v), r: 2, fill: COLORS[s.color ?? j % 4] },
          g,
        );
    paths.push(g);
    let b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-pressed", "true");
    let sw = document.createElement("i");
    sw.style.background = COLORS[s.color ?? j % 4];
    if (s.dashed) {
      sw.style.background = "none";
      sw.style.borderTop = "2px dashed " + COLORS[s.color ?? j % 4];
    }
    b.append(sw, document.createTextNode(s.name));
    b.onclick = () => {
      let on = b.getAttribute("aria-pressed") !== "true";
      b.setAttribute("aria-pressed", on);
      g.style.display = on ? "" : "none";
    };
    legend.append(b);
  });
  let guide = elem(
    "line",
    { y1: T, y2: h - B, stroke: "var(--muted)", visibility: "hidden" },
    svg,
  );
  svg.addEventListener("pointermove", (e) => {
    if (!vs.length) return;
    let px = Math.max(
        L,
        Math.min(w - R, e.clientX - svg.getBoundingClientRect().left),
      ),
      t = start + ((px - L) / (w - L - R)) * span;
    guide.setAttribute("x1", px);
    guide.setAttribute("x2", px);
    guide.setAttribute("visibility", "visible");
    const vals = m.series
      .map((s, j) => {
        if (paths[j].style.display === "none") return null;
        let pt = s.points.reduce(
          (best, p) =>
            !best || Math.abs(p[0] - t) < Math.abs(best[0] - t) ? p : best,
          null,
        );
        return (
          s.name +
          ": " +
          (pt ? fmt(pt[1]) : "Unavailable") +
          (pt ? " @ " + pt[0].toFixed(1) + "s" : "")
        );
      })
      .filter(Boolean);
    source.textContent = vals.join(" · ");
  });
  svg.addEventListener("pointerleave", () => {
    guide.setAttribute("visibility", "hidden");
    source.textContent = m.source;
  });
}

function render() {
  $("#range").textContent = `${start.toFixed(2)}–${(start + span).toFixed(2)}s`;
  $("#position").max = Math.max(0, total - span);
  $("#position").value = start;
  drawTimeline();
  drawMetrics();
}
for (let b of document.querySelectorAll("[role=tab]"))
  b.onclick = () => {
    for (let tab of document.querySelectorAll("[role=tab]")) {
      const on = b === tab;
      tab.setAttribute("aria-selected", on);
      $("#" + tab.getAttribute("aria-controls")).hidden = !on;
    }
    render();
  };
$("#window").onchange = () => {
  span = $("#window").value === "all" ? total : Number($("#window").value);
  span = Math.min(span, total);
  start = Math.min(start, total - span);
  render();
};
$("#position").oninput = () => {
  start = Number($("#position").value);
  render();
};
$("#reset").onclick = () => {
  start = 0;
  span = total;
  $("#window").value = "all";
  render();
};
const endpoint = $("#exporter");
for (const url of Object.keys(DATA.exporters)) {
  const option = document.createElement("option");
  option.value = url;
  option.textContent = url;
  endpoint.append(option);
}
endpoint.onchange = () => {
  metrics = DATA.exporters[endpoint.value];
  render();
};
$(".badge").textContent =
  DATA.summary.status.toUpperCase() +
  " · " +
  DATA.summary.errors +
  " recorded errors";
if (DATA.summary.error) $(".badge").title = DATA.summary.error;
const requestPicker = $("#request");
DATA.rows.forEach((row, index) => {
  const option = document.createElement("option");
  option.value = index;
  option.textContent = `Conversation ${row.conversation}, turn ${row.turn}${row.error ? " (failed)" : ""}`;
  requestPicker.append(option);
});
requestPicker.onchange = () => {
  selected = DATA.rows[Number(requestPicker.value)];
  details();
  drawTimeline();
};
details();
new ResizeObserver(render).observe(document.querySelector("main"));
render();

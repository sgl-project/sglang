// Launch explorer for SemiAnalysis InferenceX AgentX submissions on SGLang cookbook pages.
//
// Reads a generated `agentx/<org>/<model>.jsx` (`agentx` export, produced by the
// sa-infx-cookbook sync scripts). Pick hardware -> checkpoint -> deployment shape ->
// concurrency -> router and get every process of that deployment as raw commands (etcd /
// NATS / Mooncake, workers, frontend or router), with provenance back to the SA run.
//
// Data encoding (mirrors the Python encoder, which verifies every point decodes exactly):
//   cell.routers[r].blocks   base blocks of the cell's first point. A block is either full
//                            ({kind, cmd, args, env|envRef, ...}), `{ref:[cell, router, i], diff}`
//                            (relative to another cell's block) or `{from: i, diff}` (relative
//                            to block i of this cell's submitted router).
//   point.patch[r]           later points: null | {replace: blocks} | per-block
//                            {args: {flag: [values]|null}, env: {k: v|null}} | {block: full}.
export const AgentX = ({ data }) => {
  if (!data || !data.cells) {
    return <div style={{padding: 12, color: "#b91c1c"}}>AgentX: missing <code style={S.code}>data</code> prop</div>;
  }

  // ==== 1. Decoder ====
  const LEAD_FLAGS = [
    "--model-path", "--served-model-name", "--host", "--port", "--disaggregation-mode",
    "--disaggregation-bootstrap-port", "--dist-init-addr", "--nnodes", "--node-rank", "--request-plane",
    "--kv-events-config",
  ];
  const argRank = (name) => {
    const i = LEAD_FLAGS.indexOf(name);
    return i >= 0 ? [i, ""] : [LEAD_FLAGS.length, name];
  };
  const cmpArgs = (a, b) => {
    const [ra, na] = argRank(a[0]);
    const [rb, nb] = argRank(b[0]);
    if (ra !== rb) return ra - rb;
    return na < nb ? -1 : na > nb ? 1 : 0;
  };
  const clone = (x) => JSON.parse(JSON.stringify(x));
  const sortedObj = (o) => Object.fromEntries(Object.keys(o).sort().map((k) => [k, o[k]]));

  const applyDiff = (block, d) => {
    if (!d) return clone(block);
    if (d.block) return clone(d.block);
    const b = clone(block);
    const da = d.args || {};
    let args = b.args.filter((a) => !(a[0] in da && da[a[0]] === null));
    args = args.map((a) => (a[0] in da ? [a[0], ...da[a[0]]] : a));
    const names = new Set(args.map((a) => a[0]));
    for (const [k, v] of Object.entries(da)) if (v !== null && !names.has(k)) args.push([k, ...v]);
    if (b.kind === "worker") args.sort(cmpArgs);
    b.args = args;
    const de = d.env || {};
    const env = {};
    for (const [k, v] of Object.entries(b.env || {})) if (!(k in de && de[k] === null)) env[k] = v;
    for (const [k, v] of Object.entries(de)) if (v !== null) env[k] = v;
    b.env = sortedObj(env);
    return b;
  };
  const applyPatch = (base, p) => {
    if (!p) return clone(base);
    if (!Array.isArray(p)) return clone(p.replace);
    return base.map((b, i) => applyDiff(b, p[i]));
  };
  const fromRef = (ref, e) => {
    const r = clone(ref);
    for (const k of e.drop || []) delete r[k];
    for (const [k, v] of Object.entries(e)) if (!["from", "diff", "drop", "ref"].includes(k)) r[k] = v;
    return applyDiff(r, e.diff);
  };
  const memo = {};
  const materialize = (ci, router) => {
    const key = `${ci}|${router}`;
    if (memo[key]) return memo[key];
    const cell = data.cells[ci];
    const out = cell.routers[router].blocks.map((e) => {
      if (e.ref) return fromRef(materialize(e.ref[0], e.ref[1])[e.ref[2]], e);
      if (e.from !== undefined) return fromRef(materialize(ci, cell.submitted)[e.from], e);
      return clone(e);
    });
    for (const b of out) {
      if (b.envRef !== undefined) {
        b.env = sortedObj(data.envs[b.envRef]);
        delete b.envRef;
      }
    }
    memo[key] = out;
    return out;
  };
  const blocksFor = (ci, pi, router) => {
    const base = materialize(ci, router);
    if (pi === 0) return base;
    return applyPatch(base, ((data.cells[ci].points[pi] || {}).patch || {})[router]);
  };

  // ==== 2. Shell formatting ====
  const quote = (v) => {
    const s = String(v);
    if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(s)) return s;
    if (/^[A-Za-z0-9_@%+=:,./${}-]+$/.test(s)) return `"${s}"`; // keep $PLACEHOLDERS expandable
    return `'${s.replace(/'/g, `'\\''`)}'`;
  };
  const formatBlock = (b) => {
    const lines = [];
    for (const p of b.preamble || []) lines.push(p);
    for (const [k, v] of Object.entries(b.env || {})) lines.push(`${k}=${quote(v)} \\`);
    const args = b.args || [];
    if (!args.length) {
      lines.push(b.kind === "nginx" ? "nginx -c nginx.conf -g 'daemon off;'" : b.cmd);
      return lines.join("\n");
    }
    lines.push(`${b.cmd} \\`);
    args.forEach((a, i) => {
      const vals = a.slice(1).map(quote).join(" ");
      lines.push(`  ${a[0]}${vals ? " " + vals : ""}${i < args.length - 1 ? " \\" : ""}`);
    });
    return lines.join("\n");
  };

  // ==== 3. Labels and notes ====
  const ROLE_TITLE = { agg: "Worker", prefill: "Prefill worker", decode: "Decode worker" };
  const ROLE_HEAD = { agg: "$AGG_HEAD_IP", prefill: "$PREFILL_HEAD_IP", decode: "$DECODE_HEAD_IP" };
  const KIND_TITLE = {
    etcd: "etcd — Dynamo control plane",
    nats: "NATS — Dynamo KV-event plane",
    "mooncake-master": "Mooncake master",
    "mooncake-store": "Mooncake store",
    "dynamo-frontend": "Dynamo frontend (OpenAI API)",
    "sglang-router": "SGLang router (OpenAI API)",
    nginx: "nginx — optional frontend load balancer",
  };
  const where = (b, router, workerCount) => {
    if (b.kind === "worker") {
      const n = b.nodes_per_worker || 1;
      const parts = [];
      if (b.workers > 1) parts.push(`Start ${b.workers} ${b.role} workers${n > 1 ? `, each spanning ${n} nodes` : ", one per node"}.`);
      if (n > 1) parts.push(`Run on every node of a worker with NODE_RANK=0..${n - 1}; ${ROLE_HEAD[b.role]} is that worker's first node.`);
      if (b.gpu_pinning) parts.push(`Workers share nodes: pin GPUs per worker with CUDA_VISIBLE_DEVICES = ${b.gpu_pinning.map((w, i) => `worker ${i}: ${w.filter(Boolean).join(" / ") || "all"}`).join(" · ")}.`);
      if (router === "sglang" && workerCount === 1 && b.role === "agg" && n === 1) parts.push("Serves the OpenAI API on port 30000.");
      if (router === "dynamo") parts.push("Workers register with Dynamo through etcd; start them after etcd/NATS.");
      return parts.join(" ") || `Run on the ${b.role} node.`;
    }
    if (b.kind === "etcd") return "Start once, before everything else, on $ETCD_IP.";
    if (b.kind === "nats") return "Start once on $NATS_IP with the nats.conf shown below.";
    if (b.kind === "mooncake-master") return "Start once on $MOONCAKE_MASTER_IP before the workers.";
    if (b.kind === "mooncake-store") {
      const ports = b.ports || [];
      return `Start on each of the ${b.count_nodes || 1} decode node(s)${ports.length > 1 ? `, one instance per port (${ports.join(", ")}; change --port)` : ""}.`;
    }
    if (b.kind === "dynamo-frontend") return b.count > 1 ? `SA ran ${b.count} frontends on separate nodes behind the nginx block (port 8180 each, nginx on 8000). One frontend on port 8000 also works.` : "Start once; clients connect to port 8000.";
    if (b.kind === "nginx") return "Optional scale-out: hashes each session onto one frontend and listens on port 8000. Uses the nginx.conf below.";
    if (b.kind === "sglang-router") return "Start once after the workers; clients connect to port 8000.";
    return "";
  };
  const noteText = (S) => ({
    "rename-dp-attention": <>Flags are mirrored from the submission image: <code style={S.code}>--enable-dp-attention</code> with <code style={S.code}>--dp-size</code> / <code style={S.code}>--data-parallel-size</code>. Newer SGLang deprecates them in favor of <code style={S.code}>--attn-dp-size</code> (<a style={S.a} href="https://github.com/sgl-project/sglang/pull/41818">#41818</a>).</>,
    "hicache-size": <><code style={S.code}>--hicache-size</code> is host DRAM per rank in GB, sized to SemiAnalysis's nodes. Scale it to your host memory.</>,
    efa: <><code style={S.code}>MOONCAKE_PROTOCOL=efa</code> reflects SemiAnalysis's AWS-EFA B300 cluster. Use <code style={S.code}>rdma</code> on InfiniBand.</>,
    "ib-devices": <>Set <code style={S.code}>IB_DEVICES</code> to your node's RDMA NICs (comma-separated, e.g. <code style={S.code}>mlx5_0,mlx5_1</code>).</>,
  });
  const PLACEHOLDERS = {
    NODE_RANK: "rank of this node within its worker (0 on the worker's first node)",
    NODE_IP: "IP of the node the command runs on",
    ETCD_IP: "node running etcd",
    NATS_IP: "node running NATS",
    MOONCAKE_MASTER_IP: "node running mooncake_master",
    AGG_HEAD_IP: "first node of this worker",
    PREFILL_HEAD_IP: "first node of this prefill worker",
    DECODE_HEAD_IP: "first node of this decode worker",
    IB_DEVICES: "the node's RDMA NICs",
    EFA_DEVICES: "EFA device filter (AWS EFA fabrics only)",
  };
  const placeholderText = (name) =>
    PLACEHOLDERS[name] ||
    (/^(AGG|PREFILL|DECODE)_(\d+)_HEAD_IP$/.test(name)
      ? `first node of ${name.split("_")[0].toLowerCase()} worker ${name.split("_")[1]}`
      : /^FRONTEND_(\d+)_IP$/.test(name)
        ? `node of frontend ${name.split("_")[1]}`
        : "");

  // ==== 4. State ====
  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    const check = () => {
      const html = document.documentElement;
      setIsDark(html.classList.contains("dark") || html.getAttribute("data-theme") === "dark" || html.style.colorScheme === "dark");
    };
    check();
    const observer = new MutationObserver(check);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class", "data-theme", "style"] });
    return () => observer.disconnect();
  }, []);

  const hwList = data.hardware.map((h) => h.id);
  const [hw, setHw] = useState(hwList[0]);
  const ckptsFor = (h) => [...new Set(data.cells.filter((c) => c.hw === h).map((c) => c.ckpt))];
  const [ckpt, setCkpt] = useState(ckptsFor(hwList[0])[0]);
  const cellsFor = (h, k) => data.cells.map((c, i) => [c, i]).filter(([c]) => c.hw === h && c.ckpt === k);
  const [cellIdx, setCellIdx] = useState(cellsFor(hwList[0], ckptsFor(hwList[0])[0])[0][1]);
  const [pointIdx, setPointIdx] = useState(0);
  const [router, setRouter] = useState(data.cells[cellIdx].submitted);
  const [copied, setCopied] = useState(null);
  const [openFile, setOpenFile] = useState(null);

  const pickHw = (h) => {
    const k = ckptsFor(h)[0];
    const ci = cellsFor(h, k)[0][1];
    setHw(h); setCkpt(k); setCellIdx(ci); setPointIdx(0); setRouter(data.cells[ci].submitted);
  };
  const pickCkpt = (k) => {
    const ci = cellsFor(hw, k)[0][1];
    setCkpt(k); setCellIdx(ci); setPointIdx(0); setRouter(data.cells[ci].submitted);
  };
  const pickCell = (ci) => { setCellIdx(ci); setPointIdx(0); setRouter(data.cells[ci].submitted); };

  const cell = data.cells[cellIdx];
  const point = cell.points[Math.min(pointIdx, cell.points.length - 1)];
  const routerData = cell.routers[router];
  const blocks = blocksFor(cellIdx, Math.min(pointIdx, cell.points.length - 1), router);
  const status = router === cell.submitted ? point.status : "derived";
  const workerCount = blocks.filter((b) => b.kind === "worker").reduce((n, b) => n + (b.workers || 1), 0);
  const allText = blocks.map((b) => `# ${b.kind === "worker" ? ROLE_TITLE[b.role] : KIND_TITLE[b.kind] || b.kind}\n${formatBlock(b)}`).join("\n\n");
  const usedPlaceholders = [...new Set((allText + JSON.stringify(routerData.files.map((f) => data.files[f]))).match(/\$[A-Z][A-Z0-9_]*/g) || [])]
    .map((p) => p.slice(1))
    .filter((p) => placeholderText(p));
  const exportText = usedPlaceholders.map((p) => `export ${p}=<${placeholderText(p)}>`).join("\n");
  const copyAllText = (exportText ? `# Set per node before running the commands below\n${exportText}\n\n` : "") + allText;

  const copy = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 1500);
  };

  // ==== 5. Styles ====
  const accent = isDark ? "#E85D4D" : "#D45D44";
  const border = isDark ? "#374151" : "#e5e7eb";
  const S = {
    wrap: { maxWidth: "900px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "4px" },
    card: { padding: "6px 10px", border: `1px solid ${border}`, borderLeft: `3px solid ${accent}`, borderRadius: "4px", display: "flex", alignItems: "flex-start", gap: "10px", background: isDark ? "#1f2937" : "#fff" },
    title: { fontSize: "12px", fontWeight: 600, minWidth: "108px", flexShrink: 0, paddingTop: "4px", color: isDark ? "#e5e7eb" : "inherit" },
    chips: { display: "flex", flexWrap: "wrap", gap: "4px", flex: 1 },
    chip: (on) => ({
      padding: "3px 9px", border: `1px solid ${on ? "#D45D44" : isDark ? "#9ca3af" : "#d1d5db"}`, borderRadius: "3px", cursor: "pointer",
      fontSize: "12px", fontWeight: 500, background: on ? "#D45D44" : isDark ? "#374151" : "#fff", color: on ? "#fff" : isDark ? "#e5e7eb" : "inherit",
      textAlign: "left", lineHeight: 1.3,
    }),
    sub: { display: "block", fontSize: "10px", opacity: 0.75, fontWeight: 400 },
    badge: (kind) => {
      const c = { verified: ["#065f46", "#d1fae5"], "verified-pr": ["#065f46", "#d1fae5"], "in-progress": ["#92400e", "#fef3c7"], derived: ["#374151", "#e5e7eb"] }[kind];
      return { display: "inline-block", padding: "2px 8px", borderRadius: "10px", fontSize: "12px", fontWeight: 600, color: c[0], background: c[1] };
    },
    meta: { fontSize: "12px", color: isDark ? "#9ca3af" : "#4b5563", lineHeight: 1.6 },
    blockHead: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px", padding: "6px 10px", borderBottom: `1px solid ${border}`, background: isDark ? "#1f2937" : "#fafafa", fontSize: "12px" },
    blockWrap: { border: `1px solid ${border}`, borderRadius: "6px", overflow: "hidden", background: isDark ? "#111827" : "#f5f5f5" },
    pre: { padding: "10px 14px", margin: 0, fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace", fontSize: "12px", lineHeight: 1.5, whiteSpace: "pre", overflowX: "auto", color: isDark ? "#e5e7eb" : "#374151" },
    button: { fontSize: "11px", padding: "2px 8px", border: `1px solid ${border}`, borderRadius: "4px", cursor: "pointer", background: isDark ? "#374151" : "#fff", color: isDark ? "#e5e7eb" : "inherit" },
    where: { fontSize: "11px", color: isDark ? "#9ca3af" : "#6b7280", padding: "6px 10px 0" },
    list: { margin: "4px 0", paddingLeft: "18px", listStyleType: "disc", fontSize: "12px", lineHeight: 1.55, color: isDark ? "#d1d5db" : "#374151" },
    code: { fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace", fontSize: "11px", padding: "0 4px", borderRadius: "3px", background: isDark ? "#374151" : "#f3f4f6", color: isDark ? "#e5e7eb" : "#1f2937" },
    a: { color: accent, textDecoration: "underline" },
  };

  const statusText = {
    verified: "✓ Verified in SemiAnalysis InferenceX AgentX",
    "verified-pr": `✓ Verified in SemiAnalysis InferenceX AgentX (PR #${point.pr}, pending merge)`,
    "in-progress": `◐ In progress in SemiAnalysis InferenceX AgentX (PR #${point.pr})`,
    derived: "Derived — not run in SemiAnalysis InferenceX AgentX",
  }[status];
  const routerLabel = { sglang: "SGLang", dynamo: "Dynamo + SGLang" };
  const ckptLabel = (k) => {
    const c = data.checkpoints.find((x) => x.id === k);
    return c ? `${k} (${c.precision.toUpperCase()})` : k;
  };
  const concLabel = (p) => `c${p.concs.join(" / c")}`;
  const link = (href, text) => (href ? <a style={S.a} href={href} target="_blank" rel="noopener noreferrer">{text}</a> : null);

  // ==== 6. Render ====
  return (
    <div style={S.wrap} className="not-prose sg-agentx">
      <div style={S.card}>
        <span style={S.title} className="sg-agentx-row sg-agentx-row-hw">Hardware</span>
        <div style={S.chips}>
          {data.hardware.map((h) => (
            <button key={h.id} style={S.chip(hw === h.id)} onClick={() => pickHw(h.id)}>{h.label}</button>
          ))}
        </div>
      </div>
      <div style={S.card}>
        <span style={S.title} className="sg-agentx-row sg-agentx-row-ckpt">Checkpoint</span>
        <div style={S.chips}>
          {ckptsFor(hw).map((k) => (
            <button key={k} style={S.chip(ckpt === k)} onClick={() => pickCkpt(k)}>{ckptLabel(k)}</button>
          ))}
        </div>
      </div>
      <div style={S.card}>
        <span style={S.title} className="sg-agentx-row sg-agentx-row-cell">Deployment</span>
        <div style={{ ...S.chips, flexDirection: "column", flexWrap: "nowrap" }}>
          {cellsFor(hw, ckpt).map(([c, i]) => (
            <button key={c.id} value={c.id} style={S.chip(cellIdx === i)} onClick={() => pickCell(i)}>
              {c.label}
              <span style={S.sub}>{c.detail} · {c.gpus} GPUs · c{c.points.flatMap((p) => p.concs).join(", ")}</span>
            </button>
          ))}
          <span style={{ ...S.sub, paddingTop: "2px" }}>Ordered from low latency (top) to high throughput (bottom).</span>
        </div>
      </div>
      <div style={S.card}>
        <span style={S.title} className="sg-agentx-row sg-agentx-row-conc">Concurrency</span>
        <div style={S.chips}>
          {cell.points.map((p, i) => (
            <button key={i} value={p.concs.join(",")} style={S.chip(pointIdx === i)} onClick={() => setPointIdx(i)}>{concLabel(p)}</button>
          ))}
        </div>
      </div>
      <div style={S.card}>
        <span style={S.title} className="sg-agentx-row sg-agentx-row-router">Router</span>
        <div style={S.chips}>
          {Object.keys(cell.routers).sort().map((r) => (
            <button key={r} value={r} style={S.chip(router === r)} onClick={() => setRouter(r)}>
              {routerLabel[r]}{r === cell.submitted ? " ✓" : ""}
              <span style={S.sub}>{r === cell.submitted ? "as submitted to SA" : "derived"}</span>
            </button>
          ))}
        </div>
      </div>

      <div style={{ ...S.card, flexDirection: "column", gap: "4px" }}>
        <div><span style={S.badge(status)}>{statusText}</span></div>
        <div style={S.meta}>
          {status !== "derived" && <>
            {link(point.runUrl, "SA sweep run")}{point.runUrl && " · "}
            {point.prUrl && <>{link(point.prUrl, `InferenceX PR #${point.pr}`)} · </>}
            {point.jobUrl && <>{link(point.jobUrl, "job log")} · </>}
            {point.date && <>dashboard date {point.date} · </>}
          </>}
          config <code style={S.code}>{cell.configKey}</code>
          {point.recipeUrl && <> · {link(point.recipeUrl, "recipe")}{point.override ? <> (<code style={S.code}>{point.override}</code>)</> : null}</>}
          {" "}· image <code style={S.code}>{cell.image}</code>
        </div>
        {cell.imageStatus === "pruned" && (
          <div style={S.meta}>⚠️ This nightly tag has since been removed from Docker Hub. It was built from SGLang commit {cell.imageCommit ? link(`https://github.com/sgl-project/sglang/commit/${cell.imageCommit}`, cell.imageCommit) : "(unknown)"}; use an image or source build at or after that commit.</div>
        )}
        {cell.imageStatus === "digest-only" && (
          <div style={S.meta}>The tag has been removed from Docker Hub; pull the image by its digest (the <code style={S.code}>@sha256:...</code> part).</div>
        )}
        {status === "derived" && (
          <div style={S.meta}>
            Translated from the verified {routerLabel[cell.submitted]} launch by re-rendering the same recipe with srt-slurm's {router === "dynamo" ? "Dynamo" : "SGLang router"} frontend. Engine flags are unchanged; only the launcher, routing and router-specific flags differ.
          </div>
        )}
        <ul style={S.list}>
          {cell.disclosures.includes("synthetic-acceptance") && point.goldenAL && (
            <li>SemiAnalysis measured this point with a synthetic speculative-decoding acceptance length of {point.goldenAL} (InferenceX golden AL via <code style={S.code}>SGLANG_SIMULATE_ACC_LEN</code>), which is not part of these commands. Real speed depends on your workload's acceptance.</li>
          )}
          {cell.disclosures.includes("chat-template") && (
            <li>The SA run passed InferenceX's thinking-only chat template (<code style={S.code}>deepseek_v4_thinking.jinja</code>, which drops tool definitions). These commands keep SGLang's built-in DeepSeek-V4 encoding, which supports tool calling.</li>
          )}
          {routerData.notes.map((n) => noteText(S)[n] && <li key={n}>{noteText(S)[n]}</li>)}
          {routerData.install.map((cmd) => (
            <li key={cmd}>{cmd.startsWith("#") ? <>Dynamo: {cmd.replace(/^# /, "")}</> : <>Dynamo install used by SA: <code style={S.code}>{cmd}</code></>}</li>
          ))}
          {router === "dynamo" && !routerData.install.length && <li>Dynamo: the <code style={S.code}>ai-dynamo</code> release bundled in the image.</li>}
          {cell.setup.map((s, i) => (
            <li key={i}>{s.command ? <>Before launch, on every worker node: <code style={S.code}>{s.command}</code> — {s.reason}</> : <><code style={S.code}>{s.script}</code>: {s.reason}</>}</li>
          ))}
        </ul>
      </div>

      {usedPlaceholders.length > 0 && (
        <div style={S.blockWrap}>
          <div style={S.blockHead}>
            <strong>Placeholders — set on each node first</strong>
            <button style={S.button} onClick={() => copy(exportText, "exports")}>{copied === "exports" ? "Copied" : "⧉ Copy"}</button>
          </div>
          <pre style={S.pre} className="sg-agentx-exports">{exportText}</pre>
        </div>
      )}
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <button style={S.button} onClick={() => copy(copyAllText, "all")}>{copied === "all" ? "Copied" : "⧉ Copy all"}</button>
      </div>
      {blocks.map((b, i) => {
        const text = formatBlock(b);
        const title = b.kind === "worker" ? ROLE_TITLE[b.role] : KIND_TITLE[b.kind] || b.kind;
        return (
          <div key={`${cellIdx}-${pointIdx}-${router}-${i}`} style={S.blockWrap}>
            <div style={S.blockHead}>
              <strong>{title}{b.count > 1 ? ` ×${b.count}` : ""}</strong>
              <button style={S.button} onClick={() => copy(text, i)}>{copied === i ? "Copied" : "⧉ Copy"}</button>
            </div>
            <div style={S.where}>{where(b, router, workerCount)}</div>
            <pre style={S.pre} className="sg-agentx-block">{text}</pre>
          </div>
        );
      })}
      {routerData.files.map((fid) => {
        const f = data.files[fid];
        const open = openFile === fid;
        return (
          <div key={fid} style={S.blockWrap}>
            <div style={S.blockHead}>
              <strong>{f.name}</strong>
              <span style={{ display: "flex", gap: "6px" }}>
                <button style={S.button} onClick={() => setOpenFile(open ? null : fid)}>{open ? "Hide" : "Show"}</button>
                <button style={S.button} onClick={() => copy(f.content, fid)}>{copied === fid ? "Copied" : "⧉ Copy"}</button>
              </span>
            </div>
            {open && <pre style={S.pre}>{f.content}</pre>}
          </div>
        );
      })}
    </div>
  );
};

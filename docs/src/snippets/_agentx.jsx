// Launch explorer for SemiAnalysis InferenceX AgentX submissions on SGLang cookbook pages.
//
// Reads a generated `agentx/<org>/<model>.jsx` (`agentx` export, produced by the
// sa-infx-cookbook sync scripts). Pick hardware -> checkpoint -> deployment shape ->
// concurrency -> KV offload -> router and get every process of that deployment as raw
// commands (etcd / NATS / Mooncake, workers, frontend or router), one tab per process, with
// provenance back to the SA run. Only the measured KV tier + submitted router is verified;
// the other combinations are derived (see kvTransform).
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
    let args = b.args.filter((a) => da[a[0]] !== null);
    args = args.map((a) => (da[a[0]] !== undefined ? [a[0], ...da[a[0]]] : a));
    const names = new Set(args.map((a) => a[0]));
    for (const [k, v] of Object.entries(da)) if (v !== null && !names.has(k)) args.push([k, ...v]);
    if (b.kind === "worker") args.sort(cmpArgs);
    b.args = args;
    const de = d.env || {};
    const env = {};
    for (const [k, v] of Object.entries(b.env || {})) if (de[k] !== null) env[k] = v;
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
    const drop = e.drop || [];
    const r = Object.fromEntries(Object.entries(clone(ref)).filter(([k]) => !drop.includes(k)));
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
    const resolved = out.map((b) => (b.envRef === undefined ? b : {
      ...Object.fromEntries(Object.entries(b).filter(([k]) => k !== "envRef")),
      env: sortedObj(data.envs[b.envRef]),
    }));
    memo[key] = resolved;
    return resolved;
  };
  // KV-offload tiers: "none" | "hicache" | "mooncake" (external linker). A derived tier strips
  // the measured tier from the cache-owning workers and adds the target tier's settings
  // (cell.kv.hicache / data.kvMooncake), mirroring the generator's kv.py.
  const LINKER_FLAGS = ["--enable-unified-cache-external-linker", "--unified-cache-external-linker-backend", "--hicache-storage-backend-extra-config"];
  const STORE_ONLY_ENV = ["MOONCAKE_MASTER", "MOONCAKE_TE_META_DATA_SERVER", "MOONCAKE_GLOBAL_SEGMENT_SIZE", "MOONCAKE_STANDALONE_STORAGE"];
  const CACHE_ROLES = ["agg", "prefill"];
  const isKvFlag = (n) => n === "--enable-hierarchical-cache" || n.startsWith("--hicache-") || LINKER_FLAGS.includes(n);
  const kvTransform = (blocks, source, target, cellKv, router) => {
    if (source === target) return blocks;
    const mc = data.kvMooncake;
    const out = [];
    for (const b0 of blocks) {
      if (source === "mooncake" && (b0.kind === "mooncake-master" || b0.kind === "mooncake-store")) continue;
      const b = { ...b0, args: (b0.args || []).map((a) => [...a]), env: { ...(b0.env || {}) } };
      if (b.kind === "worker") {
        b.args = b.args.filter((a) => !isKvFlag(a[0]));
        if (source === "mooncake") b.env = Object.fromEntries(Object.entries(b.env).filter(([k]) => !STORE_ONLY_ENV.includes(k)));
        let add = [];
        let env = {};
        if (CACHE_ROLES.includes(b.role)) {
          if (target === "hicache") add = cellKv.hicache.args;
          else if (target === "mooncake") { add = mc.args; env = mc.env; }
        } else if (target === "mooncake") env = mc.decode_env;
        const names = new Set(b.args.map((a) => a[0]));
        for (const a of add) if (!names.has(a[0])) b.args.push([...a]);
        b.args.sort(cmpArgs);
        for (const [k, v] of Object.entries(env)) if (b.env[k] === undefined) b.env[k] = v;
        b.env = sortedObj(b.env);
      }
      out.push(b);
    }
    if (target === "mooncake") {
      // Right before the first worker: after etcd / NATS and the Dynamo frontend (launch order).
      out.splice(out.findIndex((b) => b.kind === "worker"), 0, clone(mc.master[router]));
    }
    return out;
  };
  const blocksFor = (ci, pi, router, kv) => {
    const cell = data.cells[ci];
    const base = materialize(ci, router);
    const measured = pi === 0 ? base : applyPatch(base, ((cell.points[pi] || {}).patch || {})[router]);
    return kvTransform(measured, cell.points[pi].kv, kv || cell.points[pi].kv, cell.kv, router);
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
    if (b.kind === "nats") return "Start once on $NATS_IP, with nats.conf (below) in the working directory.";
    if (b.kind === "mooncake-master") return "Start once on $MOONCAKE_MASTER_IP before the workers.";
    if (b.kind === "mooncake-store") {
      const ports = b.ports || [];
      return `Start on each of the ${b.count_nodes || 1} decode node(s)${ports.length > 1 ? `, one instance per port (${ports.join(", ")}; change --port)` : ""}.`;
    }
    if (b.kind === "dynamo-frontend") return (b.count > 1 ? `SA ran ${b.count} frontends on separate nodes behind nginx (previous tab; port 8180 each, nginx on 8000). One frontend on port 8000 also works.` : "Start once; clients connect to port 8000.") + " It finds the workers through etcd, so it can start before them.";
    if (b.kind === "nginx") return "Optional scale-out in front of the frontends (next tab): hashes each session onto one frontend and listens on port 8000, with nginx.conf (below) in the working directory.";
    if (b.kind === "sglang-router") return "Start once after the workers; clients connect to port 8000.";
    return "";
  };
  const noteText = (S) => ({
    "rename-dp-attention": <>Flags are mirrored from the submission image: <code style={S.code}>--enable-dp-attention</code> with <code style={S.code}>--dp-size</code> / <code style={S.code}>--data-parallel-size</code>. Newer SGLang deprecates them in favor of <code style={S.code}>--attn-dp-size</code> (<a style={S.a} href="https://github.com/sgl-project/sglang/pull/41818">#41818</a>).</>,
    "hicache-size": <><code style={S.code}>--hicache-size</code> is host DRAM per rank in GB, sized to SemiAnalysis's nodes. Scale it to your host memory.</>,
    efa: <><code style={S.code}>MOONCAKE_PROTOCOL=efa</code> reflects SemiAnalysis's AWS-EFA B300 cluster. Use <code style={S.code}>rdma</code> on InfiniBand.</>,
    "ib-devices": <>Set <code style={S.code}>IB_DEVICES</code> to your node's RDMA NICs (comma-separated, e.g. <code style={S.code}>mlx5_0,mlx5_1</code>).</>,
    "mooncake-segment": <><code style={S.code}>MOONCAKE_GLOBAL_SEGMENT_SIZE</code> is the host DRAM each worker process contributes to the Mooncake store. Size it to your host memory.</>,
  });
  const notesFor = (blocks) => {
    const names = new Set(blocks.flatMap((b) => (b.args || []).map((a) => a[0])));
    const env = Object.assign({}, ...blocks.map((b) => b.env || {}));
    const text = JSON.stringify(blocks);
    return [
      names.has("--enable-dp-attention") && "rename-dp-attention",
      names.has("--hicache-size") && "hicache-size",
      env.MOONCAKE_PROTOCOL === "efa" && "efa",
      env.MOONCAKE_GLOBAL_SEGMENT_SIZE !== undefined && "mooncake-segment",
      text.includes("$IB_DEVICES") && "ib-devices",
    ].filter(Boolean);
  };
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
  const [kvChoice, setKvChoice] = useState(null); // null = the tier the point was measured with
  const [tab, setTab] = useState(null);
  const [copied, setCopied] = useState(null);

  const resetCell = (ci) => { setCellIdx(ci); setPointIdx(0); setRouter(data.cells[ci].submitted); setKvChoice(null); };
  const pickHw = (h) => {
    const k = ckptsFor(h)[0];
    setHw(h); setCkpt(k); resetCell(cellsFor(h, k)[0][1]);
  };
  const pickCkpt = (k) => { setCkpt(k); resetCell(cellsFor(hw, k)[0][1]); };
  const pickCell = (ci) => resetCell(ci);

  const cell = data.cells[cellIdx];
  const pi = Math.min(pointIdx, cell.points.length - 1);
  const point = cell.points[pi];
  const kv = kvChoice && cell.kv.available.includes(kvChoice) ? kvChoice : point.kv;
  const routerData = cell.routers[router];
  const blocks = blocksFor(cellIdx, pi, router, kv);
  const status = router === cell.submitted && kv === point.kv ? point.status : "derived";
  const workerCount = blocks.filter((b) => b.kind === "worker").reduce((n, b) => n + (b.workers || 1), 0);
  const blockText = blocks.map(formatBlock);
  const blockTitle = (b) => (b.kind === "worker" ? ROLE_TITLE[b.role] : KIND_TITLE[b.kind] || b.kind);
  const allText = blocks.map((b, i) => `# ${blockTitle(b)}\n${blockText[i]}`).join("\n\n");
  const usedPlaceholders = [...new Set((allText + JSON.stringify(routerData.files.map((f) => data.files[f]))).match(/\$[A-Z][A-Z0-9_]*/g) || [])]
    .map((p) => p.slice(1))
    .filter((p) => placeholderText(p));
  const exportText = usedPlaceholders.map((p) => `export ${p}=<${placeholderText(p)}>`).join("\n");
  const copyAllText = (exportText ? `# Set per node before running the commands below\n${exportText}\n\n` : "") + allText;

  const shortTitle = (b) => {
    const base = b.kind === "worker" ? { agg: "Worker", prefill: "Prefill", decode: "Decode" }[b.role]
      : { etcd: "etcd", nats: "NATS", "mooncake-master": "Mooncake master", "mooncake-store": "Mooncake store", "dynamo-frontend": "Frontend", "sglang-router": "Router", nginx: "nginx (optional)" }[b.kind] || b.kind;
    const n = b.kind === "worker" ? b.workers : b.count;
    return n > 1 ? `${base} ×${n}` : base;
  };
  // Config files sit in the tab of the process that reads them (nats.conf under NATS, ...).
  const fileList = routerData.files.map((fid) => data.files[fid]);
  const looseFiles = fileList.filter((f) => !blockText.some((t) => t.includes(f.name)));
  const tabs = [
    ...(usedPlaceholders.length ? [{ id: "exports", label: "Env Vars" }] : []),
    ...blocks.map((b, i) => ({ id: `block:${b.kind}:${b.role || ""}`, label: shortTitle(b), block: b, text: blockText[i], files: fileList.filter((f) => blockText[i].includes(f.name)) })),
    ...looseFiles.map((f) => ({ id: `file:${f.name}`, label: f.name, file: f })),
  ];
  const single = blocks.length === 1 && !looseFiles.length; // one process: no tab bar
  const activeTab = tabs.some((t) => t.id === tab) ? tab : tabs[0].id;
  const tabText = (t) => (t.block ? t.text : t.file ? t.file.content : exportText);
  const KV_LABEL = { none: "No KV offload", hicache: "HiCache (host DRAM)", mooncake: "Mooncake store (external linker)" };

  const copy = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 1500);
  };

  // ==== 5. Styles ====
  const accent = isDark ? "#E85D4D" : "#D45D44";
  const border = isDark ? "#374151" : "#e5e7eb";
  const muted = isDark ? "#9ca3af" : "#6b7280";
  const mono = "'Menlo', 'Monaco', 'Courier New', monospace";
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
    // Command window (modeled on the vLLM recipes' launch-step panel).
    window: { border: `1px solid ${border}`, borderRadius: "12px", overflow: "hidden", background: isDark ? "#111827" : "#f6f7f9", marginTop: "4px" },
    winHead: { flex: 1, paddingTop: "4px", fontFamily: mono, fontSize: "11px", lineHeight: 1.5, color: muted },
    winBar: { display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "10px", padding: "10px 14px 0" },
    seg: { display: "flex", flexWrap: "wrap", gap: "2px", margin: "8px 14px 0", padding: "2px", borderRadius: "7px", background: isDark ? "rgba(255,255,255,0.06)" : "rgba(15,23,42,0.05)" },
    segTab: (on) => ({
      padding: "4px 10px", fontSize: "12px", fontWeight: on ? 600 : 500, border: "none", borderRadius: "5px", cursor: "pointer", whiteSpace: "nowrap",
      background: on ? (isDark ? "#374151" : "#fff") : "transparent", boxShadow: on ? "0 1px 2px rgba(15,23,42,0.12)" : "none",
      color: on ? (isDark ? "#f9fafb" : "#111827") : muted,
    }),
    segNum: { fontFamily: mono, opacity: 0.45, marginRight: "4px" },
    winActions: { display: "flex", gap: "6px", flexShrink: 0 },
    winButton: { fontSize: "12px", fontWeight: 500, padding: "4px 10px", border: "none", borderRadius: "6px", cursor: "pointer", background: isDark ? "rgba(255,255,255,0.08)" : "rgba(15,23,42,0.07)", color: isDark ? "#e5e7eb" : "#374151" },
    comment: { padding: "10px 14px 0", fontFamily: mono, fontSize: "11px", lineHeight: 1.5, color: muted },
    secHead: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px", margin: "4px 14px 0", padding: "8px 0 0", borderTop: `1px dashed ${border}`, fontFamily: mono, fontSize: "11px", color: muted },
    secHeadFirst: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px", margin: "0 14px", padding: "10px 0 0", fontFamily: mono, fontSize: "11px", color: muted },
    miniButton: { fontSize: "11px", padding: "1px 8px", border: "none", borderRadius: "5px", cursor: "pointer", background: isDark ? "rgba(255,255,255,0.08)" : "rgba(15,23,42,0.07)", color: isDark ? "#e5e7eb" : "#374151" },
    pre: { padding: "8px 14px 12px", margin: 0, background: "transparent", fontFamily: mono, fontSize: "12px", lineHeight: 1.5, whiteSpace: "pre", overflowX: "auto", color: isDark ? "#e5e7eb" : "#1f2937" },
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
        <span style={S.title} className="sg-agentx-row sg-agentx-row-kv">KV offload</span>
        <div style={S.chips}>
          {["none", "hicache", "mooncake"].map((t) => {
            const ok = cell.kv.available.includes(t);
            return (
              <button key={t} value={t} disabled={!ok} title={ok ? "" : cell.kv.unavailable[t]}
                style={{ ...S.chip(kv === t), ...(ok ? {} : { opacity: 0.4, cursor: "not-allowed" }) }}
                onClick={() => ok && setKvChoice(t)}>
                {KV_LABEL[t]}{t === point.kv ? " ✓" : ""}
                <span style={S.sub}>{t === point.kv ? "as measured" : ok ? "derived" : "not available for this image"}</span>
              </button>
            );
          })}
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
          <ul style={S.list}>
            <li>Derived from the verified launch ({routerLabel[cell.submitted]}, {KV_LABEL[point.kv].toLowerCase()}); SemiAnalysis did not run this combination.</li>
            {router !== cell.submitted && (
              <li>Router: re-rendered with srt-slurm's {router === "dynamo" ? "Dynamo" : "SGLang router"} frontend. Engine flags are unchanged; only the launcher, routing and router-specific flags differ.</li>
            )}
            {kv !== point.kv && kv === "none" && <li>KV offload: removed; only the GPU KV cache is used.</li>}
            {kv !== point.kv && kv === "hicache" && (
              <li>KV offload: HiCache settings from the verified <code style={S.code}>{cell.kv.hicache.donor}</code> launch ({cell.kv.hicache.donorHardware.toUpperCase()}), applied to the workers that hold the prefix cache ({blocks.some((b) => b.role === "prefill") ? "prefill" : "aggregated"}).</li>
            )}
            {kv !== point.kv && kv === "mooncake" && (
              <li>KV offload: Mooncake external-linker flags, store-client environment and <code style={S.code}>mooncake_master</code> from the verified <code style={S.code}>{data.kvMooncake.donor}</code> launch ({data.kvMooncake.donorHardware.toUpperCase()}). Its optional standalone Mooncake stores are not included.</li>
            )}
          </ul>
        )}
        <ul style={S.list}>
          {cell.disclosures.includes("synthetic-acceptance") && point.goldenAL && (
            <li>SemiAnalysis measured this point with a synthetic speculative-decoding acceptance length of {point.goldenAL} (InferenceX golden AL via <code style={S.code}>SGLANG_SIMULATE_ACC_LEN</code>), which is not part of these commands. Real speed depends on your workload's acceptance.</li>
          )}
          {cell.disclosures.includes("chat-template") && (
            <li>The SA run passed InferenceX's thinking-only chat template (<code style={S.code}>deepseek_v4_thinking.jinja</code>, which drops tool definitions). These commands keep SGLang's built-in DeepSeek-V4 encoding, which supports tool calling.</li>
          )}
          {notesFor(blocks).map((n) => noteText(S)[n] && <li key={n}>{noteText(S)[n]}</li>)}
          {routerData.install.map((cmd) => (
            <li key={cmd}>{cmd.startsWith("#") ? <>Dynamo: {cmd.replace(/^# /, "")}</> : <>Dynamo install used by SA: <code style={S.code}>{cmd}</code></>}</li>
          ))}
          {router === "dynamo" && !routerData.install.length && <li>Dynamo: the <code style={S.code}>ai-dynamo</code> release bundled in the image.</li>}
          {cell.setup.map((s, i) => (
            <li key={i}>{s.command ? <>Before launch, on every worker node: <code style={S.code}>{s.command}</code> — {s.reason}</> : <><code style={S.code}>{s.script}</code>: {s.reason}</>}</li>
          ))}
        </ul>
      </div>

      <div style={S.window} className="sg-agentx-window">
        <div style={S.winBar}>
          <span style={S.winHead}>
            {data.hardware.find((h) => h.id === hw).label} · {cell.label} · {concLabel(point)} · {KV_LABEL[kv]} · {routerLabel[router]}
          </span>
          <div style={S.winActions}>
            <button style={S.winButton} onClick={() => copy(single ? blockText[0] : tabText(tabs.find((t) => t.id === activeTab)), "tab")}>{copied === "tab" ? "Copied" : "⧉ Copy"}</button>
            <button style={S.winButton} onClick={() => copy(copyAllText, "all")}>{copied === "all" ? "Copied" : "⧉ Copy all"}</button>
          </div>
        </div>
        {!single && (
          <div role="tablist" aria-label="Processes" style={S.seg} className="sg-agentx-tabs">
            {tabs.map((t, i) => (
              <button key={t.id} value={t.id} role="tab" aria-selected={activeTab === t.id} style={S.segTab(activeTab === t.id)} onClick={() => setTab(t.id)}>
                <span style={S.segNum}>{i + 1}·</span>{t.label}
              </button>
            ))}
          </div>
        )}
        {(single ? tabs.filter((t) => t.block) : tabs).map((t) => {
          const visible = single || activeTab === t.id;
          const section = (label, text, className, id, first) => (
            <>
              <div style={first ? S.secHeadFirst : S.secHead}>
                <span>{label}</span>
                <button style={S.miniButton} onClick={() => copy(text, id)}>{copied === id ? "Copied" : "⧉ Copy"}</button>
              </div>
              <pre style={S.pre} className={className}>{text}</pre>
            </>
          );
          let body;
          if (t.block) {
            const envFirst = single && exportText;
            const split = envFirst || t.files.length > 0;
            body = (
              <>
                <div style={S.comment}># {blockTitle(t.block)}{t.block.count > 1 ? ` ×${t.block.count}` : ""}<br /># {where(t.block, router, workerCount)}</div>
                {envFirst && section("Env Vars — set first", exportText, "sg-agentx-exports", "exports", true)}
                {split ? section("Command", t.text, "sg-agentx-block", `${t.id}:cmd`, !envFirst) : <pre style={S.pre} className="sg-agentx-block">{t.text}</pre>}
                {t.files.map((f) => <div key={f.name}>{section(f.name, f.content, "sg-agentx-file", `${t.id}:${f.name}`, false)}</div>)}
              </>
            );
          } else if (t.file) {
            body = (
              <>
                <div style={S.comment}># {t.file.name}</div>
                <pre style={S.pre} className="sg-agentx-file">{t.file.content}</pre>
              </>
            );
          } else {
            body = (
              <>
                <div style={S.comment}># Env Vars — set these on every node before running the commands in the other tabs.</div>
                <pre style={S.pre} className="sg-agentx-exports">{exportText}</pre>
              </>
            );
          }
          return (
            <div key={`${cellIdx}-${pi}-${router}-${kv}-${t.id}`} style={{ display: visible ? "block" : "none" }} className="sg-agentx-panel">
              {body}
            </div>
          );
        })}
      </div>
    </div>
  );
};

// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Recipes transcribed from the K3.1 1P1D and AgentX launchers
// (B300 1×8 / 2×8, GB300 2×4 trays). Kimi K3.1 Pro is a hybrid MoE VLM: 93 layers = 69 KDA (linear)
// + 24 MLA with DSA sparse attention on every MLA layer, 896 routed experts (top-16) + 2 shared.
// K3.1 forces --kv-cache-dtype fp8_e4m3, --page-size 64 and the `dsa` attention backend, so no
// cell pins them.

export const config = {
  modelName: "Kimi-K3.1-Pro",

  // B300 (1×8) and GB300 (2×4 MNNVL) run the same recipes and have K3.1 serving
  // rounds. MI350X and MI355X have the AgentX recipes: High-Throughput (DCP8) on every PD mode (1P1D
  // on 2×8) and Low-Latency (TP8) on Unified.
  // Vera Rubin, B200, GB200 and Hopper carry placeholder cells until a recipe is collected.
  supportedHardware: ["vera-rubin", "b300", "gb300", "b200", "gb200", "h200", "h100", "mi350x", "mi355x", "m890", "810e", "tpu7x"],

  // GPUs the shared engine catalog does not carry. Vera Rubin follows the Mistral-Large-4 entry;
  // PPU is T-HEAD's SAIL device (docs/docs/hardware-platforms/sail.mdx); TPU7x runs sglang-torchtpu.
  hardware: [
    { id: "vera-rubin", label: "Vera Rubin", vram: "288GB", vendor: "rubin" },
    { id: "m890", label: "M890", vram: "144GB", vendor: "ppu" },
    { id: "810e", label: "810E", vram: "96GB", vendor: "ppu" },
    { id: "tpu7x", label: "TPU7x", vram: "192GB", vendor: "tpu" },
  ],

  isPpu(s) {
    return ["m890", "810e"].includes(s.hw);
  },

  // The vendor recipes run inside a prepared SAIL container. Device passthrough
  // is platform-specific; do not generate the shared NVIDIA Docker wrapper.
  runModes: (s) => config.isPpu(s) || s.hw === "tpu7x" ? ["python"] : ["python", "docker"],

  // ---- Cell introspection (config-internal; the engines ignore these keys) ----
  // Same pattern as the Kimi-K3 config: overlay callbacks only see the selection, so recipe
  // facts are looked up from the cell's own flags.
  cellFor(s) {
    return (config.cells || []).find(
      (c) => c.match.hw === s.hw &&
             c.match.pdMode === s.pdMode &&
             c.match.strategy === s.strategy);
  },
  flagOf(cell, name) {
    return ((cell || {}).flags || []).find((f) => f.split(/[\s=]/)[0] === name);
  },
  sizeOf(cell, name) {
    const f = config.flagOf(cell, name);
    return f ? Number(f.split(/[\s=]/)[1]) || 1 : 1;
  },
  hasDcp(s) {
    return !!config.flagOf(config.cellFor(s), "--dcp-size");
  },
  isAmd(s) {
    return ["mi350x", "mi355x"].includes(s.hw);
  },
  isPipelined(s) {
    return config.sizeOf(config.cellFor(s), "--pp-size") > 1;
  },
  // Throughput shapes verify K = 5 (block 6): at large batch the verify rows (requests ×
  // block) dominate a step. The High-Throughput decode and every MI35x cell (whose PD prefill must
  // match its decode) use block 6; everything else runs K = 7 (block 8). The MI35x unified
  // High-Throughput cell, at its 48-agent shape, verifies K = 4 (block 5): from 48 requests the
  // shorter verify outweighs its lower acceptance.
  dflashBlock(s) {
    if (config.isAmd(s) && s.pdMode === "unified" && s.strategy === "high-throughput") return 5;
    return (s.pdMode === "decode" && s.strategy === "high-throughput") || config.isAmd(s) ? 6 : 8;
  },

  matchDims: [
    {
      id: "pdMode",
      title: "PD Mode",
      options: [
        { id: "unified", label: "Unified" },
        { id: "prefill", label: "Prefill" },
        { id: "decode",  label: "Decode"  },
      ],
    },
    {
      // Unified and decode nodes are sized by the latency/throughput operating point; prefill
      // nodes by how they spend their GPUs (cache capacity vs. per-request latency).
      id: "strategy",
      title: "Strategy",
      options: [
        // Each label carries its parallel layout. B300 and GB300: Low-Latency = TP8,
        // High-Concurrency = TP8 · DCP8 · EP2, High-Throughput prefill = PP4 × PCP2,
        // decode Balanced = TP8 · DCP8 · EP2, decode High-Throughput = TP8 · DCP8.
        { id: "low-latency",        label: "Low-Latency (TP8)",            showWhen: (s) => (["b300", "gb300"].includes(s.hw) && s.pdMode !== "decode") || (config.isAmd(s) && s.pdMode === "unified") },
        { id: "high-concurrency",   label: "High-Concurrency (DCP8)",      showWhen: (s) => ["b300", "gb300"].includes(s.hw) && s.pdMode === "unified" },
        { id: "high-throughput-pp", label: "High-Throughput (PP4 × PCP2)", showWhen: (s) => ["b300", "gb300"].includes(s.hw) && s.pdMode === "prefill" },
        { id: "balanced",           label: "Balanced (DCP8 · EP2)",        showWhen: (s) => ["b300", "gb300"].includes(s.hw) && s.pdMode === "decode" },
        { id: "high-throughput",    label: "High-Throughput (DCP8)",       showWhen: (s) => (["b300", "gb300"].includes(s.hw) && s.pdMode === "decode") || config.isAmd(s) },
        { id: "balanced-ep32-dp4", label: "Balanced (TP32 · EP32 · DP4)", showWhen: (s) => s.hw === "m890" && s.pdMode === "decode" },
        { id: "balanced-pp4", label: "Balanced (TP16 · PP4)", showWhen: (s) => s.hw === "810e" && s.pdMode === "prefill" },
        // Other platforms: one placeholder shape each, named by its layout.
        { id: "balanced-tp8",       label: "Balanced (TP8)",               showWhen: (s) => s.hw === "vera-rubin" },
        { id: "balanced-dcp16",     label: "Balanced (DCP16)",             showWhen: (s) => ["b200", "gb200"].includes(s.hw) },
        { id: "balanced-ep16",      label: "Balanced (TP16 · EP16)",       showWhen: (s) => s.hw === "h200" || (s.hw === "m890" && s.pdMode === "prefill") },
        { id: "balanced-ep32",      label: "Balanced (TP32 · EP32)",       showWhen: (s) => s.hw === "h100" || (s.hw === "810e" && s.pdMode === "decode") },
        { id: "balanced-tp32-ep8",  label: "Balanced (TP32 · EP8)",        showWhen: (s) => s.hw === "tpu7x" },
      ],
    },
  ],

  overlayDims: [
    {
      id: "spec",
      title: "Spec Decode",
      default: "dflash",
      showWhen: (s) => s.hw !== "tpu7x",
      options: [
        { id: "none", label: "Non-Spec" },
        {
          id: "dflash",
          label: "DFLASH",
          // The draft path (cutedsl_mla draft attention, ReplaySSM) has only run on Blackwell and
          // Vera Rubin; MI350X / MI355X run it on the Triton draft backend.
          disabled: (s) => ["h200", "h100", "m890", "810e"].includes(s.hw),
          disableReason: "Kimi K3.1 Pro DFLASH has not run on this platform yet; use Non-Spec until a recipe is collected.",
          // K3.1 has no NextN layer, so DFLASH (a separate draft) is the only speculative
          // algorithm. It runs on every recipe here: PD prefill accepts it under PP (the draft
          // lives on the last stage), and the attention-DP decode cell carries
          // --enable-dp-lm-head, which DFLASH + attention DP requires.
          flags: (s) => [
            "--speculative-algorithm DFLASH",
            "--speculative-draft-model-path RadixArk/Kimi-K3.1-Dflash2-Preview",
            `--speculative-dflash-block-size ${config.dflashBlock(s)}`,
            "--speculative-draft-kv-cache-dtype fp8_e4m3",
            // The draft KV ring (default on) needs an MLA draft backend; DCP forces this one
            // anyway, so pinning it keeps DCP and non-DCP recipes on the same path. ROCm runs the
            // same ring on the Triton draft backend.
            `--speculative-draft-attention-backend ${config.isAmd(s) ? "triton" : "cutedsl_mla"}`,
            // Without ReplaySSM, DFLASH keeps one intermediate KDA state per draft token and the
            // KV pool collapses (5.13M -> 0.77M tokens measured on decode). The PD prefill role
            // never runs verify and rejects the flag at startup.
            ...(s.pdMode !== "prefill" ? ["--enable-linear-replayssm-spec"] : []),
            // The MI35x single-node acceptance was measured with block verification of sampled
            // drafts; greedy requests skip it.
            ...(config.isAmd(s) && s.pdMode === "unified" ? ["--speculative-use-block-verification"] : []),
          ],
          hints: (s) =>
            s.pdMode === "decode" && s.strategy === "high-throughput"
              ? ["High-throughput decode verifies K = 5 (block 6); launch the prefill role with the same block size."]
              : [],
        },
      ],
    },
    {
      // HiCache tiers the paged MLA KV, the DSA indexer, the KDA state and the DFLASH draft.
      // Unlike Kimi-K3, L2 and L3 both keep DCP on K3.1, so no option strips --dcp-size.
      id: "hicache",
      title: "HiCache",
      default: "off",
      showWhen: (s) => s.hw !== "tpu7x",
      options: [
        { id: "off", label: "Off", showWhen: (s) => !config.isAmd(s) },
        {
          id: "l2",
          showWhen: (s) => !config.isAmd(s),
          disabled: (s) => config.isPpu(s),
          disableReason: "The PPU recipes were measured without HiCache.",
          label: "L1+L2 (host)",
          stripPrefixes: (s) => (s.pdMode === "decode" ? ["--disaggregation-decode-enable-radix-cache"] : []),
          flags: (s) => [
            // A PD decode server only tiers its cache when the decode radix cache is on.
            ...(s.pdMode === "decode" ? ["--disaggregation-decode-enable-radix-cache"] : []),
            "--enable-hierarchical-cache",
            // The default `kernel` I/O gathers MLA pages on two CUs; `direct` copies them.
            "--hicache-io-backend direct",
            "--hicache-mem-layout page_first_direct",
            // Keep one KDA checkpoint per root-to-leaf path so L2 holds more distinct prefixes.
            "--mamba-max-states-per-path 1",
            ...(s.pdMode === "prefill" ? ["--hicache-write-policy write_back"] : []),
          ],
        },
        {
          id: "l3",
          showWhen: (s) => !config.isAmd(s),
          disabled: (s) => config.isPpu(s),
          disableReason: "The PPU recipes were measured without HiCache.",
          label: "+ L3 (Mooncake)",
          stripPrefixes: (s) => (s.pdMode === "decode" ? ["--disaggregation-decode-enable-radix-cache"] : []),
          flags: (s) => [
            ...(s.pdMode === "decode" ? ["--disaggregation-decode-enable-radix-cache"] : []),
            "--enable-hierarchical-cache",
            "--hicache-io-backend direct",
            "--hicache-mem-layout page_first_direct",
            "--mamba-max-states-per-path 1",
            "--hicache-storage-backend mooncake",
          ],
          env: ["SGLANG_HICACHE_MOONCAKE_CONFIG_PATH={{MOONCAKE_CONFIG}}"],
          hints: (s) => [
            "L3 also needs a mooncake_master process and the config file above on every rank.",
            "Storage stores the MLA KV once per attention-TP group; size it well above the working set:",
            "losing any object of an early page zeroes the whole prefix hit.",
            ...(config.isPipelined(s)
              ? ["On a pipelined prefill L3 is rarely queried; see HiCache in Tuning notes."]
              : []),
          ],
        },
        // MI35x: the measured host tier, first so it is the default (a hidden pick reseats onto the
        // first usable option); kernel I/O, since direct is unverified on ROCm, and every KDA state per path.
        {
          id: "amd-l2",
          label: "L1+L2 (host)",
          showWhen: (s) => config.isAmd(s),
          disabled: (s) => s.pdMode === "decode" || s.strategy === "low-latency",
          disableReason: "The MI350X / MI355X decode and Low-Latency (monokernel) recipes run without HiCache.",
          flags: (s) => [
            "--enable-hierarchical-cache",
            "--hicache-write-policy write_back",
            "--hicache-ratio 2.5",
          ],
          // Keeps every rank on the Rust TreeCore; write-back asserts on a mix with the Python one.
          env: (s) => [`SGLANG_HICACHE_MAMBA_HOST_RATIO=${s.pdMode === "prefill" ? 8 : 10}`, "RUSTUP_TOOLCHAIN=stable"],
          hints: ["The host pools pin 170-185 GB of host memory per GPU, and the 8 ranks share the node's GTT (1511 GiB on MI355X); ratio 12 overflowed it at 32 agents. Check the node's GTT limit before raising either ratio."],
        },
        { id: "amd-off", label: "Off", showWhen: (s) => config.isAmd(s) },
        {
          id: "amd-l3",
          label: "+ L3 (Mooncake)",
          showWhen: (s) => config.isAmd(s),
          disabled: () => true,
          disableReason: "HiCache L3 has not run on MI350X / MI355X.",
        },
      ],
    },
  ],

  modelNames: {
    default: "moonshotai/Kimi-K3.1-Pro",
  },

  placeholders: {
    PPU_NCCL_SOCKET_IFNAME: { target: "command", label: "PPU socket interface (NCCL / Gloo)", default: "eth0" },
    PPU_NCCL_IB_HCA: { target: "command", label: "M890 NCCL HCA names", default: "mlx5,troce" },
    PPU_NCCL_IB_GID_INDEX: { target: "command", label: "M890 GID index", default: "3" },
    PPU_IB_DEVICES: { target: "command", label: "M890 RDMA devices (comma-separated)", default: "<ib-devices>" },
    HOST_IP:   { target: "command", label: "Bind host",        default: "0.0.0.0"        },
    PORT:      { target: "command", label: "Bind port",        default: "30000"          },
    NODE0_IP:  { target: "command", label: "Head node IP",     default: "<node0-ip>"     },
    NODE_RANK: { target: "command", label: "This node rank",   default: "<node-rank>"    },
    LOCAL_IP:  { target: "command", label: "This node IP",     default: "<this-node-ip>" },
    NETWORK_IFACE: { target: "command", label: "Cross-node NIC", default: "<your-nic>"   },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    MOONCAKE_CONFIG: { target: "command", label: "Mooncake config path", default: "<mooncake.json>" },
    CURL_HOST: { target: "curl",    label: "Server host",      default: "localhost"      },
    CURL_PORT: { target: "curl",    label: "Server port",      default: "30000"          },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  benchmarkCommands: {
    speed:
`python3 -m sglang.benchmark.serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} --random-range-ratio 1.0 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --warmup-requests 64 --flush-cache`,
    numPromptsByConc: { 1: 16, 16: 80, 64: 320, 256: 1280 },
  },

  // The K3.1 image is built from docker/k31.Dockerfile; it verifies the Rust radix TreeCore,
  // flashmla_ops and the 1024-expert FlashInfer DynBlock at build time (a Python-only install
  // falls back to slower paths silently).
  dockerImages: {
    m890: "egslingjun-registry.cn-wulanchabu.cr.aliyuncs.com/egslingjun-tmp/inference-xpu-pytorch:kimi-k3.1-pro-sglang0.5.22-cuda13.0-pytorch2.13.0-20261103",
    "810e": "egslingjun-registry.cn-wulanchabu.cr.aliyuncs.com/egslingjun-tmp/inference-xpu-pytorch:kimi-k3.1-pro-sglang0.5.22-cuda13.0-pytorch2.13.0-20261103",
    b300:  "lmsysorg/sglang:dev-kimi-k31-pro",
    gb300: "lmsysorg/sglang:dev-kimi-k31-pro",
    // Placeholders until a recipe for the platform is collected.
    b200:  "lmsysorg/sglang:dev-kimi-k31-pro",
    gb200: "lmsysorg/sglang:dev-kimi-k31-pro",
    h200:  "lmsysorg/sglang:dev-kimi-k31-pro",
    h100:  "lmsysorg/sglang:dev-kimi-k31-pro",
    "vera-rubin": "lmsysorg/sglang:dev-kimi-k31-pro-cu134",
    mi350x: "lmsysorg/sglang:dev-kimi-k31-pro-mi35x",
    mi355x: "lmsysorg/sglang:dev-kimi-k31-pro-mi35x",
  },
  // MI35x PD roles take the host network and RDMA flags for the bootstrap ports and Mooncake.
  dockerHostNetworkWhen: (s) => config.isAmd(s) && ["prefill", "decode"].includes(s.pdMode),
  github: {
    cookbookModel: "moonshotai/kimi-k3.1-pro",
  },

  playgroundFeatures: {

    // ----- Card: "Parsers" -----
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser kimi_k3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser kimi_k3" },
      ],
    },

    // ----- Card: "PD Disaggregation" -----
    pdDisagg: {
      showWhen: (b) => b.pdMode === "prefill" || b.pdMode === "decode",
      transferBackends: [
        { id: "mooncake", label: "Mooncake" },
        {
          id: "nixl", label: "NiXL",
          disable: [{ when: { hw: ["m890", "810e"] },
            reason: "The PPU recipes use Mooncake; NiXL has not been validated." }, { when: { hw: ["b300", "gb300", "mi350x", "mi355x"] },
            reason: "Kimi K3.1 Pro with DCP rejects any PD transfer backend other than Mooncake on either role, and every Kimi K3.1 Pro decode recipe here runs DCP." }],
        },
      ],
      ibDevices: [
        { id: "\"{{PPU_IB_DEVICES}}\"", label: "M890 RDMA devices", defaultWhen: { hw: ["m890"] },
          hide: { hw: ["vera-rubin", "b300", "gb300", "b200", "gb200", "h200", "h100", "mi350x", "mi355x", "810e", "tpu7x"] } },
        { id: "auto", label: "Auto" },
        { id: "mlx5_0", label: "mlx5_0" },
      ],
      routerOverrides: [
        {
          // MI35x 1P1D: measured behind the ROCm image's SGLang router.
          when: { hw: ["mi350x", "mi355x"] },
          command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} 8998 \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}} \\
  --request-timeout-secs 3600`,
        },
        {
          when: { hw: ["m890"] },
          command:
`SGLANG_ROUTER_STREAM_IDLE_TIMEOUT_SECS=3600 \\
sgl-router \\
  --pd-disaggregation \\
  --prefill http://<prefill-1-head>:{{PREFILL_PORT}} 8998 \\
  --prefill http://<prefill-2-head>:{{PREFILL_PORT}} 8998 \\
  --decode http://<decode-head>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}} \\
  --policy power_of_two \\
  --model-id {{MODEL_NAME}} \\
  --tokenizer-path {{MODEL_NAME}} \\
  --request-timeout-secs 540`,
        },
        {
          when: { hw: ["810e"] },
          command:
`SGLANG_ROUTER_STREAM_IDLE_TIMEOUT_SECS=3600 \\
sgl-router \\
  --pd-disaggregation \\
  --prefill http://<prefill-head>:{{PREFILL_PORT}} 8998 \\
  --decode http://<decode-head>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}} \\
  --policy power_of_two \\
  --model-id {{MODEL_NAME}} \\
  --tokenizer-path {{MODEL_NAME}} \\
  --request-timeout-secs 540`,
        },
      ],
      router: {
        port: 8000,
        // The K3.1 build of sgl-router renders the K3 chat template and keeps content-less tool
        // messages; the stock sglang-router 0.3.2 does neither. Raise the stream-idle timeout
        // for overloaded benchmarks (the 180 s default kills long-queued streams).
        command:
`SGLANG_ROUTER_STREAM_IDLE_TIMEOUT_SECS=3600 \\
sgl-router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} 8998 \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}} \\
  --policy power_of_two \\
  --model-id moonshotai/Kimi-K3.1-Pro \\
  --tokenizer-path <Kimi-K3.1-Pro tokenizer.json> \\
  --request-timeout-secs 540`,
      },
    },

    // ----- Card: "HiCache" -----
    hicache: {
      showWhen: (b) => b.hicache !== undefined && b.hicache !== "off",
      backends: [
        { id: null,       label: "Auto" },
        { id: "file",     label: "File" },
        { id: "mooncake", label: "Mooncake" },
      ],
      writePolicies: [
        { id: "auto",          label: "Auto" },
        { id: "write_through", label: "Write-through" },
        { id: "write_back",    label: "Write-back" },
      ],
    },

    flagSelects: [
      {
        // DSA sparse-MLA kernel. FlashMLA (flashmla_kv) is the K3.1 default and matches the reference
        // fp8_ds_mla KV; TRT-LLM and Prims-TS read a raw FP8 KV with FP8 q, so PD peers must both
        // use one of them. Both phases share one KV pool, so prefill and decode get the same value.
        id: "attention", title: "Attention Backend",
        showWhen: (b) => ["b300", "gb300", "b200", "gb200"].includes(b.hw),
        stripPrefixes: ["--dsa-prefill-backend", "--dsa-decode-backend"],
        options: [
          { id: "flashmla", label: "FlashMLA" },
          { id: "trtllm",   label: "TRT-LLM",  flags: ["--dsa-prefill-backend trtllm", "--dsa-decode-backend trtllm"] },
          { id: "prims_ts", label: "Prims-TS", flags: ["--dsa-prefill-backend prims_ts", "--dsa-decode-backend prims_ts"] },
        ],
      },
      {
        // MoE runner. TRT-LLM is the Blackwell default (flashinfer_mxfp4, trtllm-gen fused MoE), so it
        // adds no flag; on the PP4 × PCP2 prefill cell it drops the MegaMoE a2a backend that cell
        // carries. MegaMoE forces EP = TP, so --ep-size is rewritten in place. On a flat TP cell it
        // also needs --shared-experts-tp-size (otherwise the BF16 shared experts are replicated,
        // ~22 GiB) and a token cap that covers the whole chunk (prefill / unified, 16384) or the
        // largest verify batch (decode, 4096). Option env is static, so the decode and non-decode
        // MegaMoE options are separate chips; only one is visible per cell. The PP4 × PCP2 cell
        // already runs MegaMoE on the default cap of 8192, so its MegaMoE chip only names it.
        id: "moe", title: "MoE Runner",
        showWhen: (b) => ["b300", "gb300", "b200", "gb200"].includes(b.hw),
        stripPrefixes: ["--moe-a2a-backend", "--shared-experts-tp-size"],
        options: [
          { id: "trtllm", label: "TRT-LLM" },
          { id: "megamoe-pp", label: "MegaMoE", flags: ["--moe-a2a-backend megamoe"],
            hide: { strategy: ["low-latency", "high-concurrency", "balanced", "high-throughput", "balanced-dcp16", "balanced-ep16", "balanced-ep32-dp4", "balanced-ep32", "balanced-pp4", "balanced-tp32-ep8", "balanced-tp8"] } },
          { id: "megamoe", label: "MegaMoE",
            hide: { strategy: ["high-throughput-pp", "balanced", "high-throughput"] },
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=16384"],
            flags: (v, b) => {
              const tp = ["b200", "gb200"].includes(b.hw) ? 16 : 8;
              return [`--ep-size ${tp}`, "--moe-a2a-backend megamoe", `--shared-experts-tp-size ${tp}`];
            } },
          { id: "megamoe-decode", label: "MegaMoE",
            hide: { strategy: ["low-latency", "high-concurrency", "high-throughput-pp", "balanced-dcp16", "balanced-ep16", "balanced-ep32-dp4", "balanced-ep32", "balanced-pp4", "balanced-tp32-ep8", "balanced-tp8"] },
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096"],
            flags: (v, b) => {
              const tp = ["b200", "gb200"].includes(b.hw) ? 16 : 8;
              return [`--ep-size ${tp}`, "--moe-a2a-backend megamoe", `--shared-experts-tp-size ${tp}`];
            } },
        ],
      },
      {
        // Tokens proposed per step (K). DFLASH spells it as the verify window, K + 1.
        id: "proposedDraftTokens", title: "Proposed Draft Tokens",
        showWhen: (b) => b.spec === "dflash" && b.hw !== "tpu7x",
        control: "slider",
        // Follow the cell's own block, so the default adds no diff to the Deploy command.
        default: (b) => String(config.dflashBlock(b) - 1),
        stripPrefixes: ["--speculative-dflash-block-size", "--speculative-num-draft-tokens"],
        options: [
          { id: "3", label: "3", flags: ["--speculative-dflash-block-size 4"] },
          { id: "4", label: "4", flags: ["--speculative-dflash-block-size 5"] },
          { id: "5", label: "5", flags: ["--speculative-dflash-block-size 6"] },
          { id: "6", label: "6", flags: ["--speculative-dflash-block-size 7"] },
          { id: "7", label: "7", flags: ["--speculative-dflash-block-size 8"] },
        ],
      },
      {
        id: "prefillGraph", title: "Prefill CUDA Graph",
        showWhen: (b) => b.pdMode !== "decode" && !config.isPpu(b) && b.hw !== "tpu7x",
        stripPrefixes: ["--cuda-graph-backend-prefill", "--cuda-graph-max-bs-prefill"],
        options: [
          { id: "auto", label: "Auto" },
          { id: "bcg",  label: "Breakable (BCG)",
            flags: ["--cuda-graph-backend-prefill breakable", "--cuda-graph-max-bs-prefill 8192"] },
        ],
      },
      {
        id: "pdDecodeRadix", title: "PD Decode Radix Cache",
        showWhen: (b) => b.pdMode === "decode" && !config.isPpu(b),
        stripPrefixes: ["--disaggregation-decode-enable-radix-cache"],
        options: [
          { id: "off", label: "Off" },
          { id: "on",  label: "On", flags: ["--disaggregation-decode-enable-radix-cache"] },
        ],
      },
      {
        // The 10-28 GB300 decode layout survey (fixed batch, 8K context): one
        // decode server on four trays, attention DP2 with TP8 · DCP8 inside each 8-GPU group and
        // MegaMoE EP16 across all 16. It beat 8-GPU workers from N = 8 up (+15% at N = 64).
        // MegaMoE runs over CUDA symmetric memory, so EP16 needs one NVLink domain: GB300 only.
        // 136 requests per DP rank x block 8 = 1088 verify rows, well under the 4096 cap.
        id: "wideEp", title: "WideEP Decode (large-scale)",
        showWhen: (b) => b.hw === "gb300" && b.pdMode === "decode",
        stripPrefixes: [
          "--tp-size", "--attn-dp-size", "--enable-dp-lm-head", "--dcp-size", "--dcp-comm-backend",
          "--ep-size", "--moe-a2a-backend", "--shared-experts-tp-size",
          "--nnodes", "--node-rank", "--dist-init-addr",
          "--max-running-requests", "--cuda-graph-max-bs-decode",
        ],
        stripEnv: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK"],
        options: [
          { id: "off", label: "Off", flags: () => null },
          {
            id: "dep16", label: "16 GPUs (DP2 · DCP8 · EP16)",
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096"],
            flags: () => [
              "--tp-size 16", "--attn-dp-size 2", "--enable-dp-lm-head",
              "--dcp-size 8", "--dcp-comm-backend fi_a2a",
              "--ep-size 16", "--moe-a2a-backend megamoe", "--shared-experts-tp-size 8",
              "--nnodes 4", "--node-rank {{NODE_RANK}}", "--dist-init-addr {{NODE0_IP}}:20000",
              "--max-running-requests 272", "--cuda-graph-max-bs-decode 136",
            ],
          },
        ],
      },
      {
        // How DCP merges the per-rank attention partials. Unset resolves to fi_a2a where the DCP
        // group sits in one MNNVL domain, else a2a.
        id: "dcpComm", title: "DCP Communication",
        showWhen: (b) => config.hasDcp(b),
        stripPrefixes: ["--dcp-comm-backend"],
        options: [
          { id: "fi_a2a", label: "FlashInfer A2A", flags: ["--dcp-comm-backend fi_a2a"] },
          { id: "a2a",    label: "A2A",            flags: ["--dcp-comm-backend a2a"] },
          { id: "ag_rs",  label: "AllGather + ReduceScatter", flags: ["--dcp-comm-backend ag_rs"] },
        ],
      },
      {
        // KDA state slots on the GPU, as a count. Replaces the memory-ratio sizing; under
        // attention DP the count is split across the DP ranks.
        id: "kdaSlots", title: "KDA State Slots",
        stripPrefixes: ["--max-mamba-cache-size", "--mamba-full-memory-ratio"],
        options: [
          { id: "128",  label: "128",  flags: ["--max-mamba-cache-size 128"] },
          { id: "256",  label: "256",  flags: ["--max-mamba-cache-size 256"] },
          { id: "512",  label: "512",  flags: ["--max-mamba-cache-size 512"] },
          { id: "1024", label: "1024", flags: ["--max-mamba-cache-size 1024"] },
        ],
      },
      {
        // KDA checkpoints kept per root-to-leaf radix path. 1 keeps more distinct prefixes.
        id: "kdaPerPath", title: "KDA States per Path",
        showWhen: (b) => b.pdMode !== "decode" && !config.isPpu(b),
        stripPrefixes: ["--mamba-max-states-per-path"],
        options: [
          { id: "unlimited", label: "Unlimited" },
          { id: "1", label: "1", flags: ["--mamba-max-states-per-path 1"] },
          { id: "2", label: "2", flags: ["--mamba-max-states-per-path 2"] },
          { id: "4", label: "4", flags: ["--mamba-max-states-per-path 4"] },
        ],
      },
      {
        // MegaMoE token cap per rank. Its symmetric workspace grows with EP size x cap (EP8:
        // ~14 GB at 8192, ~25 GB at 16384), outside --mem-fraction-static. The server refuses to
        // start below the largest prefill chunk or decode / verify batch on one rank.
        id: "megaMoeCap", title: "MegaMoE Tokens per Rank",
        // Needs per rank: flat TP prefill / unified 2048 (16384-token chunk over SP-MoE x8),
        // decode up to 176 x block 8 = 1408, WideEP 136 x 8 = 1088, PP4 x PCP2 8192 (chunk / PCP2).
        showWhen: (b, v, d) => {
          const moe = (v && v.moe) || (d && d.moe) || "";
          return moe.startsWith("megamoe") || !!(v && v.wideEp === "dep16");
        },
        stripEnv: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK"],
        options: [
          { id: "2048",  label: "2048",  flags: () => [], env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=2048"],
            disable: [{ when: { strategy: ["high-throughput-pp"] }, reason: "PP4 × PCP2 prefill needs at least 8192 tokens per rank (16384-token chunk over PCP2)." }] },
          { id: "4096",  label: "4096",  flags: () => [], env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096"],
            disable: [{ when: { strategy: ["high-throughput-pp"] }, reason: "PP4 × PCP2 prefill needs at least 8192 tokens per rank (16384-token chunk over PCP2)." }] },
          { id: "8192",  label: "8192",  flags: () => [], env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8192"] },
          { id: "16384", label: "16384", flags: () => [], env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=16384"] },
          { id: "32768", label: "32768", flags: () => [], env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=32768"] },
        ],
      },
      {
        // Layers per PP stage (93 in total). The draft lives on the last stage, so the recipe gives
        // it 21 layers; 24,24,24,21 measured 3-6% faster than the default 23,23,23,24.
        id: "ppPartition", title: "PP Layer Partition",
        showWhen: (b) => config.isPipelined(b) && !config.isPpu(b),
        stripEnv: ["SGLANG_PP_LAYER_PARTITION"],
        options: [
          { id: "recipe", label: "24,24,24,21", flags: () => [], env: ["SGLANG_PP_LAYER_PARTITION=24,24,24,21"] },
          { id: "default", label: "Default (23,23,23,24)", flags: () => [] },
        ],
      },
    ],
  },

  // Verification marks follow the measured rounds: a cell is `verified` only where accuracy
  // and a serving round exist with this exact shape.
  cells: [
    {
      match: { hw: "b300", pdMode: "unified", strategy: "low-latency" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // High-Concurrency: DCP8 shards the TP-replicated MLA KV; EP2 (no a2a) speeds up the decode down-proj.
      match: { hw: "b300", pdMode: "unified", strategy: "high-concurrency" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_DCP_PUSH_COMBINE_MAX_ROWS=1536",
        "SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB=32768",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--dcp-comm-backend fi_a2a",
        "--ep-size 2",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // TP8 prefill, no DCP: the simplest PD prefill role.
      match: { hw: "b300", pdMode: "prefill", strategy: "low-latency" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.88",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--schedule-policy shortest-prefill-first",
        "--mamba-full-memory-ratio 0.8",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend mooncake",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", pdMode: "prefill", strategy: "high-throughput-pp" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_PP_LAYER_PARTITION=24,24,24,21",
        "CUDA_MODULE_LOADING=EAGER",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 2",
        "--pp-size 4",
        "--attn-cp-size 2",
        "--enable-prefill-cp",
        "--cp-strategy interleave",
        "--ep-size 2",
        "--moe-a2a-backend megamoe",
        "--mem-fraction-static 0.84",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--pp-min-chunked-prefill-size 4096",
        "--pp-chunk-overhead-tokens 2000",
        "--cuda-graph-backend-prefill breakable",
        "--cuda-graph-max-bs-prefill 8192",
        "--schedule-policy fcfs",
        "--mamba-full-memory-ratio 0.62",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend mooncake",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // The measured 1P1D decode: TP8 · DCP8 · EP2 with the decode radix cache.
      match: { hw: "b300", pdMode: "decode", strategy: "balanced" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) => (s.hicache === "off" && s.spec === "dflash" ? "verified" : "in-progress"),
      env: [
        "SGLANG_DCP_PUSH_COMBINE_MAX_ROWS=1536",
        "SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB=32768",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--dcp-comm-backend fi_a2a",
        "--ep-size 2",
        "--mem-fraction-static 0.88",
        "--max-running-requests 176",
        "--disaggregation-decode-extra-slots 16",
        "--disaggregation-decode-enable-radix-cache",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend mooncake",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Capacity-first decode as measured without speculation: c64 at TPOT p90 ~29 ms, KV pool
      // 19.8M tokens.
      match: { hw: "b300", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 1,
      verified: false,
      verificationStatus: (s) => (s.hicache === "off" && s.spec === "none" ? "verified" : "in-progress"),
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--mem-fraction-static 0.92",
        "--max-running-requests 72",
        "--disaggregation-decode-extra-slots 16",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend mooncake",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same recipe as B300, on two GB300 trays (2×4 GPUs over MNNVL).
      match: { hw: "gb300", pdMode: "unified", strategy: "low-latency" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same recipe as B300, on two GB300 trays (2×4 GPUs over MNNVL).
      match: { hw: "gb300", pdMode: "unified", strategy: "high-concurrency" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_DCP_PUSH_COMBINE_MAX_ROWS=1536",
        "SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB=32768",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--dcp-comm-backend fi_a2a",
        "--ep-size 2",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // TP8 prefill on two GB300 trays (2×4 GPUs over MNNVL), no DCP.
      match: { hw: "gb300", pdMode: "prefill", strategy: "low-latency" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
        "MC_FORCE_MNNVL=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.88",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--schedule-policy shortest-prefill-first",
        "--mamba-full-memory-ratio 0.8",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend mooncake",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same recipe as B300, on two GB300 trays (2×4 GPUs over MNNVL).
      match: { hw: "gb300", pdMode: "prefill", strategy: "high-throughput-pp" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
        "MC_FORCE_MNNVL=1",
        "SGLANG_PP_LAYER_PARTITION=24,24,24,21",
        "CUDA_MODULE_LOADING=EAGER",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 2",
        "--pp-size 4",
        "--attn-cp-size 2",
        "--enable-prefill-cp",
        "--cp-strategy interleave",
        "--ep-size 2",
        "--moe-a2a-backend megamoe",
        "--mem-fraction-static 0.84",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--pp-min-chunked-prefill-size 4096",
        "--pp-chunk-overhead-tokens 2000",
        "--cuda-graph-backend-prefill breakable",
        "--cuda-graph-max-bs-prefill 8192",
        "--schedule-policy fcfs",
        "--mamba-full-memory-ratio 0.62",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend mooncake",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same recipe as B300, on two GB300 trays (2×4 GPUs over MNNVL).
      match: { hw: "gb300", pdMode: "decode", strategy: "balanced" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
        "MC_FORCE_MNNVL=1",
        "SGLANG_DCP_PUSH_COMBINE_MAX_ROWS=1536",
        "SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB=32768",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--dcp-comm-backend fi_a2a",
        "--ep-size 2",
        "--mem-fraction-static 0.88",
        "--max-running-requests 176",
        "--disaggregation-decode-extra-slots 16",
        "--disaggregation-decode-enable-radix-cache",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend mooncake",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same recipe as B300, on two GB300 trays (2×4 GPUs over MNNVL).
      match: { hw: "gb300", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
        "MC_FORCE_MNNVL=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--mem-fraction-static 0.92",
        "--max-running-requests 72",
        "--disaggregation-decode-extra-slots 16",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend mooncake",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ----- Placeholder cells: one Unified Balanced cell per platform without a K3.1 recipe yet. -----
    {
      // Placeholder. B200 2×8, TP16 · DCP16 (Kimi-K3's flat B200 shape under speculation).
      match: { hw: "b200", pdMode: "unified", strategy: "balanced-dcp16" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      warn: "Placeholder: no Kimi K3.1 Pro recipe has run on this platform yet; this command is a starting point, not a tested recipe.",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Placeholder. GB200 4×4, TP16 · DCP16 (the Kimi-K3 Balanced shape).
      match: { hw: "gb200", pdMode: "unified", strategy: "balanced-dcp16" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      warn: "Placeholder: no Kimi K3.1 Pro recipe has run on this platform yet; this command is a starting point, not a tested recipe.",
      env: ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Placeholder. Two Vera Rubin trays (2×4 GPUs), TP8.
      match: { hw: "vera-rubin", pdMode: "unified", strategy: "balanced-tp8" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      warn: "Placeholder: no Kimi K3.1 Pro recipe has run on this platform yet; this command is a starting point, not a tested recipe.",
      env: ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Placeholder. H200 2×8, TP16 · EP16 (the Kimi-K3 shape).
      match: { hw: "h200", pdMode: "unified", strategy: "balanced-ep16" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      warn: "Placeholder: no Kimi K3.1 Pro recipe has run on this platform yet; this command is a starting point, not a tested recipe.",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--ep-size 16",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Placeholder. H100 4×8, TP32 · EP32 (the Kimi-K3 shape).
      match: { hw: "h100", pdMode: "unified", strategy: "balanced-ep32" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      warn: "Placeholder: no Kimi K3.1 Pro recipe has run on this platform yet; this command is a starting point, not a tested recipe.",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--ep-size 32",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ----- MI350X / MI355X: the AgentX recipes of the ROCm port. -----
    {
      // MI350X 1×8, TP8: the MI355X Low-Latency recipe, on the monokernel.
      match: { hw: "mi350x", pdMode: "unified", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) => (s.spec === "dflash" ? "verified" : "in-progress"),
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
        "SGLANG_AMD_MONO_K31_WIDE=1",
        "PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.95",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB=6",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1",
        "HSA_USERPTR_FOR_PAGED_MEM=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 16384",
        "--max-running-requests 4",
        "--cuda-graph-max-bs-decode 4",
        "--schedule-policy lpm",
        "--prefill-decode-interval 16",
        // 150 state slots keep agent sessions' prefixes cached.
        "--max-mamba-cache-size 150",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--cuda-graph-backend-prefill breakable",
        "--cuda-graph-bs-prefill 256 512 768 1024 1536 2048 3072 4096",
        "--enable-amd-monokernel",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // MI350X 1×8, TP8 · DCP8: the MI355X High-Throughput recipe.
      match: { hw: "mi350x", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) => (s.spec === "dflash" && s.hicache === "amd-l2" ? "verified" : "in-progress"),
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
        "PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.95",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB=10",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1",
        "HSA_USERPTR_FOR_PAGED_MEM=0",
        // From 48 agents: the MoE front and residual stream on row shards in prefill (no extra
        // memory) and a draft block's ring locations in one launch.
        "SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_ROUTED_FRONT=1",
        "SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM=1",
        "SGLANG_OPT_DFLASH_RING_LOCS_FUSED=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--dcp-comm-backend a2a",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 32768",
        "--max-running-requests 48",
        "--cuda-graph-max-bs-decode 48",
        "--schedule-policy lpm",
        // Below 48 agents the recipe runs 16k chunks every 16 decode steps with block 6.
        "--prefill-decode-interval 8",
        "--mamba-full-memory-ratio 0.65",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--cuda-graph-backend-prefill breakable",
        "--cuda-graph-bs-prefill 256 512 768 1024 1536 2048 3072 4096",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // MI350X 1P1D prefill: the MI355X 1P1D prefill.
      match: { hw: "mi350x", pdMode: "prefill", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) => (s.spec === "dflash" && s.hicache === "amd-l2" ? "verified" : "in-progress"),
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
        "PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.95",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB=10",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1",
        "HSA_USERPTR_FOR_PAGED_MEM=0",
        "SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION=0.1",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600",
        "MC_GID_INDEX=1",
        "MC_MAX_CONCURRENT_REG_MR=8",
        "NCCL_IB_DISABLE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--dcp-comm-backend a2a",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--max-running-requests 128",
        "--cuda-graph-max-bs-decode 16",
        "--schedule-policy shortest-prefill-first",
        "--disaggregation-decode-allocation-policy early",
        "--mamba-full-memory-ratio 0.8",
        "--mamba-radix-cache-strategy extra_buffer",
        "--watchdog-timeout 3600",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend mooncake",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // MI350X 1P1D decode: the MI355X 1P1D decode.
      match: { hw: "mi350x", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) => (s.spec === "dflash" ? "verified" : "in-progress"),
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
        "PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.95",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB=10",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600",
        "MC_GID_INDEX=1",
        "MC_MAX_CONCURRENT_REG_MR=8",
        "NCCL_IB_DISABLE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--dcp-comm-backend a2a",
        "--mem-fraction-static 0.91",
        "--max-running-requests 96",
        "--cuda-graph-max-bs-decode 96",
        "--max-mamba-cache-size 160",
        "--mamba-radix-cache-strategy extra_buffer",
        "--disaggregation-decode-extra-slots 16",
        "--disaggregation-decode-enable-radix-cache",
        "--watchdog-timeout 3600",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend mooncake",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // The single-node AgentX recipe for 1-4 agents, on the decode monokernel (no KV offload).
      match: { hw: "mi355x", pdMode: "unified", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) => (s.spec === "dflash" ? "verified" : "in-progress"),
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
        "SGLANG_AMD_MONO_K31_WIDE=1",
        "PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.95",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB=6",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1",
        "HSA_USERPTR_FOR_PAGED_MEM=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 16384",
        "--max-running-requests 4",
        "--cuda-graph-max-bs-decode 4",
        "--schedule-policy lpm",
        "--prefill-decode-interval 16",
        // 150 state slots keep agent sessions' prefixes cached.
        "--max-mamba-cache-size 150",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--cuda-graph-backend-prefill breakable",
        "--cuda-graph-bs-prefill 256 512 768 1024 1536 2048 3072 4096",
        "--enable-amd-monokernel",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // The single-node AgentX recipe for 8-56 agents, at its concurrency-48 shape; HiCache comes from the overlay.
      match: { hw: "mi355x", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) => (s.spec === "dflash" && s.hicache === "amd-l2" ? "verified" : "in-progress"),
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
        "PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.95",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB=10",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1",
        "HSA_USERPTR_FOR_PAGED_MEM=0",
        // From 48 agents: the MoE front and residual stream on row shards in prefill (no extra
        // memory) and a draft block's ring locations in one launch.
        "SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_ROUTED_FRONT=1",
        "SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM=1",
        "SGLANG_OPT_DFLASH_RING_LOCS_FUSED=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--dcp-comm-backend a2a",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 32768",
        "--max-running-requests 48",
        "--cuda-graph-max-bs-decode 48",
        "--schedule-policy lpm",
        // Below 48 agents the recipe runs 16k chunks every 16 decode steps with block 6.
        "--prefill-decode-interval 8",
        "--mamba-full-memory-ratio 0.65",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--cuda-graph-backend-prefill breakable",
        "--cuda-graph-bs-prefill 256 512 768 1024 1536 2048 3072 4096",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // MI355X 1P1D prefill, measured with the decode below at AgentX concurrency 92.
      match: { hw: "mi355x", pdMode: "prefill", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) => (s.spec === "dflash" && s.hicache === "amd-l2" ? "verified" : "in-progress"),
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
        "PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.95",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB=10",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1",
        "HSA_USERPTR_FOR_PAGED_MEM=0",
        "SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION=0.1",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600",
        "MC_GID_INDEX=1",
        "MC_MAX_CONCURRENT_REG_MR=8",
        "NCCL_IB_DISABLE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--dcp-comm-backend a2a",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--max-running-requests 128",
        "--cuda-graph-max-bs-decode 16",
        "--schedule-policy shortest-prefill-first",
        "--disaggregation-decode-allocation-policy early",
        "--mamba-full-memory-ratio 0.8",
        "--mamba-radix-cache-strategy extra_buffer",
        "--watchdog-timeout 3600",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend mooncake",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // MI355X 1P1D decode: 160 KDA state slots leave the rest of the pool to the MLA KV (18.7M tokens).
      match: { hw: "mi355x", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) => (s.spec === "dflash" ? "verified" : "in-progress"),
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
        "PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.95",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB=10",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600",
        "MC_GID_INDEX=1",
        "MC_MAX_CONCURRENT_REG_MR=8",
        "NCCL_IB_DISABLE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--dcp-comm-backend a2a",
        "--mem-fraction-static 0.91",
        "--max-running-requests 96",
        "--cuda-graph-max-bs-decode 96",
        "--max-mamba-cache-size 160",
        "--mamba-radix-cache-strategy extra_buffer",
        "--disaggregation-decode-extra-slots 16",
        "--disaggregation-decode-enable-radix-cache",
        "--watchdog-timeout 3600",
        "--weight-loader-prefetch-checkpoints",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend mooncake",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "m890", pdMode: "prefill", strategy: "balanced-ep16" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      warn: "Performance results and final validation are pending.",
      env: [
        "NCCL_SOCKET_IFNAME={{PPU_NCCL_SOCKET_IFNAME}}",
        "GLOO_SOCKET_IFNAME={{PPU_NCCL_SOCKET_IFNAME}}",
        "MC_FORCE_HCA=1",
        "MOONCAKE_PROTOCOL=rdma",
        "SGLANG_WARMUP_TIMEOUT=3600",
        "NCCL_IB_HCA={{PPU_NCCL_IB_HCA}}",
        "NCCL_IB_GID_INDEX={{PPU_NCCL_IB_GID_INDEX}}",
        "NCCL_DEBUG=WARN",
        "NCCL_MNNVL_ENABLE=1",
        "SGLANG_MAMBA_SSM_DTYPE=float32",
        "SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA=1",
        "SGLANG_OPT_USE_FP4_INDEXER_CACHE=0",
        "DEEPEP_NUM_MAX_ICN_PEERS=16"
      ],
      flags: [
        "--model-path moonshotai/Kimi-K3.1-Pro",
        "--trust-remote-code",
        "--watchdog-timeout 3600",
        "--dist-timeout 3600",
        "--tp-size 16",
        "--ep-size 16",
        "--linear-attn-prefill-backend flashkda",
        "--moe-runner-backend deep_gemm",
        "--moe-a2a-backend deepep",
        "--attention-backend dsa",
        "--dsa-prefill-backend flashmla_kv",
        "--dsa-decode-backend flashmla_kv",
        "--page-size 64",
        "--chunked-prefill-size 8192",
        "--kv-cache-dtype fp8_e4m3",
        "--disable-flashinfer-autotune",
        "--weight-loader-prefetch-checkpoints",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--deepep-mode normal",
        "--mem-fraction-static 0.953",
        "--max-running-requests 48",
        "--max-mamba-cache-size 22",
        "--max-total-tokens 96192",
        "--max-prefill-tokens 16384",
        "--disable-prefill-cuda-graph",
        "--enable-symm-mem",
        "--disable-overlap-schedule",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend mooncake",
        "--disaggregation-bootstrap-port 8998",
        "--disaggregation-ib-device \"{{PPU_IB_DEVICES}}\"",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ]
    },
    {
      match: { hw: "m890", pdMode: "decode", strategy: "balanced-ep32-dp4" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      warn: "Performance results and final validation are pending.",
      env: [
        "NCCL_SOCKET_IFNAME={{PPU_NCCL_SOCKET_IFNAME}}",
        "GLOO_SOCKET_IFNAME={{PPU_NCCL_SOCKET_IFNAME}}",
        "MC_FORCE_HCA=1",
        "MOONCAKE_PROTOCOL=rdma",
        "SGLANG_WARMUP_TIMEOUT=3600",
        "NCCL_IB_HCA={{PPU_NCCL_IB_HCA}}",
        "NCCL_IB_GID_INDEX={{PPU_NCCL_IB_GID_INDEX}}",
        "NCCL_DEBUG=WARN",
        "NCCL_MNNVL_ENABLE=1",
        "SGLANG_MAMBA_SSM_DTYPE=float32",
        "SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA=1",
        "SGLANG_OPT_USE_FP4_INDEXER_CACHE=0",
        "DEEPEP_NUM_MAX_ICN_PEERS=32",
        "DG_USE_MOE_DYNAMIC_TILE=1",
        "SGLANG_SAIL_DEEPEP_RECV_HOOK=0",
        "DEEPEP_COMBINE_TOKEN_BALANCED=1"
      ],
      flags: [
        "--model-path moonshotai/Kimi-K3.1-Pro",
        "--trust-remote-code",
        "--watchdog-timeout 3600",
        "--dist-timeout 3600",
        "--tp-size 32",
        "--ep-size 32",
        "--dp-size 4",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--linear-attn-prefill-backend flashkda",
        "--moe-runner-backend deep_gemm",
        "--moe-a2a-backend deepep",
        "--attention-backend dsa",
        "--dsa-prefill-backend flashmla_kv",
        "--dsa-decode-backend flashmla_kv",
        "--page-size 64",
        "--chunked-prefill-size 8192",
        "--kv-cache-dtype fp8_e4m3",
        "--disable-flashinfer-autotune",
        "--weight-loader-prefetch-checkpoints",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--linear-attn-decode-backend triton",
        "--deepep-mode low_latency",
        "--cuda-graph-backend-prefill disabled",
        "--mem-fraction-static 0.811",
        "--max-mamba-cache-size 128",
        "--max-total-tokens 238720",
        "--max-running-requests 128",
        "--cuda-graph-max-bs-decode 32",
        "--cuda-graph-bs-decode 8 16 20 21 22 23 24 32",
        "--disable-radix-cache",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend mooncake",
        "--disaggregation-bootstrap-port 8999",
        "--disaggregation-ib-device \"{{PPU_IB_DEVICES}}\"",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ]
    },
    {
      match: { hw: "810e", pdMode: "prefill", strategy: "balanced-pp4" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      warn: "Performance results and final validation are pending.",
      env: [
        "NCCL_SOCKET_IFNAME={{PPU_NCCL_SOCKET_IFNAME}}",
        "GLOO_SOCKET_IFNAME={{PPU_NCCL_SOCKET_IFNAME}}",
        "MOONCAKE_PROTOCOL=tcp",
        "SGLANG_WARMUP_TIMEOUT=3600",
        "SGLANG_DSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8=0",
        "SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED=1",
        "SGLANG_PP_LAYER_PARTITION=24,23,23,23",
        "SGLANG_PP_LAST_STAGE_STREAM_ORDERING=1",
        "PCCL_NET_PLUGIN_ENABLE=1",
        "MC_FORCE_TCP=1"
      ],
      flags: [
        "--model-path moonshotai/Kimi-K3.1-Pro",
        "--trust-remote-code",
        "--watchdog-timeout 3600",
        "--dist-timeout 3600",
        "--tp-size 16",
        "--pp-size 4",
        "--prefill-attention-backend fa3",
        "--decode-attention-backend flashmla",
        "--linear-attn-backend flashkda",
        "--chunked-prefill-size 4096",
        "--max-running-requests 32",
        "--mem-fraction-static 0.98",
        "--pp-async-batch-depth 1",
        "--disable-overlap-schedule",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--max-mamba-cache-size 128",
        "--max-total-tokens 65536",
        "--disable-radix-cache",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend mooncake",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ]
    },
    {
      match: { hw: "810e", pdMode: "decode", strategy: "balanced-ep32" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      warn: "Performance results and final validation are pending.",
      env: [
        "NCCL_SOCKET_IFNAME={{PPU_NCCL_SOCKET_IFNAME}}",
        "GLOO_SOCKET_IFNAME={{PPU_NCCL_SOCKET_IFNAME}}",
        "MOONCAKE_PROTOCOL=tcp",
        "SGLANG_WARMUP_TIMEOUT=3600",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64",
        "SGLANG_DSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8=0",
        "DEEPEP_NUM_MAX_ICN_PEERS=16",
        "DG_USE_MOE_DYNAMIC_TILE=1",
        "SGLANG_SAIL_DEEPEP_RECV_HOOK=0",
        "DEEPEP_COMBINE_TOKEN_BALANCED=1",
        "SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16=1",
        "PCCL_NET_PLUGIN_ENABLE=1",
        "MC_FORCE_TCP=1"
      ],
      flags: [
        "--model-path moonshotai/Kimi-K3.1-Pro",
        "--trust-remote-code",
        "--watchdog-timeout 3600",
        "--dist-timeout 3600",
        "--tp-size 32",
        "--ep-size 32",
        "--moe-runner-backend deep_gemm",
        "--moe-a2a-backend deepep",
        "--prefill-attention-backend fa3",
        "--decode-attention-backend flashmla",
        "--linear-attn-backend flashkda",
        "--linear-attn-decode-backend triton",
        "--cuda-graph-backend-prefill disabled",
        "--cuda-graph-max-bs-decode 32",
        "--max-running-requests 128",
        "--mem-fraction-static 0.9",
        "--disable-radix-cache",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend mooncake",
        "--disaggregation-bootstrap-port 8999",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ]
    },
    {
      // TPU7x: the command is the sglang-torchtpu recipe kimi_k3_1/server.py (§3.4).
      match: { hw: "tpu7x", pdMode: "unified", strategy: "balanced-tp32-ep8" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      redirect: true,
      warn: "TPU7x uses the SGLang-TorchTPU recipe; see [TPU7x deployment](#tpu).",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--device tpu",
        "--tp-size 32",
        "--ep-size 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],

  multiNodeHints: {
    m890: [
      "Run inside the PPU image listed under Install SGLang with the SAIL environment configured.",
      "2P1D: two independent 2-node prefill workers and one 4-node decode worker (8 cards/node, 64 total).",
      "Use a separate head-node IP and node-rank sequence for each worker; register both prefill heads with the router.",
      "Set the PPU socket interface (NCCL / Gloo), M890 NCCL HCA names, GID index and RDMA devices in Env / placeholders to match each node; use comma-separated RDMA device names.",
    ],
    "810e": [
      "Run inside the PPU image listed under Install SGLang with the SAIL environment configured.",
      "1P1D: one 4-node prefill worker and one 2-node decode worker (16 cards/node, 96 total).",
      "Use a separate head-node IP and node-rank sequence for each worker.",
      "Set the PPU socket interface (NCCL / Gloo) in Env / placeholders to match each node.",
      "DSA backend support on 810E is still in progress.",
      "The first request may trigger online compilation and take a long time to complete. Keep --disable-radix-cache on both roles to avoid additional online compilation.",
    ],
    b200: [
      "Multi-node Kimi K3.1 Pro needs the cross-node NIC pinned on BOTH ranks:",
      "  GLOO_SOCKET_IFNAME=<your-nic>   NCCL_SOCKET_IFNAME=<your-nic>   SGLANG_HOST_IP=<this-node-ip>",
      "  NCCL_IB_HCA=<hca0,hca1,...>     # RDMA fabrics only",
    ],
    gb200: [
      "Allocate all four nodes within a single NVL72 domain. MNNVL is off by default; set on every rank:",
      "  NCCL_MNNVL_ENABLE=1   NCCL_CUMEM_ENABLE=1",
    ],
    gb300: [
      "Allocate every tray of a worker inside one IMEX / NVL72 domain: Mooncake's MNNVL transport and",
      "the cross-tray TP all-reduce cannot cross domains.",
      "Pin the cross-node NIC on every tray:",
      "  GLOO_SOCKET_IFNAME=<your-nic>   NCCL_SOCKET_IFNAME=<your-nic>   SGLANG_HOST_IP=<this-node-ip>",
      "Restart the prefill role whenever the decode role restarts (it keeps the old decode's Mooncake mapping).",
    ],
  },
};

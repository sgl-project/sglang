// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// MiniCPM5-2B: 2.5B dense `LlamaForCausalLM` (42 layers, GQA 16Q/2KV, 131072
// context). Standard architecture — no custom kernels and no model-code fork —
// so every supported card runs it single-GPU at TP=1 with the stock backend.
// That leaves one recipe per card, hence a single variant / quantization /
// strategy / node option and no parallelism flags in any cell.
//
// Recipes are the OpenBMB model card's SGLang commands, rewritten from
// `python -m sglang.launch_server` to `sglang serve`, plus the parser pair the
// model needs to be usable through the OpenAI API. Both are baked into every
// cell, so the Parsers card in the Playground reads as an opt-OUT:
//   --tool-call-parser minicpm5   the model emits XML-style
//     `<function name="f"><param name="p">v</param></function>`; without the
//     detector `tool_calls` comes back None and the XML lands in `content`.
//   --reasoning-parser qwen3      the chat template is Qwen-style
//     (`<|im_start|>` + `<think>`) and there is no `minicpm5` reasoning
//     detector, so `qwen3` is the one that applies; without it `</think>`
//     leaks into `content`.
//
// DSpark is the separately published draft checkpoint
// (openbmb/MiniCPM5-2B-DSpark). It is orthogonal to the card grid, so it is an
// overlay row rather than a match dim. No DSpark speed numbers are published:
// the speedup tracks acceptance length, which moves with the prompt
// distribution, and a random-token dataset inflates it above real traffic.

export const config = {
  modelName: "MiniCPM5-2B",

  supportedHardware: ["h200", "rtx6000", "rtx5090", "dgx-spark"],

  // RTX PRO 6000 and RTX 5090 (SM120 / Blackwell workstation + desktop) are not
  // datacenter parts, so the shared HARDWARE_CATALOG in _deployment.jsx does not
  // carry them. Ids/labels match the DeepSeek-V4 and Qwen3.8-27B configs.
  hardware: [
    { id: "rtx6000", label: "RTX PRO 6000", vram: "96GB", vendor: "blackwell" },
    { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "blackwell" },
  ],

  matchDims: [
    { id: "variant", title: "Model Variant", options: [
      { id: "default", label: "Default" },
    ] },
    { id: "quant", title: "Quantization", options: [
      { id: "bf16", label: "BF16" },
    ] },
    { id: "nodes", title: "Nodes", options: [
      { id: "single", label: "Single Node" },
    ] },
  ],

  overlayDims: [
    {
      id: "spec",
      title: "Speculative Decoding",
      default: "none",
      options: [
        { id: "none", label: "None" },
        {
          id: "dspark", label: "DSPARK",
          // Verbatim from the model card's DSpark command, including
          // `--trust-remote-code`: the base checkpoint is plain Llama and does
          // not need it, the draft checkpoint's config does. gamma = 7, so the
          // verify window is 8 tokens.
          flags: [
            "--trust-remote-code",
            "--speculative-algorithm DSPARK",
            "--speculative-draft-model-path openbmb/MiniCPM5-2B-DSpark",
            "--speculative-dspark-block-size 7",
          ],
        },
      ],
    },
  ],

  modelNames: {
    "default|bf16": "openbmb/MiniCPM5-2B",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"  },
    PORT:      { target: "command", label: "Bind port",         default: "30000"    },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"     },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Who are you? Please briefly introduce yourself."}] }'`,

  // Reproduce command for the Benchmark card's "⚡ Reproduce" modal. No
  // `accuracy` entry: the page carries no accuracy numbers yet.
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --flush-cache`,
    numPromptsByConc: { 1: 10, 128: 512 },
  },

  // MiniCPM5 support (the `minicpm5` tool-call parser and the DSPARK draft
  // worker) ships in the SGLang dev image.
  dockerImages: {
    h200:    "lmsysorg/sglang:dev",
    rtx6000: "lmsysorg/sglang:dev",
    rtx5090: "lmsysorg/sglang:dev",
    "dgx-spark": "lmsysorg/sglang:dev",
  },

  // Pre-selects the issue template's `model` field on "Submit verified cell".
  github: {
    cookbookModel: "openbmb/MiniCPM5-2B",
  },

  playgroundFeatures: {
    // The model fits one GPU on every supported card, so TP=1 is the verified
    // shape; TP=2 is exposed for experimentation only.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2] },
      ],
    },

    // ----- Card: "Parsers" -----
    // Opt-OUT: both flags are already in every cell, so the handler derives
    // each chip as on and strips the flag when one is toggled off. The
    // reasoning slug is `qwen3`, not `minicpm5` — see the header note.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser qwen3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser minicpm5" },
      ],
    },

    // ----- Card: "Speculative Decoding" -----
    // Same DSpark flags as the Deploy panel's overlay row, so the two paths
    // compose an identical command.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "dspark",  label: "DSpark",
          flags: ["--trust-remote-code",
                  "--speculative-algorithm DSPARK",
                  "--speculative-draft-model-path openbmb/MiniCPM5-2B-DSpark",
                  "--speculative-dspark-block-size 7"] },
      ],
    },
  },

  // One recipe per card — the model card's SGLang launch line, plus the
  // `minicpm5` tool-call parser it recommends for agent workloads and the
  // `qwen3` reasoning parser its `<think>` template needs.
  cells: [
    {
      match: { hw: "h200", variant: "default", quant: "bf16", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser qwen3",
        "--tool-call-parser minicpm5",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Verification round still open on this card. `verificationStatus` alone,
      // with NO `verified: true` baseline: the boolean is what the Playground
      // reads for its own badge, so leaving it on would make the Playground
      // claim "Verified" while the Deploy panel says the round is in progress.
      match: { hw: "rtx6000", variant: "default", quant: "bf16", nodes: "single" },
      // Flat string, not a predicate: this cell is in-progress with or without
      // the DSPARK overlay, so there is nothing for the selection to switch on.
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser qwen3",
        "--tool-call-parser minicpm5",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // The 32GB card is the one where the default KV pool starves decode
      // CUDA-graph capture: with defaults the pool takes 473,718 tokens / 19 GB
      // and leaves 4.6 GB, so capture stops around bs=48 and every larger batch
      // runs eager (3810 tok/s at concurrency 64). The pair below gives back
      // 4% of the pool -- still hugely oversized for a 2.5B model -- and keeps
      // batches up to 128 graph-backed (7454 tok/s at the same concurrency).
      // The two flags go together: raising the cap without freeing the memory
      // just lets SGLang clamp capture back down.
      match: { hw: "rtx5090", variant: "default", quant: "bf16", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser qwen3",
        "--tool-call-parser minicpm5",
        "--mem-fraction-static 0.75",
        "--cuda-graph-max-bs 128",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // GB10 has no discrete VRAM, so `mem_get_info()` reports all 128GB of
      // unified system memory and the default fraction claims ~89GB for KV --
      // leaving ~5GB for the OS, which kills the node during warmup with no
      // traceback and no OOMKilled event. 0.30 is required, not tuning; it
      // still leaves a 658k-token pool.
      match: { hw: "dgx-spark", variant: "default", quant: "bf16", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser qwen3",
        "--tool-call-parser minicpm5",
        "--mem-fraction-static 0.30",
        "--cuda-graph-max-bs 128",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};

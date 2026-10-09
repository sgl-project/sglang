export const config = {
  modelName: "Wan-Animate-2",
  supportedHardware: ["h200", "b300", "rtx4090", "gb200", "h100"],
  hardware: [
    { id: "rtx4090", label: "RTX 4090", vram: "24GB", vendor: "consumer" },
  ],
  groupHardware: false,
  matchDims: [],
  overlayDims: [
    {
      id: "preset",
      title: "Deployment",
      default: "single",
      options: [
        { id: "single", label: "1 GPU", flags: ["--num-gpus 1"] },
        { id: "cfg2", label: "2 GPUs: CFG", showWhen: (s) => s.hw !== "rtx4090",
          flags: ["--num-gpus 2", "--enable-cfg-parallel"] },
        { id: "sp2", label: "2 GPUs: Ulysses", showWhen: (s) => s.hw !== "rtx4090",
          flags: ["--num-gpus 2", "--ulysses-degree 2"] },
        { id: "tp2", label: "2 GPUs: TP", showWhen: (s) => s.hw !== "rtx4090",
          flags: ["--num-gpus 2", "--tp-size 2"] },
        { id: "cfg-sp4", label: "4 GPUs: CFG x Ulysses", showWhen: (s) => s.hw !== "rtx4090",
          flags: ["--num-gpus 4", "--enable-cfg-parallel", "--ulysses-degree 2"] },
        { id: "tp-sp4", label: "4 GPUs: TP x Ulysses", showWhen: (s) => s.hw !== "rtx4090",
          flags: ["--num-gpus 4", "--tp-size 2", "--ulysses-degree 2"] },
        { id: "cfg-tp-sp8", label: "8 GPUs: CFG x TP x Ulysses", showWhen: (s) => s.hw !== "rtx4090",
          flags: ["--num-gpus 8", "--enable-cfg-parallel", "--tp-size 2", "--ulysses-degree 2"] },
      ],
    },
    {
      id: "placement",
      title: "Memory policy",
      default: "auto",
      options: [
        { id: "auto", label: "Auto", showWhen: (s) => s.hw !== "rtx4090" },
        { id: "speed", label: "Speed", showWhen: (s) => s.hw !== "rtx4090",
          flags: ["--performance-mode speed"] },
        { id: "offload", label: "DiT layerwise offload", showWhen: (s) => s.hw !== "rtx4090",
          flags: ["--dit-layerwise-offload", "--layerwise-offload-components transformer"] },
        { id: "memory", label: "Memory", flags: ["--performance-mode memory"] },
        {
          id: "cfg-regression",
          label: "CFG parity",
          showWhen: (s) => s.preset === "cfg2" && s.hw !== "rtx4090",
          disabled: (s) => s.preset !== "cfg2",
          disableReason: "This preset requires 2 GPUs with CFG parallelism.",
          env: ["SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=1"],
          flags: [
            "--dit-layerwise-offload", "--layerwise-offload-components transformer",
            "--encoder-parallel replicate", "--vae-config.use-parallel-decode false",
          ],
          hints: ["Replicated encoders and serial VAE decode; see the benchmark for output-parity evidence."],
        },
      ],
    },
    {
      id: "encoder",
      title: "Text encoder",
      showWhen: (s) => s.hw !== "rtx4090",
      default: "auto",
      options: [
        { id: "auto", label: "Auto" },
        { id: "offload", label: "CPU offload", showWhen: (s) => s.hw !== "rtx4090",
          flags: ["--text-encoder-cpu-offload"] },
      ],
    },
  ],
  cells: [
    { match: { hw: "h200" }, nnodes: 1,
      verificationStatus: (s) => s.preset === "single" && s.encoder === "auto"
        && ["auto", "memory"].includes(s.placement) ? "verified" : "unverified",
      flags: ["--model-path {{MODEL_NAME}}", "--port {{PORT}}"] },
    { match: { hw: "b300" }, nnodes: 1,
      verificationStatus: (s) => s.encoder === "auto" && (
        (s.preset === "single" && ["auto", "memory"].includes(s.placement))
        || (s.preset === "cfg2" && ["auto", "cfg-regression"].includes(s.placement))
      ) ? "verified" : "unverified",
      flags: ["--model-path {{MODEL_NAME}}", "--port {{PORT}}"] },
    { match: { hw: "rtx4090" }, nnodes: 1,
      verificationStatus: (s) => s.preset === "single" && s.placement === "memory"
        && s.encoder === "auto" ? "verified" : "unverified",
      env: ["PYTORCH_ALLOC_CONF=expandable_segments:True"],
      flags: ["--model-path {{MODEL_NAME}}", "--port {{PORT}}"],
      warn: "24 GB recipe: memory mode plus an expandable allocator. The 24-frame benchmark uses nearly all VRAM; longer clips or larger pixel budgets may OOM. Offload also needs sufficient host RAM." },
    { match: { hw: "gb200" }, nnodes: 1, verified: false,
      flags: ["--model-path {{MODEL_NAME}}", "--port {{PORT}}"] },
    { match: { hw: "h100" }, nnodes: 1, verified: false,
      flags: ["--model-path {{MODEL_NAME}}", "--port {{PORT}}"],
      warn: "Use DiT layerwise offload for the documented 80 GB memory preset. H100 recipes have not been measured in this cookbook." },
  ],
  modelNames: { default: "Wan-AI/Wan2.2-Animate-2-14B-Diffusers" },
  placeholders: {
    PORT: { target: "command", label: "Server port", default: "30010" },
    CURL_HOST: { target: "curl", label: "Server host", default: "127.0.0.1" },
    CURL_PORT: { target: "curl", label: "Request port", default: "30010" },
    INPUT_IMAGE: { target: "curl", label: "Reference PNG on the client", default: "/path/to/reference.png" },
    INPUT_VIDEO: { target: "curl", label: "Reference video on the server", default: "/path/to/reference_video.mp4" },
  },
  curl: `curl -sS -X POST http://{{CURL_HOST}}:{{CURL_PORT}}/v1/videos \\
  --form-string "prompt=a person dancing" \\
  --form-string "video_path={{INPUT_VIDEO}}" \\
  --form-string "clip_len=37" \\
  --form-string "size=640x800" \\
  --form-string "num_inference_steps=40" \\
  --form-string "guidance_scale=3.0" \\
  --form-string "fps=16" \\
  --form-string "seed=42" \\
  --form-string "enable_audio=false" \\
  --form "input_reference=@{{INPUT_IMAGE}};type=image/png"`,
  runModes: ["python"],
  showPlaygroundLink: false,
  github: { cookbookModel: "Wan-AI/Wan2.2-Animate-2-14B-Diffusers" },
};

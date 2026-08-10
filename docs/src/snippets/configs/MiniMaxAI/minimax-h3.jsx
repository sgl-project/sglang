// MiniMax-H3 diffusion deployment matrix. Consumed by _deployment.jsx.
//
// Checkpoint and request-mode choices stay in the picker because they determine
// the model partition and generated cURL. Orthogonal runtime features such as
// attention, quantization, and encoder scheduling live in the cookbook recipes
// instead of multiplying the deployment matrix.
// Hardware/profile cells remain deliberately small and carry an honest
// verification state for the exact platform, rather than inheriting a result
// measured on a different GPU.


export const config = {
  modelName: "MiniMax-H3",

  supportedHardware: [
    "b200",
    "b300",
    "h200",
    "h100",
    "mi300x",
    "mi355x",
    "rtx5090",
  ],
  hardware: [
    { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "consumer" },
  ],
  groupHardware: false,

  matchDims: [
    {
      id: "profile",
      title: "Deployment Profile",
      showWhen: (s) => ["b200", "b300", "h200", "h100"].includes(s.hw),
      options: [
        { id: "resident", label: "Resident" },
        {
          id: "fsdp",
          label: "FSDP sharded",
          showWhen: (s) =>
            ["b200", "b300", "h200", "h100"].includes(s.hw),
        },
        {
          id: "offload",
          label: "Layerwise offload",
          showWhen: (s) => s.hw === "rtx5090",
        },
        {
          id: "cross_node",
          label: "Cross-node (2 nodes)",
          showWhen: (s) => s.hw === "h200",
        },
      ],
    },
  ],

  overlayDims: [
    {
      id: "weights",
      title: "Checkpoint Weights",
      default: "fl2va",
      options: [
        {
          id: "fl2va",
          label: "FL2VA (First-and-Last-Frame-to-Video-and-Audio)",
          flags: ["--model-variant fl2va"],
        },
        {
          id: "ref2va",
          label: "Ref2VA (Reference-to-Video-and-Audio)",
          flags: ["--model-variant ref2va"],
        },
      ],
    },
    {
      id: "mode",
      title: "Request Mode",
      default: "t2va",
      options: [
        {
          id: "t2va",
          label: "Text only",
          showWhen: (s) => s.weights === "fl2va",
        },
        {
          id: "i2va",
          label: "First frame",
          showWhen: (s) => s.weights === "fl2va",
        },
        {
          id: "l2va",
          label: "Last frame",
          showWhen: (s) => s.weights === "fl2va",
        },
        {
          id: "fl2va",
          label: "First + last frames",
          showWhen: (s) => s.weights === "fl2va",
        },
        {
          id: "ref_image",
          label: "Image reference",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "ref_image_audio",
          label: "Image + audio",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "v2v",
          label: "Video reference",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "video_audio",
          label: "Video + soundtrack",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "audio_only",
          label: "Audio reference",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "mixed_ref",
          label: "Mixed references",
          showWhen: (s) => s.weights === "ref2va",
        },
      ],
    },
  ],

  modelNames: {
    default: "MiniMaxAI/MiniMax-H3",
  },

  placeholders: {
    HOST_IP: {
      target: "command",
      label: "Bind host",
      default: "0.0.0.0",
    },
    PORT: {
      target: "command",
      label: "Bind port",
      default: "30010",
    },
    HF_TOKEN: {
      target: "command",
      label: "HF token (Docker)",
      default: "<your-hf-token>",
    },
    MEDIA_DIR: {
      target: "command",
      label: "Host media directory (Docker)",
      default: "/data/minimax-h3",
    },
    CURL_HOST: {
      target: "curl",
      label: "Server host",
      default: "localhost",
    },
    CURL_PORT: {
      target: "curl",
      label: "Server port",
      default: "30010",
    },
    NUM_OUTPUTS: {
      target: "curl",
      label: "Outputs per prompt (1-10)",
      default: "1",
    },
    DURATION_SECONDS: {
      target: "curl",
      label: "Duration (seconds, 4-15)",
      default: "5",
    },
    FIRST_FRAME: {
      target: "curl",
      label: "FL2VA first frame URI",
      default: "file:///data/minimax-h3/first-frame.png",
    },
    LAST_FRAME: {
      target: "curl",
      label: "FL2VA last frame URI",
      default: "file:///data/minimax-h3/last-frame.png",
    },
    INPUT_VIDEO: {
      target: "curl",
      label: "First video URI",
      default: "file:///data/minimax-h3/video-1.mp4",
    },
    INPUT_VIDEO_START_SECONDS: {
      target: "curl",
      label: "First video start (seconds)",
      default: "0",
    },
    SECOND_INPUT_VIDEO: {
      target: "curl",
      label: "Second video URI (mixed ref)",
      default: "file:///data/minimax-h3/video-2.mp4",
    },
    SECOND_INPUT_VIDEO_START_SECONDS: {
      target: "curl",
      label: "Second video start (seconds)",
      default: "0",
    },
    REFERENCE_IMAGE: {
      target: "curl",
      label: "First reference image URI",
      default: "file:///data/minimax-h3/reference-1.png",
    },
    SECOND_REFERENCE_IMAGE: {
      target: "curl",
      label: "Second reference image URI",
      default: "file:///data/minimax-h3/reference-2.png",
    },
    REFERENCE_AUDIO: {
      target: "curl",
      label: "First reference audio URI",
      default: "file:///data/minimax-h3/reference-1.mp3",
    },
    SECOND_REFERENCE_AUDIO: {
      target: "curl",
      label: "Second reference audio URI",
      default: "file:///data/minimax-h3/reference-2.mp3",
    },
  },

  curl: (s) => {
    const request = {
      model: "{{MODEL_NAME}}",
      prompt:
        "Night-vision bedroom footage: while the owner sleeps, three cats burst in playing tiny brass instruments at full volume, freeze, then march out as if nothing happened.",
      seconds: "{{DURATION_SECONDS}}",
      task: "t2va",
      conditions: [],
      target: {
        short_edge: 768,
        aspect_ratio: "16:9",
        duration_seconds: "{{DURATION_SECONDS}}",
      },
      num_outputs_per_prompt: "{{NUM_OUTPUTS}}",
      num_inference_steps: 50,
      flow_shift: 12.0,
      audio_flow_shift: 3.0,
      seed: 1101,
    };
    const imageReference = (uri) => ({
      type: "image",
      uri,
      role: "reference",
    });
    const audioReference = (uri) => ({
      type: "audio",
      uri,
      role: "reference",
    });
    const videoReference = (uri, start, type = "video") => ({
      type,
      uri,
      role: "reference",
      start_time_seconds: start,
    });

    if (["i2va", "l2va", "fl2va"].includes(s.mode)) {
      request.task = "fl2va";
      request.prompt =
        "Continue naturally between the supplied endpoint frame or frames, with synchronized ambient sound.";
      request.target.aspect_ratio = "auto";
      request.seed = 2101;
      request.conditions = [];
      if (s.mode !== "l2va") {
        request.conditions.push({
          type: "image",
          uri: "{{FIRST_FRAME}}",
          role: "keyframe",
          frame_index: 0,
        });
      }
      if (s.mode !== "i2va") {
        request.conditions.push({
          type: "image",
          uri: "{{LAST_FRAME}}",
          role: "keyframe",
          frame_index: -1,
        });
      }
    } else if (s.mode === "ref_image") {
      request.task = "ref2va";
      request.prompt = "Use <Picture 1> as the visual subject and style reference.";
      request.target.aspect_ratio = "auto";
      request.conditions = [imageReference("{{REFERENCE_IMAGE}}")];
      request.seed = 3101;
    } else if (s.mode === "ref_image_audio") {
      request.task = "ref2va";
      request.prompt =
        "Use <Picture 1> as the visual subject and <Audio 1> as the sound reference.";
      request.target.aspect_ratio = "auto";
      request.conditions = [
        imageReference("{{REFERENCE_IMAGE}}"),
        audioReference("{{REFERENCE_AUDIO}}"),
      ];
      request.seed = 3102;
    } else if (s.mode === "v2v" || s.mode === "video_audio") {
      request.task = "ref2va";
      request.prompt =
        s.mode === "video_audio"
          ? "Follow <Video 1> and its required <Audio 1> soundtrack with coherent synchronized motion."
          : "Follow the appearance and motion of <Video 1>; use its soundtrack when present.";
      request.conditions = [
        videoReference(
          "{{INPUT_VIDEO}}",
          "{{INPUT_VIDEO_START_SECONDS}}",
          s.mode === "video_audio" ? "video_audio" : "video",
        ),
      ];
      request.seed = s.mode === "video_audio" ? 4102 : 4101;
    } else if (s.mode === "audio_only") {
      request.task = "ref2va";
      request.prompt = "Build a coherent visual scene around <Audio 1>.";
      request.conditions = [audioReference("{{REFERENCE_AUDIO}}")];
      request.seed = 3103;
    } else if (s.mode === "mixed_ref") {
      request.task = "ref2va";
      request.prompt =
        "Combine <Picture 1>, <Picture 2>, <Audio 1>, <Audio 2>, <Video 1>, and <Video 2> in their one-based modality order.";
      request.conditions = [
        imageReference("{{REFERENCE_IMAGE}}"),
        imageReference("{{SECOND_REFERENCE_IMAGE}}"),
        audioReference("{{REFERENCE_AUDIO}}"),
        audioReference("{{SECOND_REFERENCE_AUDIO}}"),
        videoReference("{{INPUT_VIDEO}}", "{{INPUT_VIDEO_START_SECONDS}}"),
        videoReference(
          "{{SECOND_INPUT_VIDEO}}",
          "{{SECOND_INPUT_VIDEO_START_SECONDS}}",
        ),
      ];
      request.seed = 3104;
    }

    const body = JSON.stringify(request, null, 2).replace(
      /"{{(NUM_OUTPUTS|DURATION_SECONDS|INPUT_VIDEO_START_SECONDS|SECOND_INPUT_VIDEO_START_SECONDS)}}"/g,
      "{{$1}}",
    );
    return `curl -sS -X POST http://{{CURL_HOST}}:{{CURL_PORT}}/v1/videos \\
  -H 'Content-Type: application/json' \\
  -d '${body}'`;
  },

  dockerMounts: ["{{MEDIA_DIR}}:/data/minimax-h3:ro"],

  dockerRunCommand: (s) =>
    ["mi300x", "mi355x"].includes(s.hw)
      ? `bash -lc 'python -m pip install -e "/sgl-workspace/sglang/python[diffusion_hip]" && exec sglang serve "$@"' --`
      : `bash -lc 'python -m pip install -e "/sgl-workspace/sglang/python[diffusion]" && exec sglang serve "$@"' --`,

  // Publish AMD Docker only after an H3-capable ROCm image has been validated.
  runModes: (s) =>
    ["mi300x", "mi355x"].includes(s.hw)
      ? ["python"]
      : ["python", "docker"],

  dockerImages: {
    b200: "lmsysorg/sglang:dev",
    b300: "lmsysorg/sglang:dev",
    h200: "lmsysorg/sglang:dev",
    h100: "lmsysorg/sglang:dev",
  },

  showPlaygroundLink: false,

  cells: [
    {
      match: { hw: "b200", profile: "resident" },
      nnodes: 1,
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--num-gpus 8",
        "--ulysses-degree 8",
        "--encoder-parallel auto",
        "--performance-mode speed",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", profile: "resident" },
      nnodes: 1,
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--num-gpus 8",
        "--ulysses-degree 8",
        "--encoder-parallel auto",
        "--performance-mode speed",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
      warn:
        "This is the B300 topology used for the documented benchmark sweep, not a claimed minimum GPU count.",
    },
    {
      match: { hw: "h200", profile: "resident" },
      nnodes: 1,
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--num-gpus 4",
        "--ulysses-degree 4",
        "--encoder-parallel auto",
        "--performance-mode speed",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", profile: "fsdp" },
      nnodes: 1,
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--num-gpus 8",
        "--ulysses-degree 8",
        "--encoder-parallel auto",
        "--performance-mode speed",
        "--use-fsdp-inference true",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
      warn:
        "FSDP reduces resident DiT memory but adds per-block parameter collectives. Prefer Resident when the full pipeline fits.",
    },
    {
      match: { hw: "h200", profile: "fsdp" },
      nnodes: 1,
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--num-gpus 4",
        "--ulysses-degree 4",
        "--encoder-parallel auto",
        "--performance-mode speed",
        "--use-fsdp-inference true",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
      warn:
        "FSDP reduces resident DiT memory but adds per-block parameter collectives. Prefer Resident when the full pipeline fits.",
    },
    {
      match: { hw: "h200", profile: "cross_node" },
      nnodes: 2,
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--num-gpus 16",
        "--sp-degree 16",
        "--ulysses-degree 8",
        "--ring-degree 2",
        "--encoder-parallel replicate",
        "--performance-mode speed",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
      warn:
        "Verified on 2 nodes of 8× H200 each (Ulysses8 within a node, Ring2 across nodes). Requires --encoder-parallel replicate: --encoder-parallel auto's fold decision is not yet node-boundary aware and will crash across nodes.",
    },
    {
      match: { hw: "b200", profile: "fsdp" },
      nnodes: 1,
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--num-gpus 4",
        "--ulysses-degree 4",
        "--encoder-parallel auto",
        "--performance-mode speed",
        "--use-fsdp-inference true",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
      warn:
        "The 4-GPU FSDP path is lossless but slower than the 8-GPU resident recipe.",
    },
    {
      match: { hw: "h100", profile: "resident" },
      nnodes: 1,
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--num-gpus 4",
        "--tp-size 2",
        "--ulysses-degree 2",
        "--encoder-parallel auto",
        "--performance-mode speed",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
      warn:
        "Fastest measured 4× H100 80 GB topology. TP4 + Ulysses1 lowers peak memory at a small latency cost.",
    },
    {
      match: { hw: "h100", profile: "fsdp" },
      nnodes: 1,
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--num-gpus 4",
        "--ulysses-degree 4",
        "--encoder-parallel auto",
        "--performance-mode speed",
        "--use-fsdp-inference true",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
      warn:
        "Capacity path on 4× H100 80 GB. Prefer the resident TP2 + Ulysses2 profile for latency.",
    },
    {
      match: { hw: "mi300x", profile: "resident" },
      nnodes: 1,
      verified: true,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--num-gpus 8",
        "--ulysses-degree 8",
        "--encoder-parallel auto",
        "--performance-mode speed",
        "--attention-backend aiter",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
      warn:
        "Validated on 1×, 2×, 4×, and 8× MI300X with BF16 and AITER packed attention. The picker emits the fastest measured 8-GPU topology; set --num-gpus and --ulysses-degree to the same lower count for a measured capacity recipe.",
    },
    {
      match: { hw: "mi355x", profile: "resident" },
      nnodes: 1,
      verified: true,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--num-gpus 8",
        "--ulysses-degree 8",
        "--encoder-parallel auto",
        "--performance-mode speed",
        "--attention-backend aiter",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
      warn:
        "Validated on 1×, 2×, 4×, and 8× MI355X with BF16 and AITER packed attention. The picker emits the fastest measured 8-GPU topology; set --num-gpus and --ulysses-degree to the same lower count for a measured capacity recipe.",
    },
    {
      match: { hw: "rtx5090", profile: "offload" },
      nnodes: 1,
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--num-gpus 2",
        "--tp-size 2",
        "--ulysses-degree 1",
        "--encoder-parallel auto",
        "--performance-mode memory",
        "--layerwise-offload-components dit,text_encoder,vae",
        "--dit-offload-prefetch-size 1",
        "--dit-layerwise-resident-layers 20",
        "--enable-torch-compile false",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
      warn:
        "Validated lossless BF16/FP32 recipe on 2× RTX 5090 (32 GB each) with a 384 GiB-class host. TP2 avoids the full per-rank DiT replication observed with Ulysses2 on PCIe.",
    },
  ],
};

export const featureGuide = {
  serveFeatures: [
    {
      id: "attention",
      title: "Attention backend",
      summary: "Change the attention kernel.",
      quality: { label: "Picker default", tone: "included" },
      scope: "Platform-dependent",
      defaultValue: "The picker keeps the platform default and selects AITER on verified AMD recipes.",
      contract: "FlashAttention keeps native input dtype; SageAttention quantizes attention math.",
      compatibility: "Use component overrides only for measured modules. SageAttention requires its packed-varlen dependency.",
      recipes: [
        {
          label: "Platform default",
          quality: { label: "Already handled", tone: "included" },
          description: "No action needed. Use the generated command for the production and consistency baseline.",
        },
        {
          label: "FlashAttention",
          quality: { label: "Native dtype", tone: "native" },
          description: "Use for an explicit kernel comparison; floating-point ordering can still differ.",
          code: "--attention-backend fa",
        },
        {
          label: "SageAttention",
          quality: { label: "Approximate", tone: "approximate" },
          description: "Quantized attention math; inspect both video and audio quality.",
          code: "--attention-backend sage_attn",
        },
      ],
    },
    {
      id: "quantization",
      title: "Online FP8",
      summary: "Use less GPU memory by quantizing transformer weights.",
      quality: { label: "Approximate", tone: "approximate" },
      scope: "B200 / B300 resident",
      defaultValue: "Native BF16/FP32 weights",
      contract: "Transformer linear weights are quantized; required patch, time, and output projections remain FP32.",
      compatibility: "Validated only on resident B200/B300. Combining it with Cache-DiT compounds approximations.",
      recipes: [
        {
          label: "Online FP8",
          quality: { label: "Approximate", tone: "approximate" },
          code: "--quantization fp8",
        },
        {
          label: "Protect extra layers",
          quality: { label: "Selective FP8", tone: "approximate" },
          code: "--quantization fp8 \\\n--quantization-ignored-layers blocks.0.attn token_refiner",
        },
      ],
    },
    {
      id: "encoder",
      title: "Encoder scheduling",
      summary: "Choose how the text encoder uses multiple GPUs.",
      quality: { label: "Included by picker", tone: "included" },
      scope: "Topology-dependent",
      defaultValue: "The picker emits auto on one host and replicate across nodes.",
      contract: "Fold and replicate preserve native weights; DP changes batching and is not bitwise-identical to fold.",
      compatibility: "DP requires TP1 and DiT DP1. Fold expects fast peer-to-peer access; cross-node H3 uses replicate.",
      recipes: [
        {
          label: "Single-node default",
          quality: { label: "Already included", tone: "included" },
          description: "Auto folds on verified P2P hosts, keeps pure-TP sharding, and avoids a costly fold on PCIe-only hosts.",
          code: "--encoder-parallel auto",
        },
        {
          label: "DP for a request batch",
          quality: { label: "Throughput", tone: "baseline" },
          code: "--encoder-parallel dp \\\n--batching-max-size 2",
        },
        {
          label: "Cross-node default",
          quality: { label: "Already included", tone: "included" },
          description: "The cross-node picker recipe sets this explicitly because auto is not node-boundary aware yet.",
          code: "--encoder-parallel replicate",
        },
      ],
    },
    {
      id: "graphs",
      title: "Graph execution",
      summary: "Reuse captured execution for repeated request shapes.",
      quality: { label: "Experimental", tone: "experimental" },
      scope: "B200 Ref2VA / H200",
      defaultValue: "Eager DiT",
      contract: "BCG preserves matching-signature eager output; torch.compile currently changes H3 numerical output.",
      compatibility: "BCG takes precedence over Cache-DiT and reserves capture memory. Compile is not a consistency mode.",
      recipes: [
        {
          label: "Breakable CUDA graph",
          quality: { label: "Matching signature", tone: "native" },
          code: "--enable-breakable-cuda-graph true \\\n--warmup-resolutions 1344x768 \\\n--bcg-text-buckets 5504",
        },
        {
          label: "torch.compile experiment",
          quality: { label: "Numerically different", tone: "experimental" },
          code: "--enable-torch-compile true",
        },
      ],
    },
  ],
  requestFeatures: [
    {
      id: "quality",
      title: "Quality level",
      summary: "Choose reference quality or audited Cache-DiT acceleration.",
      quality: { label: "Approximate option", tone: "approximate" },
      scope: "4× H200 audited",
      defaultValue: "quality: lossless",
      contract: "lossless is reference-exact; high measured 1.40× with SSIM 0.931 and PSNR 28.16 dB.",
      compatibility: "high is fail-closed to the audited workload and cannot use FSDP, DiT offload, or BCG.",
      recipes: [
        {
          label: "Reference path",
          quality: { label: "Lossless", tone: "native" },
          code: "\"quality\": \"lossless\"",
        },
        {
          label: "Audited acceleration",
          quality: { label: "1.40× measured", tone: "approximate" },
          code: "\"quality\": \"high\"",
        },
      ],
    },
    {
      id: "outputs",
      title: "Multiple outputs",
      summary: "Generate more than one result from a prompt.",
      quality: { label: "Independent variants", tone: "baseline" },
      scope: "2× RTX 5090 measured",
      defaultValue: "One output",
      contract: "Each output runs its own denoise and decode path; model math is unchanged.",
      compatibility: "The 32 GB offload recipe runs outputs sequentially to keep peak memory bounded.",
      recipes: [
        {
          label: "Two variants",
          quality: { label: "Request field", tone: "baseline" },
          code: "\"num_outputs_per_prompt\": 2",
        },
      ],
    },
  ],
};

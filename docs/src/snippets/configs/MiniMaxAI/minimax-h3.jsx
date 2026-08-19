// MiniMax-H3 diffusion deployment matrix. Consumed by _deployment.jsx.
//
// The mode, quantization, and encoder choices are deployment overlays because
// they do not change which base hardware topology fits. Request sampling
// controls remain in the generated cURL instead of being mixed into this
// deployment matrix.
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
    {
      id: "quant",
      title: "Online Quantization",
      default: "bf16",
      showWhen: (s) => ["b200", "b300"].includes(s.hw),
      options: [
        { id: "bf16", label: "Off — Native BF16/FP32" },
        {
          id: "fp8",
          label: "FP8 — Approximate",
          showWhen: (s) => ["b200", "b300"].includes(s.hw),
          disabled: (s) => s.profile !== "resident",
          disableReason:
            "The documented FP8 operating point keeps the transformer resident; FSDP combinations have not been validated.",
          flags: ["--quantization fp8"],
          hints: [
            "Online FP8 is approximate. Validate both video and audio quality;",
            "verified B200 and B300 runs reduced memory; re-benchmark latency on the target workload.",
          ],
        },
      ],
    },
    {
      id: "encoder",
      title: "Text Encoder Parallel",
      default: "auto",
      options: [
        {
          id: "auto",
          label: "Auto (recommended)",
          hints: [
            "Auto uses folding for the single-request recipes below and can",
            "select data parallel encoding for a compatible TP1 request batch.",
          ],
        },
        {
          id: "fold",
          label: "Fold (single-request)",
          flags: ["--encoder-parallel fold"],
          hints: [
            "Fold shards the resident Qwen3-VL encoder across the replica and is",
            "best suited to single-node GPUs with fast peer-to-peer links.",
          ],
        },
        {
          id: "dp",
          label: "DP (batched throughput)",
          disabled: (s) =>
            s.hw === "rtx5090" ||
            (s.hw === "h100" && s.profile === "resident"),
          disableReason:
            "Encoder DP requires TP1 and DiT DP1; this verified recipe uses TP2.",
          flags: [
            "--encoder-parallel dp",
            "--batching-max-size {{BATCHING_MAX_SIZE}}",
          ],
          hints: [
            "DP distributes a compatible multi-request text batch across ranks;",
            "it does not improve a batch of one and replicates encoder weights.",
          ],
        },
        {
          id: "replicate",
          label: "Replicate (compatibility)",
          flags: ["--encoder-parallel replicate"],
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
    BATCHING_MAX_SIZE: {
      target: "command",
      label: "Maximum request batch size",
      default: "2",
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

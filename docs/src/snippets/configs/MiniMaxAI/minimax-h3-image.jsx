export const config = (() => {
  // Derived recipes estimate memory, not latency. Default multi-GPU recipes
  // require NVLink or Infinity Fabric; RTX 3090 supports a two-card NVLink pair only.
  const hardware = [
    ["b300", "B300", 288, 1], ["gb300", "GB300", 288, 1],
    ["b200", "B200", 192, 1], ["gb200", "GB200", 192, 1],
    ["h200", "H200", 141, 1], ["h20-3e", "H20-3e", 141, 1],
    ["h100", "H100", 80, 1], ["h800", "H800", 80, 1],
    ["mi355x", "MI355X", 288, 1], ["mi350x", "MI350X", 288, 1],
    ["mi325x", "MI325X", 256, 1], ["mi300x", "MI300X", 192, 1],
    ["dgx-spark", "DGX Spark", 128, 1],
    ["rtxpro6000", "RTX PRO 6000 Blackwell", 96, 1],
    ["rtx6000ada", "RTX 6000 Ada", 48, 1],
    ["rtx5090", "RTX 5090", 32, 1],
    ["rtx4090", "RTX 4090", 24, 1], ["rtx3090", "RTX 3090", 24, 1],
    ["rtx5080", "RTX 5080", 16, 1], ["rtx5070ti", "RTX 5070 Ti", 16, 1],
    ["rtx4080", "RTX 4080", 16, 1], ["rtx4060ti", "RTX 4060 Ti", 16, 1],
    ["rtx5070", "RTX 5070", 12, 1], ["rtx4070", "RTX 4070", 12, 1],
    ["rtx3060", "RTX 3060", 12, 1],
  ];
  const profile = (hw) => hardware.find(([id]) => id === hw);
  // MI355X/MI350X share gfx950 and MI325X/MI300X share gfx942; each pair uses
  // the measured MI350X or MI325X recipe. The DiT runs dense AITER attention.
  const isAmd = (hw) => ["mi355x", "mi350x", "mi325x", "mi300x"].includes(hw);
  const maxRecipeGpus = (hw) => hw === "rtx3090" ? 2 : hw === "dgx-spark" || hw.startsWith("rtx") ? 1 : 8;
  // Whole-component placement must fit the encoder; layerwise placement streams
  // its blocks and keeps the native BF16 checkpoint usable on smaller cards.
  const encoderFits = (hw, gpus) => (profile(hw)?.[2] || 0) * gpus >= 48.1;
  const derivedPlan = (hw, gpus) => {
    const [, , vram, minimum] = profile(hw) || [hw, hw, 0, 1];
    if (hw === "dgx-spark") return { tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "auto" };
    if (!encoderFits(hw, gpus)) return { tp_size: 1, ulysses_degree: gpus, ring_degree: 1, placement: "layerwise" };
    // Conservative working-set estimate from the measured H200 load/runtime
    // peaks: sharded DiT, folded encoder, VAE/activations and allocator margin.
    const residentTp = [1, 2, 4, 8].find((tp) => tp >= minimum && gpus % tp === 0
      && 42 / tp + 54 / gpus + 8 <= vram);
    const tp = residentTp || gpus;
    return { tp_size: tp, ulysses_degree: Math.max(1, gpus / tp), ring_degree: 1,
      placement: vram >= 130 ? "auto" : residentTp ? "resident" : "offload" };
  };
  const derivedRecipes = hardware.flatMap(([hw, , , minimum]) => hw === "dgx-spark" ? []
    : [1, 2, 4, 8].filter((gpus) => gpus <= maxRecipeGpus(hw)
      && (["h200", "b300"].includes(hw) ? gpus > 2 : !isAmd(hw) && hw !== "rtxpro6000")).map((gpus) => ({
        id: `${hw}-${gpus}-derived`, hw, nodes: 1, gpus_per_node: gpus,
        ...derivedPlan(hw, gpus), encoder: gpus > 1 ? "fold" : "auto",
        adaln: ["h200", "b300", "rtxpro6000"].includes(hw) ? "auto" : "online",
        default: !["h200", "b300"].includes(hw) && gpus === minimum, unverified: true,
      })));
  const uncachedServer = (s) => s.hw === "h200" && Number(s.nodes) === 1
    && Number(s.gpus_per_node) === 1 && s.placement === "auto" && s.encoder === "auto"
    && s.adaln === "auto" && s.warmup === "server";
  const uncachedWorkload = (s) => uncachedServer(s) && s.mode === "edit"
    && s.size === "2048x2048" && s.references === "one" && s.quality === "lossless";
  const guide4kWorkload = (s) => (["h200", "b300", "rtxpro6000", "dgx-spark"].includes(s.hw) || isAmd(s.hw)) && (
    (s.mode === "t2i" && ["4736x3552", "6272x2688"].includes(s.size))
    || (s.mode === "edit" && s.size === "4096x4096" && s.references === "one")
  );
  const localVaeServer = (s) => ["h200", "b300"].includes(s.hw) && Number(s.nodes) === 1
    && Number(s.gpus_per_node) === 2 && s.encoder === "fold"
    && s.placement === "auto" && s.adaln === "auto" && s.warmup === "server"
    && s.conditioning === "on" && (s.topology_mode !== "manual"
      || (Number(s.tp_size) === 1 && Number(s.ulysses_degree) === 2 && Number(s.ring_degree) === 1));
  const localVaeWorkload = (s) => localVaeServer(s) && guide4kWorkload(s);
  const uncompressedPngWorkload = (s) => localVaeWorkload(s)
    && s.vae === "auto" && s.quality === "lossless";
  const fileUrlServer = (s) => s.hw === "b300" && localVaeServer(s) && s.vae === "auto";
  const fileUrlWorkload = (s) => fileUrlServer(s) && guide4kWorkload(s) && s.quality === "lossless";
  const requestPrompt = (s) => s.mode === "edit"
    ? s.references === "two"
      ? "Keep the scene from <Picture 1> and use the color palette of <Picture 2>."
      : "Keep the scene and change the sky to a warm sunset."
    : "A quiet London street after rain, warm shop lights reflected on the pavement, detailed photograph";
  const config = {
  modelName: "MiniMax-H3-Image",
  runModes: ["python"],
  supportedHardware: hardware.map(([id]) => id),
  hardware: hardware.map(([id, label, vram]) => ({ id, label,
    vram: `${vram}GB${id === "dgx-spark" ? " unified" : ""}`,
    vendor: isAmd(id) ? "amd" : id.startsWith("rtx") ? "consumer" : ["h200", "h100", "h800", "h20-3e"].includes(id) ? "hopper" : "blackwell" })),
  groupHardware: false,
  matchDims: [],
  overlayDims: [
    {
      id: "mode", title: "Task", scope: "base", default: "t2i",
      description: "One checkpoint serves text-to-image and ordered image-reference editing.",
      options: [{ id: "t2i", label: "Text to image" }, { id: "edit", label: "Image editing" }],
    },
    {
      id: "delivery", title: "Image delivery", scope: "base", default: "b64_json",
      description: "Base64 uses temporary files. File URL retains images in the server output directory and downloads them separately. Switching requires a server restart. URLs last for the server process; manage the retained files.",
      options: [
        { id: "b64_json", label: "Base64", recommended: true },
        { id: "url", label: "File URL", soft: (s) => !fileUrlWorkload(s), softReason: "File URL delivery has been measured only with the documented B300 Ulysses2 lossless 4K workloads." },
      ],
    },
    {
      id: "placement", title: "Placement", scope: "serve", default: "auto",
      description: "Resident keeps weights on the GPU. Component offload transfers whole modules; layerwise offload streams blocks for smaller GPUs and needs host RAM.",
      options: [
        { id: "auto", label: "Auto", recommended: true },
        { id: "resident", label: "Resident", flags: ["--component-residency all=resident"] },
        { id: "offload", label: "Component offload", flags: ["--component-residency all=component-offload"] },
        { id: "layerwise", label: "Layerwise offload", flags: ["--component-residency all=layerwise-offload", "--layerwise-prefetch-size transformer=1,text_encoder=1,video_vae=1"] },
      ],
    },
    {
      id: "quantization", title: "Weight quantization", scope: "serve", default: "native",
      description: "Native keeps release precision. Online FP8 and MXFP8 reduce memory and can accelerate inference; image details and reference identity can change. AdaLN projections and the VAE retain native precision. Restart to change this setting.",
      options: [
        { id: "native", label: "Native BF16", recommended: true },
        { id: "fp8-dit", label: "FP8 DiT", flags: ["--quantization fp8"] },
        { id: "mxfp8-dit", label: "MXFP8 DiT", flags: ["--quantization mxfp8"] },
        { id: "fp8-all", label: "FP8 DiT + encoder", flags: ["--quantization fp8", "--component-quantizations.text_encoder fp8"] },
        { id: "mxfp8-all", label: "MXFP8 DiT + encoder", flags: ["--quantization mxfp8", "--component-quantizations.text_encoder mxfp8"] },
      ],
    },
    {
      id: "encoder", title: "Encoder parallelism", scope: "serve", default: "auto",
      description: "Auto uses the runtime policy. Fold shards the large text encoder across the GPUs serving a request.",
      options: [
        { id: "auto", label: "Auto" },
        { id: "fold", label: "Fold", flags: ["--encoder-parallel fold"] },
      ],
    },
    {
      id: "adaln", title: "AdaLN projections", scope: "serve", default: "auto",
      description: "Auto precomputes timestep projections on H200, B300, GB10 and RTX PRO 6000 Blackwell. This releases about 24.6–24.7 GiB on one GPU. New plans read the checkpoint; resident GB10 requires precomputation.",
      options: [
        { id: "auto", label: "Auto", recommended: true },
        { id: "online", label: "Precompute", flags: ["--minimax-h3-adaln-online true"] },
        { id: "resident", label: "Keep weights", flags: ["--minimax-h3-adaln-online false"] },
      ],
    },
    {
      id: "execution", title: "Execution", scope: "serve", default: "eager",
      description: "Graphs capture the selected request shape at startup and replay matching inputs; other shapes run eager. Changing task, size or reference layout requires a restart to capture it. Regional compile adds startup work and can change floating-point rounding.",
      options: [
        { id: "eager", label: "Eager", recommended: true },
        { id: "bcg", label: "Breakable CUDA graph" },
        { id: "compile", label: "Regional compile", flags: ["--enable-torch-compile true", "--regional-compile true"] },
      ],
    },
    {
      id: "warmup", title: "Startup warmup", scope: "serve", default: "server",
      description: "Prepare both task grids at startup. Skipping warmup starts serving sooner but moves initialization into the first request.",
      options: [
        { id: "server", label: "Prepare at startup", recommended: true },
        { id: "off", label: "Skip", flags: ["--warmup-mode off"] },
      ],
    },
    {
      id: "conditioning", title: "Conditioning reuse", scope: "serve", default: "on",
      description: "Reuse exact text conditioning and VAE moments for matching inputs. Disable it to measure uncached requests; sampling still uses the request's generator.",
      options: [
        { id: "on", label: "Exact reuse", recommended: true },
        { id: "off", label: "Disabled", flags: ["--disable-conditioning-cache"], soft: (s) => !uncachedWorkload(s), softReason: "This uncached server recipe has not been validated with the selected workload." },
      ],
    },
    {
      id: "vae", title: "VAE decode", scope: "serve", default: "auto",
      description: "Auto shares complete decode tiles across GPUs for lossless requests. Local runs all tiles on each GPU. Exact retains the release scheduling.",
      options: [
        { id: "auto", label: "Auto", recommended: true },
        { id: "local", label: "Local tiles", flags: ["--vae-config.use-parallel-tiling false"], soft: (s) => !localVaeWorkload(s), softReason: "Local decoding has not been validated with this server and workload combination." },
      ],
    },
    {
      id: "quality", title: "Quality", scope: "request", default: "lossless",
      description: "Lossless enables native fusion and batched VAE tiles, shared across GPUs when available. Exact retains the release arithmetic and tile scheduling; both use six steps.",
      options: [{ id: "lossless", label: "Lossless", recommended: true }, { id: "exact", label: "Exact" }],
    },
    {
      id: "size", title: "Output size", scope: "request", default: "2048x2048",
      description: "Smaller outputs still use the default 2048² minimum generation canvas, then resize. Size does not change the six-step schedule.",
      options: [
        { id: "2048x2048", label: "2048 × 2048" },
        { id: "1024x1024", label: "1024 × 1024" },
        ...["4096x4096", "4736x3552", "3552x4736", "5024x3360", "3360x5024", "5472x3072", "3072x5472", "6272x2688"].map((id) => ({
          id, label: id.replace("x", " × "), soft: (s) => !guide4kWorkload({ ...s, size: id }),
          softReason: "This size has not been measured with the selected hardware, task and reference count.",
        })),
      ],
    },
    {
      id: "png", title: "PNG compression", scope: "request", default: "default",
      description: "Default produces smaller PNGs. No compression can shorten encoding and saving on a fast connection, with larger responses; pixels are unchanged.",
      options: [
        { id: "default", label: "Default", recommended: true },
        { id: "none", label: "No compression", soft: (s) => !uncompressedPngWorkload(s), softReason: "Uncompressed PNG has been measured only with the documented H200/B300 Ulysses2 4K workloads." },
      ],
    },
    {
      id: "seed", title: "Seed", scope: "request", kind: "number", default: 42,
      min: 0, max: 4294967295, step: 1, unit: "",
      description: "Use the same seed, inputs, checkpoint and server settings to reproduce a request. The generator stays on the CPU.",
    },
    {
      id: "references", title: "Reference images", scope: "request", default: "one",
      showWhen: (s) => s.mode === "edit",
      description: "Upload references in prompt order: <Picture 1>, then <Picture 2>. Each file is encoded separately.",
      options: [{ id: "one", label: "One image" }, { id: "two", label: "Two images" }],
    },
  ],
  commandBuilder: {
    defaultSelection: { hw: "b300", nodes: 1, gpus_per_node: 1, topology_mode: "auto", tp_size: 1, ulysses_degree: 1, ring_degree: 1 },
    resource: {
      limits: {
        nodes: { min: 1, max: 1 },
        gpus_per_node: { min: 1, max: 8, disabledWhen: (s) => maxRecipeGpus(s.hw) === 1, disabledValue: 1 },
      },
      verifiedRecipes: [
        { id: "spark-single", hw: "dgx-spark", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "auto", encoder: "auto", adaln: "auto", default: true },
        { id: "b300-single", hw: "b300", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "auto", encoder: "auto" },
        { id: "b300-single-resident", hw: "b300", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "resident", encoder: "auto" },
        { id: "b300-single-layerwise", hw: "b300", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "layerwise", encoder: "auto", adaln: "online" },
        ...[[2, 1, 1], [1, 2, 1], [1, 1, 2]].flatMap(([tp_size, ulysses_degree, ring_degree]) => ["auto", "resident"].map((placement) => ({ id: `b300-two-${tp_size}-${ulysses_degree}-${ring_degree}-${placement}`, hw: "b300", nodes: 1, gpus_per_node: 2, tp_size, ulysses_degree, ring_degree, placement, encoder: "fold" }))),
        ...["auto", "resident", "offload"].map((placement) => ({ id: `h200-single-${placement}`, hw: "h200", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement, encoder: "auto" })),
        ...[[2, 1, 1], [1, 2, 1], [1, 1, 2]].flatMap(([tp_size, ulysses_degree, ring_degree]) => ["auto", "resident"].map((placement) => ({ id: `h200-two-${tp_size}-${ulysses_degree}-${ring_degree}-${placement}`, hw: "h200", nodes: 1, gpus_per_node: 2, tp_size, ulysses_degree, ring_degree, placement, encoder: "fold" }))),
        ...["mi355x", "mi350x", "mi325x", "mi300x"].flatMap((hw) => [[1, 1, 1, 1], [2, 2, 1, 1], [2, 1, 2, 1], [4, 4, 1, 1], [4, 1, 4, 1], [8, 1, 8, 1], [8, 8, 1, 1]]
          .map(([gpus_per_node, tp_size, ulysses_degree, ring_degree]) => ({ id: `${hw}-${gpus_per_node}-${tp_size}-${ulysses_degree}-${ring_degree}`, hw, nodes: 1, gpus_per_node, tp_size, ulysses_degree, ring_degree, placement: "auto", encoder: "auto", default: gpus_per_node === 8 && ulysses_degree === 8 }))),
        ...["resident", "offload", "layerwise"].map((placement) => ({ id: `rtxpro6000-single-${placement}`, hw: "rtxpro6000", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement, encoder: "auto", adaln: placement === "layerwise" ? "online" : "auto", default: placement === "resident" })),
        ...derivedRecipes,
      ].map((recipe) => ({
        adaln: "auto",
        conditioning: "on",
        vae: "auto",
        ...recipe,
        verifiedWhen: (s) => config.commandBuilder.resolveDeployment({
          ...s, ...recipe, topology_mode: "manual", adaln: recipe.adaln || "auto", warmup: "server", conditioning: "on", vae: "auto",
        }).verified,
      })),
      recommendedRecipe: (s) => {
        const t = config.commandBuilder.resource.autoTopology(s);
        const recipes = config.commandBuilder.resource.verifiedRecipes;
        return recipes.find((r) =>
          r.hw === s.hw && r.nodes === Number(s.nodes) && r.gpus_per_node === Number(s.gpus_per_node)
          && r.tp_size === t.tp_size && r.ulysses_degree === t.ulysses_degree && r.ring_degree === t.ring_degree
        );
      },
      autoTopology: (s) => {
        const { tp_size, ulysses_degree, ring_degree } = derivedPlan(s.hw, Number(s.gpus_per_node));
        return { tp_size, ulysses_degree, ring_degree };
      },
      validateTopology: (s, t) => {
        const degrees = [t.tp_size, t.ulysses_degree, t.ring_degree];
        const errors = [];
        if (!profile(s.hw)) errors.push("Choose a listed GPU.");
        if (s.hw === "dgx-spark" && Number(s.gpus_per_node) !== 1) errors.push("This DGX Spark recipe uses one node with one GB10 GPU.");
        if (s.hw === "dgx-spark" && s.adaln === "resident" && ["auto", "resident"].includes(s.placement)) errors.push("Resident GB10 requires native AdaLN precomputation. Choose Auto or Precompute.");
        if (isAmd(s.hw) && t.ring_degree > 1) errors.push("Ring parallelism requires FlashAttention. On AMD, use Ulysses or TP.");
        if (Number(s.gpus_per_node) > maxRecipeGpus(s.hw)) errors.push("Default multi-GPU recipes require NVLink. This GPU is limited to a single-card recipe, or a two-card NVLink pair for RTX 3090.");
        if (s.quantization?.startsWith("mxfp8") && ["rtx6000ada", "rtx4090", "rtx4080", "rtx4060ti", "rtx4070", "rtx3090", "rtx3060"].includes(s.hw)) errors.push("Choose FP8 or Native BF16 on this GPU; MXFP8 requires Hopper or newer.");
        if (s.execution === "bcg" && s.warmup === "off") errors.push("Choose Prepare at startup to capture CUDA graphs.");
        if (profile(s.hw) && !encoderFits(s.hw, Number(s.gpus_per_node))) {
          if (s.placement !== "layerwise") errors.push("Choose Layerwise offload for this GPU capacity, or apply the derived recipe.");
          if (s.adaln !== "online") errors.push("Choose AdaLN Precompute for the small-GPU recipe.");
        }
        if (!degrees.every((n) => Number.isInteger(n) && n > 0)) errors.push("Parallel degrees must be positive integers.");
        if (Number(s.nodes) !== 1 || Number(s.gpus_per_node) !== degrees.reduce((a, b) => a * b, 1)) errors.push("Use one node with GPU count equal to TP × Ulysses × Ring.");
        if (![1, 2, 4, 8].includes(t.tp_size)) errors.push("TP must divide both the DiT and text-encoder head counts: choose 1, 2, 4, or 8.");
        if (56 % (t.tp_size * t.ulysses_degree)) errors.push("TP × Ulysses must divide the 56 DiT attention heads.");
        return errors;
      },
    },
    resolveDeployment: (s) => {
      const resource = config.commandBuilder.resource;
      const topology = s.topology_mode === "manual"
        ? { tp_size: Number(s.tp_size), ulysses_degree: Number(s.ulysses_degree), ring_degree: Number(s.ring_degree) }
        : resource.autoTopology(s);
      const recipe = resource.verifiedRecipes.find((r) => r.hw === s.hw && r.gpus_per_node === Number(s.gpus_per_node) && r.tp_size === topology.tp_size && r.ulysses_degree === topology.ulysses_degree && r.ring_degree === topology.ring_degree && r.placement === s.placement && r.encoder === s.encoder);
      const suggestedRecipe = resource.recommendedRecipe(s);
      const derivedRecipe = suggestedRecipe?.unverified ? suggestedRecipe : undefined;
      const errors = resource.validateTopology(s, topology);
      // AMD Auto keeps the AdaLN weights; its rows measured lossless one-reference requests.
      const measuredAmd = isAmd(s.hw) && s.quality === "lossless" && s.adaln !== "online" && s.warmup === "server"
        && (guide4kWorkload(s) || (s.size === "2048x2048" && (s.mode === "t2i" || s.references === "one")));
      const measured2k = !isAmd(s.hw) && s.size === "2048x2048" && (Number(s.gpus_per_node) === 1 || s.quality === "lossless")
        && !(s.mode === "edit" && s.references === "two" && s.quality === "exact");
      const measured4k = !isAmd(s.hw) && guide4kWorkload(s) && s.adaln !== "resident" && s.warmup === "server"
        && (s.hw === "h200" ? s.placement === "auto"
          : s.hw === "b300" ? Number(s.gpus_per_node) <= 2
          : Number(s.gpus_per_node) === 1);
      const serveVerified = !!recipe && !recipe.unverified && !errors.length
        && ((!s.quantization || s.quantization === "native") || (s.adaln !== "resident" && (
          (Number(s.gpus_per_node) === 1 && (
            (s.hw === "b300" && (s.placement === "resident" || (s.placement === "layerwise" && s.quantization.endsWith("-all"))))
            || (s.hw === "rtxpro6000" && s.placement === "resident" && s.quantization.endsWith("-all"))
            || (s.hw === "dgx-spark" && s.placement === "auto" && s.quantization === "mxfp8-all")))
          || (s.hw === "b300" && Number(s.gpus_per_node) === 2 && s.placement === "resident"
            && s.encoder === "fold" && topology.tp_size === 1 && topology.ulysses_degree === 2
            && topology.ring_degree === 1 && s.quantization === "mxfp8-all"))))
        && ((!s.execution || s.execution === "eager") || (s.hw === "b300" && Number(s.gpus_per_node) === 1 && s.adaln !== "resident" && (
          (s.execution === "bcg" && ((s.placement === "resident" && (!s.quantization || ["native", "mxfp8-all"].includes(s.quantization)))
            || (s.placement === "layerwise" && (!s.quantization || s.quantization === "native"))))
          || (s.execution === "compile" && s.placement === "resident" && (!s.quantization || s.quantization === "native")))))
        && (s.placement !== "layerwise" || s.adaln === "online")
        && (s.delivery !== "url" || fileUrlServer(s))
        && (s.conditioning !== "off" || uncachedServer(s))
        && (s.vae !== "local" || localVaeServer(s))
        && (s.hw !== "rtxpro6000" || s.adaln !== "resident")
        && (s.warmup !== "off" || (s.hw === "h200" && Number(s.gpus_per_node) === 1));
      const verified = serveVerified && (s.conditioning === "off" ? uncachedWorkload(s)
        : s.vae === "local" ? localVaeWorkload(s) : measured2k || measured4k || measuredAmd)
        && (s.png !== "none" || uncompressedPngWorkload(s))
        && (s.delivery !== "url" || fileUrlWorkload(s));
      const flags = ["--model-path {{MODEL_NAME}}", "--backend sglang"];
      if (Number(s.gpus_per_node) > 1) flags.push(`--num-gpus ${s.gpus_per_node}`, `--tp-size ${topology.tp_size}`, `--ulysses-degree ${topology.ulysses_degree}`, `--ring-degree ${topology.ring_degree}`);
      flags.push("--host {{HOST_IP}}", "--port {{PORT}}");
      if (s.execution === "bcg") {
        const [width, height] = s.size.split("x").map(Number);
        const warmup = { prompt: requestPrompt(s), width, height, quality: "lossless" };
        if (s.mode === "edit") warmup.image_path = s.references === "two"
          ? ["{{WARMUP_IMAGE_PATH}}", "{{WARMUP_IMAGE_PATH_2}}"] : ["{{WARMUP_IMAGE_PATH}}"];
        flags.push("--enable-breakable-cuda-graph true", `--warmup-resolutions ${s.size}`,
          "--warmup-steps 6", "--bcg-text-buckets 256", `--warmup-sampling-params '${JSON.stringify(warmup)}'`);
      }
      if (s.delivery === "url") flags.push("--output-path {{OUTPUT_DIR}}");
      return {
        match: { hw: s.hw }, nnodes: 1, verified, flags,
        builder: {
          topology, errors,
          topologySummary: `TP ${topology.tp_size}, Ulysses ${topology.ulysses_degree}, Ring ${topology.ring_degree}`,
          resolvedSettings: {
            placement: s.placement === "auto" ? (s.hw === "dgx-spark" ? "Resident (≥106 GiB free)" : (profile(s.hw)?.[2] || 0) >= 130 ? "Resident" : "Component offload") : undefined,
            adaln: s.adaln === "resident" || (s.adaln === "auto" && !["b300", "h200", "rtxpro6000", "dgx-spark"].includes(s.hw)) ? "Keep weights" : "Precompute",
            attention: isAmd(s.hw) ? "AITER (auto)" : undefined,
            vae: s.vae === "local" || Number(s.gpus_per_node) === 1 ? "Local tiles" : "Parallel tiles for lossless",
          },
          warnings: s.hw === "dgx-spark"
            ? ["One GB10 shares its 128 GB pool with the CPU. The verified Auto recipe needs at least 106 GiB available before startup; component offload does not add capacity.",
              ...(!verified ? ["This Spark option and workload combination has not been verified."] : []),
            ]
            : verified ? [] : [
              "This exact hardware, topology, placement and workload recipe is not yet verified.",
              ...(derivedRecipe ? ["Derived from checkpoint sizes and measured native placement paths. Start at 2048 × 2048; this GPU's latency is unmeasured.",
                ...(["offload", "layerwise"].includes(derivedRecipe.placement) ? ["Plan for 256 GB host RAM and local NVMe. Layerwise offload reduces GPU memory; host-to-GPU transfers can dominate latency."] : []),
              ] : []),
            ],
          verification: { serve: errors.length ? "error" : serveVerified ? "verified" : "unverified", request: errors.length ? "error" : verified ? "verified" : "unverified" },
        },
      };
    },
  },
  modelNames: { default: "MiniMaxAI/MiniMax-H3-Image" },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
    OUTPUT_DIR: { target: "command", label: "Saved image directory", default: "./outputs/h3-image", showWhen: (s) => s.delivery === "url" },
    IMAGE_PATH: { target: "curl", label: "Reference image", default: "/path/to/reference.png", showWhen: (s) => s.mode === "edit" },
    IMAGE_PATH_2: { target: "curl", label: "Second reference image", default: "/path/to/reference-2.png", showWhen: (s) => s.mode === "edit" && s.references === "two" },
    WARMUP_IMAGE_PATH: { target: "command", label: "Warmup reference on server", default: "/path/to/reference.png", showWhen: (s) => s.execution === "bcg" && s.mode === "edit" },
    WARMUP_IMAGE_PATH_2: { target: "command", label: "Second warmup reference on server", default: "/path/to/reference-2.png", showWhen: (s) => s.execution === "bcg" && s.mode === "edit" && s.references === "two" },
  },
  curl: (s) => {
    const save = s.delivery === "url"
      ? " | python3 -c 'import json,sys,urllib.parse,urllib.request; urllib.request.urlretrieve(urllib.parse.urljoin(\"http://{{CURL_HOST}}:{{CURL_PORT}}\",json.load(sys.stdin)[\"data\"][0][\"url\"]),\"generated.png\")'"
      : " | python3 -c 'import base64,json,sys; open(\"generated.png\",\"wb\").write(base64.b64decode(json.load(sys.stdin)[\"data\"][0][\"b64_json\"]))'";
    const references = s.references === "two" ? " \\\n  -F 'image=@{{IMAGE_PATH_2}}'" : "";
    const prompt = requestPrompt(s);
    const compression = s.png === "none" ? " -F 'output_compression=0'" : "";
    if (s.mode === "edit") return `curl --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/edits \\
  -F 'model={{MODEL_NAME}}' \\
  -F 'image=@{{IMAGE_PATH}}'${references} \\
  -F 'prompt=${prompt}' \\
  -F 'size=${s.size}' -F 'quality=${s.quality}' -F 'seed=${s.seed}' -F 'generator_device=cpu' \\
  -F 'response_format=${s.delivery}' -F 'output_format=png'${compression}${save}`;
    return `curl --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify({ model: "{{MODEL_NAME}}", prompt, size: s.size, quality: s.quality, seed: Number(s.seed), generator_device: "cpu", response_format: s.delivery, output_format: "png", output_compression: s.png === "none" ? 0 : undefined }, null, 2)}'${save}`;
  },
  cells: [],
  };
  return config;
})();

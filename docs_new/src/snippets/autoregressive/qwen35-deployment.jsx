export const Qwen35Deployment = () => {
  // Qwen3.5 Configuration Generator
  //
  // MoE models (Gated Delta Networks + sparse MoE, hybrid architecture):
  //   397B-A17B, 122B-A10B, 35B-A3B
  //
  // Dense models (standard transformer):
  //   27B, 9B, 4B, 2B, 0.8B
  //
  // GPU requirements (BF16):
  //   397B-A17B: H100 tp=16 (2 nodes), H200 tp=8, B200 tp=8, B300 tp=8, MI300X tp=8, MI325X tp=4, MI355X tp=4
  //   122B-A10B: H100 tp=4,  H200 tp=4, B200 tp=2, B300 tp=2, MI300X tp=2, MI325X tp=1, MI355X tp=1
  //   35B-A3B:   H100 tp=1 (tp=2 w/ MTP), H200 tp=1, B200 tp=1, B300 tp=1, MI300X tp=1, MI325X tp=1, MI355X tp=1
  //   27B:       H100 tp=1 (tp=2 w/ MTP); tp=1 on all other hardware
  //   9B/4B/2B/0.8B: tp=1 on all hardware (including MI300X, MI325X, MI355X)
  //
  // GPU requirements (FP8, where available):
  //   397B-A17B: H100 tp=8, H200 tp=8 ep=8, B200 tp=4, B300 tp=4, MI300X tp=4, MI325X tp=2, MI355X tp=2
  //   122B-A10B: H100 tp=2 (tp=4 w/ MTP), H200 tp=2, B200 tp=1, B300 tp=1, MI300X tp=1, MI325X tp=1, MI355X tp=1
  //   35B-A3B:   H100 tp=1, H200 tp=1, B200 tp=1, B300 tp=1, MI300X tp=1, MI325X tp=1, MI355X tp=1
  //   27B:       tp=1 on all hardware (including MI300X, MI325X, MI355X)
  //
  // FP4 (397B only): NVFP4 on Blackwell B200/B300 tp=4; AMD MXFP4 on MI355X tp=2

  const MOE_MODELS = new Set(['397b', '122b', '35b']);
  const FP8_MODELS = new Set(['397b', '122b', '35b', '27b']);

  // Maps model id -> HuggingFace model name suffix
  const MODEL_SUFFIX = {
    '397b': '397B-A17B',
    '122b': '122B-A10B',
    '35b':  '35B-A3B',
    '27b':  '27B',
    '9b':   '9B',
    '4b':   '4B',
    '2b':   '2B',
    '0.8b': '0.8B',
  };

  const options = {
    model: {
      name: 'model',
      title: 'Model Variant',
      items: [
        { id: '397b',  label: '397B', subtitle: 'MoE', default: true  },
        { id: '122b',  label: '122B', subtitle: 'MoE', default: false },
        { id: '35b',   label: '35B',  subtitle: 'MoE', default: false },
        { id: '27b',   label: '27B',  subtitle: 'Dense', default: false },
        { id: '9b',    label: '9B',   subtitle: 'Dense', default: false },
        { id: '4b',    label: '4B',   subtitle: 'Dense', default: false },
        { id: '2b',    label: '2B',   subtitle: 'Dense', default: false },
        { id: '0.8b',  label: '0.8B', subtitle: 'Dense', default: false },
      ]
    },
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      getDynamicItems: (values) => {
        const isNvfp4 = values.quantization === 'fp4';
        return [
          { id: 'h100',   label: 'H100',   default: !isNvfp4, disabled: isNvfp4 },
          { id: 'h200',   label: 'H200',   default: false,     disabled: isNvfp4 },
          { id: 'b200',   label: 'B200',   default: false,     disabled: false },
          { id: 'b300',   label: 'B300',   default: isNvfp4,   disabled: false },
          { id: 'mi300x', label: 'MI300X', default: false,     disabled: isNvfp4 },
          { id: 'mi325x', label: 'MI325X', default: false,     disabled: isNvfp4 },
          { id: 'mi355x', label: 'MI355X', default: false,     disabled: false },
          { id: 'xeon',   label: 'XEON',   default: false,     disabled: isNvfp4 }
        ];
      }
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      getDynamicItems: (values) => {
        const hasFp8 = FP8_MODELS.has(values.model);
        const hasFp4 = values.model === '397b';
        const isXeon = values.hardware === 'xeon';
        return [
          { id: 'bf16', label: 'BF16', default: !hasFp8 || isXeon },
          { id: 'fp8',  label: 'FP8',  default: hasFp8 && !isXeon, disabled: !hasFp8,
            disabledReason: 'No FP8 variant available for this model' },
          { id: 'fp4',  label: 'FP4',  default: false,   disabled: !hasFp4 || isXeon,
            disabledReason: isXeon ? 'FP4 is not supported on Xeon' : 'FP4 is only available for Qwen3.5-397B-A17B' }
        ];
      }
    },
    reasoning: {
      name: 'reasoning',
      title: 'Reasoning Parser',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled',  label: 'Enabled',  default: true  }
      ]
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled',  label: 'Enabled',  default: true  }
      ]
    },
    speculative: {
      name: 'speculative',
      title: 'Speculative Decoding (MTP)',
      condition: (values) => values.hardware !== 'xeon',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled',  label: 'Enabled',  default: true  }
      ]
    },
    mambaCache: {
      name: 'mambaCache',
      title: 'Mamba Radix Cache',
      condition: (values) => MOE_MODELS.has(values.model) && values.hardware !== 'xeon',
      getDynamicItems: (currentValues) => {
        const amdGpus = ['mi300x', 'mi325x', 'mi355x'];
        const isAmdGpu = amdGpus.includes(currentValues.hardware);
        const mtpEnabled = currentValues.speculative === 'enabled';

        // MTP requires V2 mamba radix cache
        if (mtpEnabled && !isAmdGpu) {
          return [
            { id: 'v1', label: 'V1', default: false, disabled: true },
            { id: 'v2', label: 'V2', default: true }
          ];
        }

        // Show V2 as disabled for AMD GPUs (V2 requires FLA backend, NVIDIA only)
        if (isAmdGpu) {
          return [
            { id: 'v1', label: 'V1', default: true },
            { id: 'v2', label: 'V2', default: false, disabled: true }
          ];
        }

        // Show both V1 and V2 enabled for NVIDIA GPUs
        return [
          { id: 'v1', label: 'V1', default: true },
          { id: 'v2', label: 'V2', default: false }
        ];
      }
    }
  };

  const modelConfigs = {
    '397b': {
      h100:   { bf16: { tp: 16, mem: 0.8, multinode: true, nnodes: 2 }, fp8: { tp: 8, mem: 0.8 } },
      h200:   { bf16: { tp: 8,  mem: 0.8 }, fp8: { tp: 8, ep: 8, mem: 0.8 } },
      b200:   { bf16: { tp: 8,  mem: 0.8 }, fp8: { tp: 4, mem: 0.8 }, fp4: { tp: 4, mem: 0.85 } },
      b300:   { bf16: { tp: 8,  mem: 0.8 }, fp8: { tp: 4, mem: 0.8 }, fp4: { tp: 4, mem: 0.8 } },
      mi300x: { bf16: { tp: 8, mem: 0.8 }, fp8: { tp: 4, mem: 0.8 } },
      mi325x: { bf16: { tp: 4, mem: 0.8 }, fp8: { tp: 2, mem: 0.8 } },
      mi355x: { bf16: { tp: 4, mem: 0.8 }, fp8: { tp: 2, mem: 0.8 }, fp4: { tp: 2, mem: 0.8 } },
      xeon:   { bf16: { tp: 6 },           fp8: { tp: 6 } }
    },
    '122b': {
      h100:   { bf16: { tp: 4, mem: 0.88 }, fp8: { tp: 2, mem: 0.8 } },
      h200:   { bf16: { tp: 4 },            fp8: { tp: 2 } },
      b200:   { bf16: { tp: 2, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      b300:   { bf16: { tp: 2 },           fp8: { tp: 1, mem: 0.8 } },
      mi300x: { bf16: { tp: 2, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      mi325x: { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      mi355x: { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      xeon:   { bf16: { tp: 6 },           fp8: { tp: 6 } }
    },
    '35b': {
      h100:   { bf16: { tp: 1, mem: 0.88 }, fp8: { tp: 1, mem: 0.8 } },
      h200:   { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      b200:   { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      b300:   { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      mi300x: { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      mi325x: { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      mi355x: { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      xeon:   { bf16: { tp: 3 }, fp8: { tp: 3 } }
    },
    '27b': {
      h100:   { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      h200:   { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      b200:   { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      b300:   { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      mi300x: { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      mi325x: { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      mi355x: { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
      xeon:   { bf16: { tp: 6 }, fp8: { tp: 6 } }
    },
    '9b': {
      h100:   { bf16: { tp: 1, mem: 0.8 } },
      h200:   { bf16: { tp: 1, mem: 0.8 } },
      b200:   { bf16: { tp: 1, mem: 0.8 } },
      b300:   { bf16: { tp: 1, mem: 0.8 } },
      mi300x: { bf16: { tp: 1, mem: 0.8 } },
      mi325x: { bf16: { tp: 1, mem: 0.8 } },
      mi355x: { bf16: { tp: 1, mem: 0.8 } },
      xeon:   { bf16: { tp: 3 } }
    },
    '4b': {
      h100:   { bf16: { tp: 1, mem: 0.8 } },
      h200:   { bf16: { tp: 1, mem: 0.8 } },
      b200:   { bf16: { tp: 1, mem: 0.8 } },
      b300:   { bf16: { tp: 1, mem: 0.8 } },
      mi300x: { bf16: { tp: 1, mem: 0.8 } },
      mi325x: { bf16: { tp: 1, mem: 0.8 } },
      mi355x: { bf16: { tp: 1, mem: 0.8 } },
      xeon:   { bf16: { tp: 3 } }
    },
    '2b': {
      h100:   { bf16: { tp: 1, mem: 0.8 } },
      h200:   { bf16: { tp: 1, mem: 0.8 } },
      b200:   { bf16: { tp: 1, mem: 0.8 } },
      b300:   { bf16: { tp: 1, mem: 0.8 } },
      mi300x: { bf16: { tp: 1, mem: 0.8 } },
      mi325x: { bf16: { tp: 1, mem: 0.8 } },
      mi355x: { bf16: { tp: 1, mem: 0.8 } },
      xeon:   { bf16: { tp: 3 } }
    },
    '0.8b': {
      h100:   { bf16: { tp: 1, mem: 0.8 } },
      h200:   { bf16: { tp: 1, mem: 0.8 } },
      b200:   { bf16: { tp: 1, mem: 0.8 } },
      b300:   { bf16: { tp: 1, mem: 0.8 } },
      mi300x: { bf16: { tp: 1, mem: 0.8 } },
      mi325x: { bf16: { tp: 1, mem: 0.8 } },
      mi355x: { bf16: { tp: 1, mem: 0.8 } },
      xeon:   { bf16: { tp: 3 } }
    }
  };

  const resolveItems = (option, vals) =>
    typeof option.getDynamicItems === 'function' ? option.getDynamicItems(vals) : option.items;

  const getInitialState = () => {
    const initialState = {};
    for (const [key, option] of Object.entries(options)) {
      const items = resolveItems(option, initialState);
      const def = items.find(i => i.default && !i.disabled) || items.find(i => !i.disabled) || items[0];
      initialState[key] = def.id;
    }
    return initialState;
  };

  const [values, setValues] = useState(getInitialState);
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const checkDarkMode = () => {
      const html = document.documentElement;
      const isDarkMode = html.classList.contains('dark') ||
                         html.getAttribute('data-theme') === 'dark' ||
                         html.style.colorScheme === 'dark';
      setIsDark(isDarkMode);
    };
    checkDarkMode();
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme', 'style'] });
    return () => observer.disconnect();
  }, []);

  // When hardware or model changes, re-resolve dynamic selections to stay consistent.
  useEffect(() => {
    setValues(prev => {
      const next = { ...prev };
      for (const [key, option] of Object.entries(options)) {
        if (typeof option.getDynamicItems !== 'function') continue;
        const items = option.getDynamicItems(next);
        const current = items.find(i => i.id === next[key]);
        if (!current || current.disabled) {
          const fallback = items.find(i => i.default && !i.disabled) || items.find(i => !i.disabled);
          if (fallback) next[key] = fallback.id;
        }
      }
      return next;
    });
  }, [values.hardware, values.model]);

  const handleRadioChange = (optionName, value) => {
    setValues(prev => ({ ...prev, [optionName]: value }));
  };

  // Multi-node flag template — mirrors DeepSeek-V4 cookbook's multiNodeFlags.
  // Each launcher must be invoked on every node with <node-rank> set to its rank
  // (0 on the head node) and <node0-ip> resolvable from every node.
  const multiNodeFlags = (nnodes) => [
    `--nnodes ${nnodes}`,
    `--node-rank <node-rank>`,
    `--dist-init-addr <node0-ip>:20000`,
  ];

  const prependMultiNodeNote = (cmd, nnodes) =>
    `# Multi-node (${nnodes} nodes). Run the same command on every node with:\n` +
    `#   <node-rank> = 0 on the head node, 1..${nnodes - 1} on the others\n` +
    `#   <node0-ip>  = IP of the head node (reachable from all others)\n` +
    cmd;

  // Generate command — must produce byte-identical output to sgl-cookbook's
  // config.generateCommand(values) for every valid combination.
  const generateCommand = () => {
    const { model, hardware, quantization, speculative, mambaCache } = values;

    let hwConfig = modelConfigs[model]?.[hardware]?.[quantization];
    if (!hwConfig) {
      if (quantization === 'fp4') {
        return '# FP4 requires B200/B300 (Blackwell) and is only available for Qwen3.5-397B-A17B';
      }
      return '# Please select a valid hardware and quantization combination';
    }

    // 35B / 27B H100 BF16 with MTP: bump TP to 2 and skip --mem-fraction-static.
    // Spread the base spec so any future fields (multinode, ep, ...) survive.
    if ((model === '35b' || model === '27b') && hardware === 'h100' && quantization === 'bf16' && speculative === 'enabled') {
      hwConfig = { ...hwConfig, tp: 2, mem: undefined };
    }
    // 122B H100 FP8 with MTP: bump TP to 4 and skip --mem-fraction-static.
    if (model === '122b' && hardware === 'h100' && quantization === 'fp8' && speculative === 'enabled') {
      hwConfig = { ...hwConfig, tp: 4, mem: undefined };
    }

    let modelName;
    if (quantization === 'fp4') {
      // AMD MI355X uses the MXFP4 checkpoint; Blackwell uses NVFP4.
      modelName = hardware === 'mi355x'
        ? 'amd/Qwen3.5-397B-A17B-MXFP4'
        : 'nvidia/Qwen3.5-397B-A17B-NVFP4';
    } else {
      const suffix = MODEL_SUFFIX[model];
      const quantSuffix = quantization === 'fp8' ? '-FP8' : '';
      modelName = `Qwen/Qwen3.5-${suffix}${quantSuffix}`;
    }

    const tpValue = hwConfig.tp;
    const epValue = hwConfig.ep;
    const memFraction = hwConfig.mem;
    const isMultinode = !!hwConfig.multinode;
    const nnodes = hwConfig.nnodes || 1;

    // Initialize the base command
    let cmd = `sglang serve --model-path ${modelName}`;
    if (hardware === 'xeon') {
      cmd += ` \\\n  --device cpu \\\n  --disable-overlap-schedule`;
    }
    if (tpValue > 1) {
      cmd += ` \\\n  --tp ${tpValue}`;
    }
    if (epValue) {
      cmd += ` \\\n  --expert-parallel-size ${epValue}`;
    }

    // Multi-node wiring goes right after --tp / --expert-parallel-size so the
    // distributed-init flags sit next to the parallelism flags they configure.
    if (isMultinode) {
      for (const flag of multiNodeFlags(nnodes)) {
        cmd += ` \\\n  ${flag}`;
      }
    }

    // Force Mamba V1 for AMD GPUs and Xeon CPUs (V2 requires FLA backend).
    // Force Mamba V2 when MTP is enabled.
    // Dense models with MTP off: force V1 — values.mambaCache is not
    // re-resolved on a speculative toggle (useEffect deps are hardware/model),
    // so it can stay at 'v2' from a prior MTP-on state. Reading it directly
    // would emit a spurious --mamba-scheduler-strategy extra_buffer. The UI
    // radio is hidden for dense models, so users can't manually correct it.
    // MoE keeps the old behavior — the UI radio is the recovery path there.
    const mamba_v1_dev = ['mi300x', 'mi325x', 'mi355x', 'xeon'];
    const actualMambaCache = mamba_v1_dev.includes(hardware)
      ? 'v1'
      : (speculative === 'enabled' ? 'v2' : (MOE_MODELS.has(model) ? mambaCache : 'v1'));

    // Apply commandRules from options (reasoning, toolcall, speculative, mambaCache)
    // Skip quantization and model (handled via model name)
    const commandRules = {
      reasoning: (value) => value === 'enabled' ? '--reasoning-parser qwen3' : null,
      toolcall: (value) => value === 'enabled' ? '--tool-call-parser qwen3_coder' : null,
      speculative: (value) => value === 'enabled' ? '--speculative-algorithm NEXTN \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 4' : null,
      mambaCache: (value) => value === 'v2' ? '--mamba-scheduler-strategy extra_buffer' : null,
    };

    // Iterate options in order, applying commandRules
    for (const [key, option] of Object.entries(options)) {
      if (key === 'quantization' || key === 'model') continue;
      // Skip options that don't pass their condition. mambaCache is special:
      // its condition gates only the UI radio (hidden for dense models), but
      // the rule still fires for dense models on NVIDIA + MTP to emit
      // --mamba-scheduler-strategy extra_buffer.
      if (option.condition && !option.condition(values) && (key !== 'mambaCache' || speculative !== 'enabled')) continue;
      const rule = commandRules[key];
      if (rule) {
        const adjustedValue = key === 'mambaCache' ? actualMambaCache : values[key];
        const result = rule(adjustedValue);
        if (result) {
          cmd += ` \\\n  ${result}`;
        }
      }
    }

    // Enable NCCL symmetric memory for H100 FP8 deployments.
    if (hardware === 'h100' && quantization === 'fp8' && hwConfig.tp > 1) {
      cmd += ` \\\n  --enable-symm-mem`;
    }

    // Chunked prefill tuning for H200 FP8 + MTP (validated on H200 only)
    if (hardware === 'h200' && quantization === 'fp8' && speculative === 'enabled') {
      cmd += ` \\\n  --max-running-requests 128`;
      cmd += ` \\\n  --chunked-prefill-size 16384`;
      cmd += ` \\\n  --tokenizer-worker-num 6`;
    }

    // Enable FlashInfer allreduce fusion for NVIDIA Qwen3.5 configs (skip for FP4:
    // benchmark only enables this for TP>=8). AMD MI GPUs use the AITER allreduce
    // fusion flag instead, handled in the AMD backend block below.
    const amdGpu = hardware === 'mi300x' || hardware === 'mi325x' || hardware === 'mi355x';
    if (quantization !== 'fp4' && hardware !== 'xeon' && !amdGpu) {
      cmd += ` \\\n  --enable-flashinfer-allreduce-fusion`;
    }

    // H200 FP8-specific optimizations
    if (hardware === 'h200' && quantization === 'fp8') {
      cmd += ` \\\n  --attention-backend flashinfer`;
      if (MOE_MODELS.has(model)) {
        cmd += ` \\\n  --mamba-ssm-dtype bfloat16`;
      }
    }

    // Append backend configurations
    if (hardware === 'b200' || (hardware === 'b300' && quantization === 'fp4')) {
      cmd += ` \\\n  --attention-backend trtllm_mha`;
    }
    if (hardware === 'b300' && quantization !== 'fp4') {
      cmd += ` \\\n  --attention-backend flashinfer`;
    }

    // Append AMD GPU-specific backend configurations.
    // All AMD MI GPUs use the AITER unified-attention backend (pair with
    // SGLANG_USE_AITER=1 and SGLANG_USE_AITER_UNIFIED_ATTN=1; see cookbook prose),
    // which requires --page-size 16. Enable AITER allreduce fusion for multi-GPU.
    if (amdGpu) {
      let amdEnv = "SGLANG_USE_AITER=1 \\\nSGLANG_USE_AITER_UNIFIED_ATTN=1 \\\nAITER_FLYDSL_FORCE=1 \\\n";
      if (MOE_MODELS.has(model)) {
        amdEnv += "SGLANG_MAMBA_SSM_DTYPE=bfloat16 \\\n";
      }
      cmd = amdEnv + cmd;
      cmd += " \\\n  --attention-backend aiter";
      cmd += " \\\n  --page-size 16";
      if (hwConfig.tp > 1) {
        cmd += " \\\n  --enable-aiter-allreduce-fusion";
      }
    }

    // Tokenizer workers for H200 and B200/B300
    if (hardware === 'h200' || hardware === 'b200' || hardware === 'b300') {
      if (speculative === 'disabled') {
        cmd += ` \\\n  --tokenizer-worker-num 6`;
      }
    }

    // Workaround: FlashInfer autotune's warmup dummy_run trips a CUDA grid-dim
    // overflow in the GDN packed_decode Triton kernel (B*HV >= 65536). Remove
    // once the kernel is fixed upstream.
    if (hardware === 'b300' && quantization === 'bf16' && (model === '0.8b' || model === '2b')) {
      cmd += ` \\\n  --max-running-requests 4064`;
    }

    // FP4-specific backend settings
    if (quantization === 'fp4') {
      if (hardware === 'mi355x') {
        // AMD MXFP4 on MI355X: backend / --page-size 16 / AITER allreduce fusion
        // are emitted by the AMD backend block above. Add the FP4-specific flags here.
        cmd += ' \\\n  --disable-radix-cache';
        // Cap concurrency under MTP to avoid OOM at tp=2.
        if (speculative === 'enabled') {
          cmd += ' \\\n  --max-running-requests 128';
        }
      } else {
        // NVIDIA NVFP4 on Blackwell (B200 / B300).
        if (hardware === 'b300') {
          cmd += ' \\\n  --moe-runner-backend flashinfer_trtllm';
          cmd += ' \\\n  --fp4-gemm-backend flashinfer_cutlass';
        } else {
          cmd += ' \\\n  --quantization modelopt_fp4';
          cmd += ' \\\n  --fp4-gemm-backend flashinfer_cutlass';
          cmd += ' \\\n  --kv-cache-dtype fp8_e4m3';
          cmd += ' \\\n  --moe-runner-backend flashinfer_trtllm';
          cmd += ' \\\n  --chunked-prefill-size 32768';
          cmd += ' \\\n  --max-prefill-tokens 32768';
          cmd += ' \\\n  --max-running-requests 128';
          cmd += ' \\\n  --stream-interval 30';
          cmd += ' \\\n  --disable-radix-cache';
        }
      }
    }

    // Add memory fraction last
    if (memFraction !== undefined) {
      cmd += ` \\\n  --mem-fraction-static ${memFraction}`;
    }

    if (isMultinode) {
      cmd = prependMultiNodeNote(cmd, nnodes);
    }

    return cmd;
  };

  // Styles
  const containerStyle = { maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '4px' };
  const cardStyle = { padding: '8px 12px', border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`, borderLeft: `3px solid ${isDark ? '#E85D4D' : '#D45D44'}`, borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '12px', background: isDark ? '#1f2937' : '#fff' };
  const titleStyle = { fontSize: '13px', fontWeight: '600', minWidth: '140px', flexShrink: 0, color: isDark ? '#e5e7eb' : 'inherit' };
  const itemsStyle = { display: 'flex', rowGap: '2px', columnGap: '6px', flexWrap: 'wrap', alignItems: 'center', flex: 1 };
  const labelBaseStyle = { padding: '4px 10px', border: `1px solid ${isDark ? '#9ca3af' : '#d1d5db'}`, borderRadius: '3px', cursor: 'pointer', display: 'inline-flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', fontWeight: '500', fontSize: '13px', transition: 'all 0.2s', userSelect: 'none', minWidth: '45px', textAlign: 'center', flex: 1, background: isDark ? '#374151' : '#fff', color: isDark ? '#e5e7eb' : 'inherit' };
  const checkedStyle = { background: '#D45D44', color: 'white', borderColor: '#D45D44' };
  const disabledStyle = { cursor: 'not-allowed', opacity: 0.4 };
  const subtitleStyle = { display: 'block', fontSize: '9px', marginTop: '1px', lineHeight: '1.1', opacity: 0.7 };
  const commandDisplayStyle = { flex: 1, padding: '12px 16px', background: isDark ? '#111827' : '#f5f5f5', borderRadius: '6px', fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace", fontSize: '12px', lineHeight: '1.5', color: isDark ? '#e5e7eb' : '#374151', whiteSpace: 'pre-wrap', overflowX: 'auto', margin: 0, border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}` };

  return (
    <div style={containerStyle} className="not-prose">
      {Object.entries(options).map(([key, option]) => {
        if (typeof option.condition === 'function' && !option.condition(values)) return null;
        const items = resolveItems(option, values);
        return (
          <div key={key} style={cardStyle}>
            <div style={titleStyle}>{option.title}</div>
            <div style={itemsStyle}>
              {items.map(item => {
                const isChecked = values[option.name] === item.id;
                const isDisabled = !!item.disabled;
                return (
                  <label
                    key={item.id}
                    style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}), ...(isDisabled ? disabledStyle : {}) }}
                    title={item.disabledReason || ''}
                  >
                    <input
                      type="radio"
                      name={option.name}
                      value={item.id}
                      checked={isChecked}
                      disabled={isDisabled}
                      onChange={() => !isDisabled && handleRadioChange(option.name, item.id)}
                      style={{ display: 'none' }}
                    />
                    {item.label}
                    {item.subtitle && <small style={{ ...subtitleStyle, color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit' }}>{item.subtitle}</small>}
                  </label>
                );
              })}
            </div>
          </div>
        );
      })}
      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{generateCommand()}</pre>
      </div>
    </div>
  );
};

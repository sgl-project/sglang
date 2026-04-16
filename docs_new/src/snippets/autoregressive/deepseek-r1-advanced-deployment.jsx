export const DeepSeekR1AdvancedDeployment = () => {
const lookupData = {
  "model": "deepseek-r1",
  "version": "v0.5.6",
  "ui_options": {
    "hardware": [
      {
        "id": "b200",
        "label": "B200",
        "default": true
      },
      {
        "id": "h200",
        "label": "H200",
        "default": false
      },
      {
        "id": "mi300x",
        "label": "MI300X",
        "default": false
      },
      {
        "id": "mi325x",
        "label": "MI325X",
        "default": false
      },
      {
        "id": "mi355x",
        "label": "MI355X",
        "default": false
      }
    ],
    "quantization": [
      {
        "id": "fp8",
        "label": "FP8",
        "default": true
      },
      {
        "id": "fp4",
        "label": "FP4",
        "default": false
      }
    ],
    "scenario": [
      {
        "id": "low-latency",
        "label": "Low Latency",
        "subtitle": "Concurrency 4-8",
        "default": true
      },
      {
        "id": "high-throughput",
        "label": "High Throughput",
        "subtitle": "Concurrency 16-128",
        "default": false
      }
    ],
    "gpu_count": [
      {
        "id": 4,
        "label": "4 GPUs",
        "default": false
      },
      {
        "id": 8,
        "label": "8 GPUs",
        "default": true
      }
    ]
  },
  "configs": [
    {
      "hardware": "b200",
      "quantization": "fp4",
      "gpu_count": 4,
      "scenario": "low-latency",
      "parameters": {
        "model_path": "nvidia/DeepSeek-R1-0528-FP4-v2",
        "tensor_parallel_size": 4,
        "cuda_graph_max_bs": 256,
        "max_running_requests": 256,
        "mem_fraction_static": 0.85,
        "ep_size": 4,
        "scheduler_recv_interval": 10,
        "enable_symm_mem": true,
        "stream_interval": 10
      }
    },
    {
      "hardware": "b200",
      "quantization": "fp4",
      "gpu_count": 4,
      "scenario": "high-throughput",
      "parameters": {
        "model_path": "nvidia/DeepSeek-R1-0528-FP4-v2",
        "tensor_parallel_size": 4,
        "cuda_graph_max_bs": 256,
        "max_running_requests": 256,
        "mem_fraction_static": 0.85,
        "ep_size": 4,
        "scheduler_recv_interval": 30,
        "enable_symm_mem": true,
        "stream_interval": 10
      }
    },
    {
      "hardware": "b200",
      "quantization": "fp4",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "model_path": "nvidia/DeepSeek-R1-0528-FP4-v2",
        "tensor_parallel_size": 8,
        "cuda_graph_max_bs": 256,
        "max_running_requests": 256,
        "mem_fraction_static": 0.85,
        "kv_cache_dtype": "fp8_e4m3",
        "chunked_prefill_size": 16384,
        "ep_size": 8,
        "scheduler_recv_interval": 10,
        "enable_symm_mem": true,
        "stream_interval": 10
      }
    },
    {
      "hardware": "b200",
      "quantization": "fp4",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "model_path": "nvidia/DeepSeek-R1-0528-FP4-v2",
        "tensor_parallel_size": 8,
        "cuda_graph_max_bs": 256,
        "max_running_requests": 256,
        "mem_fraction_static": 0.85,
        "kv_cache_dtype": "fp8_e4m3",
        "chunked_prefill_size": 16384,
        "ep_size": 8,
        "scheduler_recv_interval": 30,
        "enable_symm_mem": true,
        "stream_interval": 10
      }
    },
    {
      "hardware": "b200",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "env_vars": "SGLANG_ENABLE_JIT_DEEPGEMM=false",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "tensor_parallel_size": 8,
        "cuda_graph_max_bs": 128,
        "max_running_requests": 128,
        "mem_fraction_static": 0.82,
        "kv_cache_dtype": "fp8_e4m3",
        "chunked_prefill_size": 32768,
        "max_prefill_tokens": 32768,
        "scheduler_recv_interval": 10,
        "stream_interval": 30,
        "fp8_gemm_backend": "flashinfer_trtllm"
      }
    },
    {
      "hardware": "b200",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "env_vars": "SGLANG_ENABLE_JIT_DEEPGEMM=false",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "tensor_parallel_size": 8,
        "cuda_graph_max_bs": 128,
        "max_running_requests": 128,
        "mem_fraction_static": 0.82,
        "kv_cache_dtype": "fp8_e4m3",
        "chunked_prefill_size": 32768,
        "max_prefill_tokens": 32768,
        "scheduler_recv_interval": 30,
        "stream_interval": 30,
        "fp8_gemm_backend": "flashinfer_trtllm"
      }
    },
    {
      "hardware": "h200",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "disable_radix_cache": true,
        "max_running_requests": 256,
        "cuda_graph_max_bs": 256,
        "chunked_prefill_size": 32768,
        "max_prefill_tokens": 32768,
        "mem_fraction_static": 0.82,
        "attention_backend": "flashinfer",
        "stream_interval": 10,
        "decode_log_interval": 1
      }
    },
    {
      "hardware": "h200",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "disable_radix_cache": true,
        "max_running_requests": 512,
        "cuda_graph_max_bs": 512,
        "chunked_prefill_size": 32768,
        "max_prefill_tokens": 32768,
        "mem_fraction_static": 0.82,
        "attention_backend": "flashinfer",
        "stream_interval": 10,
        "decode_log_interval": 1
      }
    },
    {
      "hardware": "mi300x",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 SGLANG_AITER_MLA_PERSIST=1",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "cuda_graph_max_bs": 128,
        "chunked_prefill_size": 131072,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 131072,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "aiter",
        "disable_radix_cache": true
      }
    },
    {
      "hardware": "mi300x",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 SGLANG_AITER_MLA_PERSIST=1",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "cuda_graph_max_bs": 512,
        "chunked_prefill_size": 131072,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 131072,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "aiter",
        "disable_radix_cache": true
      }
    },
    {
      "hardware": "mi325x",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 SGLANG_AITER_MLA_PERSIST=1",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "cuda_graph_max_bs": 128,
        "chunked_prefill_size": 131072,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 131072,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "aiter",
        "disable_radix_cache": true
      }
    },
    {
      "hardware": "mi325x",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 SGLANG_AITER_MLA_PERSIST=1",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "cuda_graph_max_bs": 512,
        "chunked_prefill_size": 131072,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 131072,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "aiter",
        "disable_radix_cache": true
      }
    },
    {
      "hardware": "mi355x",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 RCCL_MSCCL_ENABLE=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "disable_radix_cache": true,
        "chunked_prefill_size": 196608,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 196608,
        "cuda_graph_max_bs": 128,
        "attention_backend": "aiter",
        "kv_cache_dtype": "fp8_e4m3"
      }
    },
    {
      "hardware": "mi355x",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 RCCL_MSCCL_ENABLE=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "disable_radix_cache": true,
        "chunked_prefill_size": 196608,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 196608,
        "cuda_graph_max_bs": 512,
        "attention_backend": "aiter",
        "kv_cache_dtype": "fp8_e4m3"
      }
    },
    {
      "hardware": "mi355x",
      "quantization": "fp4",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 ROCM_QUICK_REDUCE_QUANTIZATION=INT4",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "disable_radix_cache": true,
        "chunked_prefill_size": 196608,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 196608,
        "cuda_graph_max_bs": 128,
        "attention_backend": "aiter",
        "kv_cache_dtype": "fp8_e4m3"
      }
    },
    {
      "hardware": "mi355x",
      "quantization": "fp4",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 ROCM_QUICK_REDUCE_QUANTIZATION=INT4",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "disable_radix_cache": true,
        "chunked_prefill_size": 196608,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 196608,
        "cuda_graph_max_bs": 512,
        "attention_backend": "aiter",
        "kv_cache_dtype": "fp8_e4m3"
      }
    }
  ],
  "validation": [
    {
      "hardware": "h200",
      "quantization": "fp4",
      "error": "FP4 is only available for B200 hardware. Please select FP8 quantization."
    }
  ]
};

const fieldToFlag = {
  model_path: 'model-path',
  trust_remote_code: 'trust-remote-code',
  tensor_parallel_size: 'tp',
  data_parallel_size: 'dp',
  ep_size: 'ep-size',
  cuda_graph_max_bs: 'cuda-graph-max-bs',
  max_running_requests: 'max-running-requests',
  mem_fraction_static: 'mem-fraction-static',
  kv_cache_dtype: 'kv-cache-dtype',
  chunked_prefill_size: 'chunked-prefill-size',
  max_prefill_tokens: 'max-prefill-tokens',
  enable_flashinfer_allreduce_fusion: 'enable-flashinfer-allreduce-fusion',
  scheduler_recv_interval: 'scheduler-recv-interval',
  enable_symm_mem: 'enable-symm-mem',
  disable_radix_cache: 'disable-radix-cache',
  attention_backend: 'attention-backend',
  moe_runner_backend: 'moe-runner-backend',
  stream_interval: 'stream-interval',
  quantization: 'quantization',
  decode_log_interval: 'decode-log-interval',
  fp8_gemm_backend: 'fp8-gemm-backend',
  num_continuous_decode_steps: 'num-continuous-decode-steps',
};

const findConfig = (hardware, quantization, gpuCount, scenario) => {
  const match = lookupData.configs.find((entry) => {
    const hardwareMatch = entry.hardware === hardware;
    const quantizationMatch = entry.quantization === quantization;
    const gpuCountMatch = !entry.gpu_count || entry.gpu_count === Number.parseInt(gpuCount, 10);
    const scenarioMatch = entry.scenario === scenario;
    return hardwareMatch && quantizationMatch && gpuCountMatch && scenarioMatch;
  });
  return match ? match.parameters : null;
};

const getAvailableGpuCounts = (hardware, quantization) => {
  const entries = lookupData.configs.filter(
    (entry) => entry.hardware === hardware && entry.quantization === quantization
  );
  const gpuCounts = [...new Set(entries.map((entry) => entry.gpu_count))].filter(Boolean);
  return gpuCounts.length > 0 ? gpuCounts.sort((a, b) => a - b) : [8];
};

const generateCommandFromConfig = (config) => {
  if (!config) {
    return '# Error: Configuration not found';
  }

  let command = '';
  if (config.env_vars) {
    command = `${config.env_vars} `;
  }

  command += 'python3 -m sglang.launch_server \\\n';
  command += `  --model-path ${config.model_path}`;

  for (const [key, value] of Object.entries(config)) {
    if (key === 'model_path' || key === 'env_vars') {
      continue;
    }

    const flagName = fieldToFlag[key];
    if (!flagName) {
      continue;
    }

    if (typeof value === 'boolean') {
      if (value) {
        command += ` \\\n  --${flagName}`;
      }
      continue;
    }

    command += ` \\\n  --${flagName} ${value}`;
  }

  return command;
};

const validateSelection = (hardware, quantization) => {
  for (const rule of lookupData.validation || []) {
    const hardwareMatch = Array.isArray(rule.hardware)
      ? rule.hardware.includes(hardware)
      : rule.hardware === hardware;
    const quantizationMatch = Array.isArray(rule.quantization)
      ? rule.quantization.includes(quantization)
      : rule.quantization === quantization;
    if (hardwareMatch && quantizationMatch) {
      return rule.error;
    }
  }
  return null;
};

const resolveItems = (option, values) =>
  typeof option.getDynamicItems === 'function' ? option.getDynamicItems(values) : option.items;

  const uiOptions = lookupData.ui_options;
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: uiOptions.hardware
        .filter((option) =>
          ['b200', 'h200', 'mi300x', 'mi325x', 'mi355x'].includes(option.id)
        )
        .map((option) => ({
          id: option.id,
          label: option.label,
          default: option.id === 'b200',
        })),
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      getDynamicItems: (values) =>
        uiOptions.quantization.map((option) => {
          const fp4Disabled = ['h200', 'mi300x', 'mi325x'].includes(values.hardware) && option.id === 'fp4';
          return {
            id: option.id,
            label: option.label,
            default:
              ['h200', 'mi300x', 'mi325x'].includes(values.hardware)
                ? option.id === 'fp8'
                : option.default,
            disabled: fp4Disabled,
            disabledReason: fp4Disabled ? 'FP4 not supported on H200, MI300X, MI325X' : '',
          };
        }),
    },
    gpuCount: {
      name: 'gpuCount',
      title: 'GPU Count',
      getDynamicItems: (values) => {
        const availableGpuCounts = getAvailableGpuCounts(values.hardware, values.quantization);
        const allGpuCounts = uiOptions.gpu_count.map((option) =>
          typeof option.id === 'number' ? option.id : Number.parseInt(option.id, 10)
        );
        const defaultGpuCount = Math.max(...availableGpuCounts);

        return allGpuCounts.map((count) => ({
          id: String(count),
          label: `${count} GPUs`,
          default: count === defaultGpuCount,
          disabled: !availableGpuCounts.includes(count),
          disabledReason: availableGpuCounts.includes(count)
            ? ''
            : `${count} GPUs not available for ${values.hardware.toUpperCase()} ${values.quantization.toUpperCase()}`,
        }));
      },
    },
    scenario: {
      name: 'scenario',
      title: 'Scenario',
      items: uiOptions.scenario.map((option) => ({
        id: option.id,
        label: option.label,
        subtitle: option.subtitle,
        default: option.default,
      })),
    },
  };

  const getInitialState = () => {
    const initialState = {};
    for (const [key, option] of Object.entries(options)) {
      const items = resolveItems(option, initialState) || [];
      const fallback =
        items.find((item) => item.default && !item.disabled) ||
        items.find((item) => !item.disabled) ||
        items[0];
      initialState[key] = fallback ? fallback.id : '';
    }
    return initialState;
  };

  const [values, setValues] = useState(getInitialState);
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const checkDarkMode = () => {
      const html = document.documentElement;
      const isDarkMode =
        html.classList.contains('dark') ||
        html.getAttribute('data-theme') === 'dark' ||
        html.style.colorScheme === 'dark';
      setIsDark(isDarkMode);
    };
    checkDarkMode();
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class', 'data-theme', 'style'],
    });
    return () => observer.disconnect();
  }, []);

  const handleRadioChange = (optionName, value) => {
    setValues((prev) => {
      const next = { ...prev, [optionName]: value };
      for (const [key, option] of Object.entries(options)) {
        if (typeof option.getDynamicItems !== 'function') {
          continue;
        }
        const items = option.getDynamicItems(next);
        const current = items.find((item) => item.id === next[key]);
        if (!current || current.disabled) {
          const fallback =
            items.find((item) => item.default && !item.disabled) ||
            items.find((item) => !item.disabled);
          if (fallback) {
            next[key] = fallback.id;
          }
        }
      }
      return next;
    });
  };

  const handleCheckboxChange = (optionName, itemId, isChecked) => {
    setValues((prev) => {
      const currentValues = prev[optionName] || [];
      if (isChecked) {
        return { ...prev, [optionName]: [...currentValues, itemId] };
      }
      return {
        ...prev,
        [optionName]: currentValues.filter((id) => id !== itemId),
      };
    });
  };

  const handleTextChange = (optionName, value) => {
    setValues((prev) => ({ ...prev, [optionName]: value }));
  };

  const generateCommand = (vals) => {
    const validationError = validateSelection(vals.hardware, vals.quantization);
    if (validationError) {
      return `# Error: ${validationError}`;
    }

    const config = findConfig(
      vals.hardware,
      vals.quantization,
      vals.gpuCount || '8',
      vals.scenario
    );
    if (!config) {
      return `# Error: No configuration found for:
# Hardware: ${vals.hardware}
# Quantization: ${vals.quantization}
# GPU Count: ${vals.gpuCount}
# Scenario: ${vals.scenario}
# This combination is not yet supported.`;
    }

    return generateCommandFromConfig(config);
  };

  const command = generateCommand(values);

  const containerStyle = {
    maxWidth: '900px',
    margin: '0 auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  };
  const cardStyle = {
    padding: '8px 12px',
    border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`,
    borderLeft: `3px solid ${isDark ? '#E85D4D' : '#D45D44'}`,
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    background: isDark ? '#1f2937' : '#fff',
  };
  const titleStyle = {
    fontSize: '13px',
    fontWeight: '600',
    minWidth: '140px',
    flexShrink: 0,
    color: isDark ? '#e5e7eb' : 'inherit',
  };
  const itemsStyle = {
    display: 'flex',
    rowGap: '2px',
    columnGap: '6px',
    flexWrap: 'wrap',
    alignItems: 'center',
    flex: 1,
  };
  const labelBaseStyle = {
    padding: '4px 10px',
    border: `1px solid ${isDark ? '#9ca3af' : '#d1d5db'}`,
    borderRadius: '3px',
    cursor: 'pointer',
    display: 'inline-flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: '500',
    fontSize: '13px',
    transition: 'all 0.2s',
    userSelect: 'none',
    minWidth: '45px',
    textAlign: 'center',
    flex: 1,
    background: isDark ? '#374151' : '#fff',
    color: isDark ? '#e5e7eb' : 'inherit',
  };
  const checkedStyle = {
    background: '#D45D44',
    color: 'white',
    borderColor: '#D45D44',
  };
  const disabledStyle = {
    cursor: 'not-allowed',
    opacity: 0.5,
  };
  const subtitleStyle = {
    display: 'block',
    fontSize: '9px',
    marginTop: '1px',
    lineHeight: '1.1',
    opacity: 0.7,
  };
  const textInputStyle = {
    flex: 1,
    padding: '8px 10px',
    borderRadius: '4px',
    border: `1px solid ${isDark ? '#4b5563' : '#d1d5db'}`,
    background: isDark ? '#111827' : '#fff',
    color: isDark ? '#e5e7eb' : '#111827',
    fontSize: '13px',
  };
  const commandDisplayStyle = {
    flex: 1,
    padding: '12px 16px',
    background: isDark ? '#111827' : '#f5f5f5',
    borderRadius: '6px',
    fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
    fontSize: '12px',
    lineHeight: '1.5',
    color: isDark ? '#e5e7eb' : '#374151',
    whiteSpace: 'pre-wrap',
    overflowX: 'auto',
    margin: 0,
    border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`,
  };

  return (
    <div style={containerStyle} className="not-prose">
      {Object.entries(options).map(([key, option]) => {
        if (option.condition && !option.condition(values)) {
          return null;
        }
        const items = option.getDynamicItems ? option.getDynamicItems(values) : option.items || [];
        return (
          <div key={key} style={cardStyle}>
            <div style={titleStyle}>{option.title}</div>
            <div style={itemsStyle}>
              {option.type === 'text' ? (
                <input
                  type="text"
                  value={values[option.name] || ''}
                  placeholder={option.placeholder || ''}
                  onChange={(event) => handleTextChange(option.name, event.target.value)}
                  style={textInputStyle}
                />
              ) : option.type === 'checkbox' ? (
                (option.items || []).map((item) => {
                  const isChecked = (values[option.name] || []).includes(item.id);
                  const isDisabled =
                    item.required ||
                    (typeof item.disabledWhen === 'function' && item.disabledWhen(values));
                  return (
                    <label
                      key={item.id}
                      title={item.disabledReason || ''}
                      style={{
                        ...labelBaseStyle,
                        ...(isChecked ? checkedStyle : {}),
                        ...(isDisabled ? disabledStyle : {}),
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        disabled={isDisabled}
                        onChange={(event) =>
                          handleCheckboxChange(option.name, item.id, event.target.checked)
                        }
                        style={{ display: 'none' }}
                      />
                      {item.label}
                      {item.subtitle && (
                        <small
                          style={{
                            ...subtitleStyle,
                            color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit',
                          }}
                        >
                          {item.subtitle}
                        </small>
                      )}
                    </label>
                  );
                })
              ) : (
                items.map((item) => {
                  const isChecked = values[option.name] === item.id;
                  const isDisabled = Boolean(item.disabled);
                  return (
                    <label
                      key={item.id}
                      title={item.disabledReason || ''}
                      style={{
                        ...labelBaseStyle,
                        ...(isChecked ? checkedStyle : {}),
                        ...(isDisabled ? disabledStyle : {}),
                      }}
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
                      {item.subtitle && (
                        <small
                          style={{
                            ...subtitleStyle,
                            color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit',
                          }}
                        >
                          {item.subtitle}
                        </small>
                      )}
                    </label>
                  );
                })
              )}
            </div>
          </div>
        );
      })}

      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{command}</pre>
      </div>
    </div>
  );
};

export const DeepSeekR1AdvancedDeployment = () => {
  // Config options
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'b200', label: 'B200', default: true },
        { id: 'h200', label: 'H200', default: false },
        { id: 'mi300x', label: 'MI300X', default: false },
        { id: 'mi325x', label: 'MI325X', default: false },
        { id: 'mi355x', label: 'MI355X', default: false }
      ]
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      items: [
        { id: 'fp8', label: 'FP8', default: true },
        { id: 'fp4', label: 'FP4', default: false }
      ]
    },
    gpuCount: {
      name: 'gpuCount',
      title: 'GPU Count',
      items: [
        { id: '1', label: '1 GPU', default: false },
        { id: '2', label: '2 GPUs', default: false },
        { id: '4', label: '4 GPUs', default: false },
        { id: '8', label: '8 GPUs', default: true }
      ]
    },
    scenario: {
      name: 'scenario',
      title: 'Scenario',
      items: [
        { id: 'latency', label: 'Latency-Sensitive', subtitle: 'Low latency', default: true },
        { id: 'throughput', label: 'Throughput-Sensitive', subtitle: 'High throughput', default: false }
      ]
    }
  };

  // Optimal configurations (simplified from lookup data)
  const optimalConfigs = {
    'b200-fp8-latency': {
      modelPath: 'deepseek-ai/DeepSeek-R1-0528',
      tp: 1,
      dp: 1,
      cudaGraphMaxBs: 1,
      maxRunningRequests: 32,
      memFractionStatic: 0.8,
      kvCacheDtype: 'fp8_e4m3',
      chunkedPrefillSize: 4096,
      maxPrefillTokens: 8192,
      enableFlashinferAllreduceFusion: true,
      schedulerRecvInterval: 20,
      enableSymmMem: true
    },
    'b200-fp8-throughput': {
      modelPath: 'deepseek-ai/DeepSeek-R1-0528',
      tp: 8,
      dp: 1,
      cudaGraphMaxBs: 256,
      maxRunningRequests: 256,
      memFractionStatic: 0.9,
      kvCacheDtype: 'fp8_e4m3',
      chunkedPrefillSize: 8192,
      maxPrefillTokens: 16384,
      enableFlashinferAllreduceFusion: true,
      schedulerRecvInterval: 10,
      enableSymmMem: true
    },
    'b200-fp4-latency': {
      modelPath: 'nvidia/DeepSeek-R1-0528-FP4-v2',
      tp: 1,
      dp: 1,
      cudaGraphMaxBs: 1,
      maxRunningRequests: 32,
      memFractionStatic: 0.8,
      kvCacheDtype: 'fp8_e4m3',
      chunkedPrefillSize: 4096,
      maxPrefillTokens: 8192,
      enableFlashinferAllreduceFusion: true,
      schedulerRecvInterval: 20,
      enableSymmMem: true
    },
    'b200-fp4-throughput': {
      modelPath: 'nvidia/DeepSeek-R1-0528-FP4-v2',
      tp: 8,
      dp: 1,
      cudaGraphMaxBs: 256,
      maxRunningRequests: 256,
      memFractionStatic: 0.9,
      kvCacheDtype: 'fp8_e4m3',
      chunkedPrefillSize: 8192,
      maxPrefillTokens: 16384,
      enableFlashinferAllreduceFusion: true,
      schedulerRecvInterval: 10,
      enableSymmMem: true
    },
    'h200-fp8-latency': {
      modelPath: 'deepseek-ai/DeepSeek-R1-0528',
      tp: 1,
      dp: 1,
      cudaGraphMaxBs: 1,
      maxRunningRequests: 32,
      memFractionStatic: 0.8,
      chunkedPrefillSize: 4096,
      maxPrefillTokens: 8192,
      enableFlashinferAllreduceFusion: true,
      schedulerRecvInterval: 20,
      enableSymmMem: true
    },
    'h200-fp8-throughput': {
      modelPath: 'deepseek-ai/DeepSeek-R1-0528',
      tp: 8,
      dp: 1,
      cudaGraphMaxBs: 256,
      maxRunningRequests: 256,
      memFractionStatic: 0.9,
      chunkedPrefillSize: 8192,
      maxPrefillTokens: 16384,
      enableFlashinferAllreduceFusion: true,
      schedulerRecvInterval: 10,
      enableSymmMem: true
    },
    'mi300x-fp8-latency': {
      modelPath: 'deepseek-ai/DeepSeek-R1-0528',
      tp: 8,
      dp: 1,
      cudaGraphMaxBs: 1,
      maxRunningRequests: 32,
      memFractionStatic: 0.8,
      chunkedPrefillSize: 4096,
      maxPrefillTokens: 8192,
      schedulerRecvInterval: 20
    },
    'mi300x-fp8-throughput': {
      modelPath: 'deepseek-ai/DeepSeek-R1-0528',
      tp: 8,
      dp: 1,
      cudaGraphMaxBs: 256,
      maxRunningRequests: 256,
      memFractionStatic: 0.9,
      chunkedPrefillSize: 8192,
      maxPrefillTokens: 16384,
      schedulerRecvInterval: 10
    },
    'mi325x-fp8-latency': {
      modelPath: 'deepseek-ai/DeepSeek-R1-0528',
      tp: 8,
      dp: 1,
      cudaGraphMaxBs: 1,
      maxRunningRequests: 32,
      memFractionStatic: 0.8,
      chunkedPrefillSize: 4096,
      maxPrefillTokens: 8192,
      schedulerRecvInterval: 20
    },
    'mi325x-fp8-throughput': {
      modelPath: 'deepseek-ai/DeepSeek-R1-0528',
      tp: 8,
      dp: 1,
      cudaGraphMaxBs: 256,
      maxRunningRequests: 256,
      memFractionStatic: 0.9,
      chunkedPrefillSize: 8192,
      maxPrefillTokens: 16384,
      schedulerRecvInterval: 10
    },
    'mi355x-fp8-latency': {
      modelPath: 'deepseek-ai/DeepSeek-R1-0528',
      tp: 8,
      dp: 1,
      cudaGraphMaxBs: 1,
      maxRunningRequests: 32,
      memFractionStatic: 0.8,
      chunkedPrefillSize: 4096,
      maxPrefillTokens: 8192,
      schedulerRecvInterval: 20
    },
    'mi355x-fp8-throughput': {
      modelPath: 'deepseek-ai/DeepSeek-R1-0528',
      tp: 8,
      dp: 1,
      cudaGraphMaxBs: 256,
      maxRunningRequests: 256,
      memFractionStatic: 0.9,
      chunkedPrefillSize: 8192,
      maxPrefillTokens: 16384,
      schedulerRecvInterval: 10
    }
  };

  // Initialize state
  const getInitialState = () => {
    const initialState = {};
    Object.entries(options).forEach(([key, option]) => {
      const defaultItem = option.items.find(item => item.default);
      initialState[key] = defaultItem ? defaultItem.id : option.items[0].id;
    });
    return initialState;
  };

  const [values, setValues] = useState(getInitialState);
  const [isDark, setIsDark] = useState(false);

  // Detect dark mode
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

  const handleRadioChange = (optionName, value) => {
    setValues(prev => ({ ...prev, [optionName]: value }));
  };

  // Check if option should be disabled
  const isOptionDisabled = (optionName, itemId) => {
    if (optionName === 'quantization' && itemId === 'fp4' &&
        (values.hardware === 'h200' || values.hardware === 'mi300x' || values.hardware === 'mi325x' || values.hardware === 'mi355x')) {
      return true;
    }
    return false;
  };

  // Get disabled reason
  const getDisabledReason = (optionName, itemId) => {
    if (optionName === 'quantization' && itemId === 'fp4' &&
        (values.hardware === 'h200' || values.hardware === 'mi300x' || values.hardware === 'mi325x' || values.hardware === 'mi355x')) {
      return 'FP4 not supported on H200, MI300X, MI325X, MI355X';
    }
    return '';
  };

  // Generate command
  const generateCommand = () => {
    const { hardware, quantization, gpuCount, scenario } = values;
    const configKey = `${hardware}-${quantization}-${scenario}`;
    const config = optimalConfigs[configKey];

    if (!config) {
      return `# Error: No configuration found for:\n# Hardware: ${hardware}\n# Quantization: ${quantization}\n# Scenario: ${scenario}\n# This combination is not yet supported.`;
    }

    let cmd = 'python3 -m sglang.launch_server \\\n';
    cmd += `  --model-path ${config.modelPath}`;

    if (config.tp > 1) cmd += ` \\\n  --tp ${config.tp}`;
    if (config.dp > 1) cmd += ` \\\n  --dp ${config.dp}`;
    if (config.cudaGraphMaxBs) cmd += ` \\\n  --cuda-graph-max-bs ${config.cudaGraphMaxBs}`;
    if (config.maxRunningRequests) cmd += ` \\\n  --max-running-requests ${config.maxRunningRequests}`;
    if (config.memFractionStatic) cmd += ` \\\n  --mem-fraction-static ${config.memFractionStatic}`;
    if (config.kvCacheDtype) cmd += ` \\\n  --kv-cache-dtype ${config.kvCacheDtype}`;
    if (config.chunkedPrefillSize) cmd += ` \\\n  --chunked-prefill-size ${config.chunkedPrefillSize}`;
    if (config.maxPrefillTokens) cmd += ` \\\n  --max-prefill-tokens ${config.maxPrefillTokens}`;
    if (config.enableFlashinferAllreduceFusion) cmd += ` \\\n  --enable-flashinfer-allreduce-fusion`;
    if (config.schedulerRecvInterval) cmd += ` \\\n  --scheduler-recv-interval ${config.schedulerRecvInterval}`;
    if (config.enableSymmMem) cmd += ` \\\n  --enable-symm-mem`;

    return cmd;
  };

  // Styles - with dark mode support
  const containerStyle = { maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '4px' };
  const cardStyle = { padding: '8px 12px', border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`, borderLeft: `3px solid ${isDark ? '#E85D4D' : '#D45D44'}`, borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '12px', background: isDark ? '#1f2937' : '#fff' };
  const titleStyle = { fontSize: '13px', fontWeight: '600', minWidth: '140px', flexShrink: 0, color: isDark ? '#e5e7eb' : 'inherit' };
  const itemsStyle = { display: 'flex', rowGap: '2px', columnGap: '6px', flexWrap: 'wrap', alignItems: 'center', flex: 1 };
  const labelBaseStyle = { padding: '4px 10px', border: `1px solid ${isDark ? '#9ca3af' : '#d1d5db'}`, borderRadius: '3px', cursor: 'pointer', display: 'inline-flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', fontWeight: '500', fontSize: '13px', transition: 'all 0.2s', userSelect: 'none', minWidth: '45px', textAlign: 'center', flex: 1, background: isDark ? '#374151' : '#fff', color: isDark ? '#e5e7eb' : 'inherit' };
  const checkedStyle = { background: '#D45D44', color: 'white', borderColor: '#D45D44' };
  const disabledStyle = { cursor: 'not-allowed', opacity: 0.5 };
  const subtitleStyle = { display: 'block', fontSize: '9px', marginTop: '1px', lineHeight: '1.1', opacity: 0.7 };
  const commandDisplayStyle = { flex: 1, padding: '12px 16px', background: isDark ? '#111827' : '#f5f5f5', borderRadius: '6px', fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace", fontSize: '12px', lineHeight: '1.5', color: isDark ? '#e5e7eb' : '#374151', whiteSpace: 'pre-wrap', overflowX: 'auto', margin: 0, border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}` };

  return (
    <div style={containerStyle} className="not-prose">
      {Object.entries(options).map(([key, option]) => (
        <div key={key} style={cardStyle}>
          <div style={titleStyle}>{option.title}</div>
          <div style={itemsStyle}>
            {option.items.map(item => {
              const isChecked = values[option.name] === item.id;
              const isDisabled = isOptionDisabled(option.name, item.id);
              const disabledReason = getDisabledReason(option.name, item.id);
              return (
                <label key={item.id} style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}), ...(isDisabled ? disabledStyle : {}) }} title={disabledReason}>
                  <input type="radio" name={option.name} value={item.id} checked={isChecked} disabled={isDisabled} onChange={() => handleRadioChange(option.name, item.id)} style={{ display: 'none' }} />
                  {item.label}
                  {item.subtitle && <small style={{ ...subtitleStyle, color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit' }}>{item.subtitle}</small>}
                </label>
              );
            })}
          </div>
        </div>
      ))}
      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{generateCommand()}</pre>
      </div>
    </div>
  );
};

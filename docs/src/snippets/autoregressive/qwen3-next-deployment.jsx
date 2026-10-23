export const Qwen3NextDeployment = () => {
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'b200', label: 'B200', default: true },
        { id: 'b300', label: 'B300', default: false },
        { id: 'h200', label: 'H200', default: false },
        { id: 'h100', label: 'H100', default: false },
        { id: 'mi300x', label: 'MI300X', default: false },
        { id: 'mi325x', label: 'MI325X', default: false },
        { id: 'mi355x', label: 'MI355X', default: false },
        { id: 'a3', label: 'A3 (NPU)', default: false },
        { id: 'xeon', label: 'Xeon', default: false }
      ]
    },
    modelsize: {
      name: 'modelsize',
      title: 'Model Size',
      items: [
        { id: '80b', label: '80B', subtitle: 'MOE', default: true },
      ]
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      getDynamicItems: (values) => values.hardware === 'a3'
        ? [{ id: 'w8a8', label: 'W8A8', subtitle: 'ModelSlim', default: true }]
        : [
            { id: 'bf16', label: 'BF16', subtitle: 'Full Weights', default: true },
            { id: 'fp8', label: 'FP8', subtitle: 'High Throughput', default: false }
          ]
    },
    thinking: {
      name: 'thinking',
      title: 'Thinking Capabilities',
      getDynamicItems: (values) => values.hardware === 'a3'
        ? [{ id: 'instruct', label: 'Instruct', subtitle: 'General Purpose', default: true }]
        : [
            { id: 'instruct', label: 'Instruct', subtitle: 'General Purpose', default: true },
            { id: 'thinking', label: 'Thinking', subtitle: 'Reasoning / CoT', default: false }
          ],
      commandRule: (value) => value === 'thinking' ? '--reasoning-parser qwen3' : null
    },
    modelPath: {
      name: 'modelPath',
      title: 'Main Model Path',
      type: 'text',
      condition: (values) => values.hardware === 'a3',
      default: '/path/to/Qwen3-Next-80B-A3B-Instruct-W8A8',
      placeholder: '/path/to/Qwen3-Next-80B-A3B-Instruct-W8A8',
    },
    draftModelPath: {
      name: 'draftModelPath',
      title: 'Draft Model Path',
      type: 'text',
      condition: (values) => values.hardware === 'a3',
      default: '/path/to/Qwen3-Next-80B-A3B-Instruct',
      placeholder: '/path/to/Qwen3-Next-80B-A3B-Instruct',
    },
    networkIface: {
      name: 'networkIface',
      title: 'HCCL/Gloo Interface',
      type: 'text',
      condition: (values) => values.hardware === 'a3',
      default: 'eth0',
      placeholder: 'eth0',
    },
    hostIp: {
      name: 'hostIp',
      title: 'Host IP',
      type: 'text',
      condition: (values) => values.hardware === 'a3',
      default: '127.0.0.1',
      placeholder: '127.0.0.1',
    },
    port: {
      name: 'port',
      title: 'Port',
      type: 'text',
      condition: (values) => values.hardware === 'a3',
      default: '8888',
      placeholder: '8888',
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      condition: (values) => values.hardware !== 'a3',
      items: [
        { id: 'disabled', label: 'Disabled', default: true },
        { id: 'enabled', label: 'Enabled', default: false }
      ],
      commandRule: (value) => value === 'enabled' ? '--tool-call-parser qwen' : null
    },
    speculative: {
      name: 'speculative',
      title: 'Speculative Decoding',
      condition: (values) => values.hardware !== 'xeon' && values.hardware !== 'a3',
      items: [
        { id: 'disabled', label: 'Disabled', default: true },
        { id: 'enabled', label: 'Enabled', default: false }
      ],
      commandRule: (value) => value === 'enabled' ? '--speculative-algorithm EAGLE \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 4' : null
    },
    mambaCache: {
      name: 'mambaCache',
      title: 'Mamba Radix Cache',
      condition: (values) => values.hardware !== 'xeon' && values.hardware !== 'a3',
      items: [
        { id: 'v1', label: 'V1', default: true },
        { id: 'v2', label: 'V2', default: false }
      ],
      commandRule: (value) => value === 'v2' ? '--mamba-radix-cache-strategy extra_buffer \\\n  --page-size 64' : null
    }
  };

  const modelConfigs = {
    '80b': {
      baseName: '80B-A3B',
      isMOE: true,
      h100: { tp: 4, ep: 0, bf16: true, fp8: true },
      h200: { tp: 2, ep: 0, bf16: true, fp8: true },
      b200: { tp: 2, ep: 0, bf16: true, fp8: true },
      b300: { tp: 2, ep: 0, bf16: true, fp8: true },
      mi300x: { tp: 2, ep: 0, bf16: true, fp8: true },
      mi325x: { tp: 2, ep: 0, bf16: true, fp8: true },
      mi355x: { tp: 2, ep: 0, bf16: true, fp8: true },
      a3: { tp: 2, ep: 0, w8a8: true },
      xeon: { tp: 3, ep: 0, bf16: true, fp8: true }
    }
  };

  const generateCommand = (values) => {
    const { hardware, modelsize: modelSize, quantization, thinking } = values;
    const commandKey = `${hardware}-${modelSize}-${quantization}-${thinking}`;

    const modelSizeConfig = modelConfigs[modelSize];
    if (!modelSizeConfig) {
      return `# Error: Unknown model size: ${modelSize}`;
    }

    const hwConfig = modelSizeConfig[hardware];
    if (!hwConfig) {
      return `# Error: Unknown hardware platform: ${hardware}`;
    }

    if (hardware === 'a3') {
      if (quantization !== 'w8a8' || thinking !== 'instruct') {
        return '# A3 supports the W8A8 Instruct recipe only';
      }

      const npuEnv = [
        'ASCEND_USE_FIA=1',
        'DEEPEP_HCCL_BUFFSIZE=2000',
        'DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048',
        'DEEPEP_NORMAL_LONG_SEQ_ROUND=10',
        'FORCE_DRAFT_MODEL_NON_QUANT=1',
        'GLOO_SOCKET_IFNAME={{networkIface}}',
        'HCCL_OP_EXPANSION_MODE=AIV',
        'HCCL_SOCKET_IFNAME={{networkIface}}',
        'PYTORCH_NPU_ALLOC_CONF=expandable_segments:True',
        'SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=400',
        'SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1',
        'SGLANG_ENABLE_SPEC_V2=1',
        'SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0',
        'SGLANG_NPU_USE_MULTI_STREAM=0',
        'SGLANG_SET_CPU_AFFINITY=1',
        'SGLANG_WARMUP_TIMEOUT=3600',
        'STREAMS_PER_DEVICE=32',
        'TASK_QUEUE_ENABLE=1',
        'ZBCCL_BOOTSTRAP_URL=tcp://127.0.0.1:24669',
        'ZBCCL_ENABLE_GRAPH=1',
        'ZBCCL_LOCAL_MEM_SIZE=60416',
        'ZBCCL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True',
      ];
      const npuFlags = [
        '--trust-remote-code',
        '--model-path {{modelPath}}',
        '--attention-backend ascend',
        '--device npu',
        '--quantization modelslim',
        '--page-size 128',
        '--tp-size 2',
        '--watchdog-timeout 9000',
        '--mem-fraction-static 0.85',
        '--disable-radix-cache',
        '--max-prefill-tokens 28672',
        '--context-length 26384',
        '--max-total-tokens 122304',
        '--speculative-algorithm NEXTN',
        '--speculative-num-steps 3',
        '--speculative-eagle-topk 1',
        '--speculative-num-draft-tokens 4',
        '--speculative-draft-model-quantization unquant',
        '--speculative-draft-model-path {{draftModelPath}}',
        '--chunked-prefill-size -1',
        '--max-running-requests 2',
        '--cuda-graph-bs-decode 2',
        '--mamba-ssm-dtype bfloat16',
        '--reasoning-parser qwen3',
        '--tool-call-parser qwen3_coder',
        '--host {{hostIp}}',
        '--port {{port}}',
      ];
      const replacePlaceholders = (line) => line
        .replaceAll('{{networkIface}}', values.networkIface)
        .replaceAll('{{modelPath}}', values.modelPath)
        .replaceAll('{{draftModelPath}}', values.draftModelPath)
        .replaceAll('{{hostIp}}', values.hostIp)
        .replaceAll('{{port}}', values.port);
      return `${npuEnv.map(replacePlaceholders).join(' \\\n')} \\\npython -m sglang.launch_server \\\n${npuFlags.map(replacePlaceholders).map((flag) => `  ${flag}`).join(' \\\n')}`;
    }

    const quantSuffix = quantization === 'fp8' ? '-FP8' : '';
    const thinkingSuffix = thinking === 'thinking' ? '-Thinking' : '-Instruct';
    const modelName = `Qwen/Qwen3-Next-${modelSizeConfig.baseName}${thinkingSuffix}${quantSuffix}`;

    let cmd = 'python -m sglang.launch_server \\\n';
    cmd += `  --model ${modelName}`;

    if (hardware === 'xeon') {
      cmd += ` \\\n  --device cpu \\\n  --disable-overlap-schedule`;
    }

    if (hwConfig.tp > 1) {
      cmd += ` \\\n  --tp ${hwConfig.tp}`;
    }

    let ep = hwConfig.ep;
    if (quantization === 'fp8' && hwConfig.tp === 8) {
      ep = 2;
    }

    if (ep > 0) {
      cmd += ` \\\n  --ep ${ep}`;
    }

    for (const [key, option] of Object.entries(options)) {
      if (option.condition && !option.condition(values)) {
        continue;
      }
      if (option.commandRule) {
        const rule = option.commandRule(values[key], values);
        if (rule) {
          cmd += ` \\\n  ${rule}`;
        }
      }
    }

    // AMD GPUs require triton attention backend
    if (hardware === 'mi300x' || hardware === 'mi325x' || hardware === 'mi355x') {
      cmd += ` \\\n  --attention-backend triton`;
    }
    if (hardware === 'b300') {
      cmd += ` \\\n  --attention-backend flashinfer`;
      cmd += ` \\\n  --enforce-disable-flashinfer-allreduce-fusion`;
    }

    return cmd;
  };

  const getInitialState = () => {
    const initialState = {};
    Object.entries(options).forEach(([key, option]) => {
      if (option.type === 'checkbox') {
        initialState[key] = (option.items || [])
          .filter((item) => item.default)
          .map((item) => item.id);
        return;
      }
      if (option.type === 'text') {
        initialState[key] = option.default || '';
        return;
      }
      let items = option.items || [];
      if (option.getDynamicItems) {
        const defaultValues = {};
        Object.entries(options).forEach(([innerKey, innerOption]) => {
          if (innerOption.type === 'checkbox') {
            defaultValues[innerKey] = (innerOption.items || [])
              .filter((item) => item.default)
              .map((item) => item.id);
          } else if (innerOption.type === 'text') {
            defaultValues[innerKey] = innerOption.default || '';
          } else if (innerOption.items && innerOption.items.length > 0) {
            const defaultItem = innerOption.items.find((item) => item.default);
            defaultValues[innerKey] = defaultItem ? defaultItem.id : innerOption.items[0].id;
          }
        });
        items = option.getDynamicItems(defaultValues);
      }
      const defaultItem = items && items.find((item) => item.default);
      initialState[key] = defaultItem ? defaultItem.id : items && items[0] ? items[0].id : '';
    });
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
      if (optionName !== 'hardware') {
        return { ...prev, [optionName]: value };
      }
      return {
        ...prev,
        hardware: value,
        quantization: value === 'a3' ? 'w8a8' : 'bf16',
        thinking: 'instruct',
      };
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

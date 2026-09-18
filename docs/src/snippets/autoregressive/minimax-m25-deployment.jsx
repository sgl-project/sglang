export const MiniMaxM25Deployment = () => {
  const modelFamily = 'MiniMaxAI';

  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h200', label: 'H200', default: true },
        { id: 'b200', label: 'B200', default: false },
        { id: 'a100', label: 'A100', default: false },
        { id: 'h100', label: 'H100', default: false },
        { id: 'mi300x', label: 'MI300X', default: false },
        { id: 'mi325x', label: 'MI325X', default: false },
        { id: 'mi355x', label: 'MI355X', default: false },
        { id: 'a3', label: 'Ascend A3', default: false }
      ]
    },
    gpuCount: {
      name: 'gpuCount',
      title: 'GPU Count',
      condition: (values) => values.hardware !== 'a3',
      getDynamicItems: (values) => {
        const isAMD = values.hardware === 'mi300x' || values.hardware === 'mi325x' || values.hardware === 'mi355x';
        return [
          {
            id: '2gpu',
            label: '2',
            default: isAMD,
            disabled: !isAMD
          },
          {
            id: '4gpu',
            label: '4',
            default: !isAMD,
            disabled: false
          },
          {
            id: '8gpu',
            label: '8',
            default: false,
            disabled: false
          }
        ];
      }
    },
    thinking: {
      name: 'thinking',
      title: 'Thinking Capabilities',
      condition: (values) => values.hardware !== 'a3',
      items: [
        { id: 'disabled', label: 'Disabled', default: true },
        { id: 'enabled', label: 'Enabled', default: false }
      ],
      commandRule: (value) => value === 'enabled' ? '--reasoning-parser minimax-append-think' : null
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      condition: (values) => values.hardware !== 'a3',
      items: [
        { id: 'disabled', label: 'Disabled', default: true },
        { id: 'enabled', label: 'Enabled', default: false }
      ],
      commandRule: (value) => value === 'enabled' ? '--tool-call-parser minimax-m2' : null
    },
    ascendPreset: {
      name: 'ascendPreset',
      title: 'A3 Configuration',
      condition: (values) => values.hardware === 'a3',
      items: [
        {
          id: '8p-in3k5-out1k5',
          label: '1 node / 8 cards / 16 dies',
          subtitle: 'PD co-located · W8A8 + EAGLE3 · 3.5k input / 1.5k output',
          default: true
        }
      ]
    },
    modelPath: {
      name: 'modelPath',
      title: 'W8A8 Model Directory',
      type: 'text',
      default: '/models/MiniMax-M2.5-w8a8-QuaRot',
      condition: (values) => values.hardware === 'a3'
    },
    draftModelPath: {
      name: 'draftModelPath',
      title: 'EAGLE3 Directory',
      type: 'text',
      default: '/models/MiniMax-M2.5-eagel-model-0318',
      condition: (values) => values.hardware === 'a3'
    },
    networkInterface: {
      name: 'networkInterface',
      title: 'HCCL / Gloo Interface',
      type: 'text',
      default: 'lo',
      condition: (values) => values.hardware === 'a3'
    }
  };

  const generateCommand = (values) => {
    const { hardware, gpuCount, thinking, toolcall } = values;

    if (hardware === 'a3') {
      const { ascendPreset, modelPath, draftModelPath, networkInterface } = values;
      if (ascendPreset !== '8p-in3k5-out1k5') {
        return '# Select the A3 configuration: 1 node / 8 cards / 16 dies.';
      }
      const isAbsolutePath = (path) => typeof path === 'string' && path.startsWith('/') && !/[\u0000-\u001f\u007f]/.test(path);
      if (!isAbsolutePath(modelPath) || !isAbsolutePath(draftModelPath)) {
        return '# Enter absolute paths to both model directories inside the container.';
      }
      if (typeof networkInterface !== 'string' || !/^[a-zA-Z0-9_.:-]+$/.test(networkInterface)) {
        return '# Enter a valid network interface name from ip -brief address.';
      }
      const shellQuote = (value) => "'" + value.replace(/'/g, "'\\''") + "'";
      const environment = [
        '# Ascend A3: 1 node / 8 cards / 16 dies, PD co-located',
        '# Complete the Ascend prerequisites in this guide before launching.',
        `MODEL_PATH=${shellQuote(modelPath)}`,
        `DRAFT_MODEL_PATH=${shellQuote(draftModelPath)}`,
        'export PYTHONPATH="${DRAFT_MODEL_PATH}${PYTHONPATH:+:${PYTHONPATH}}"',
        '',
        'unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING',
        'source /usr/local/Ascend/ascend-toolkit/set_env.sh',
        'source /usr/local/Ascend/nnal/atb/set_env.sh',
        '',
        'export ASCEND_USE_FIA=1',
        'export DEEPEP_HCCL_BUFFSIZE=1024',
        `export GLOO_SOCKET_IFNAME=${shellQuote(networkInterface)}`,
        `export HCCL_SOCKET_IFNAME=${shellQuote(networkInterface)}`,
        'export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True',
        'export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=204800',
        'export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1',
        'export SGLANG_EXTERNAL_MODEL_PACKAGE=custom_eagle3',
        'export SGLANG_SET_CPU_AFFINITY=1',
        'export STREAMS_PER_DEVICE=32',
        'export TASK_QUEUE_ENABLE=1',
        ''
      ];
      const launch = [
        'python3 -m sglang.launch_server',
        '    --model-path "$MODEL_PATH"',
        '    --host 127.0.0.1 --port 6688',
        '    --tp-size 16',
        '    --enable-dp-attention',
        '    --dp-size 16',
        '    --mem-fraction-static 0.75',
        '    --max-running-requests 320',
        '    --disable-radix-cache',
        '    --reasoning-parser minimax-append-think',
        '    --tool-call-parser minimax-m2',
        '    --prefill-delayer-max-delay-passes 500',
        '    --enable-prefill-delayer',
        '    --chunked-prefill-size 196608',
        '    --max-prefill-tokens 8192',
        '    --cuda-graph-bs-decode 1 2 4 8 12 16 20',
        '    --moe-a2a-backend ascend_fuseep',
        '    --fuseep-mode 2',
        '    --quantization modelslim',
        '    --speculative-algorithm EAGLE3',
        '    --speculative-draft-model-path "$DRAFT_MODEL_PATH"',
        '    --speculative-num-steps 3',
        '    --speculative-eagle-topk 1',
        '    --speculative-num-draft-tokens 4',
        '    --speculative-draft-model-quantization unquant',
        '    --dtype bfloat16',
        '    --device npu'
      ];
      return environment.join('\n') + '\n' + launch.join(' \\\n');
    }

    const isAMD = hardware === 'mi300x' || hardware === 'mi325x' || hardware === 'mi355x';
    if (gpuCount === '2gpu' && !isAMD) {
      return '# Please select compatible hardware\n# 2-GPU requires AMD MI300X/MI325X/MI355X';
    }

    const modelName = `${modelFamily}/MiniMax-M2.5`;

    const isBlackwell = hardware === 'b200';
    const useAllreduceFusion = hardware === 'h200' || hardware === 'b200';

    let cmd = '';
    if (useAllreduceFusion) {
      cmd += 'SGLANG_USE_FUSED_PARALLEL_QKNORM=1 \\\n';
    }
    cmd += 'python -m sglang.launch_server \\\n';
    cmd += `  --model-path ${modelName}`;

    if (gpuCount === '8gpu') {
      cmd += ` \\\n  --tp 8`;
      cmd += ` \\\n  --ep 8`;
    } else if (gpuCount === '4gpu') {
      cmd += ` \\\n  --tp 4`;
      if (isAMD) {
        cmd += ` \\\n  --ep 4`;
      }
    } else if (gpuCount === '2gpu') {
      cmd += ` \\\n  --tp 2`;
      if (isAMD) {
        cmd += ` \\\n  --ep 2`;
      }
    }

    if (toolcall === 'enabled') {
      cmd += ` \\\n  --tool-call-parser minimax-m2`;
    }

    if (thinking === 'enabled') {
      cmd += ` \\\n  --reasoning-parser minimax-append-think`;
    }

    cmd += ` \\\n  --trust-remote-code`;
    cmd += ` \\\n  --mem-fraction-static 0.85`;

    if (isBlackwell) {
      cmd += ` \\\n  --moe-runner-backend flashinfer_trtllm_routed`;
      cmd += ` \\\n  --fp8-gemm-backend flashinfer_trtllm`;
      cmd += ` \\\n  --dtype bfloat16`;
    }

    if (useAllreduceFusion) {
      cmd += ` \\\n  --enable-flashinfer-allreduce-fusion`;
    }

    if (isAMD) {
      cmd += ` \\\n  --kv-cache-dtype fp8_e4m3`;
      cmd += ` \\\n  --attention-backend triton`;
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

  const getUpdatedValues = (previous, optionName, value) => {
    const next = { ...previous, [optionName]: value };
    if (optionName === 'hardware' && value !== 'a3') {
      const items = options.gpuCount.getDynamicItems(next);
      if (!items.some((item) => item.id === next.gpuCount && !item.disabled)) {
        next.gpuCount = items.find((item) => item.default && !item.disabled).id;
      }
    }
    return next;
  };

  const [values, setValues] = useState(getInitialState);
  const [isDark, setIsDark] = useState(false);
  const [copyState, setCopyState] = useState({ command: null, failed: false });

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
    setValues((prev) => getUpdatedValues(prev, optionName, value));
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
  const handleCopyCommand = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopyState({ command, failed: false });
    } catch {
      setCopyState({ command, failed: true });
    }
  };
  const copyLabel = copyState.command !== command
    ? 'Copy command'
    : copyState.failed ? 'Select text to copy' : 'Copied!';

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
    maxHeight: '420px',
    overflowY: 'auto',
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
                  aria-label={option.title}
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
      <div style={{ ...cardStyle, flexDirection: 'column', alignItems: 'stretch' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
          <div style={titleStyle}>Run this Command:</div>
          <button
            type="button"
            onClick={handleCopyCommand}
            aria-live="polite"
            style={{ ...labelBaseStyle, flex: 'none' }}
          >
            {copyLabel}
          </button>
        </div>
        <pre style={commandDisplayStyle}>{command}</pre>
      </div>
    </div>
  );
};

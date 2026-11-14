export const DeepSeekR1BasicDeployment = () => {
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h100', label: 'H100', default: false },
        { id: 'h200', label: 'H200', default: false },
        { id: 'b200', label: 'B200', default: true },
        { id: 'b300', label: 'B300', default: false },
        { id: 'mi300x', label: 'MI300X', default: false },
        { id: 'mi325x', label: 'MI325X', default: false },
        { id: 'mi355x', label: 'MI355X', default: false },
        { id: 'xeon', label: 'XEON', default: false },
        { id: 'ascend-a3', label: 'Ascend A3', default: false },
      ],
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      getDynamicItems: (values) => {
        if (values.hardware === 'ascend-a3') {
          return [{ id: 'w4a8', label: 'W4A8', subtitle: 'ModelSlim', default: true }];
        }
        const isXeon = values.hardware === 'xeon';
        const fp4Disabled = values.hardware === 'h100' || values.hardware === 'mi300x' || isXeon;
        return [
          { id: 'fp8', label: 'FP8', default: true },
          {
            id: 'fp4',
            label: 'FP4',
            default: false,
            disabled: fp4Disabled,
            disabledReason: isXeon
              ? 'Intel Xeon CPUs do not support FP4 quantization'
              : 'H100 and MI300X only support FP8 quantization',
          },
          {
            id: 'int8',
            label: 'INT8',
            default: false,
            disabled: !isXeon,
            disabledReason: 'INT8 is only available when XEON hardware is selected',
          },
        ];
      },
    },
    deployshape: {
      name: 'deployshape',
      title: 'A3 deployment',
      condition: (values) => values.hardware === 'ascend-a3',
      items: [
        { id: 'colocated', label: '1×A3 · PD co-located', default: true },
        { id: 'pd2node', label: '2×A3 · PD disaggregated', default: false },
      ],
    },
    ascendRole: {
      name: 'ascendRole',
      title: 'PD role',
      condition: (values) => values.hardware === 'ascend-a3' && values.deployshape === 'pd2node',
      items: [
        { id: 'prefill', label: 'Prefill', default: true },
        { id: 'decode', label: 'Decode', default: false },
        { id: 'router', label: 'Router', default: false },
      ],
    },
    strategy: {
      name: 'strategy',
      title: 'Deployment Strategy',
      condition: (values) => values.hardware !== 'ascend-a3',
      type: 'checkbox',
      items: [
        { id: 'tp', label: 'TP', subtitle: 'Tensor Parallel', default: true, required: true },
        { id: 'dp', label: 'DP', subtitle: 'Data Parallel', default: false },
        { id: 'ep', label: 'EP', subtitle: 'Expert Parallel', default: false, disabledWhen: (v) => v.hardware === 'xeon', disabledReason: 'Intel Xeon CPUs only support Tensor Parallel (TP)' },
        { id: 'mtp', label: 'MTP', subtitle: 'Multi-token Prediction', default: false, disabledWhen: (v) => v.hardware === 'xeon', disabledReason: 'Intel Xeon CPUs do not support Multi-token Prediction' },
      ],
    },
    thinking: {
      name: 'thinking',
      title: 'Reasoning Parser',
      condition: (values) => values.hardware !== 'ascend-a3',
      items: [
        { id: 'disabled', label: 'Disabled', default: true },
        { id: 'enabled', label: 'Enabled', default: false },
      ],
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      condition: (values) => values.hardware !== 'ascend-a3',
      items: [
        { id: 'disabled', label: 'Disabled', default: true },
        { id: 'enabled', label: 'Enabled', default: false },
      ],
    },
  };

  const resolveItems = (option, values) =>
    typeof option.getDynamicItems === 'function' ? option.getDynamicItems(values) : option.items;

  const getInitialState = () => {
    const initialState = {};
    for (const [key, option] of Object.entries(options)) {
      if (option.type === 'checkbox') {
        initialState[key] = (option.items || [])
          .filter((item) => item.default)
          .map((item) => item.id);
        continue;
      }

      const items = resolveItems(option, initialState) || [];
      const fallback =
        items.find((item) => item.default && !item.disabled) ||
        items.find((item) => !item.disabled) ||
        items[0];
      initialState[key] = fallback ? fallback.id : '';
    }
    return initialState;
  };

  // Measured A3 recipes; see the Cookbook for results and validation limits.
  const ascendA3ReferenceFlags = {
  "prefill": [
    "--model-path \"$MODEL_PATH\"",
    "--disaggregation-mode prefill",
    "--host \"<prefill-ip>\"",
    "--port 30000",
    "--disaggregation-bootstrap-port 8998",
    "--node-rank 0",
    "--nnodes 1",
    "--tp-size 16",
    "--mem-fraction-static 0.8",
    "--quantization modelslim",
    "--max-running-requests 32",
    "--context-length 8192",
    "--disable-radix-cache",
    "--chunked-prefill-size 20480",
    "--max-prefill-tokens 20480",
    "--moe-a2a-backend deepep",
    "--deepep-mode normal",
    "--speculative-algorithm NEXTN",
    "--speculative-num-steps 1",
    "--speculative-eagle-topk 1",
    "--speculative-num-draft-tokens 2",
    "--attn-dp-size 8",
    "--disable-shared-experts-fusion",
    "--dtype bfloat16",
    "--reasoning-parser deepseek-r1",
    "--tool-call-parser deepseekv3",
    "--max-total-tokens 80000",
    "--disaggregation-transfer-backend ascend",
    "--trust-remote-code",
    "--attention-backend ascend",
    "--device npu"
  ],
  "decode": [
    "--model-path \"$MODEL_PATH\"",
    "--disaggregation-mode decode",
    "--host \"<decode-ip>\"",
    "--port 30000",
    "--nnodes 1",
    "--tp-size 16",
    "--attn-dp-size 16",
    "--mem-fraction-static 0.9",
    "--max-running-requests 416",
    "--quantization modelslim",
    "--moe-a2a-backend deepep",
    "--deepep-mode low_latency",
    "--enable-dp-lm-head",
    "--cuda-graph-bs-decode 2 4 6 8 10 12 14 16 18 20 22 24 26",
    "--watchdog-timeout 9000",
    "--context-length 8192",
    "--speculative-algorithm NEXTN",
    "--speculative-num-steps 2",
    "--speculative-eagle-topk 1",
    "--speculative-num-draft-tokens 3",
    "--disable-shared-experts-fusion",
    "--dtype bfloat16",
    "--tokenizer-worker-num 4",
    "--load-balance-method round_robin",
    "--reasoning-parser deepseek-r1",
    "--tool-call-parser deepseekv3",
    "--max-total-tokens 145792",
    "--disaggregation-transfer-backend ascend",
    "--trust-remote-code",
    "--attention-backend ascend",
    "--device npu"
  ],
  "colocated": [
    "--model-path \"$MODEL_PATH\"",
    "--host 127.0.0.1",
    "--port 30000",
    "--tp-size 16",
    "--trust-remote-code",
    "--attention-backend ascend",
    "--device npu",
    "--quantization modelslim",
    "--watchdog-timeout 9000",
    "--cuda-graph-bs-decode 4 8 12 14",
    "--mem-fraction-static 0.9",
    "--max-running-requests 224",
    "--context-length 8188",
    "--disable-radix-cache",
    "--chunked-prefill-size 65536",
    "--max-prefill-tokens 3000",
    "--moe-a2a-backend deepep",
    "--deepep-mode auto",
    "--attn-dp-size 16",
    "--enable-dp-lm-head",
    "--speculative-algorithm NEXTN",
    "--speculative-num-steps 3",
    "--speculative-eagle-topk 1",
    "--speculative-num-draft-tokens 4",
    "--dtype bfloat16",
    "--reasoning-parser deepseek-r1",
    "--tool-call-parser deepseekv3",
    "--max-total-tokens 100000"
  ]
};
  const ascendA3Environment = {
  "colocated": {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_USE_FIA_NZ": "1",
    "STREAMS_PER_DEVICE": "32",
    "DEEPEP_HCCL_BUFFSIZE": "2000",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "72",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1"
  },
  "prefill": {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_USE_FIA_NZ": "1",
    "STREAMS_PER_DEVICE": "32",
    "ENABLE_MOE_NZ": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "DEEPEP_HCCL_BUFFSIZE": "4000",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "ASCEND_MF_TRANSFER_PROTOCOL": "sdma"
  },
  "decode": {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_USE_FIA_NZ": "1",
    "STREAMS_PER_DEVICE": "32",
    "ENABLE_MOE_NZ": "1",
    "DEEPEP_HCCL_BUFFSIZE": "800",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "78",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "TASK_QUEUE_ENABLE": "1",
    "ASCEND_MF_TRANSFER_PROTOCOL": "sdma"
  }
};

  // Pack words onto lines of at most `width` characters; a word is never split.
  const packWords = (words, first, rest, width, tail) => {
    const out = [];
    let line = first;
    let empty = true;
    words.forEach((word) => {
      const candidate = empty ? line + word : `${line} ${word}`;
      if (!empty && candidate.length + tail.length > width) {
        out.push(line + tail);
        line = rest + word;
      } else {
        line = candidate;
      }
      empty = false;
    });
    out.push(line);
    return out;
  };

  const generateAscendA3Command = (values) => {
    if (values.quantization !== 'w4a8') {
      return '# Select ModelSlim W4A8 for Ascend A3.';
    }
    if (!['colocated', 'pd2node'].includes(values.deployshape)) {
      return '# Select one A3 deployment scenario.';
    }
    const role = values.deployshape === 'colocated' ? 'colocated' : values.ascendRole;
    if (!['colocated', 'prefill', 'decode', 'router'].includes(role)) {
      return '# Select Prefill, Decode, or Router.';
    }
    const lines = ['# Ascend A3 ModelSlim W4A8: replace the <...> placeholders and the model path.'];
    if (role === 'router') {
      lines.push(
        ...packWords(
          ['--pd-disaggregation', '--policy cache_aware', '--prefill "http://<prefill-ip>:30000" 8998',
            '--decode "http://<decode-ip>:30000"', '--host 127.0.0.1', '--port 6688'],
          'python -m sglang_router.launch_router ', '  ', 80, ' \\',
        ),
      );
      return lines.join('\n');
    }
    lines.push(
      'MODEL_PATH="/path/to/DeepSeek-R1-0528-w4a8"',
      'export HCCL_SOCKET_IFNAME=<network-interface> GLOO_SOCKET_IFNAME=<network-interface>',
    );
    if (role !== 'colocated') {
      lines.push(
        'export ASCEND_MF_STORE_URL="tcp://<prefill-ip>:24670"',
        'export MF_CONFIG_STORE_URL="tcp://<prefill-ip>:24670"',
      );
    }
    lines.push(
      'source /usr/local/Ascend/ascend-toolkit/set_env.sh',
      'source /usr/local/Ascend/nnal/atb/set_env.sh',
      'unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING',
      ...packWords(
        Object.entries(ascendA3Environment[role]).map(([key, value]) => `${key}=${value}`),
        'export ', 'export ', 80, '',
      ),
      ...packWords(ascendA3ReferenceFlags[role], 'python3 -m sglang.launch_server ', '  ', 80, ' \\'),
    );
    return lines.join('\n');
  };

  const generateCommand = (values) => {
    if (values.hardware === 'ascend-a3') return generateAscendA3Command(values);
    const { hardware, quantization, strategy, thinking, toolcall } = values;
    const strategyValues = Array.isArray(strategy) ? strategy : [];

    if ((hardware === 'h100' || hardware === 'mi300x') && quantization === 'fp4') {
      return '# Error: H100 and MI300X only support FP8 quantization';
    }

    const isXeon = hardware === 'xeon';
    const modelPath =
      quantization === 'fp4'
        ? 'nvidia/DeepSeek-R1-0528-FP4-v2'
        : quantization === 'int8'
        ? 'Conexis/DeepSeek-R1-0528-Channel-INT8'
        : 'deepseek-ai/DeepSeek-R1-0528';

    let command = 'python3 -m sglang.launch_server \\\n';
    command += `  --model-path ${modelPath}`;

    if (strategyValues.includes('tp')) {
      command += isXeon ? ' \\\n  --tp 6' : ' \\\n  --tp 8';
    }
    if (strategyValues.includes('dp')) {
      if(!isXeon) {
        command += ' \\\n  --attn-dp-size 8';
      }
      else {
        command += ' \\\n  --dp 6';
        command += ' \\\n  --enable-dp-attention';
      }
    }
    if (strategyValues.includes('ep')) {
      command += ' \\\n  --ep 8';
    }
    if (strategyValues.includes('mtp')) {
      command +=
        ' \\\n  --speculative-algorithm EAGLE' +
        ' \\\n  --speculative-num-steps 3' +
        ' \\\n  --speculative-eagle-topk 1' +
        ' \\\n  --speculative-num-draft-tokens 4';
    }

    if (!isXeon) {
      command += ' \\\n  --enable-symm-mem # Optional: improves performance, but may be unstable';
    }

    if (hardware === 'b200' || (hardware === 'mi355x' && quantization === 'fp8')) {
      command +=
        ' \\\n  --kv-cache-dtype fp8_e4m3 # Optional: enables fp8 kv cache and fp8 attention kernels to improve performance';
    }

    if (hardware === 'b300') {
      command += ' \\\n  --kv-cache-dtype fp8_e4m3';
      command += ' \\\n  --attention-backend flashinfer';
      command += ' \\\n  --enforce-disable-flashinfer-allreduce-fusion';
      if (quantization === 'fp4') {
        command += ' \\\n  --moe-runner-backend flashinfer_cutlass';
      }
      if (quantization === 'fp4' || strategyValues.includes('mtp')) {
        command += ' \\\n  --mem-fraction-static 0.85';
      }
    }

    if (isXeon) {
      command += ' \\\n  --device cpu \\\n  --disable-overlap-schedule';
      if (quantization === 'int8') {
        command += ' \\\n  --quantization w8a8_int8';
      }
    }

    if (thinking === 'enabled') {
      command += ' \\\n  --reasoning-parser deepseek-r1';
    }
    if (toolcall === 'enabled') {
      command +=
        ' \\\n  --tool-call-parser deepseekv3' +
        ' \\\n  --chat-template examples/chat_template/tool_chat_template_deepseekr1.jinja';
    }

    return command;
  };

  const getUpdatedValues = (prev, optionName, value) => {
    const next = { ...prev, [optionName]: value };
    if (optionName === 'hardware') {
      const quantizationItems = resolveItems(options.quantization, next);
      const current = quantizationItems.find((item) => item.id === next.quantization);
      if (!current || current.disabled) {
        const fallback =
          quantizationItems.find((item) => item.default && !item.disabled) ||
          quantizationItems.find((item) => !item.disabled);
        if (fallback) {
          next.quantization = fallback.id;
        }
      }
    }
    if (optionName === 'hardware') {
      if (next.hardware === 'xeon') {
        next.quantization = 'int8';
      } else if (next.quantization === 'int8') {
        next.quantization = 'fp8';
      }
    }
    const strategyItems = options.strategy.items || [];
    const currentStrategy = Array.isArray(next.strategy) ? next.strategy : [];
    next.strategy = currentStrategy.filter((id) => {
      const item = strategyItems.find((s) => s.id === id);
      if (!item) {
        return false;
      }
      if (typeof item.disabledWhen === 'function' && item.disabledWhen(next)) {
        return false;
      }
      return true;
    });
    return next;
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
                        name={`deepseek-r1-basic-${option.name}`}
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

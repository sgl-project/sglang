export const MiMoV2FlashDeployment = () => {
  // Config mirrors sgl-cookbook src/components/autoregressive/MiMoConfigGenerator/index.js.
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h200', label: 'H200', default: true },
        { id: 'h100', label: 'H100', default: false },
        { id: 'mi355x', label: 'MI355X', default: false },
        { id: 'ascend-a3', label: 'A3', default: false }
      ]
    },
    modelname: {
      name: 'modelname',
      title: 'Model Name',
      items: [
        { id: 'mimo-v2-flash', label: 'MiMo-V2-Flash', default: true }
      ]
    },
    strategy: {
      name: 'strategy',
      title: 'Deployment Strategy',
      type: 'checkbox',
      items: [
        { id: 'tp', label: 'TP 8 (Required)', default: true, disabled: true },
        { id: 'dp', label: 'DP Attention (DP 2)', default: true },
        { id: 'mtp', label: 'Multi-token Prediction (MTP)', default: true },
        { id: 'optimization', label: 'Performance Optimizations', default: true }
      ]
    },
    topology: {
      name: 'topology',
      title: 'Topology',
      type: 'checkbox',
      items: [
        { id: 'pd', label: 'PD Disaggregation', subtitle: '1 prefill + 1 decode + router', default: true, disabled: true }
      ]
    },
    reasoning: {
      name: 'reasoning',
      title: 'Reasoning & Tools',
      type: 'checkbox',
      items: [
        { id: 'reasoning', label: 'Reasoning Parser (Qwen3)', default: true },
        { id: 'toolcall', label: 'Tool Call Parser', default: true }
      ]
    }
  };

  // Initialize state
  const getInitialState = () => {
    const initialState = {};
    Object.entries(options).forEach(([key, option]) => {
      if (option.type === 'checkbox') {
        initialState[key] = option.items.filter(item => item.default).map(item => item.id);
      } else {
        const defaultItem = option.items.find(item => item.default);
        initialState[key] = defaultItem ? defaultItem.id : option.items[0].id;
      }
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
    setValues(prev => {
      const next = { ...prev, [optionName]: value };
      if (optionName === 'hardware') {
        // The A3 recipe is a fixed multi-node PD command; the strategy row is locked to it,
        // and reasoning/tool-call parsers don't apply.
        if (value === 'ascend-a3') {
          next.strategy = options.strategy.items.filter(item => item.default).map(item => item.id);
          next.reasoning = [];
        } else if ((prev.reasoning || []).length === 0) {
          next.reasoning = options.reasoning.items.filter(item => item.default).map(item => item.id);
        }
      }
      return next;
    });
  };

  const handleCheckboxChange = (optionName, itemId, isChecked) => {
    setValues(prev => {
      const currentValues = prev[optionName] || [];
      if (isChecked) {
        return { ...prev, [optionName]: [...currentValues, itemId] };
      } else {
        return { ...prev, [optionName]: currentValues.filter(id => id !== itemId) };
      }
    });
  };

  // Ascend A3: validated 2-node PD-disaggregated deployment (1 prefill + 1 decode + router).
  // Mirrors the official Ascend best practice for MiMo-V2-Flash W8A8 on A3.
  const generateAscendA3Command = (strategyArray) => {
    const eagleFlags = strategyArray.includes('mtp')
      ? ' --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-multi-layer-eagle'
      : '';
    const prefillDp = strategyArray.includes('dp') ? ' --dp-size 2' : '';
    const decodeDp = strategyArray.includes('dp') ? ' --attn-dp-size 2 --enable-dp-lm-head' : '';

    const lines = [
      '# ============================================================',
      '# Ascend A3 PD disaggregation: 1 prefill node + 1 decode node.',
      '# Replace <prefill-ip>, <decode-ip>, <nic> and MODEL_PATH first.',
      '# ============================================================',
      '',
      '# ---- Run on BOTH nodes: system tuning + environment ----',
      'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor',
      'sysctl -w vm.swappiness=0',
      'sysctl -w kernel.numa_balancing=0',
      'sysctl -w kernel.sched_migration_cost_ns=50000',
      '',
      'source /usr/local/Ascend/ascend-toolkit/set_env.sh',
      'source /usr/local/Ascend/nnal/atb/set_env.sh',
      '',
      'export ASCEND_USE_FIA=1',
      'export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=3584',
      'export DEEPEP_NORMAL_LONG_SEQ_ROUND=32',
      'export DEEP_NORMAL_MODE_USE_INT8_QUANT=1',
      'export HCCL_CONNECT_TIMEOUT=1800',
      'export HCCL_OP_EXPANSION_MODE=AIV',
      'export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True',
      'export SGLANG_DEEPEP_BF16_DISPATCH=0',
      'export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600',
      'export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600',
      'export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0',
      'export SGLANG_SET_CPU_AFFINITY=1',
      'export STREAMS_PER_DEVICE=32',
      'export GLOO_SOCKET_IFNAME=<nic>',
      'export HCCL_SOCKET_IFNAME=<nic>',
      '# KV transfer store runs on the prefill node; the decode node must reach it',
      'export ASCEND_MF_STORE_URL=tcp://<prefill-ip>:24670',
      'MODEL_PATH=/path/to/MiMo-V2-Flash-W8A8',
      '',
      '# ---- Prefill node (A3, TP8 / DP2) ----',
      'export DEEPEP_HCCL_BUFFSIZE=1024',
      'export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32',
      'export SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK=1',
      'sglang serve --model-path ${MODEL_PATH} --disaggregation-mode prefill --host <prefill-ip> --port 30000 --disaggregation-bootstrap-port 8998 --node-rank 0 --attention-backend ascend --device npu --tp-size 8 --nnodes 1 --chunked-prefill-size 8192 --trust-remote-code --max-running-requests 64 --mem-fraction-static 0.8 --swa-full-tokens-ratio 0.3 --disaggregation-transfer-backend ascend --disable-radix-cache --disable-cuda-graph --cuda-graph-backend-prefill=disabled' + prefillDp,
      '',
      '# ---- Decode node (A3, TP16) ----',
      'export DEEPEP_HCCL_BUFFSIZE=800',
      'export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128',
      'sglang serve --model-path ${MODEL_PATH} --disaggregation-mode decode --host <decode-ip> --port 30000 --attention-backend ascend --device npu --tp-size 16 --nnodes 1 --trust-remote-code --max-running-requests 64 --mem-fraction-static 0.8 --swa-full-tokens-ratio 0.3 --cuda-graph-bs-decode 1 2 4 8 12 16 20 24 28 32 --disaggregation-transfer-backend ascend --disable-radix-cache' + decodeDp + ' --moe-a2a-backend deepep --deepep-mode low_latency' + eagleFlags,
      '',
      '# ---- Router (any host that can reach both nodes) ----',
      'python -m sglang_router.launch_router --pd-disaggregation --policy cache_aware --prefill http://<prefill-ip>:30000 8998 --decode http://<decode-ip>:30000 --host 0.0.0.0 --port 6688 --health-check-interval-secs 3600 --mini-lb',
    ];
    return lines.join('\n');
  };

  // Generate command — mirrors sgl-cookbook's config.generateCommand(values) exactly
  const generateCommand = () => {
    const { hardware, strategy, reasoning } = values;
    const isMI355X = hardware === 'mi355x';
    const isAscendA3 = hardware === 'ascend-a3';

    const modelPath = 'XiaomiMiMo/MiMo-V2-Flash';
    const strategyArray = Array.isArray(strategy) ? strategy : [];
    const reasoningArray = Array.isArray(reasoning) ? reasoning : [];

    if (isAscendA3) {
      return generateAscendA3Command(strategyArray);
    }

    if (isMI355X && strategyArray.includes('mtp')) {
      return '# MI355X Speculative Decoding (EAGLE): Work In Progress\n'
        + '# Uncheck "Multi-token Prediction (MTP)" to view the validated non-speculative MI355X command.';
    }

    const commandPrefix = isMI355X
      ? 'PYTHONPATH=/sgl-workspace/aiter SGLANG_USE_AITER=0 USE_ROCM_AITER_ROPE_BACKEND=0 '
      : '';
    const tpSize = isMI355X ? 4 : 8;

    let cmd = `${commandPrefix}sglang serve \\\n`;
    cmd += `  --model-path ${modelPath} \\\n`;
    cmd += `  --trust-remote-code \\\n`;
    cmd += `  --tp-size ${tpSize}`;

    // DP settings
    if (!isMI355X && strategyArray.includes('dp')) {
      cmd += ` \\\n  --attn-dp-size 2`;
    }

    // Performance Optimizations
    if (strategyArray.includes('optimization')) {
      cmd += ` \\\n  --mem-fraction-static 0.75 \\\n  --max-running-requests 128 \\\n  --chunked-prefill-size 16384 \\\n  --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}'`;
      cmd += isMI355X
        ? ` \\\n  --attention-backend triton \\\n  --prefill-attention-backend triton \\\n  --decode-attention-backend triton \\\n  --disable-custom-all-reduce`
        : ` \\\n  --attention-backend fa3`;
    }

    // MTP/Speculative settings
    if (!isMI355X && strategyArray.includes('mtp')) {
      cmd += ` \\\n  --speculative-algorithm EAGLE \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 4 \\\n  --enable-multi-layer-eagle`;
    }

    // Reasoning Parser
    if (reasoningArray.includes('reasoning')) {
      cmd += ` \\\n  --reasoning-parser qwen3`;
    }

    // Tool Call Parser
    if (reasoningArray.includes('toolcall')) {
      cmd += ` \\\n  --tool-call-parser mimo`;
    }

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
      {Object.entries(options).filter(([key]) => key !== 'topology' || values.hardware === 'ascend-a3').map(([key, option]) => (
        <div key={key} style={cardStyle}>
          <div style={titleStyle}>{option.title}</div>
          <div style={itemsStyle}>
            {option.type === 'checkbox' ? (
              option.items.map(item => {
                const isChecked = (values[option.name] || []).includes(item.id);
                const isDisabled = item.disabled || (values.hardware === 'ascend-a3' && (option.name === 'reasoning' || option.name === 'strategy'));
                return (
                  <label key={item.id} style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}), ...(isDisabled ? disabledStyle : {}) }}>
                    <input type="checkbox" checked={isChecked} disabled={isDisabled} onChange={(e) => handleCheckboxChange(option.name, item.id, e.target.checked)} style={{ display: 'none' }} />
                    {item.label}
                    {item.subtitle && <small style={{ ...subtitleStyle, color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit' }}>{item.subtitle}</small>}
                  </label>
                );
              })
            ) : (
              option.items.map(item => {
                const isChecked = values[option.name] === item.id;
                return (
                  <label key={item.id} style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}) }}>
                    <input type="radio" name={option.name} value={item.id} checked={isChecked} onChange={() => handleRadioChange(option.name, item.id)} style={{ display: 'none' }} />
                    {item.label}
                    {item.subtitle && <small style={{ ...subtitleStyle, color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit' }}>{item.subtitle}</small>}
                  </label>
                );
              })
            )}
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

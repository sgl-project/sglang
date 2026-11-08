export const DeepSeekV32Deployment = () => {
  // Config mirrors sgl-cookbook src/components/autoregressive/DeepSeekConfigGenerator/index.js.
  //
  // Model variants:
  //   DeepSeek-V3.2, V3.2-Exp, V3.2-Speciale    → deepseek-ai/ family, TP=8
  //   DeepSeek-V3.2-NVFP4                         → nvidia/ family, B200/B300 only, TP=4
  //   DeepSeek-V3.2-MXFP4                         → amd/ family, MI300X/MI355X only, TP=8
  //
  // Ascend A3: 4-node 1P1D PD disaggregation with DeepSeek-V3.2-W8A8 (2 prefill + 2 decode
  // nodes, 8 cards / 16 dies each). Mirrors the frozen best practice in
  // docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v3_2.mdx,
  // with the per-node IP detection loop replaced by the Node Role / Node Rank selectors and
  // the server port moved to the Cookbook's 30000; docs/scripts/tests/deepseek-v32-ascend.test.mjs
  // checks that nothing else diverges.
  const isAscend = (values) => values.hardware === 'a3';
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h200',   label: 'H200',          default: true  },
        { id: 'b200',   label: 'B200',          default: false },
        { id: 'b300',   label: 'B300',          default: false },
        { id: 'mi300x', label: 'MI300X',        default: false },
        { id: 'mi355x', label: 'MI355X',        default: false },
        { id: 'a3',     label: 'Ascend A3',     default: false }
      ]
    },
    modelname: {
      name: 'modelname',
      title: 'Model Name',
      getDynamicItems: (values) => {
        const hw = values.hardware;
        const isBlackwell = hw === 'b200' || hw === 'b300';
        const isAMD = hw === 'mi300x' || hw === 'mi355x';
        const isA3 = hw === 'a3';
        const a3Reason = 'Ascend A3 is validated with DeepSeek-V3.2 (W8A8) only';
        return [
          { id: 'v32',         label: 'DeepSeek-V3.2',           default: !isBlackwell && !isAMD, subtitle: isA3 ? 'W8A8' : undefined },
          { id: 'v32speciale', label: 'DeepSeek-V3.2-Speciale',  default: false, disabled: isA3, disabledReason: isA3 ? a3Reason : '' },
          { id: 'v32exp',      label: 'DeepSeek-V3.2-Exp',       default: false, disabled: isA3, disabledReason: isA3 ? a3Reason : '' },
          { id: 'v32nvfp4',    label: 'DeepSeek-V3.2-NVFP4',     default: isBlackwell, disabled: !isBlackwell, disabledReason: 'NVFP4 requires B200/B300 (Blackwell)' },
          { id: 'v32mxfp4',    label: 'DeepSeek-V3.2-MXFP4',     default: isAMD,   disabled: !isAMD,  disabledReason: 'MXFP4 requires AMD MI300X/MI355X' }
        ];
      }
    },
    npuDeployment: {
      name: 'npuDeployment',
      title: 'Deployment Scenario',
      condition: isAscend,
      items: [
        { id: '4a3-pd', label: '4 × A3 PD Disaggregation', subtitle: '2 prefill + 2 decode crads', default: true }
      ]
    },
    npuRole: {
      name: 'npuRole',
      title: 'Node Role',
      condition: isAscend,
      items: [
        { id: 'prefill', label: 'Prefill', subtitle: 'TP 32 · CP 32',        default: true  },
        { id: 'decode',  label: 'Decode',  subtitle: 'TP 32 · DP 8 · EP 32', default: false },
        { id: 'router',  label: 'Router',  subtitle: 'PD router',            default: false }
      ]
    },
    npuNodeRank: {
      name: 'npuNodeRank',
      title: 'Node Rank',
      condition: (values) => isAscend(values) && values.npuRole !== 'router',
      items: [
        { id: '0', label: 'Node 0', subtitle: 'head node', default: true  },
        { id: '1', label: 'Node 1',                        default: false }
      ]
    },
    strategy: {
      name: 'strategy',
      title: 'Deployment Strategy',
      type: 'checkbox',
      condition: (values) => !isAscend(values) && values.modelname !== 'v32nvfp4' && values.modelname !== 'v32mxfp4',
      items: [
        { id: 'tp',  label: 'TP',                       default: true,  required: true },
        { id: 'dp',  label: 'DP attention',              default: false },
        { id: 'ep',  label: 'EP',                        default: false },
        { id: 'mtp', label: 'Multi-token Prediction',    default: false }
      ]
    },
    reasoningParser: {
      name: 'reasoningParser',
      title: 'Reasoning Parser',
      condition: (values) => !isAscend(values) && values.modelname !== 'v32nvfp4' && values.modelname !== 'v32mxfp4',
      items: [
        { id: 'disabled', label: 'Disabled', default: true  },
        { id: 'enabled',  label: 'Enabled',  default: false }
      ]
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      condition: (values) => !isAscend(values) && values.modelname !== 'v32nvfp4' && values.modelname !== 'v32mxfp4' && values.modelname !== 'v32speciale',
      items: [
        { id: 'disabled', label: 'Disabled', default: true  },
        { id: 'enabled',  label: 'Enabled',  default: false }
      ]
    }
  };

  const resolveItems = (option, vals) => {
    if (typeof option.getDynamicItems === 'function') return option.getDynamicItems(vals);
    return option.items;
  };

  const getInitialState = () => {
    const initialState = {};
    for (const [key, option] of Object.entries(options)) {
      if (option.type === 'checkbox') {
        const items = resolveItems(option, initialState);
        initialState[key] = items.filter(i => i.default).map(i => i.id);
      } else {
        const items = resolveItems(option, initialState);
        const def = items.find(i => i.default && !i.disabled) || items.find(i => !i.disabled) || items[0];
        initialState[key] = def.id;
      }
    }
    return initialState;
  };

  // When hardware changes, re-resolve model name defaults (NVFP4→Blackwell, MXFP4→AMD).
  const getUpdatedValues = (prev, optionName, value) => {
    const next = { ...prev, [optionName]: value };
    if (optionName !== 'hardware') return next;
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
  };

  // Ascend A3 4-node PD disaggregation. Flags and env vars are copied verbatim from the
  // "DeepSeek-V3.2 W8A8 1P1D 32P IN128K OUT1K" best practice; only node addressing and the
  // server port differ (the best practice serves prefill on 8000 and decode on 8001).
  const npuServerPort = 30000;
  const generateAscendCommand = (values) => {
    const { npuDeployment, npuRole, npuNodeRank } = values;

    if (npuDeployment !== '4a3-pd') {
      return '# Error: unknown deployment scenario\n# Please select "4 × A3 PD Disaggregation" for Deployment Scenario';
    }
    if (!['prefill', 'decode', 'router'].includes(npuRole)) {
      return '# Error: unknown node role\n# Please select "Prefill", "Decode", or "Router" for Node Role';
    }

    if (npuRole === 'router') {
      return [
        '# Ascend A3 · 4-node PD disaggregation · router',
        '# Run on any host that can reach prefill node 0 and decode node 0.',
        '# Replace <prefill-node0-ip> and <decode-node0-ip> before running.',
        'python -m sglang_router.launch_router \\',
        '  --pd-disaggregation \\',
        '  --policy cache_aware \\',
        `  --prefill http://<prefill-node0-ip>:${npuServerPort} 8998 \\`,
        `  --decode http://<decode-node0-ip>:${npuServerPort} \\`,
        '  --host 127.0.0.1 \\',
        '  --port 6688 \\',
        '  --mini-lb'
      ].join('\n');
    }

    if (!['0', '1'].includes(npuNodeRank)) {
      return '# Error: unknown node rank\n# Please select "Node 0" or "Node 1" for Node Rank';
    }
    const isPrefill = npuRole === 'prefill';
    const rank = Number(npuNodeRank);
    const roleName = isPrefill ? 'prefill' : 'decode';

    const header = [
      `# Ascend A3 · 4-node PD disaggregation · ${roleName} node ${rank} of 2`,
      '# Run inside the quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3 container on this node.',
      '# Replace the <...> placeholders (node IPs, the NIC carrying them) and the model path.',
      'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor',
      'sysctl -w vm.swappiness=0',
      'sysctl -w kernel.numa_balancing=0',
      'sysctl -w kernel.sched_migration_cost_ns=50000',
      '',
      'unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING',
      'source /usr/local/Ascend/ascend-toolkit/set_env.sh',
      'source /usr/local/Ascend/nnal/atb/set_env.sh',
      '',
      'export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True',
      'export SGLANG_SET_CPU_AFFINITY=1',
      'export STREAMS_PER_DEVICE=32',
      'export ASCEND_MF_STORE_URL="tcp://<prefill-node0-ip>:24670"',
      'export GLOO_SOCKET_IFNAME=<network-interface>',
      'export HCCL_SOCKET_IFNAME=<network-interface>'
    ];

    const roleEnv = isPrefill
      ? [
          'export DEEPEP_HCCL_BUFFSIZE=1200',
          'export DEEP_NORMAL_MODE_USE_INT8_QUANT=1',
          'export TASK_QUEUE_ENABLE=2'
        ]
      : [
          'export DEEPEP_HCCL_BUFFSIZE=400',
          'export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8',
          'export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1',
          'export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1',
          'export TASK_QUEUE_ENABLE=0'
        ];

    const commonFlags = [
      '--model-path /path/to/DeepSeek-V3.2-W8A8',
      `--disaggregation-mode ${roleName}`,
      `--host <${roleName}-node${rank}-ip>`
    ];

    const roleFlags = isPrefill
      ? [
          `--port ${npuServerPort}`,
          '--dist-init-addr <prefill-node0-ip>:5000',
          '--disaggregation-bootstrap-port 8998',
          `--node-rank ${rank}`,
          '--nnodes 2',
          '--tp 32',
          '--watchdog-timeout 9000',
          '--mem-fraction-static 0.73',
          '--disable-radix-cache',
          '--chunked-prefill-size -1',
          '--max-prefill-tokens 68000',
          '--max-running-requests 1',
          '--moe-a2a-backend deepep',
          '--deepep-mode normal',
          '--quantization modelslim',
          '--disaggregation-transfer-backend ascend',
          '--disable-cuda-graph',
          '--moe-dense-tp-size 1',
          '--enable-prefill-cp',
          '--cp-strategy zigzag',
          '--attn-cp-size 32',
          '--speculative-algorithm NEXTN',
          '--speculative-num-steps 1',
          '--speculative-eagle-topk 1',
          '--speculative-num-draft-tokens 2'
        ]
      : [
          `--port ${npuServerPort}`,
          '--dist-init-addr <decode-node0-ip>:5000',
          `--node-rank ${rank}`,
          '--nnodes 2',
          '--tp 32',
          '--enable-dp-attention --dp-size 8',
          '--ep 32',
          '--moe-dense-tp-size 1',
          '--enable-dp-lm-head',
          '--watchdog-timeout 9000',
          '--mem-fraction-static 0.79',
          '--disable-radix-cache',
          '--chunked-prefill-size -1',
          '--max-prefill-tokens 68000',
          '--max-running-requests 32',
          '--cuda-graph-max-bs-decode 4',
          '--moe-a2a-backend deepep',
          '--deepep-mode low_latency',
          '--quantization modelslim',
          '--speculative-algorithm NEXTN',
          '--speculative-num-steps 3',
          '--speculative-eagle-topk 1',
          '--speculative-num-draft-tokens 4',
          '--disaggregation-transfer-backend ascend'
        ];

    const tailFlags = [
      '--reasoning-parser deepseek-v3',
      '--tool-call-parser deepseekv32',
      '--trust-remote-code',
      '--attention-backend ascend',
      '--device npu'
    ];

    const launch = ['python3 -m sglang.launch_server', ...[...commonFlags, ...roleFlags, ...tailFlags].map(f => `  ${f}`)];
    return [...header, ...roleEnv, ''].join('\n') + '\n' + launch.join(' \\\n');
  };

  const generateCommand = (values) => {
    const { hardware, modelname, strategy, reasoningParser, toolcall } = values;

    if (hardware === 'a3') {
      if (modelname !== 'v32') {
        return `# Error: Ascend A3 deployment is validated with DeepSeek-V3.2 (W8A8) only\n# Please select "DeepSeek-V3.2" for Model Name`;
      }
      return generateAscendCommand(values);
    }

    const isNvfp4 = modelname === 'v32nvfp4';
    const isMxfp4 = modelname === 'v32mxfp4';
    const isAMD = hardware === 'mi300x' || hardware === 'mi355x';
    const isB300 = hardware === 'b300';
    const isBlackwell = hardware === 'b200' || isB300;

    // Validation: NVFP4 requires Blackwell
    if (isNvfp4 && !isBlackwell) {
      return `# Error: DeepSeek-V3.2-NVFP4 requires NVIDIA B200/B300 (Blackwell) hardware\n# Please select "B200" or "B300" for Hardware Platform or choose a different model`;
    }

    // Validation: MXFP4 requires AMD MI300X/MI355X
    if (isMxfp4 && !isAMD) {
      return `# Error: DeepSeek-V3.2-MXFP4 requires AMD MI300X/MI355X hardware\n# Please select "MI300X" or "MI355X" for Hardware Platform or choose a different model`;
    }

    // Validation: Speciale doesn't support tool calling
    if (modelname === 'v32speciale' && toolcall === 'enabled') {
      return `# Error: DeepSeek-V3.2-Speciale doesn't support tool calling\n# Please select "Disabled" for Tool Call Parser or choose a different model`;
    }

    // Model name mapping
    const modelMap = {
      'v32':         'DeepSeek-V3.2',
      'v32exp':      'DeepSeek-V3.2-Exp',
      'v32speciale': 'DeepSeek-V3.2-Speciale',
      'v32nvfp4':    'DeepSeek-V3.2-NVFP4',
      'v32mxfp4':    'DeepSeek-V3.2-mxfp4'
    };

    let modelFamily;
    if (isNvfp4) modelFamily = 'nvidia';
    else if (isMxfp4) modelFamily = 'amd';
    else modelFamily = 'deepseek-ai';

    const modelName = `${modelFamily}/${modelMap[modelname]}`;

    // NVFP4: fixed config
    if (isNvfp4) {
      let cmd = 'sglang serve \\\n';
      cmd += `  --model-path ${modelName}`;
      cmd += ' \\\n  --tp 4';
      if (isB300) {
        cmd += ' \\\n  --attention-backend flashinfer';
        cmd += ' \\\n  --enforce-disable-flashinfer-allreduce-fusion';
        cmd += ' \\\n  --cuda-graph-backend-prefill disabled';
        cmd += ' \\\n  --moe-runner-backend flashinfer_cutlass';
        cmd += ' \\\n  --disable-flashinfer-autotune';
      } else {
        cmd += ' \\\n  --quantization modelopt_fp4';
        cmd += ' \\\n  --moe-runner-backend flashinfer_trtllm';
      }
      return cmd;
    }

    // MXFP4: fixed config for AMD
    if (isMxfp4) {
      let cmd = 'sglang serve \\\n';
      cmd += `  --model-path ${modelName}`;
      cmd += ' \\\n  --tp 8';
      cmd += ' \\\n  --trust-remote-code';
      return cmd;
    }

    let cmd = 'sglang serve \\\n';
    cmd += `  --model-path ${modelName}`;

    // Hardware platform specific parameters
    if (isAMD) {
      cmd += ' \\\n  --trust-remote-code';
      cmd += ' \\\n  --dsa-prefill-backend tilelang';
      cmd += ' \\\n  --dsa-decode-backend tilelang';
      cmd += ' \\\n  --cuda-graph-max-bs-decode 64';
    }

    // Strategy configurations
    const strategyArray = Array.isArray(strategy) ? strategy : [];
    const tpSize = 8;
    const dpSize = 8;
    const epSize = 8;
    cmd += ` \\\n  --tp ${tpSize}`;
    if (strategyArray.includes('dp')) {
      cmd += ` \\\n  --attn-dp-size ${dpSize}`;
    }
    if (strategyArray.includes('ep')) {
      cmd += ` \\\n  --ep ${epSize}`;
    }

    // Multi-token prediction (MTP) configuration
    if (strategyArray.includes('mtp')) {
      cmd += ' \\\n  --speculative-algorithm EAGLE';
      cmd += ' \\\n  --speculative-num-steps 3';
      cmd += ' \\\n  --speculative-eagle-topk 1';
      cmd += ' \\\n  --speculative-num-draft-tokens 4';
    }

    if (isB300) {
      cmd += ' \\\n  --attention-backend flashinfer';
      if (!strategyArray.includes('dp') || strategyArray.includes('ep') || strategyArray.includes('mtp')) {
        cmd += ' \\\n  --enforce-disable-flashinfer-allreduce-fusion';
        cmd += ' \\\n  --cuda-graph-backend-prefill disabled';
      }
    }

    // Add tool-call-parser if enabled (not supported for Speciale)
    if (toolcall === 'enabled' && modelname !== 'v32speciale') {
      if (modelname === 'v32exp') {
        cmd += ' \\\n  --tool-call-parser deepseekv31';
      } else if (modelname === 'v32') {
        cmd += ' \\\n  --tool-call-parser deepseekv32';
      }
    }

    // Add reasoning-parser when enabled
    if (reasoningParser === 'enabled') {
      cmd += ' \\\n  --reasoning-parser deepseek-v3';
    }

    // Add chat-template if tool calling is enabled (only for v32exp)
    if (toolcall === 'enabled' && modelname === 'v32exp') {
      cmd += ' \\\n  --chat-template ./examples/chat_template/tool_chat_template_deepseekv32.jinja';
    }

    return cmd;
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

  const handleRadioChange = (optionName, value) => {
    setValues(prev => getUpdatedValues(prev, optionName, value));
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
              {option.type === 'checkbox' ? (
                items.map(item => {
                  const isChecked = (values[option.name] || []).includes(item.id);
                  const isDisabled = item.required || !!item.disabled;
                  return (
                    <label
                      key={item.id}
                      style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}), ...(isDisabled ? { ...disabledStyle, ...(item.required ? {} : {}) } : {}) }}
                      title={item.disabledReason || ''}
                    >
                      <input type="checkbox" checked={isChecked} disabled={isDisabled} onChange={(e) => handleCheckboxChange(option.name, item.id, e.target.checked)} style={{ display: 'none' }} />
                      {item.label}
                      {item.subtitle && <small style={{ ...subtitleStyle, color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit' }}>{item.subtitle}</small>}
                    </label>
                  );
                })
              ) : (
                items.map(item => {
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
                })
              )}
            </div>
          </div>
        );
      })}
      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{generateCommand(values)}</pre>
      </div>
    </div>
  );
};

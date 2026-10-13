// Presentation only: which fields show and what each folded header summarizes.
// Combination rules live in core/preflight.py and run when the loader executes.
export const GROUPS = {
  execution: 'Execution',
  attention: 'Attention',
  cache: 'Cache-DiT',
  memory: 'Memory',
  parallel: 'Manual parallel layout',
  weights: 'Weights & quantization',
};

export function visibleField(name, v) {
  if (['tp_size', 'ulysses_degree', 'ring_degree'].includes(name)) return v.parallel_mode === 'manual';
  if (name === 'regional_compile') return v.execution_mode === 'torch_compile';
  if (['resident_layers', 'prefetch_layers'].includes(name)) return v.memory_mode === 'layerwise_offload';
  return true;
}

export function groupSummary(group, v) {
  switch (group) {
    case 'execution': return v.execution_mode;
    case 'attention': return v.attention_backend;
    case 'cache': return v.enable_cache_dit ? 'on' : 'off';
    case 'memory': return v.memory_mode;
    case 'parallel': return v.parallel_mode === 'manual'
      ? `TP ${v.tp_size} / Ulysses ${v.ulysses_degree} / Ring ${v.ring_degree}` : 'auto';
    case 'weights': {
      const parts = [];
      if (v.transformer_weights_path) parts.push('override');
      if (v.quantization !== 'none') parts.push(v.quantization);
      if (v.int8_kernel !== 'auto') parts.push(`int8 ${v.int8_kernel}`);
      return parts.join(' / ') || 'checkpoint';
    }
    default: return '';
  }
}

export function statusLine(v) {
  const layout = v.parallel_mode === 'manual' ? 'manual layout' : 'auto layout';
  return `${v.model_type} / ${v.num_gpus} GPU(s), ${layout} / ${v.execution_mode}`;
}

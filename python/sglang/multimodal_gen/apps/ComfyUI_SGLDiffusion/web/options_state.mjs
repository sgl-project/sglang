// Presentation only: which fields show and what each folded header summarizes.
// Combination rules live in core/preflight.py and run when the loader executes.
export const GROUPS = {
  parallel: 'Parallel',
  attention: 'Attention',
  quantization: 'Quantization',
  cache: 'Cache',
  execution: 'Execution',
  memory: 'Memory',
};

const SHORT = {
  torch_compile: 'compile',
  cuda_graph: 'CUDA graph',
  layerwise_offload: 'offload',
  comfy_kitchen: 'kitchen',
  sglang_jit: 'JIT',
  flash_attention: 'flash attention',
  sage_attention2: 'sage 2',
  sage_attention3: 'sage 3',
  sol_attention: 'sol',
  cache_dit: 'Cache-DiT',
};
const short = value => SHORT[value] ?? value;

const CACHE_KNOBS = ['cache_warmup_steps', 'cache_threshold', 'cache_max_cached_steps', 'cache_fn_blocks'];

// Written into an empty attention_backend_config when the backend is picked, so
// its knobs are visible and editable; the values match the backend's defaults.
const BACKEND_DEFAULT_CONFIGS = {
  sol_attention: 'dense_steps=10,dense_layers=0-1,dense_backend=fa,tau=1.0,thresh_type=diag',
};

export function defaultBackendConfig(backend, current) {
  return current ? current : (BACKEND_DEFAULT_CONFIGS[backend] ?? current);
}

export function visibleField(name, v) {
  if (['tp_size', 'ulysses_degree', 'ring_degree'].includes(name)) return v.parallel_mode === 'manual';
  if (name === 'regional_compile') return v.execution_mode === 'torch_compile';
  if (['resident_layers', 'prefetch_layers'].includes(name)) return v.memory_mode === 'layerwise_offload';
  if (name === 'cache_dit_tuning') return v.cache_mode === 'cache_dit';
  if (CACHE_KNOBS.includes(name)) return v.cache_mode === 'cache_dit' && v.cache_dit_tuning === 'manual';
  return true;
}

// Kept to one or two words so the folded headers line up.
export function groupSummary(group, v) {
  switch (group) {
    case 'execution': return short(v.execution_mode);
    case 'attention': return short(v.attention_backend);
    case 'cache': return short(v.cache_mode);
    case 'memory': return short(v.memory_mode);
    case 'parallel': return v.parallel_mode === 'manual'
      ? `TP${v.tp_size} U${v.ulysses_degree} R${v.ring_degree}` : 'auto';
    case 'quantization': return v.int8_gemm === 'auto'
      ? v.quantization : `${v.quantization}, ${short(v.int8_gemm)}`;
    default: return '';
  }
}

export function headerLabel(group, v, open) {
  return `${open ? '▾' : '▸'} ${GROUPS[group]} · ${groupSummary(group, v)}`;
}

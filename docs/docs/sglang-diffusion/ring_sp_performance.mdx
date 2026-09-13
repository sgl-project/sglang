---
title: "Sequence Parallelism"
tag: "preserve"
metatags:
    description: "Configure sequence parallelism, TP plus SP, Ulysses, K/V gather, and ring-based sequence splitting for SGLang Diffusion workloads."
---

Sequence parallelism splits long image or video latent sequences across GPUs. In SGLang Diffusion, the public controls are:

- `--sp-degree`: total sequence parallel degree
- `--ulysses-degree`: Ulysses parallel degree
- `--ring-degree`: ring parallel degree
- `--sp-attention-mode`: attention exchange used inside each SP group

The degrees must satisfy:

```text
sp_degree = ulysses_degree * ring_degree
```

The default `--sp-attention-mode ulysses` uses all-to-all to redistribute
sequence shards over attention heads. `--sp-attention-mode kv_gather` keeps
queries sequence-sharded and all-gathers keys and values, then computes each
rank's local output directly. The K/V-gather mode currently supports
non-causal attention with `--ring-degree 1`. Varlen calls through the legacy
`UlyssesAttention` adapter and video sparse attention are not supported.

Use SP when sequence length or video shape makes the DiT forward pass the bottleneck and the model supports sequence sharding. For latency-oriented multi-GPU Qwen/Wan deployments, also compare against CFG parallelism and FSDP; SP is not automatically the best multi-GPU setting for every model.

## Choosing The Attention Exchange

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "20%"}} />
    <col style={{width: "30%"}} />
    <col style={{width: "25%"}} />
    <col style={{width: "25%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Mode</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Communication</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Memory</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Constraints</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px"}}><code>ulysses</code></td>
      <td style={{padding: "9px 12px"}}>All-to-all before and after attention</td>
      <td style={{padding: "9px 12px"}}>Full sequence with a shard of the attention heads during attention</td>
      <td style={{padding: "9px 12px"}}>Attention head divisibility must match the Ulysses degree</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><code>kv_gather</code></td>
      <td style={{padding: "9px 12px"}}>All-gather K and V; Q and output remain sequence-sharded</td>
      <td style={{padding: "9px 12px"}}>Replicates full K and V within the SP group</td>
      <td style={{padding: "9px 12px"}}>Non-causal attention and <code>ring_degree=1</code>; no legacy varlen or video sparse attention</td>
    </tr>
  </tbody>
</table>

Neither exchange is universally faster. K/V gather avoids the reverse
all-to-all and can help when its local attention shape or collective is more
efficient, while Ulysses can use less attention activation memory. Benchmark
both on the target model, resolution, accelerator, and interconnect.

For SP degree `P`, the approximate per-rank network payload of K/V gather
relative to Ulysses is `P / 2`, excluding each rank's local shard. The payloads
are therefore similar at SP2, while K/V gather moves about 2x as much data at
SP4 and 4x at SP8. K/V gather may still be faster when all-gather and its local
attention layout are more efficient, especially at low SP degrees, but this
scaling makes the interconnect and input shape part of the selection policy.

## Recommended Commands

### Ulysses Sequence Parallelism

The default mode needs only the total SP degree when ring parallelism is not
used:

```bash
sglang serve \
  --model-path Qwen/Qwen-Image \
  --num-gpus 4 \
  --sp-degree 4 \
  --port 8898
```

### K/V-Gather Sequence Parallelism

Use the same SP process-group layout and select the alternative attention
exchange explicitly:

```bash
sglang serve \
  --model-path Qwen/Qwen-Image \
  --num-gpus 4 \
  --sp-degree 4 \
  --sp-attention-mode kv_gather \
  --port 8898
```

### Tensor Plus Sequence Parallelism

TP and SP use independent dimensions. With DP and CFG parallelism disabled,
the required GPU count is `tp_size * sp_degree`. This example creates two TP
groups across a two-rank SP dimension:

```bash
sglang serve \
  --model-path Qwen/Qwen-Image \
  --num-gpus 4 \
  --tp-size 2 \
  --sp-degree 2 \
  --sp-attention-mode kv_gather \
  --port 8898
```

Omit `--sp-attention-mode kv_gather` to use TP plus Ulysses with the same
`tp=2, sp=2` topology.

#### How TP Plus SP Works

TP and SP form orthogonal dimensions of the DiT process mesh. For `tp=2,
sp=2`, ranks `[0, 1]` and `[2, 3]` are TP groups, while ranks `[0, 2]` and
`[1, 3]` are SP groups. Each rank therefore belongs to one group of each type:

- TP shards supported attention and MLP projection weights and computation,
  then communicates partial projection results inside the TP group.
- SP shards the latent sequence and attention activations, then uses Ulysses
  or K/V gather inside the SP group.

Pure SP replicates the DiT weights on every SP rank. Adding TP reduces the
per-rank memory used by TP-sharded weights and keeps the sequence activation
sharding from SP, at the cost of adding TP communication to every applicable
DiT block. The exact memory reduction is model-dependent because not every
parameter or runtime buffer is TP-sharded.

TP plus SP should therefore be treated as a capacity and memory-latency Pareto
option, not as the default latency winner. On a single NVSwitch node, pure SP
often wins when the complete DiT weights fit on every GPU because it avoids
the repeated TP collectives. Try TP plus SP when pure SP does not fit, when
more memory headroom is required, or when its measured memory reduction is
worth a small latency increase.

The following representative eager results used eight H200 GPUs in one
NVSwitch node. Times are median scheduler-side end-to-end latency. They
illustrate the tradeoff rather than define a universal policy:

| Model and workload | Fastest tested topology | TP plus SP Pareto point | Tradeoff |
| --- | --- | --- | --- |
| Qwen-Image, 1536x1536 | CFG2xSP4 Ulysses: 972.6 ms, 62.8 GiB/GPU | CFG2xTP2xSP2 K/V: 1017.4 ms, 48.4 GiB/GPU | 4.6% slower, 22.9% less peak memory |
| Wan2.2-A14B, 832x480x81 | CFG2xSP4 K/V: 6573.5 ms, 61.9 GiB/GPU | CFG2xTP2xSP2 K/V: 7243.1 ms, 34.2 GiB/GPU | 10.2% slower, 44.7% less peak memory |
| LTX2.3, 768x512x241 | SP8 K/V: 7258.2 ms, 55.4 GiB/GPU | TP2xSP4 K/V: 10372.3 ms, 37.8 GiB/GPU | 42.9% slower, 31.8% less peak memory |

K/V gather can still improve TP plus SP at the same topology even when that
topology is not the global latency winner. In the same experiment it improved
TP2xSP4 by 4.2% for FLUX and 8.4% for LTX2.3, and improved
CFG2xTP2xSP2 by 6.0% for Qwen-Image and 2.7% for Wan2.2-A14B, relative to
Ulysses. Always compare the full candidate set, including pure SP, TP, CFG,
and their feasible combinations, rather than selecting the SP attention
backend first.

### FSDP Plus Sequence Parallelism

FSDP can shard DiT weights across the same workers that participate in SP.
Unlike TP times SP, the FSDP and SP degrees do not multiply the required GPU
count. This is useful when pure SP is fast enough but replicated DiT weights
or long-sequence activations leave too little memory headroom:

```bash
sglang serve \
  --model-path Lightricks/LTX-2.3 \
  --num-gpus 2 \
  --use-fsdp-inference true \
  --sp-degree 2 \
  --sp-attention-mode kv_gather \
  --port 8898
```

FSDP adds weight all-gather communication, so compare it with pure SP when
both fit. K/V gather has the same non-causal and `ring_degree=1` constraints
under FSDP.

### Ring Sequence Parallelism

This example uses two GPUs with `sp=2`, `ulysses=1`, and `ring=2`.

```bash
sglang serve \
  --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 1 \
  --ring-degree 2 \
  --port 8898
```

### Single-GPU Baseline

Use an explicit single-GPU baseline before attributing a gain to sequence parallelism.

```bash
sglang serve \
  --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --port 8898
```

## Choosing The Degrees

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "28%"}} />
    <col style={{width: "32%"}} />
    <col style={{width: "40%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Setting</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Typical use</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px"}}><code>--sp-degree 1</code></td>
      <td style={{padding: "9px 12px"}}>Single-GPU or no sequence splitting</td>
      <td style={{padding: "9px 12px"}}>Use this as the baseline.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><code>--ulysses-degree N</code></td>
      <td style={{padding: "9px 12px"}}>Ulysses-only sequence parallelism</td>
      <td style={{padding: "9px 12px"}}>When ring parallelism is not needed, keep <code>--ring-degree 1</code>.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><code>--ring-degree N</code></td>
      <td style={{padding: "9px 12px"}}>Ring-based sequence splitting over long sequences</td>
      <td style={{padding: "9px 12px"}}>Keep <code>--sp-degree</code> equal to <code>ulysses_degree * ring_degree</code>.</td>
    </tr>
  </tbody>
</table>

## Cross-Node Sequence Parallelism

Ulysses alone cannot scale sequence parallelism past the GPU count of one
node: going wider either violates head-count divisibility or exposes an
all-to-all across the slower inter-node link. Ring's point-to-point KV
rotation is designed to overlap with attention compute, which tolerates a
slower cross-node link far better than an all-to-all does — so the pattern
for scaling SP across nodes is **node-local Ulysses × cross-node Ring**, not
Ulysses alone.

Cross-node launches add three flags on top of the usual SP degrees:

- `--nnodes`: number of nodes. `--num-gpus` stays the *total* GPU count
  across every node; each node runs `num_gpus // nnodes` local workers.
- `--node-rank`: this node's rank, `0` on the head node (which keeps the
  HTTP/TokenizerManager surface) and `1..nnodes-1` on the others (worker-only).
- `--dist-init-addr`: a `host:port` rendezvous address reachable from every
  node — typically the head node's address.

Run the same command on every node, changing only `--node-rank`:

```bash
# node 0 (head)
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant ref2va \
  --num-gpus 16 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr <node0-ip>:23456 \
  --sp-degree 16 \
  --ulysses-degree 8 \
  --ring-degree 2 \
  --encoder-parallel replicate \
  --port 30010

# node 1 (worker)
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant ref2va \
  --num-gpus 16 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr <node0-ip>:23456 \
  --sp-degree 16 \
  --ulysses-degree 8 \
  --ring-degree 2 \
  --encoder-parallel replicate \
  --port 30010
```

`--encoder-parallel replicate` is required for cross-node deployments today:
the `auto` fold decision is not yet node-boundary aware and will try to fold
the text encoder across nodes, which crashes reference-conditioned encoders.
See [Encoder Parallelism](/docs/sglang-diffusion/encoder_parallel).

<Warning>
Cross-node ring support is model-specific, not a property of the launch
flags alone. Confirm support in the model cookbook and see
[Parallelism](/docs/sglang-diffusion/parallelism) for topology requirements.
Passing `--ring-degree > 1` for a model that only
has single-node Ulysses may either raise or, in some cases, silently
compute incorrect output; check the model's cookbook page before assuming
cross-node scaling is supported.
</Warning>

### Numerics across node boundaries

Ring's online-softmax merge across P2P hops accumulates floating-point
operations in a different order than single-node attention, so a cross-node
run is **not** expected to bit-match a single-node run of the same prompt and
seed — this is the same class of difference as choosing a different
attention backend, not a correctness regression. What *is* expected: the same
request run twice against the same cross-node deployment must produce
byte-identical output. Use that repeat-request check, not a cross-topology
comparison, to validate a cross-node deployment's determinism.

## Benchmarking Guidance

When benchmarking SP, compare the same model, precision, resolution, frame count, step count, scheduler settings, prompt type, and output path. Report both stage latency and peak GPU memory; SP can reduce per-GPU memory while adding communication overhead.

Useful metrics:

- End-to-end latency
- Denoising stage latency
- Decoding stage latency
- Peak GPU memory and peak allocated memory
- Communication or runtime overhead when available

## Reference Benchmark

The following numbers are a reference measurement for one setup. They are not a general promise for all Wan2.2 deployments.

- Model: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- Hardware: two 48 GB RTX 40-series GPUs for sequence parallelism, one 48 GB RTX 40-series GPU for baseline
- Sequence parallel config: `sp=2, ulysses=1, ring=2` (`u1r2`)
- Baseline config: `sp=1, ulysses=1, ring=1` (`u1r1`)

### Stage Time Breakdown

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "25%"}} />
    <col style={{width: "25%"}} />
    <col style={{width: "25%"}} />
    <col style={{width: "25%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Stage / Metric</th>
      <th><code>u1r2</code> (s)</th>
      <th><code>u1r1</code> baseline (s)</th>
      <th>Speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>InputValidation</td>
      <td>0.1060</td>
      <td>0.1029</td>
      <td>0.97x</td>
    </tr>
    <tr>
      <td>TextEncoding</td>
      <td>1.3965</td>
      <td>2.2261</td>
      <td>1.59x</td>
    </tr>
    <tr>
      <td>LatentPreparation</td>
      <td>0.0002</td>
      <td>0.0002</td>
      <td>1.00x</td>
    </tr>
    <tr>
      <td>TimestepPreparation</td>
      <td>0.0003</td>
      <td>0.0004</td>
      <td>1.33x</td>
    </tr>
    <tr>
      <td>Denoising</td>
      <td>52.6358</td>
      <td>71.6785</td>
      <td>1.36x</td>
    </tr>
    <tr>
      <td>Decoding</td>
      <td>7.6708</td>
      <td>13.4314</td>
      <td>1.75x</td>
    </tr>
    <tr>
      <td><strong>Total</strong></td>
      <td><strong>63.74</strong></td>
      <td><strong>90.63</strong></td>
      <td><strong>1.42x</strong></td>
    </tr>
  </tbody>
</table>

### Memory Usage

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "25%"}} />
    <col style={{width: "25%"}} />
    <col style={{width: "25%"}} />
    <col style={{width: "25%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Memory Metric</th>
      <th><code>u1r2</code> (GB)</th>
      <th><code>u1r1</code> baseline (GB)</th>
      <th>Delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Peak GPU Memory</td>
      <td>20.07</td>
      <td>27.40</td>
      <td>-7.33</td>
    </tr>
    <tr>
      <td>Peak Allocated</td>
      <td>13.35</td>
      <td>20.40</td>
      <td>-7.05</td>
    </tr>
    <tr>
      <td>Memory Overhead</td>
      <td>6.72</td>
      <td>7.00</td>
      <td>-0.28</td>
    </tr>
    <tr>
      <td>Overhead Ratio</td>
      <td>33.5%</td>
      <td>25.6%</td>
      <td>+7.9pp</td>
    </tr>
  </tbody>
</table>

In this setup, end-to-end latency improved from `90.63s` to `63.74s` (`1.42x`) and peak GPU memory dropped by `7.33GB`. The overhead ratio increased, so future tuning should still check communication and runtime overhead on the target hardware.

## Cross-Node Reference Benchmark

The following numbers are a reference measurement for MiniMax-H3's cross-node
Ulysses × Ring deployment. They are not a general promise for every
model or topology — see each model's cookbook page for its own verified
cross-node status.

- Model: `MiniMaxAI/MiniMax-H3`
- Hardware: 2 nodes × 8× NVIDIA H200 SXM, same cluster, InfiniBand between nodes
- Cross-node config: `--num-gpus 16 --sp-degree 16 --ulysses-degree 8 --ring-degree 2`
- Single-node baseline: `--num-gpus 8 --sp-degree 8 --ulysses-degree 8 --ring-degree 1`

Denoise-stage-only comparison, holding prompt, seed, and step count fixed:

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "34%"}} />
    <col style={{width: "22%"}} />
    <col style={{width: "22%"}} />
    <col style={{width: "22%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Task</th>
      <th>Single-node (s/step)</th>
      <th>Cross-node (s/step)</th>
      <th>Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>T2VA denoise</td>
      <td>0.749</td>
      <td>0.477</td>
      <td>-36.3%</td>
    </tr>
    <tr>
      <td>Ref2VA / V2V denoise</td>
      <td>2.572</td>
      <td>1.494</td>
      <td>-41.9%</td>
    </tr>
  </tbody>
</table>

The gain grows with sequence length: ring's per-hop communication cost stays
roughly constant while attention compute grows quadratically with sequence
length, so V2V's longer packed sequence benefits more than T2VA's shorter
one. With the point-to-point KV rotation pipelined against attention compute
(rather than a blocking `all_gather`), one V2V request's full denoise stage
completed in 68.1-68.3s versus 128.6s on the single-node 8-GPU baseline
(-47.0%), with byte-identical output to the unpipelined cross-node path.

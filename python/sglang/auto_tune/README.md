# All-reduce tuner

- [tuner](run_allreduce.py)
- [communicator scan](aiter_comms.py)

The commands below need the `sglang` package to be importable, for example from the `python` directory of this repository.

One process cannot all-reduce with itself. `torchrun` starts one process per GPU; `--nproc_per_node` sets how many. Each `--sizes` value is a message size in bytes, timed as a bf16 tensor.

`--graph 0` (the default) times eager calls and writes `allreduce_configs_eager.json` for `SGLANG_TUNED_EAGER_CONFIG`. `--graph 1` times calls captured into a CUDA graph and writes `allreduce_configs_graph.json` for `SGLANG_TUNED_GRAPH_CONFIG`. Each winning row's `backend` is the full class name, for example `aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4` on aiter pin `90596736`, and its `method` is the collective that row timed: `allreduce`, `quick_all_reduce`, `custom_all_reduce`, `outplace_all_reduce`, or `all_reduce`.

At startup the tuner parses `/sgl-workspace/aiter` (`--aiter-dir`) and the SGLang package (`--sglang-dir`). The class names that the scan returns are the competitors. Each one that defines a collective method is imported by that name and opened. HIP QuickReduce is opened only when `--qr-quant` is a quant that AITER declares for it. The FlyDSL INT4 classes (`FlyQuickAllReduce`, `QuickAllReduceInt4`) are opened only when `--qr-quant` is `INT4`. A class with no collective method is logged and not timed.

`--skip` leaves named communicators out of the timing. Pass a full class name, or the class name alone. Several names can follow one `--skip`, and the flag can be repeated:

```bash
--skip aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4
```

Use it when a class is obsolete on the AITER pin under test, or when its accuracy is not acceptable. The built-in skip list is empty, so `QuickAllReduceInt4` competes unless a run names it. A skipped class is logged under `Not opened` and is not written into the table. A name that matches no scanned class is logged and does not stop the run. The names are also stored in the JSON `skip` field.

## Tuner

```bash
python -m sglang.auto_tune.run_allreduce --help
```

```bash
torchrun --nproc_per_node=8 -m sglang.auto_tune.run_allreduce \
    --tp 8 \
    --qr-quant INT4 \
    --fly-algorithm mesh \
    --qr-cast-bf16-to-fp16 1 \
    --ignore-acr-max-size 1 \
    --ignore-scr-max-size 0 \
    --graph 0 \
    --median 1 \
    --num-warmups 100 \
    -n 5000 \
    --timeout 240 \
    --output-dir /workspace/inference-testing/ar-tune \
    --sizes 16384 40960 65536 98304 131072 196608 \
            262144 393216 524288 786432 1048576 1572864 \
            2097152 3145728 4194304 6291456 8388608 12582912 \
            16777216 25165824 33554432 50331648 67108864 \
            100663296 134217728 201326592 268435456
```

Tested on aiter pin `90596736`, the output looks like this:

```text
============================================================
All-reduce tuning summary (eager, median)
============================================================
     16384B  sglang.srt.distributed.device_communicators.custom_all_reduce_v2.CustomAllReduceV2     11.64 us
     40960B  sglang.srt.distributed.device_communicators.custom_all_reduce_v2.CustomAllReduceV2     12.08 us
     65536B  sglang.srt.distributed.device_communicators.custom_all_reduce_v2.CustomAllReduceV2     12.16 us
     98304B  sglang.srt.distributed.device_communicators.custom_all_reduce_v2.CustomAllReduceV2     11.92 us
    131072B  sglang.srt.distributed.device_communicators.custom_all_reduce_v2.CustomAllReduceV2     12.32 us
    196608B  sglang.srt.distributed.device_communicators.custom_all_reduce_v2.CustomAllReduceV2     13.12 us
    262144B  sglang.srt.distributed.device_communicators.pynccl.PyNcclCommunicator     14.92 us
    393216B  sglang.srt.distributed.device_communicators.custom_all_reduce_v2.CustomAllReduceV2     16.80 us
    524288B  sglang.srt.distributed.device_communicators.custom_all_reduce.CustomAllreduce     19.04 us
    786432B  aiter.dist.device_communicators.communicator_cuda.CudaCommunicator     20.12 us
   1048576B  aiter.dist.device_communicators.communicator_cuda.CudaCommunicator     19.80 us
   1572864B  aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce     20.28 us
   2097152B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     20.60 us
   3145728B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     20.76 us
   4194304B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     21.24 us
   6291456B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     25.32 us
   8388608B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     31.04 us
  12582912B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     39.88 us
  16777216B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     49.88 us
  25165824B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     68.68 us
  33554432B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     82.16 us
  50331648B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4    107.84 us
  67108864B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4    136.16 us
 100663296B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4    190.52 us
 134217728B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4    243.88 us
 201326592B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4    352.68 us
 268435456B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4    467.00 us
============================================================
  Use it with SGLANG_TUNED_EAGER_CONFIG=ar-tune/allreduce_configs_eager.json
```

`--ingraph-calls` is how many all-reduces are captured back to back in one graph. The reported time is per call.

```bash
torchrun --nproc_per_node=8 -m sglang.auto_tune.run_allreduce \
    --tp 8 \
    --qr-quant INT4 \
    --fly-algorithm mesh \
    --qr-cast-bf16-to-fp16 1 \
    --ignore-acr-max-size 1 \
    --ignore-scr-max-size 0 \
    --graph 1 \
    --ingraph-calls 100 \
    --median 1 \
    --num-warmups 20 \
    -n 500 \
    --timeout 240 \
    --output-dir /workspace/inference-testing/ar-tune \
    --sizes 16384 40960 65536 98304 131072 196608 \
            262144 393216 524288 786432 1048576 1572864 \
            2097152 3145728 4194304 6291456 8388608 12582912 \
            16777216 25165824 33554432 50331648 67108864 \
            100663296 134217728 201326592 268435456
```

The output looks like this:

```text
============================================================
All-reduce tuning summary (graph, median)
============================================================
     16384B  sglang.srt.distributed.device_communicators.custom_all_reduce_v2.CustomAllReduceV2      4.77 us
     40960B  sglang.srt.distributed.device_communicators.custom_all_reduce_v2.CustomAllReduceV2      4.75 us
     65536B  sglang.srt.distributed.device_communicators.custom_all_reduce_v2.CustomAllReduceV2      5.88 us
     98304B  aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce      6.79 us
    131072B  aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce      6.81 us
    196608B  aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce      6.88 us
    262144B  aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce      7.03 us
    393216B  aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce      7.50 us
    524288B  aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce      8.11 us
    786432B  aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce     10.20 us
   1048576B  aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce     10.70 us
   1572864B  aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce     13.27 us
   2097152B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     14.77 us
   3145728B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     16.32 us
   4194304B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     18.17 us
   6291456B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     22.75 us
   8388608B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     28.32 us
  12582912B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     36.77 us
  16777216B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     46.72 us
  25165824B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     66.09 us
  33554432B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4     80.39 us
  50331648B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4    105.05 us
  67108864B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4    133.75 us
 100663296B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4    188.12 us
 134217728B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4    241.58 us
 201326592B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4    349.63 us
 268435456B  aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4    463.43 us
============================================================
  Use it with SGLANG_TUNED_GRAPH_CONFIG=ar-tune/allreduce_configs_graph.json
```

The tuner writes the JSON files and logs its progress and results.

## Stack dumps

If you see messages like

```
  File "/sgl-workspace/sglang/python/sglang/auto_tune/allreduce_tuner.py", line 1214 in run_allreduce_tuning
  File "/sgl-workspace/sglang/python/sglang/auto_tune/run_allreduce.py", line 303 in main
  File "/sgl-workspace/sglang/python/sglang/auto_tune/run_allreduce.py", line 330 in <module>
  File "<frozen runpy>", line 88 in _run_code
  File "<frozen runpy>", line 198 in _run_module_as_main
```

(especially in graph capture mode), wait; the process most likely continues.

While the tuner runs, each rank writes its Python stack to `/tmp/ar-tune-stacks/rank<N>.txt`, where `N` is the global rank. Set `ALLREDUCE_TUNE_STACK_DIR` to use another directory.

- `--timeout S` (default `20`) sets how often a rank dumps. The timer starts when the process starts and repeats every `S` seconds. It is not a stall detector: a slow but healthy size also produces dumps. Use `--timeout 0` to turn dumps off. Raise it, for example to `240`, when sizes take a long time, so the dumps do not bury the results.
- Each run overwrites the files at startup. Later dumps in the same run are appended.
- Rank 0 also prints the last 8 lines of every rank's file to the terminal, under `===== where each rank is =====`, starting `S + 5` seconds after launch and then every `S` seconds.

To look at one rank:

```bash
tail -n 60 /tmp/ar-tune-stacks/rank1.txt
grep -n 'File' /tmp/ar-tune-stacks/rank1.txt | tail -n 20
```

How to read a dump:

- A rank inside `torch.cuda.synchronize` or a kernel launch is waiting on a GPU kernel. The custom kernels spin-wait on their peers.
- A rank inside `barrier`, `broadcast_object_list` or another collective is waiting on another rank.
- If a rank's file has a `Fatal Python error: Segmentation fault` section, that section is the crash. If it has none, the process did not reach the fatal handler. Then find the size and backend that were running in the last terminal lines: `  [i/N] <bytes>B` is the size, and the `  <class>: ...` line after it is the backend.

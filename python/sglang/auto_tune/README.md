# All-reduce tuner

- [config generation](generate_selection_config.py)
- [consumer tuner](run_allreduce.py)
- [shell script](run_allreduce.sh)
- [README](README.md)

Commands below run from this directory (`python/sglang/auto_tune`).

One process cannot all-reduce with itself. `torchrun` starts as many processes, as how many GPUs specified by the parameter `nproc_per_node`: one process per GPU. Each `--sizes` value is a message size in bytes, timed as a bf16 tensor.

`--graph 0` (the default) times eager calls and writes `allreduce_configs_eager.json` for `SGLANG_TUNED_EAGER_CONFIG`. `--graph 1` times calls captured into a CUDA graph and writes `allreduce_configs_graph.json` for `SGLANG_TUNED_GRAPH_CONFIG`.

The selection JSON names the kernels that compete. `generate_selection_config` writes it. `run_allreduce` opens that file and does not build a second list. `run_allreduce.sh` runs those two steps in order.

HIP QuickReduce is the row `quick_allreduce_<codec>`, the name the server dispatches. FlyDSL INT4 is a second row, `quick_allreduce_flydsl_INT4`.

## Config generation

```bash
python -m sglang.auto_tune.generate_selection_config --help

python -m sglang.auto_tune.generate_selection_config \
    --output ar-tune/selection_config.json \
    --qr-quant INT4 \
    --qr-cast-bf16-to-fp16 1 \
    --ignore-acr-max-size 1 \
    --ignore-scr-max-size 0
```

## Tuner

```bash
python -m sglang.auto_tune.run_allreduce --help

torchrun --nproc_per_node=8 -m sglang.auto_tune.run_allreduce \
    --selection-config ar-tune/selection_config.json \
    --tp 8 \
    --graph 0 \
    --median 1 \
    -n 200 \
    --output-dir ar-tune \
    --sizes 16384 65536 131072 262144 524288 1048576 \
            2097152 4194304 8388608 16777216 33554432 \
            67108864 134217728 268435456

`--ingraph-calls` is how many all-reduces are captured back to back in one graph. The reported time is per call.

torchrun --nproc_per_node=8 -m sglang.auto_tune.run_allreduce \
    --selection-config ar-tune/selection_config.json \
    --tp 8 \
    --graph 1 \
    --ingraph-calls 10 \
    --median 1 \
    -n 500 \
    --output-dir ar-tune \
    --sizes 6384 65536 131072 262144 524288 1048576 \
            2097152 4194304 8388608 16777216 33554432 \
            67108864 134217728 268435456
```

## Both steps

`run_allreduce.sh` writes the selection, then starts the tuner with `--selection-config` pointing at that file. `--nproc-per-node` defaults to `--tp`. The JSON defaults to `<output-dir>/selection_config.json`.

```bash
./run_allreduce.sh --help

./run_allreduce.sh \
    --tp 8 \
    --qr-quant INT4 \
    --qr-cast-bf16-to-fp16 1 \
    --ignore-acr-max-size 1 \
    --ignore-scr-max-size 0 \
    --median 1 \
    -n 200 \
    --graph 0 \
    --output-dir ar-tune \
    --sizes 16384 65536 131072 262144 524288 1048576 \
            2097152 4194304 8388608 16777216 33554432 \
            67108864 134217728 268435456

./run_allreduce.sh \
    --tp 8 \
    --qr-quant INT4 \
    --qr-cast-bf16-to-fp16 1 \
    --ignore-acr-max-size 1 \
    --ignore-scr-max-size 0 \
    --median 1 \
    -n 500 \
    --graph 1 \
    --ingraph-calls 10 \
    --output-dir ar-tune \
    --sizes 6384 65536 131072 262144 524288 1048576 \
            2097152 4194304 8388608 16777216 33554432 \
            67108864 134217728 268435456
```

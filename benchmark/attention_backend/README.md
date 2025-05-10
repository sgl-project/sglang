## Run benchmark

```shell
python bench.py --config bench_serving.yaml bench_serving_spec_decode.yaml gsm8k.yaml gsm8k_spec_decode.yaml
```

Benchmark results will be stored in `<torch.cuda.get_device_name(0).replace(' ', '_')>.md` file

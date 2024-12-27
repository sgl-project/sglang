### v0.1.0

#### How to run with TE

We use NVIDIA [TE](https://github.com/NVIDIA/TransformerEngine/tree/main) to run the code. To start the server, run:

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --port 30001 --host 0.0.0.0 --enable-te
```

in the recommended docker container. You can start the docker with a pre-built image with TE (v1.14.0.dev0+994f19d) and torch (2.5.1+cu124). The docker image can be used via `docker pull zhuohaol/sglang-te:latest`

#### Benchmark code

very soon

# R-Fork

R-Fork (Tensor Remote Fork) is a novel weight loading methodology that leverages efficient inter-node GPU-to-GPU data transfer path to load tensors from a running SGLang instance to a new instance with zero-copy. It can significantly optimize the SGLang instance boot-up time by reducing model weights loading from several minutes to mere seconds.

To learn more details about R-Fork, please check **<a href=https://lmsys.org/blog/2025-12-10-rfork/> R-Fork blog </a>**

## Usage

| Argument     | Usage                                      |
|--------------|--------------------------------------------|
| load-format  | set to `remote_instance` to enable R-Fork. |
| remote-instance-weight-loader-backend | `nccl`, `transfer_engine`, or `modelexpress`. Default is `nccl`. |
| remote-instance-weight-loader-seed-instance-ip | IP address of the seed instance who will provide the model weight. Used by `nccl` and `transfer_engine` backends. |
| remote-instance-weight-loader-seed-instance-service-port | the port that the seed instance's HTTP server is listening on. Used by `nccl` and `transfer_engine` backends. |
| remote-instance-weight-loader-send-weights-group-ports | the list of available ports on the seed instance that will be used to build NCCL communication groups between seed and client instance. Only needed by `nccl` backend. |
| remote-instance-weight-loader-start-seed-via-transfer-engine | set to start seed service that supports TransferEngine as backend. Needed for seed instances when using `transfer_engine` as backend. |
| modelexpress-config | JSON config for `modelexpress` backend. Keys: `"url"` (required, gRPC host:port of ModelExpress server), `"model_name"` (optional, defaults to `--model-path`), `"source"` (optional bool, `true` for seed mode). |

### NCCL as backend

seed instance:
```shell
python -m sglang.launch_server [args]
```

client instance:
```shell
python -m sglang.launch_server [args] \
  --load-format remote_instance \
  --remote-instance-weight-loader-seed-instance-ip [seed_instance_ip] \
  --remote-instance-weight-loader-seed-instance-service-port [seed_instance_service_port] \
  --remote-instance-weight-loader-send-weights-group-ports [send_weights_nccl_group_ports_list]  \
  --remote-instance-weight-loader-backend nccl
```

### TransferEngine as backend

seed instance:
```shell
python -m sglang.launch_server [args] \
  --remote-instance-weight-loader-start-seed-via-transfer-engine
```

```shell
python -m sglang.launch_server [args] \
  --load-format remote_instance \
  --remote-instance-weight-loader-seed-instance-ip [seed_instance_ip] \
  --remote-instance-weight-loader-seed-instance-service-port [seed_instance_service_port] \
  --remote-instance-weight-loader-backend transfer_engine
```

### ModelExpress as backend

[ModelExpress](https://github.com/ai-dynamo/modelexpress) is a coordination service that manages P2P weight transfer metadata. It removes the need for direct seed IP/port configuration by providing a centralized registry that seeds publish to and clients discover from. Under the hood it uses TransferEngine (Mooncake) for the actual RDMA data transfer.

A running ModelExpress server is required. See the [ModelExpress documentation](https://github.com/ai-dynamo/modelexpress) for setup instructions.

seed instance:
```shell
python -m sglang.launch_server [args] \
  --modelexpress-config '{"url": "[modelexpress_grpc_host:port]", "model_name": "[model_name]", "source": true}'
```

client instance:
```shell
python -m sglang.launch_server [args] \
  --load-format remote_instance \
  --remote-instance-weight-loader-backend modelexpress \
  --modelexpress-config '{"url": "[modelexpress_grpc_host:port]", "model_name": "[model_name]"}'
```

The seed publishes its TransferEngine session ID and tensor layout to ModelExpress. The client queries ModelExpress to discover the seed, then pulls weights directly via RDMA. This enables dynamic seed discovery without hardcoding IPs, and supports multiple models through a single ModelExpress instance.

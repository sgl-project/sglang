# R-Fork

R-Fork (Tensor Remote Fork) is a novel weight loading methodology that leverages efficient inter-node GPU-to-GPU data transfer path to load tensors from a running SGLang instance to a new instance with zero-copy. It can significantly optimize the SGLang instance boot-up time by reducing model weights loading from several minutes to mere seconds.

To learn more details about R-Fork, please check **<a href=https://lmsys.org/blog/2025-12-10-rfork/> R-Fork blog </a>**

## Usage

| Argument     | Usage                                      |
|--------------|--------------------------------------------|
| load-format  | set to `remote_instance` to enable R-Fork. |
| remote-instance-weight-loader-backend | `nccl` or `transfer_engine`, default value is `nccl` |
| remote-instance-weight-loader-seed-instance-ip | IP address of the seed instance who will provide the model weight |
| remote-instance-weight-loader-seed-instance-service-port | the port that the seed instance's HTTP server is listening on |
| remote-instance-weight-loader-send-weights-group-ports | the list of available ports on the seed instance that will be used to build NCCL communication groups between seed and client instance. This argument is only needed by `nccl` backend.  |
| remote-instance-weight-loader-start-seed-via-transfer-engine | set to start seed service that supports TransferEngine as backend. It is needed for seed instances when using `transfer_engine` as backend. |

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

### Multi-node scenarios

when users are trying to launch servers with multi-node settings,  `weight` R-fork will use zmq to build the communication between different nodes,
so that the api call `get_remote_instance_transfer_engine_info` could also work for the gpus of nodes with the `node_rank > 1`.

Users need to add `--node-hosts` and `--inter-node-transfer-engine-info-port`.

`--node-hosts`: a dict of {`node_rank`: `address_of_node`} ;
`--inter-node-transfer-engine-info-port`: port for zmq communication .

For example:

```
# Node 0:
python3 -m sglang.launch_server --model-path /data/models/Moonlight-16B-A3B-Instruct --tp 16 --trust-remote-code  --mem-frac 0.85 --host 0.0.0.0 --port 30000 --dist-init-addr node-0-addr:5000 --nnodes 2 --node-rank 0 --remote-instance-weight-loader-start-seed-via-transfer-engine --node-hosts '{"0":"node-0-addr","1":"node-1-addr"}' --inter-node-transfer-engine-info-port 15036

# Node 1:
python3 -m sglang.launch_server --model-path /data/models/Moonlight-16B-A3B-Instruct --tp 16 --trust-remote-code  --mem-frac 0.85 --host 0.0.0.0 --port 30000 --dist-init-addr node-0-addr:5000 --nnodes 2 --node-rank 1 --remote-instance-weight-loader-start-seed-via-transfer-engine --node-hosts '{"0":"node-0-addr","1":"node-1-addr"}' --inter-node-transfer-engine-info-port 15036
```

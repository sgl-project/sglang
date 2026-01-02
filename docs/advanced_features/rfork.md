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

### NCCL as backend

```shell
python -m sglang.launch_server [args] \
  --load-format remote_instance	\
  --remote-instance-weight-loader-seed-instance-ip [seed_instance_ip] \
  --remote-instance-weight-loader-seed-instance-service-port [seed_instance_service_port] \
  --remote-instance-weight-loader-send-weights-group-ports [send_weights_nccl_group_ports_list]  \
  --remote-instance-weight-loader-backend nccl
```

### TransferEngine as backend

```shell
python -m sglang.launch_server [args] \
  --load-format remote_instance	\
  --remote-instance-weight-loader-seed-instance-ip [seed_instance_ip] \
  --remote-instance-weight-loader-seed-instance-service-port [seed_instance_service_port] \
  --remote-instance-weight-loader-backend transfer_engine
```

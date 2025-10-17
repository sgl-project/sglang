### RDMA Need To Know

SGLang transfer engine is required to enable RDMA, you can verify RDMA with `ibv_devices` commands, the output of the ibv_devices command should look similar to the following result.

```
    device                 node GUID
    ------              ----------------
    mlx5_0              58a2e10300a5399e
    mlx5_1              b83fd20300ce434e
    mlx5_2              58a2e10300a539ae
    mlx5_3              58a2e10300a5413e
    mlx5_4              58a2e10300a539de
    mlx5_5              58a2e10300a7e7c6
    mlx5_6              b83fd20300ce4350
    mlx5_7              58a2e10300a7e9e6
    mlx5_8              58a2e10300a7ea06
    mlx5_9              58a2e10300a7ea56

```

You might be wondering why there are ten devices here instead of eight. By using the `ibstatus` command, you will notice that **mlx5_1** and **mlx5_6** devices have different rates (this may vary across different machines).

```
Infiniband device 'mlx5_0' port 1 status:
        default gid:     fe80:0000:0000:0000:5aa2:e1ff:fea5:399e
        base lid:        0x0
        sm lid:          0x0
        state:           4: ACTIVE
        phys state:      5: LinkUp
        rate:            400 Gb/sec (4X NDR)
        link_layer:      Ethernet

Infiniband device 'mlx5_1' port 1 status:
        default gid:     fe80:0000:0000:0000:ba3f:d2ff:fece:434e
        base lid:        0x0
        sm lid:          0x0
        state:           4: ACTIVE
        phys state:      5: LinkUp
        rate:            100 Gb/sec (4X EDR)
        link_layer:      Ethernet
```

The reason is that **mlx5_1 mlx5_6** and other devices come from different network cards. and you can verify this by using the commands `ibstat mlx5_1` and `ibstat mlx5_0`.

when setting **IBDEVICES**, you should skip the **mlx5_1** and **mlx5_6** devices.

```
IBDEVICES="mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9"
```

# Run SGlang

SGLang supports Prefill-Decode (PD) disaggregation on AMD Instinct GPUs, which uses Mooncake to transfer the KV cache. From a system architecture perspective, SGLang PD disaggregation includes 3 distinct components: a proxy server, prefill server, and decode server. When a request comes in, the proxy server selects a pair of prefill and decode servers based on a workload-balancing scheme. The selected prefill server and decode server pair using a handshake, establishing a local sender and receiver, respectively. The decode server preallocates the KV cache, then signals the prefill server to begin LLM prefill inference and compute the KV caches. After the prefill work is done, the KV cache data is transferred to the decode server, which handles iterative token generation.

## Run the prefill server

Use the `sglang.launch_server` command to launch the prefill server. **host_ip** is the IP address of the current prefill machine.

```
MODEL_PATH="/workspace/models/deepseek-ai/DeepSeek-V3"
IBDEVICES="mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9"
host_ip="xx.xx.xx.xx"
python3 -m sglang.launch_server --model-path $MODEL_PATH \
                        --disaggregation-mode prefill --disaggregation-ib-device ${IBDEVICES} \
                        --host ${host_ip} --port 30000  --trust-remote-code  \
                        --tp 8 2>&1 | tee server_prefill_log.txt
```

## Run the decode server

Use the `sglang.launch_server` command to launch the decode server with a new Terminal. **host_ip** is the IP address of the current decode machine.

```
MODEL_PATH="/workspace/models/deepseek-ai/DeepSeek-V3"
IBDEVICES="mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9"
host_ip="xx.xx.xx.xx"
python3 -m sglang.launch_server --model-path $MODEL_PATH \
    --disaggregation-mode decode --disaggregation-ib-device ${IBDEVICES} \
    --host ${host_ip} --port 30001 --trust-remote-code \
    --tp 8  2>&1 | tee server_decode_log.txt
```

## Run the proxy server

this server can run on the prefill node. You can run it on a standalone node in the same cluster.

In this step, the IP addresses and ports of the prefill and decode node pools are configured. The IP address and port of the proxy server are also provided for the test client program to connect to.

```
MASTER_ADDR="xx.xx.xx.xx"
MASTER_ADDR_DECODE="xx.xx.xx.xx"
python -m sglang_router.launch_router --pd-disaggregation \
        --prefill http://${MASTER_ADDR}:30000 \
        --decode http://${MASTER_ADDR_DECODE}:30001 \
        --host 127.0.0.1 --port 8000
```
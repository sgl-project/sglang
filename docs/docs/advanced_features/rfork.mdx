---
title: "R-Fork"
metatags:
    description: "SGLang R-Fork: zero-copy GPU-to-GPU weight loading, reduce boot-up time from minutes to seconds. NCCL and TransferEngine backends."
---
R-Fork (Tensor Remote Fork) is a novel weight loading methodology that leverages efficient inter-node GPU-to-GPU data transfer path to load tensors from a running SGLang instance to a new instance with zero-copy. It can significantly optimize the SGLang instance boot-up time by reducing model weights loading from several minutes to mere seconds.

To learn more details about R-Fork, please check **<a href="https://lmsys.org/blog/2025-12-10-rfork/"> R-Fork blog </a>**

## Usage

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "50%"}} />
    <col style={{width: "50%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Argument</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Usage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>load-format</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>set to `remote_instance` to enable R-Fork.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>remote-instance-weight-loader-backend</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><code>nccl</code>, <code>transfer_engine</code>, or <code>modelexpress</code>. Default is <code>nccl</code>.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>remote-instance-weight-loader-seed-instance-ip</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>IP address of the seed instance who will provide the model weight. Used by <code>nccl</code> and <code>transfer_engine</code> backends.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>remote-instance-weight-loader-seed-instance-service-port</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>the port that the seed instance's HTTP server is listening on. Used by <code>nccl</code> and <code>transfer_engine</code> backends.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>remote-instance-weight-loader-send-weights-group-ports</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>the list of available ports on the seed instance that will be used to build NCCL communication groups between seed and client instance. Only needed by <code>nccl</code> backend.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>remote-instance-weight-loader-start-seed-via-transfer-engine</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>set to start seed service that supports TransferEngine as backend. Needed for seed instances when using <code>transfer_engine</code> as backend.</td>
    </tr>
<tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>modelexpress-config</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>JSON config for <code>modelexpress</code> backend. Keys: <code>"url"</code> (optional gRPC host:port override) and <code>"transport"</code> (<code>"nixl"</code> or <code>"transfer_engine"</code>, defaults to <code>"nixl"</code>).</td>
    </tr>
</tbody>
</table>

### NCCL as backend

seed instance:
```shell Command
python -m sglang.launch_server [args]
```

client instance:
```shell Command
python -m sglang.launch_server [args] \
  --load-format remote_instance \
  --remote-instance-weight-loader-seed-instance-ip [seed_instance_ip] \
  --remote-instance-weight-loader-seed-instance-service-port [seed_instance_service_port] \
  --remote-instance-weight-loader-send-weights-group-ports [send_weights_nccl_group_ports_list]  \
  --remote-instance-weight-loader-backend nccl
```

### TransferEngine as backend

seed instance:
```shell Command
python -m sglang.launch_server [args] \
  --remote-instance-weight-loader-start-seed-via-transfer-engine
```

```shell Command
python -m sglang.launch_server [args] \
  --load-format remote_instance \
  --remote-instance-weight-loader-seed-instance-ip [seed_instance_ip] \
  --remote-instance-weight-loader-seed-instance-service-port [seed_instance_service_port] \
  --remote-instance-weight-loader-backend transfer_engine
```

### ModelExpress as backend

[ModelExpress](https://github.com/ai-dynamo/modelexpress) is a coordination service that manages P2P weight transfer metadata. It removes the need for direct seed IP/port configuration by providing a centralized registry that instances publish to and discover from. The ModelExpress Python package must be installed in the SGLang image.

A running ModelExpress server is required. See the [ModelExpress documentation](https://github.com/ai-dynamo/modelexpress) for setup instructions.

server instance:
```bash Command
python -m sglang.launch_server [args] \
  --load-format remote_instance \
  --remote-instance-weight-loader-backend modelexpress \
  --modelexpress-config '{"url": "[modelexpress_grpc_host:port]", "transport": "nixl"}'
```

All SGLang instances use the same command shape. If no ready source exists, the instance loads weights natively and publishes metadata to ModelExpress. If a compatible source exists, it loads weights through ModelExpress P2P transfer. Set <code>"transport": "transfer_engine"</code> to use Mooncake TransferEngine instead of the default NIXL transport.

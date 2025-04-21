# SGLang Monitoring Setup

This directory contains a ready-to-use monitoring setup for SGLang using Prometheus and Grafana.

## Prerequisites

- Docker and Docker Compose installed
- SGLang server running with metrics enabled

## Usage

1. Start your SGLang server with metrics enabled:

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --port 30000 --enable-metrics
```

By default, the metrics server will run on `127.0.0.1:30000`.

2. Start the monitoring stack:

```bash
cd examples/monitoring
docker compose up
```

3. Access the monitoring interfaces:
   - Grafana: [http://localhost:3000](http://localhost:3000)
   - Prometheus: [http://localhost:9090](http://localhost:9090)

Default Grafana login credentials:
- Username: `admin`
- Password: `admin`

You'll be prompted to change the password on first login.

4. The SGLang dashboard will be automatically available in the "SGLang Monitoring" folder.

## Troubleshooting

### Port Conflicts
If you see errors like "port is already allocated":

1. Check if you already have Prometheus or Grafana running:
   ```bash
   docker ps | grep -E 'prometheus|grafana'
   ```

2. Stop any conflicting containers:
   ```bash
   docker stop <container_id>
   ```

3. Ensure no other services are using ports 9090 and 3000:
   ```bash
   lsof -i :9090
   lsof -i :3000
   ```

### Connection Issues
If Grafana cannot connect to Prometheus:
1. Check that both services are running
2. Verify the datasource configuration in Grafana
3. Check that your SGLang server is properly exposing metrics

## Configuration

- Prometheus configuration: `prometheus.yaml`
- Docker Compose configuration: `docker-compose.yaml`
- Grafana datasource: `grafana/datasources/datasource.yaml`
- Grafana dashboard configuration: `grafana/dashboards/config/dashboard.yaml`
- SGLang dashboard JSON: `grafana/dashboards/json/sglang-dashboard.json`

## Customization

You can customize the monitoring setup by modifying the configuration files as needed.

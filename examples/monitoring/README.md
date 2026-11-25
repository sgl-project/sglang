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

2. Start the monitoring stack, choosing a Grafana admin password. Put the
   password in a `.env` file next to `docker-compose.yaml` (the repository's
   `.gitignore` already excludes `.env`), so every later `docker compose`
   command (`ps`, `logs`, `down`) can read it too:

```bash
cd examples/monitoring
(umask 077; echo "GRAFANA_ADMIN_PASSWORD='<a strong password>'" > .env)
docker compose up
```

Exporting `GRAFANA_ADMIN_PASSWORD` in your shell works as well. Every
`docker compose` command for this stack fails while the variable is unset.

Grafana applies `GRAFANA_ADMIN_PASSWORD` only when it first creates its
database. To change the password later, change it in Grafana under your
profile, or reset it in place:

```bash
docker compose exec grafana grafana cli admin reset-admin-password '<new password>'
```

Then update `.env` to match. Do not change it by running `docker compose down`
and starting again: this example does not persist Grafana's database or
Prometheus's data, so that discards the dashboards, users and alert rules you
created in Grafana and the metric history Prometheus collected.

3. Access the monitoring interfaces:
   - Grafana: [http://localhost:3000](http://localhost:3000)
   - Prometheus: [http://localhost:9090](http://localhost:9090)

Log in to Grafana as `admin` with the password you set. Dashboards are also visible without logging in.

Grafana and Prometheus listen on `127.0.0.1` only. To view them from another machine, use an SSH tunnel (`ssh -L 3000:127.0.0.1:3000 <host>`) instead of exposing the ports.

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

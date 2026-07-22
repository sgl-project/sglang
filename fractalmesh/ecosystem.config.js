// FractalMesh Omega Titan v2.1.0 — PM2 Ecosystem
// Samuel James Hiotis | ABN 56628117363 | Sole Trader | Albury NSW
// Usage: pm2 start ecosystem.config.js --env production
// Agents: 49 nodes | Memory budget: ~2.9GB total

const ROOT = process.env.FRACTALMESH_HOME || require('path').join(process.env.HOME, 'fmsaas');
const REPO = process.env.REPO_ROOT        || require('path').join(process.env.HOME, 'sglang');

module.exports = {
    apps: [

        // ── MASTER API (port 5058) ────────────────────────────────────
        {
            name:              'fm-pod',
            script:            'agents/fm_pod.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'30M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                FLASK_PORT:       '5058',
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-pod-error.log`,
            out_file:   `${ROOT}/logs/fm-pod-out.log`,
            time:       true,
        },

        // ── GEOSIGNAL (port 5057) ─────────────────────────────────────
        {
            name:              'fm-geosignal',
            script:            'agents/fm_geosignal.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'28M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                GEOSIGNAL_PORT:   '5057',
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-geosignal-error.log`,
            out_file:   `${ROOT}/logs/fm-geosignal-out.log`,
            time:       true,
        },

        // ── ANALYTICS (port 5060) ─────────────────────────────────────
        {
            name:              'fm-analytics',
            script:            'agents/fm_analytics.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'16M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                ANALYTICS_PORT:   '5060',
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-analytics-error.log`,
            out_file:   `${ROOT}/logs/fm-analytics-out.log`,
            time:       true,
        },

        // ── TRADING ORCHESTRATOR ──────────────────────────────────────
        {
            name:              'fm-trading',
            script:            'agents/fm_trading.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'16M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-trading-error.log`,
            out_file:   `${ROOT}/logs/fm-trading-out.log`,
            time:       true,
        },

        // ── WHITEPAPER PUBLISHER (cron 10 min) ────────────────────────
        {
            name:              'fm-whitepaper',
            script:            'agents/fm_whitepaper.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            cron_restart:      '*/10 * * * *',
            autorestart:       false,
            max_memory_restart:'12M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-whitepaper-error.log`,
            out_file:   `${ROOT}/logs/fm-whitepaper-out.log`,
            time:       true,
        },

        // ── PRODUCT DELIVERY ──────────────────────────────────────────
        {
            name:              'fm-delivery',
            script:            'agents/fm_delivery.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'10M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-delivery-error.log`,
            out_file:   `${ROOT}/logs/fm-delivery-out.log`,
            time:       true,
        },

        // ── RF SALES BRIDGE ───────────────────────────────────────────
        {
            name:              'rf-bridge',
            script:            'agents/rf_bridge.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'10M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/rf-bridge-error.log`,
            out_file:   `${ROOT}/logs/rf-bridge-out.log`,
            time:       true,
        },

        // ── DORKING — OSINT Lead Discovery ───────────────────────────
        {
            name:              'fm-dorking',
            script:            'agents/fm_dorking.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'25M',
            restart_delay:     10000,
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-dorking-error.log`,
            out_file:   `${ROOT}/logs/fm-dorking-out.log`,
            time:       true,
        },

        // ── FIGMA — TerraMesh Design Sync ─────────────────────────────
        {
            name:              'fm-figma',
            script:            'agents/fm_figma.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'25M',
            restart_delay:     10000,
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-figma-error.log`,
            out_file:   `${ROOT}/logs/fm-figma-out.log`,
            time:       true,
        },

        // ── ADVERT — Gmail Outreach ───────────────────────────────────
        {
            name:              'fm-advert',
            script:            'agents/fm_advert.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'11M',
            restart_delay:     5000,
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-advert-error.log`,
            out_file:   `${ROOT}/logs/fm-advert-out.log`,
            time:       true,
        },

        // ── NOTES IP REGISTRAR (port 5061) ────────────────────────────
        {
            name:              'fm-notes-ip-registrar',
            script:            'agents/fm_notes_registrar.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'10M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NOTES_PORT:       '5061',
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-notes-error.log`,
            out_file:   `${ROOT}/logs/fm-notes-out.log`,
            time:       true,
        },

        // ── TUNNEL — Public URL Manager (port 5062) ───────────────────
        {
            name:              'fm-tunnel',
            script:            'agents/fm_tunnel.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'9M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                TUNNEL_PORT:      '5062',
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-tunnel-error.log`,
            out_file:   `${ROOT}/logs/fm-tunnel-out.log`,
            time:       true,
        },

        // ── AI DJ (Synthwave Empire) ──────────────────────────────────
        {
            name:              'fm-ai-dj',
            script:            `${process.env.HOME}/synthwave/ai_dj.py`,
            interpreter:       '/usr/bin/python3',
            watch:             false,
            autorestart:       true,
            max_restarts:      50,
            restart_delay:     8000,
            max_memory_restart:'100M',
            env_production: {
                PYTHONUNBUFFERED: '1',
                DJ_INTERVAL:      '600',
                DJ_MODEL:         'mistralai/mistral-7b-instruct:free',
                NODE_ENV:         'production',
            },
            error_file: `${process.env.HOME}/.fm_logs/ai_dj_err.log`,
            out_file:   `${process.env.HOME}/.fm_logs/ai_dj_out.log`,
            time:       true,
        },

        // ── NFT MINTER (Synthwave Empire) ─────────────────────────────
        {
            name:              'fm-nft-minter',
            script:            `${process.env.HOME}/synthwave/nft_minter.py`,
            interpreter:       '/usr/bin/python3',
            watch:             false,
            autorestart:       true,
            max_restarts:      50,
            restart_delay:     15000,
            max_memory_restart:'80M',
            env_production: {
                PYTHONUNBUFFERED: '1',
                NODE_ENV:         'production',
            },
            error_file: `${process.env.HOME}/.fm_logs/nft_minter_err.log`,
            out_file:   `${process.env.HOME}/.fm_logs/nft_minter_out.log`,
            time:       true,
        },

        // ── NEXUS GATEWAY (port 8000) ─────────────────────────────────
        {
            name:              'unified-gateway',
            script:            'agents/gateway.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'16M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                GATEWAY_PORT:     '8000',
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/gateway-error.log`,
            out_file:   `${ROOT}/logs/gateway-out.log`,
            time:       true,
        },

        // ── DASHBOARD (serve static, port 8090) ───────────────────────
        {
            name:              'fm-dashboard',
            script:            'serve',
            args:              ['-s', 'www', '-l', '8090', '--no-clipboard'],
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'30M',
            env_production: {
                NODE_ENV: 'production',
            },
            error_file: `${ROOT}/logs/fm-dashboard-error.log`,
            out_file:   `${ROOT}/logs/fm-dashboard-out.log`,
            time:       true,
        },

        // ── v10003.42 APEX OMNI-MATRIX NODES ─────────────────────────

        // Cloudflare tunnel — bash wrapper prevents PM2 crash loop
        {
            name:              'fm-cloudflared',
            script:            'tunnels/start_tunnel.sh',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'100M',
            env_production: {
                NODE_ENV: 'production',
            },
            error_file: `${ROOT}/logs/fm-cloudflared-error.log`,
            out_file:   `${ROOT}/logs/fm-cloudflared-out.log`,
            time:       true,
        },

        // Secure HMAC-signed pulse event bus (port 5060)
        {
            name:              'fm-bus',
            script:            'agents/fm_pulse_bus.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'50M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                BUS_PORT:         '5060',
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-bus-error.log`,
            out_file:   `${ROOT}/logs/fm-bus-out.log`,
            time:       true,
        },

        // React/Flask API gateway (port 8080)
        {
            name:              'fm-gateway',
            script:            'agents/fm_dashboard.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'100M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-gateway-error.log`,
            out_file:   `${ROOT}/logs/fm-gateway-out.log`,
            time:       true,
        },

        // External LLM directive bridge (port 8090)
        {
            name:              'fm-integrator',
            script:            'agents/fm_mesh_integrator.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'100M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                INTEGRATOR_PORT:  '8090',
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-integrator-error.log`,
            out_file:   `${ROOT}/logs/fm-integrator-out.log`,
            time:       true,
        },

        // GitOps webhook runner (port 8092)
        {
            name:              'fm-gitops-runner',
            script:            'agents/fm_gitops_runner.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'100M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                GITOPS_PORT:      '8092',
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-gitops-error.log`,
            out_file:   `${ROOT}/logs/fm-gitops-out.log`,
            time:       true,
        },

        // Harmonic φ-logic/memory engine
        {
            name:              'fm-harmonic',
            script:            'modules/harmonic_logic_memory.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'70M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-harmonic-error.log`,
            out_file:   `${ROOT}/logs/fm-harmonic-out.log`,
            time:       true,
        },

        // Network topology warden
        {
            name:              'fm-warden',
            script:            'modules/network_topology_warden.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'80M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-warden-error.log`,
            out_file:   `${ROOT}/logs/fm-warden-out.log`,
            time:       true,
        },

        // ── OMEGA v40.0 AGENTS (repo-path agents) ────────────────────

        // Omni-dashboard Flask UI (port 8090) — hot-upgrade, log viewer, vault status
        {
            name:              'omni-dashboard',
            script:            `${REPO}/fractalmesh/api/omni_dashboard.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'150M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                REPO_ROOT:        REPO,
                APP_PORT:         '8090',
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/omni-dashboard-error.log`,
            out_file:   `${ROOT}/logs/omni-dashboard-out.log`,
            time:       true,
        },

        // OSINT spider — 18-dork Google CSE lead discovery (interval 7200s)
        {
            name:              'fm-osint-spider',
            script:            `${REPO}/fractalmesh/agents/fm_osint_spider.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'100M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-osint-spider-error.log`,
            out_file:   `${ROOT}/logs/fm-osint-spider-out.log`,
            time:       true,
        },

        // 4A Negotiator — pricing tiers, Gmail proposals, OSINT pipeline (interval 3600s)
        {
            name:              'fm-negotiator',
            script:            `${REPO}/fractalmesh/agents/fm_negotiator.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'80M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-negotiator-error.log`,
            out_file:   `${ROOT}/logs/fm-negotiator-out.log`,
            time:       true,
        },

        // ── v2.0.0 SATELLITE INTELLIGENCE AGENTS ─────────────────────

        // Sentinel ingest — S5P CH4 + NASA EMIT + S2 NDVI (interval 6h)
        {
            name:              'fm-sentinel-ingest',
            script:            `${REPO}/fractalmesh/agents/fm_sentinel_ingest.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'100M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-sentinel-ingest-error.log`,
            out_file:   `${ROOT}/logs/fm-sentinel-ingest-out.log`,
            time:       true,
        },

        // Methane reports — super-emitter detection + client report generation
        {
            name:              'fm-methane-reports',
            script:            `${REPO}/fractalmesh/agents/fm_methane_reports.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'80M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                REPO_ROOT:        REPO,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-methane-reports-error.log`,
            out_file:   `${ROOT}/logs/fm-methane-reports-out.log`,
            time:       true,
        },

        // AIS monitor — dark fleet gap + spoofing detection (interval 15min)
        {
            name:              'fm-ais-monitor',
            script:            `${REPO}/fractalmesh/agents/fm_ais_monitor.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'80M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-ais-monitor-error.log`,
            out_file:   `${ROOT}/logs/fm-ais-monitor-out.log`,
            time:       true,
        },

        // ── v2.0.0 ENTERPRISE + AUTOMATION AGENTS ────────────────────

        // Enterprise Bus — circuit breakers, HMAC, monetization tracking
        {
            name:              'fm-enterprise-bus',
            script:            `${REPO}/fractalmesh/agents/fm_enterprise_bus.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'100M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                REPO_ROOT:        REPO,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-enterprise-bus-error.log`,
            out_file:   `${ROOT}/logs/fm-enterprise-bus-out.log`,
            time:       true,
        },

        // RAG API — semantic knowledge search (port 8001)
        {
            name:              'rag-api',
            script:            `${REPO}/fractalmesh/api/rag_api.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'120M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                RAG_API_PORT:     '8001',
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/rag-api-error.log`,
            out_file:   `${ROOT}/logs/rag-api-out.log`,
            time:       true,
        },

        // Billing API — Stripe/PayPal/LemonSqueezy metering (port 8003)
        {
            name:              'billing-api',
            script:            `${REPO}/fractalmesh/api/billing_api.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'80M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                BILLING_API_PORT: '8003',
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/billing-api-error.log`,
            out_file:   `${ROOT}/logs/billing-api-out.log`,
            time:       true,
        },

        // Licensing agent — auto copyright/license stamping (interval 3600s)
        {
            name:              'fm-licensing',
            script:            `${REPO}/fractalmesh/agents/fm_licensing.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'50M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                REPO_ROOT:        REPO,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-licensing-error.log`,
            out_file:   `${ROOT}/logs/fm-licensing-out.log`,
            time:       true,
        },

        // Watermark agent — document + image watermarking (interval 3600s)
        {
            name:              'fm-watermark',
            script:            `${REPO}/fractalmesh/agents/fm_watermark.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'60M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                REPO_ROOT:        REPO,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-watermark-error.log`,
            out_file:   `${ROOT}/logs/fm-watermark-out.log`,
            time:       true,
        },

        // Geo validator — coordinate validation + geotagging (interval 1800s)
        {
            name:              'fm-geo-validator',
            script:            `${REPO}/fractalmesh/agents/fm_geo_validator.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'60M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-geo-validator-error.log`,
            out_file:   `${ROOT}/logs/fm-geo-validator-out.log`,
            time:       true,
        },

        // Carbon credits — fleet energy + CO2 analysis (interval 86400s)
        {
            name:              'fm-carbon-credits',
            script:            `${REPO}/fractalmesh/agents/fm_carbon_credits.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'50M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-carbon-credits-error.log`,
            out_file:   `${ROOT}/logs/fm-carbon-credits-out.log`,
            time:       true,
        },

        // Smart contracts — ERC-20 + royalty split generation (interval 7200s)
        {
            name:              'fm-smart-contracts',
            script:            `${REPO}/fractalmesh/agents/fm_smart_contracts.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'60M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                REPO_ROOT:        REPO,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-smart-contracts-error.log`,
            out_file:   `${ROOT}/logs/fm-smart-contracts-out.log`,
            time:       true,
        },

        // AdMob bridge — Google AdMob revenue tracking (interval 3600s)
        {
            name:              'fm-admob',
            script:            `${REPO}/fractalmesh/agents/fm_admob_bridge.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'50M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-admob-error.log`,
            out_file:   `${ROOT}/logs/fm-admob-out.log`,
            time:       true,
        },

        // Campaign manager — multi-channel outreach lifecycle (interval 3600s)
        {
            name:              'fm-campaign-manager',
            script:            `${REPO}/fractalmesh/agents/marketing/fm_campaign_manager.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'80M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-campaign-manager-error.log`,
            out_file:   `${ROOT}/logs/fm-campaign-manager-out.log`,
            time:       true,
        },

        // ── v2.1.0 AFFILIATE + CONTENT + DRIP + NEXUS ────────────────

        // Research agent — NOAA Kp, CoinGecko, analytics aggregation (interval 1800s)
        {
            name:              'research-agent',
            script:            `${REPO}/fractalmesh/agents/research_agent.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'80M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/research-agent-error.log`,
            out_file:   `${ROOT}/logs/research-agent-out.log`,
            time:       true,
        },

        // Affiliate manager — 17 programs, click tracking, drip enrollment (interval 3600s)
        {
            name:              'affiliate-manager',
            script:            `${REPO}/fractalmesh/agents/affiliate_manager.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'60M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/affiliate-manager-error.log`,
            out_file:   `${ROOT}/logs/affiliate-manager-out.log`,
            time:       true,
        },

        // Content generator — GPT-4o-mini articles + dev.to publishing (interval 24h)
        {
            name:              'content-generator',
            script:            `${REPO}/fractalmesh/agents/content_generator.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'80M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/content-generator-error.log`,
            out_file:   `${ROOT}/logs/content-generator-out.log`,
            time:       true,
        },

        // Drip agent — 5-step email sequences, 3-day spacing, Gmail SMTP (interval 3600s)
        {
            name:              'fm-drip-agent',
            script:            `${REPO}/fractalmesh/agents/fm_drip_agent.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'60M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-drip-agent-error.log`,
            out_file:   `${ROOT}/logs/fm-drip-agent-out.log`,
            time:       true,
        },

        // Omni Nexus — FastAPI terminal dashboard (port 8095, 30s auto-refresh)
        {
            name:              'fm-omni-nexus',
            script:            `${REPO}/fractalmesh/api/fm_omni_nexus.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'150M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                REPO_ROOT:        REPO,
                NEXUS_PORT:       '8095',
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/fm-omni-nexus-error.log`,
            out_file:   `${ROOT}/logs/fm-omni-nexus-out.log`,
            time:       true,
        },

        // ── APEX OMNI-MATRIX EXTENDED SWARM ──────────────────────────

        ...[
            ['fm-dork-engine',    'fm_dork_engine',    '80M'],
            ['fm-contract-forge', 'fm_contract_forge', '80M'],
            ['fm-wigle-oracle',   'fm_wigle_oracle',   '80M'],
            ['fm-tokenomics',     'fm_live_tokenomics','80M'],
            ['fm-workspace',      'fm_workspace_sync', '80M'],
            ['fm-device',         'fm_device_bridge',  '80M'],
            ['fm-enochian',       'fm_enochian_hash',  '50M'],
            ['fm-stripe-gate',    'fm_stripe_gateway', '80M'],
            ['fm-salvage',        'fm_salvage_crew',   '50M'],
            ['fm-auto-advert',    'fm_auto_advert',    '50M'],
            ['fm-oversight',      'fm_oversight',      '50M'],
            ['fm-rl-quad',        'fm_rl_quad',        '50M'],
            ['fm-azr-rl',         'fm_azr_rl',         '50M'],
            ['fm-bounty',         'fm_bounty',         '50M'],
            ['fm-domain',         'fm_domain',         '50M'],
            ['fm-toolkit',        'fm_toolkit',        '50M'],
            ['fm-synthwave',      'fm_synthwave',      '50M'],
            ['fm-affiliate',      'fm_affiliate',      '50M'],
            ['fm-ipfs',           'fm_ipfs',           '50M'],
            ['fm-immortality',    'fm_immortality',    '50M'],
            ['fm-healer',         'fm_healer',         '50M'],
        ].map(([name, script, mem]) => ({
            name,
            script:            `agents/${script}.py`,
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart: mem,
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
                PYTHONUNBUFFERED: '1',
            },
            error_file: `${ROOT}/logs/${name}-error.log`,
            out_file:   `${ROOT}/logs/${name}-out.log`,
            time:       true,
        })),
    ],
};

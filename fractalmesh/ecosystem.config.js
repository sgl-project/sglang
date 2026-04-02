// FractalMesh PM2 Ecosystem — v600 Nexus Unified
// Samuel James Hiotis | ABN 56628117363 | Albury NSW
// Usage: pm2 start ecosystem.config.js --env production
// Agents: 14 processes | Memory budget: 250MB total

const ROOT = process.env.FRACTALMESH_HOME || require('path').join(process.env.HOME, 'fmsaas');

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
    ],
};

// FractalMesh PM2 Ecosystem — v401.6
// Samuel James Hiotis | ABN 56628117363 | Albury NSW
// Usage: pm2 start ecosystem.config.js --env production

const ROOT = process.env.FRACTALMESH_HOME || require('path').join(process.env.HOME, 'fmsaas');

module.exports = {
    apps: [
        // ── MASTER API ───────────────────────────────────────────────
        {
            name:              'fm-pod',
            script:            'agents/fm_pod.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'512M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                FLASK_PORT:       '5058',
                NODE_ENV:         'production',
            },
            error_file: `${ROOT}/logs/fm-pod-error.log`,
            out_file:   `${ROOT}/logs/fm-pod-out.log`,
        },

        // ── GEOSIGNAL ────────────────────────────────────────────────
        {
            name:              'fm-geosignal',
            script:            'agents/fm_geosignal.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'256M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                GEOSIGNAL_PORT:   '5057',
                NODE_ENV:         'production',
            },
            error_file: `${ROOT}/logs/fm-geosignal-error.log`,
            out_file:   `${ROOT}/logs/fm-geosignal-out.log`,
        },

        // ── TRADING ORCHESTRATOR ─────────────────────────────────────
        {
            name:              'fm-trading',
            script:            'agents/fm_trading.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'384M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
            },
            error_file: `${ROOT}/logs/fm-trading-error.log`,
            out_file:   `${ROOT}/logs/fm-trading-out.log`,
        },

        // ── WHITEPAPER PUBLISHER (cron) ───────────────────────────────
        {
            name:              'fm-whitepaper',
            script:            'agents/fm_whitepaper.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            cron_restart:      '0 */10 * * *',
            autorestart:       false,
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
            },
            error_file: `${ROOT}/logs/fm-whitepaper-error.log`,
            out_file:   `${ROOT}/logs/fm-whitepaper-out.log`,
        },

        // ── PRODUCT DELIVERY ENGINE ───────────────────────────────────
        {
            name:              'fm-delivery',
            script:            'agents/fm_delivery.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'64M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
            },
            error_file: `${ROOT}/logs/fm-delivery-error.log`,
            out_file:   `${ROOT}/logs/fm-delivery-out.log`,
        },

        // ── RF SALES BRIDGE ───────────────────────────────────────────
        {
            name:              'rf-bridge',
            script:            'agents/rf_bridge.py',
            interpreter:       '/usr/bin/python3',
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            max_memory_restart:'64M',
            env_production: {
                FRACTALMESH_HOME: ROOT,
                NODE_ENV:         'production',
            },
            error_file: `${ROOT}/logs/rf-bridge-error.log`,
            out_file:   `${ROOT}/logs/rf-bridge-out.log`,
        },

        // ── DASHBOARD (serve static) ──────────────────────────────────
        {
            name:              'fm-dashboard',
            script:            'serve',
            args:              ['-s', 'www', '-l', '8090', '--no-clipboard'],
            cwd:               ROOT,
            autorestart:       true,
            watch:             false,
            env_production: {
                NODE_ENV: 'production',
            },
            error_file: `${ROOT}/logs/fm-dashboard-error.log`,
            out_file:   `${ROOT}/logs/fm-dashboard-out.log`,
        },
    ],
};

#!/usr/bin/env bash
# ─────────────────────────────────────────────────────
#  FractalMesh Startup — Kelly's version
#  Run this in Termux if FractalMesh has stopped.
#  Just type:  bash ~/fmsaas/START_FRACTALMESH.sh
# ─────────────────────────────────────────────────────

echo ""
echo "  Starting FractalMesh..."
echo ""

cd ~/fmsaas || { echo "ERROR: ~/fmsaas folder not found. Is this Sam's phone?"; exit 1; }

# Try to wake up any saved processes first
pm2 resurrect 2>/dev/null

sleep 3

# Check if anything is running
RUNNING=$(pm2 list 2>/dev/null | grep -c "online" || echo "0")

if [ "$RUNNING" -gt "0" ]; then
    echo "  FractalMesh is running. ($RUNNING agents online)"
    echo ""
    pm2 list
else
    echo "  Starting all agents from config..."
    pm2 start ecosystem.config.js --env production
    sleep 5
    pm2 save
    echo ""
    echo "  Done. FractalMesh should now be running."
    pm2 list
fi

echo ""
echo "  Dashboard: open browser and go to http://localhost:8090"
echo ""

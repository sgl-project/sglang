"""
FractalMesh Carbon Credit Analysis Agent v2.0.0
Estimates node fleet energy consumption, calculates CO2 footprint,
reports offset opportunities, and tracks carbon credit metrics.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import json
import time
import signal
import sqlite3
import math
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("CARBON_INTERVAL", "86400"))   # daily

PHI      = 1.6180339887

# Device power profiles (watts)
DEVICES = {
    "samsung_android":  {"watts": 3.5,  "count": 1, "hours": 24},
    "raspberry_pi_4":   {"watts": 6.0,  "count": 2, "hours": 24},
    "linux_server":     {"watts": 80.0, "count": 1, "hours": 24},
    "router_modem":     {"watts": 12.0, "count": 1, "hours": 24},
}

# Australian NEM grid emission factor (kg CO2 per kWh) — NSW avg 2026
AUS_EMISSION_FACTOR = float(os.getenv("AUS_EMISSION_FACTOR", "0.77"))

# Carbon credit price AUD per tonne CO2 (Gold Standard ~AUD 20-40)
CREDIT_PRICE_AUD    = float(os.getenv("CARBON_CREDIT_PRICE", "30.0"))

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS carbon_reports (
        id INTEGER PRIMARY KEY, period TEXT,
        kwh_total REAL, co2_kg REAL, co2_tonnes REAL,
        offset_cost_aud REAL, credits_to_buy REAL,
        phi_score REAL, renewable_pct REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS carbon_devices (
        id INTEGER PRIMARY KEY, device_type TEXT, watts REAL, count INTEGER,
        hours REAL, kwh_day REAL, co2_kg_day REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _calculate_fleet_energy() -> dict:
    """Calculate daily kWh and CO2 for full node fleet."""
    total_kwh = 0.0
    breakdown = []

    for device, spec in DEVICES.items():
        kwh_day  = (spec["watts"] * spec["count"] * spec["hours"]) / 1000.0
        co2_day  = kwh_day * AUS_EMISSION_FACTOR
        total_kwh += kwh_day
        breakdown.append({
            "device":    device,
            "watts":     spec["watts"],
            "count":     spec["count"],
            "kwh_day":   round(kwh_day, 4),
            "co2_kg_day": round(co2_day, 4),
        })

    total_co2_kg = total_kwh * AUS_EMISSION_FACTOR
    return {
        "total_kwh_day":  round(total_kwh, 4),
        "total_co2_kg":   round(total_co2_kg, 4),
        "breakdown":      breakdown,
    }


def _offset_analysis(co2_tonnes: float) -> dict:
    """Calculate offset cost and credit requirements."""
    credits_needed  = co2_tonnes
    cost_aud        = round(credits_needed * CREDIT_PRICE_AUD, 2)
    phi_score       = round(cost_aud * PHI / 100, 6)

    # Renewable alternatives analysis
    solar_kwh_day   = 4.5    # avg NSW solar (5kW system)
    solar_offsets_pct = min(100.0, round(
        (solar_kwh_day / max(_calculate_fleet_energy()["total_kwh_day"], 0.001)) * 100, 1))

    return {
        "co2_tonnes":         co2_tonnes,
        "credits_to_buy":     round(credits_needed, 6),
        "offset_cost_aud":    cost_aud,
        "credit_price_aud":   CREDIT_PRICE_AUD,
        "solar_offset_pct":   solar_offsets_pct,
        "phi_score":          phi_score,
        "providers": [
            "Gold Standard (goldstandard.org)",
            "Verra VCS (verra.org)",
            "Australian Carbon Credit Units (ACCU via CER)",
        ],
    }


def _annual_projection(daily_co2_kg: float) -> dict:
    annual_kg     = daily_co2_kg * 365
    annual_tonnes = annual_kg / 1000
    annual_cost   = round(annual_tonnes * CREDIT_PRICE_AUD, 2)
    return {
        "annual_co2_kg":     round(annual_kg, 2),
        "annual_co2_tonnes": round(annual_tonnes, 4),
        "annual_offset_aud": annual_cost,
        "depin_node_equiv":  round(annual_tonnes * 3.5, 2),  # tonnes per DePIN node year
    }


def run_cycle():
    ts     = datetime.utcnow().isoformat()
    print(f"[fm-carbon-credits] {ts} | emission_factor={AUS_EMISSION_FACTOR}kg/kWh")

    fleet      = _calculate_fleet_energy()
    co2_tonnes = fleet["total_co2_kg"] / 1000
    offset     = _offset_analysis(co2_tonnes)
    annual     = _annual_projection(fleet["total_co2_kg"])

    # Log device breakdown
    conn = sqlite3.connect(DB, timeout=10)
    for dev in fleet["breakdown"]:
        conn.execute("""INSERT INTO carbon_devices
            (device_type,watts,count,hours,kwh_day,co2_kg_day) VALUES (?,?,?,?,?,?)""",
            (dev["device"], dev["watts"], dev["count"], 24,
             dev["kwh_day"], dev["co2_kg_day"]))

    # Log summary report
    conn.execute("""INSERT INTO carbon_reports
        (period,kwh_total,co2_kg,co2_tonnes,offset_cost_aud,credits_to_buy,phi_score,renewable_pct)
        VALUES (?,?,?,?,?,?,?,?)""",
        ("daily", fleet["total_kwh_day"], fleet["total_co2_kg"], co2_tonnes,
         offset["offset_cost_aud"], offset["credits_to_buy"],
         offset["phi_score"], offset["solar_offset_pct"]))
    conn.commit(); conn.close()

    print(f"   Fleet:  {fleet['total_kwh_day']:.3f} kWh/day | "
          f"{fleet['total_co2_kg']:.3f} kg CO2/day")
    print(f"   Offset: {offset['credits_to_buy']:.4f} t | "
          f"${offset['offset_cost_aud']:.2f} AUD/day")
    print(f"   Annual: {annual['annual_co2_tonnes']:.3f} t | "
          f"${annual['annual_offset_aud']:.2f} AUD")
    print(f"   Solar:  {offset['solar_offset_pct']}% of fleet could be offset by 5kW system")
    print(f"   ACCU:   https://cer.gov.au/markets/australian-carbon-credit-units")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)
    _db_init()
    print(f"[fm-carbon-credits] Active | interval={INTERVAL}s | "
          f"emission_factor={AUS_EMISSION_FACTOR}kg/kWh | "
          f"credit_price=${CREDIT_PRICE_AUD}AUD/t")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-carbon-credits] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-carbon-credits] Stopped.")

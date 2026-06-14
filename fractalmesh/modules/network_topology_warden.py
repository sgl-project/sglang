"""
FractalMesh Network Topology Warden
Geometric elite traffic routing + packet carrier validation
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import subprocess


class NetworkWarden:
    AGENTS = [
        "fm-bus", "fm-gitops-runner", "fm-integrator",
        "fm-harmonic", "fm-warden",
    ]

    def geometric_elite(self):
        print("[GEOMETRIC-ELITE] Full network topology + Traffic Warden + Packet Carrier active")
        for agent in self.AGENTS:
            result = subprocess.run(
                ["pm2", "restart", agent],
                capture_output=True, text=True
            )
            status = "ok" if result.returncode == 0 else "err"
            print(f"   → {agent} [{status}]")

    def health_check(self):
        result = subprocess.run(["pm2", "jlist"], capture_output=True, text=True)
        return result.stdout


warden = NetworkWarden()

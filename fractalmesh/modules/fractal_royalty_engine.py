"""
FractalMesh Fractal Royalty Engine
φ-geometric stacking of royalty layers with live sovereign.db integration
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import sqlite3
import math

PHI = 1.6180339887


class FractalRoyaltyEngine:
    def geometric_stacking(self, base_royalty: float, layers: int = 5,
                            fib: float = PHI) -> float:
        """
        Sum geometric series: base * φ^0 + base * φ^1 + ... + base * φ^(layers-1)
        Represents compounding royalty value across n distribution layers.
        """
        return sum(base_royalty * (fib ** i) for i in range(layers))

    def phi_decay(self, base_royalty: float, layers: int = 5) -> float:
        """Inverse φ-decay: sum base / φ^i for convergent diminishing returns."""
        return sum(base_royalty / (PHI ** i) for i in range(1, layers + 1))

    def harmonic_split(self, total: float, n: int) -> list:
        """Split total into n φ-proportioned shares that sum to total."""
        weights = [PHI ** (-i) for i in range(n)]
        w_sum   = sum(weights)
        return [round(total * w / w_sum, 4) for w in weights]

    def phi_score(self, pct: float, idx: int) -> float:
        """φ-weight a percentage by layer index."""
        return round(pct * (PHI ** idx) / 100, 6)

    def stack_from_db(self, db_path: str = None) -> list:
        """
        Read royalty_pools from sovereign.db and compute stacked values.
        Returns list of dicts with pool metadata + stacked_royalty.
        """
        if db_path is None:
            root    = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
            db_path = os.path.join(root, "database", "sovereign.db")
        if not os.path.exists(db_path):
            return []
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM royalty_pools").fetchall()
        except Exception:
            conn.close()
            return []
        conn.close()
        results = []
        for i, row in enumerate(rows):
            base    = float(row["aud_balance"] if "aud_balance" in row.keys() else 0.0)
            pct     = float(row["pct"] if "pct" in row.keys() else 0.0)
            stacked = self.geometric_stacking(base)
            decayed = self.phi_decay(base)
            results.append({
                "pool_id":        row["pool_id"],
                "label":          row["label"],
                "pct":            pct,
                "aud_balance":    base,
                "stacked":        round(stacked, 4),
                "decayed":        round(decayed, 4),
                "phi_score":      self.phi_score(pct, i),
                "harmonic_share": self.harmonic_split(base, 3) if base > 0 else [],
            })
        return results

    def update_balance(self, pool_id: str, delta_aud: float,
                       db_path: str = None) -> float:
        """Add delta_aud to pool balance; return new balance."""
        if db_path is None:
            root    = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
            db_path = os.path.join(root, "database", "sovereign.db")
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("""UPDATE royalty_pools
            SET aud_balance = aud_balance + ?, updated = CURRENT_TIMESTAMP
            WHERE pool_id = ?""", (delta_aud, pool_id))
        conn.commit()
        row = conn.execute("SELECT aud_balance FROM royalty_pools WHERE pool_id=?",
                           (pool_id,)).fetchone()
        conn.close()
        return float(row[0]) if row else 0.0


engine = FractalRoyaltyEngine()

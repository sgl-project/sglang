"""
FractalMesh Fractal Royalty Engine
Geometric stacking of royalty layers using φ (golden ratio)
Samuel James Hiotis | ABN 56 628 117 363
"""
import sqlite3
import os

PHI = 1.6180339887


class FractalRoyaltyEngine:
    def geometric_stacking(
        self,
        base_royalty: float,
        layers: int = 5,
        fib: float = PHI,
    ) -> float:
        """
        Sum a geometric series:  base * φ^0 + base * φ^1 + ... + base * φ^(layers-1)
        Represents compounding royalty value across n distribution layers.
        """
        return sum(base_royalty * (fib ** i) for i in range(layers))

    def stack_from_db(self, db_path: str | None = None) -> list[dict]:
        """
        Read all royalty_pools from sovereign.db and compute stacked values.
        Returns list of dicts with pool metadata + stacked_royalty.
        """
        if db_path is None:
            root    = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
            db_path = os.path.join(root, "database", "sovereign.db")

        if not os.path.exists(db_path):
            return []

        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM royalty_pools").fetchall()
        conn.close()

        results = []
        for row in rows:
            base    = float(row["base_royalty"])
            stacked = self.geometric_stacking(base)
            results.append({
                "id":             row["id"],
                "pool_name":      row["pool_name"],
                "industries":     row["industries"],
                "base_royalty":   base,
                "stacked_royalty": round(stacked, 6),
            })
        return results


# Module-level singleton
engine = FractalRoyaltyEngine()

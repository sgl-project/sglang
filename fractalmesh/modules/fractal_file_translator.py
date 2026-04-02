"""
FractalMesh Fractal File Translator + Local/Cloud Balancer
Two-level recursive compression with SHA-256 fractal ID + optional cloud sync
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import hashlib
import zlib


class FractalTranslator:
    """
    Compresses files using nested zlib passes (fractal principle:
    the output of pass 1 is seeded with its own hash before pass 2).
    Files are stored locally under compression_dir.
    Optional cloud sync via Supabase (enabled when env vars are present).
    """

    def __init__(self, compression_dir: str | None = None):
        root = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
        self.compression_dir = compression_dir or os.path.join(root, "compression")
        os.makedirs(self.compression_dir, exist_ok=True)

        # Optional Supabase cloud sync
        self._supabase = None
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_ANON_KEY", "")
        if url and key:
            try:
                from supabase import create_client
                self._supabase = create_client(url, key)
            except ImportError:
                pass  # supabase package not installed — local only

    # ── Compression ──────────────────────────────────────────────────────────

    def compress(self, data: bytes) -> tuple[str, bytes]:
        """
        Apply two-level fractal compression.
        Returns (fractal_id, compressed_bytes).
        """
        level1    = zlib.compress(data, level=9)
        seed      = hashlib.sha256(level1).hexdigest().encode()
        level2    = zlib.compress(level1 + seed, level=9)
        fractal_id = hashlib.sha256(level2).hexdigest()[:16]
        return fractal_id, level2

    def decompress(self, data: bytes) -> bytes:
        """Reverse of compress (strips hash seed before returning raw bytes)."""
        level1_plus_seed = zlib.decompress(data)
        level1           = level1_plus_seed[:-64]   # remove 64-char hex seed
        return zlib.decompress(level1)

    # ── File operations ───────────────────────────────────────────────────────

    def translate_and_balance(self, file_path: str) -> str:
        """
        Compress file → save locally → optionally sync to Supabase storage.
        Returns the fractal_id (16-char hex).
        """
        with open(file_path, "rb") as f:
            raw = f.read()

        fractal_id, compressed = self.compress(raw)
        local_path = os.path.join(self.compression_dir, f"{fractal_id}.fractal")

        with open(local_path, "wb") as f:
            f.write(compressed)

        print(f"[FRACTAL-TRANSLATOR] {file_path} → {fractal_id}.fractal (local)")

        if self._supabase:
            try:
                self._supabase.storage.from_("fractal-mesh").upload(
                    f"files/{fractal_id}", compressed
                )
                print(f"[FRACTAL-TRANSLATOR] {fractal_id} synced to Supabase cloud")
            except Exception as e:
                print(f"[FRACTAL-TRANSLATOR] Cloud sync skipped: {e}")

        return fractal_id

    def restore(self, fractal_id: str, out_path: str) -> bool:
        """Decompress a stored fractal file back to out_path."""
        src = os.path.join(self.compression_dir, f"{fractal_id}.fractal")
        if not os.path.exists(src):
            print(f"[FRACTAL-TRANSLATOR] Not found: {fractal_id}")
            return False
        with open(src, "rb") as f:
            compressed = f.read()
        with open(out_path, "wb") as f:
            f.write(self.decompress(compressed))
        print(f"[FRACTAL-TRANSLATOR] Restored {fractal_id} → {out_path}")
        return True


# Module-level singleton
translator = FractalTranslator()

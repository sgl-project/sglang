#!/usr/bin/env python3
"""
FractalMesh NFT Minter v4.1
Watches .latest_track → Pinata IPFS → Sugar Solana mint → Dev.to post
Vault: PINATA_KEY or PINATA_JWT, DEVTO_KEY, SOLANA_KEYPAIR_PATH
"""
import os, sys, json, time, subprocess, logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MINTER] %(message)s",
    handlers=[
        logging.FileHandler(Path.home() / ".fm_logs/nft_minter.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "--break-system-packages", "requests"], check=True)
    import requests

# Load vault
VAULT_PATH = os.getenv("VAULT_PATH", str(Path.home() / ".secrets/fractal.env"))
if Path(VAULT_PATH).exists():
    for line in Path(VAULT_PATH).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

PINATA_KEY    = os.getenv("PINATA_KEY") or os.getenv("PINATA_JWT", "")
DEVTO_KEY     = os.getenv("DEVTO_KEY", "")
KEYPAIR_PATH  = os.getenv("SOLANA_KEYPAIR_PATH",
                           str(Path.home() / ".secrets" / "solana-keypair.json"))
RPC_URL       = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
LATEST_FILE   = Path.home() / "synthwave" / ".latest_track"
MINTED_LOG    = Path.home() / "synthwave" / ".minted_tracks"
ROYALTY_BPS   = 700
POLL_INTERVAL = 60


def is_minted(path: str) -> bool:
    if not MINTED_LOG.exists():
        return False
    return path in MINTED_LOG.read_text()


def mark_minted(path: str) -> None:
    with open(MINTED_LOG, "a") as f:
        f.write(path + "\n")


def solana_address() -> str:
    try:
        r = subprocess.run(
            ["solana", "address", "--keypair", KEYPAIR_PATH],
            capture_output=True, text=True, timeout=10
        )
        addr = r.stdout.strip()
        return addr if addr else "unknown"
    except Exception:
        return "unknown"


def pin_file(file_path: str) -> str:
    if not PINATA_KEY:
        log.warning("PINATA_KEY / PINATA_JWT missing — skipping IPFS upload")
        return ""
    with open(file_path, "rb") as fh:
        resp = requests.post(
            "https://api.pinata.cloud/pinning/pinFileToIPFS",
            files={"file": (Path(file_path).name, fh)},
            headers={"Authorization": f"Bearer {PINATA_KEY}"},
            timeout=180,
        )
    resp.raise_for_status()
    h = resp.json().get("IpfsHash", "")
    log.info(f"File pinned: ipfs://{h}")
    return h


def pin_metadata(name: str, audio_hash: str, file_path: str) -> str:
    if not PINATA_KEY:
        return ""
    ext      = Path(file_path).suffix.lower()
    mime     = "audio/wav" if ext == ".wav" else "audio/midi"
    creator  = solana_address()
    metadata = {
        "name":                    name,
        "symbol":                  "FMSW",
        "description":             "FractalMesh Synthwave — autonomous AI-generated audio on Solana.",
        "seller_fee_basis_points": ROYALTY_BPS,
        "image":                   f"ipfs://{audio_hash}",
        "animation_url":           f"ipfs://{audio_hash}",
        "external_url":            "https://fractalmesh.io",
        "properties": {
            "files":    [{"uri": f"ipfs://{audio_hash}", "type": mime}],
            "category": "audio",
            "creators": [{"address": creator, "share": 100}],
        },
        "attributes": [
            {"trait_type": "Genre",     "value": "Synthwave"},
            {"trait_type": "Generator", "value": "FractalMesh AI DJ"},
            {"trait_type": "Chain",     "value": "Solana"},
            {"trait_type": "Format",    "value": ext.lstrip(".")},
        ],
    }
    resp = requests.post(
        "https://api.pinata.cloud/pinning/pinJSONToIPFS",
        json={"pinataContent": metadata, "pinataMetadata": {"name": f"{name}_metadata"}},
        headers={"Authorization": f"Bearer {PINATA_KEY}", "Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    h = resp.json().get("IpfsHash", "")
    log.info(f"Metadata pinned: ipfs://{h}")
    return h


def mint_with_sugar(meta_uri: str, name: str) -> bool:
    if not Path(KEYPAIR_PATH).exists():
        log.warning(f"Solana keypair not found: {KEYPAIR_PATH} — minting skipped")
        return False
    if not Path("config.json").exists():
        cfg = {
            "price": 0.0, "number": 1,
            "symbol": "FMSW", "sellerFeeBasisPoints": ROYALTY_BPS,
            "solTreasuryAccount": solana_address(),
            "splTokenAccount": None, "splToken": None,
            "goLiveDate": "now", "endSettings": None,
            "whitelistMintSettings": None, "hiddenSettings": None,
            "uploadMethod": "bundlr", "retainAuthority": True, "isMutable": True,
            "creators": [{"address": solana_address(), "share": 100}],
        }
        Path("config.json").write_text(json.dumps(cfg, indent=2))
    try:
        r = subprocess.run(
            ["sugar", "mint", "--keypair", KEYPAIR_PATH, "--rpc-url", RPC_URL, "--number", "1"],
            capture_output=True, text=True, timeout=180,
        )
        if r.returncode == 0:
            log.info(f"Mint OK: {r.stdout.strip()[:300]}")
            return True
        log.error(f"Sugar error: {r.stderr.strip()[:300]}")
    except FileNotFoundError:
        log.warning("sugar not in PATH — install: bash <(curl -sSfL https://sugar.metaplex.com/install.sh)")
    except subprocess.TimeoutExpired:
        log.error("Sugar mint timed out")
    return False


def post_devto(name: str, audio_hash: str, meta_hash: str) -> None:
    if not DEVTO_KEY:
        log.warning("DEVTO_KEY missing — skipping Dev.to post")
        return
    body = f"""## New Drop: {name}

> Autonomous AI-generated synthwave, minted on Solana by the FractalMesh Rolls-Royce engine.

### Links
- 🎵 **Audio (IPFS):** [ipfs://{audio_hash}](https://ipfs.io/ipfs/{audio_hash})
- 📄 **Metadata:** [ipfs://{meta_hash}](https://ipfs.io/ipfs/{meta_hash})

### About FractalMesh
Autonomous AI mesh — live crypto trading, multi-agent LLM orchestration,
generative music → NFT pipeline, running 24/7 on sovereign infrastructure.
ABN 56 628 117 363 | Samuel James Hiotis | Albury NSW

---
*Auto-posted by FractalMesh Synthwave Empire v4.1*
"""
    resp = requests.post(
        "https://dev.to/api/articles",
        headers={"api-key": DEVTO_KEY, "Content-Type": "application/json"},
        json={"article": {
            "title": name,
            "body_markdown": body,
            "tags": ["synthwave", "nft", "solana", "ai"],
            "published": True,
        }},
        timeout=30,
    )
    if resp.status_code in (200, 201):
        log.info(f"Dev.to posted: {resp.json().get('url', '')}")
    else:
        log.error(f"Dev.to failed {resp.status_code}: {resp.text[:200]}")


def main():
    log.info("═══ FractalMesh NFT Minter v4.1 started ═══")
    last_seen = ""
    cycle     = 0

    while True:
        try:
            if LATEST_FILE.exists():
                track = LATEST_FILE.read_text().strip()
                if track and track != last_seen and not is_minted(track) and Path(track).exists():
                    cycle += 1
                    name = f"FractalMesh Synthwave #{cycle:04d}"
                    log.info(f"Processing: {Path(track).name}  ({name})")
                    audio_hash = pin_file(track)
                    meta_hash  = ""
                    if audio_hash:
                        meta_hash = pin_metadata(name, audio_hash, track)
                    if meta_hash:
                        mint_with_sugar(f"ipfs://{meta_hash}", name)
                        post_devto(name, audio_hash, meta_hash)
                    elif audio_hash:
                        post_devto(name, audio_hash, "")
                    mark_minted(track)
                    last_seen = track
        except Exception as e:
            log.exception(f"Minter error: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

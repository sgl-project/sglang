#!/usr/bin/env python3
"""
fm_nft_engine.py — NFT Generation + Listing Engine (Port 7828)
FractalMesh OMEGA Titan — Full NFT pipeline: AI image generation via HuggingFace
Stable Diffusion SDXL, ERC-721 metadata, OpenSea v2 listings, batch minting,
dynamic pricing strategies, portfolio tracking.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import json
import logging
import os
import random
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT             = int(os.getenv("NFT_ENGINE_PORT", "7828"))
ROOT             = Path(os.path.expanduser("~/fmsaas"))
DB_PATH          = ROOT / "database" / "sovereign.db"
LOG_PATH         = ROOT / "logs" / "nft_engine.log"
IMAGES_DIR       = ROOT / "nft" / "images"
METADATA_DIR     = ROOT / "nft" / "metadata"

OPENSEA_API_KEY  = os.getenv("OPENSEA_API_KEY", "")
OPENSEA_WALLET   = os.getenv("OPENSEA_WALLET_ADDRESS", "")
OPENSEA_CHAIN    = os.getenv("OPENSEA_CHAIN", "base")
ALCHEMY_API_KEY  = os.getenv("ALCHEMY_API_KEY", "")
ALCHEMY_ETH      = os.getenv("ALCHEMY_ETH_MAINNET", "")
HF_TOKEN         = os.getenv("HF_TOKEN", "")

OPENSEA_BASE     = "https://api.opensea.io/api/v2"
HF_SDXL_URL      = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
CONDUIT_KEY      = "0x0000007b02230091a7ed01230072f7006a004d60a8d4e71d599b8104250f0000"
SEAPORT_ADDR     = "0x0000000000000068F116a894984e2DB1123eB395"

TRAIT_BACKGROUNDS = ["Cyber", "Void", "Neon", "Matrix", "Quantum"]
TRAIT_ENERGIES    = ["High", "Medium", "Low", "Ultra", "Zero"]
TRAIT_RARITIES    = ["Common", "Uncommon", "Rare", "Epic", "Legendary"]

RARITY_MULTIPLIERS = {
    "Common": 1.0, "Uncommon": 1.5, "Rare": 2.0, "Epic": 3.0, "Legendary": 5.0
}

ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [NFT-ENGINE] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("nft_engine")


# ── database ──────────────────────────────────────────────────────────────────
def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS nft_collections (
                id               INTEGER PRIMARY KEY,
                name             TEXT,
                symbol           TEXT,
                description      TEXT,
                chain            TEXT,
                contract_address TEXT,
                status           TEXT,
                created_at       REAL
            );

            CREATE TABLE IF NOT EXISTS nft_tokens (
                id           INTEGER PRIMARY KEY,
                collection_id INTEGER,
                token_id     TEXT,
                name         TEXT,
                description  TEXT,
                image_url    TEXT,
                metadata_uri TEXT,
                traits       TEXT,
                floor_price  REAL,
                status       TEXT,
                opensea_url  TEXT,
                created_at   REAL
            );

            CREATE TABLE IF NOT EXISTS nft_listings (
                id                 INTEGER PRIMARY KEY,
                token_id           TEXT,
                collection_id      INTEGER,
                price_eth          REAL,
                currency           TEXT,
                start_time         REAL,
                end_time           REAL,
                status             TEXT,
                opensea_listing_id TEXT,
                created_at         REAL
            );

            CREATE TABLE IF NOT EXISTS nft_sales (
                id               INTEGER PRIMARY KEY,
                token_id         TEXT,
                collection_id    INTEGER,
                price_eth        REAL,
                buyer            TEXT,
                seller           TEXT,
                tx_hash          TEXT,
                opensea_event_id TEXT UNIQUE,
                created_at       REAL
            );
        """)
    log.info("Database tables initialised.")


# ── helpers ───────────────────────────────────────────────────────────────────
def _ensure_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def _eth_to_wei(eth: float) -> int:
    return int(eth * 1e18)


def _opensea_get(path: str) -> dict:
    url = f"{OPENSEA_BASE}{path}"
    req = urllib.request.Request(url, headers={
        "X-API-KEY": OPENSEA_API_KEY,
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"OpenSea GET {path} → {exc.code}: {body}") from exc


def _opensea_post(path: str, body: dict) -> dict:
    url = f"{OPENSEA_BASE}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "X-API-KEY": OPENSEA_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        raise RuntimeError(f"OpenSea POST {path} → {exc.code}: {body_text}") from exc


def _hf_generate_image(prompt: str, width: int = 512, height: int = 512) -> bytes:
    payload = json.dumps({
        "inputs": prompt,
        "parameters": {"width": width, "height": height},
    }).encode()
    req = urllib.request.Request(HF_SDXL_URL, data=payload, method="POST", headers={
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HuggingFace inference → {exc.code}: {body}") from exc


def _json_response(handler: BaseHTTPRequestHandler, code: int, data) -> None:
    body = json.dumps(data).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode())


# ── route handlers ────────────────────────────────────────────────────────────
def _handle_health(handler: BaseHTTPRequestHandler):
    _json_response(handler, 200, {
        "status": "ok",
        "service": "fm-nft-engine",
        "port": PORT,
    })


def _handle_generate_image(handler: BaseHTTPRequestHandler):
    try:
        body = _read_body(handler)
        prompt    = body.get("prompt", "fractal mesh cyberpunk landscape")
        style     = body.get("style", "digital art")
        width     = int(body.get("width", 512))
        height    = int(body.get("height", 512))

        full_prompt = f"{prompt}, {style}, high quality, detailed"
        log.info("Generating image: %s", full_prompt)

        image_bytes = _hf_generate_image(full_prompt, width, height)

        timestamp_ms = int(time.time() * 1000)
        image_path = IMAGES_DIR / f"{timestamp_ms}.png"
        image_path.write_bytes(image_bytes)

        _json_response(handler, 200, {
            "image_path": str(image_path),
            "prompt": full_prompt,
            "generated_at": timestamp_ms,
        })
    except Exception as exc:
        log.error("generate/image error: %s", exc)
        _json_response(handler, 500, {"error": str(exc)})


def _handle_generate_metadata(handler: BaseHTTPRequestHandler):
    try:
        body = _read_body(handler)
        name        = body.get("name", "FractalMesh Token")
        description = body.get("description", "")
        image_path  = body.get("image_path", "")
        traits      = body.get("traits", [])

        metadata = {
            "name": name,
            "description": description,
            "image": image_path,
            "attributes": [
                {"trait_type": t.get("trait_type", ""), "value": t.get("value", "")}
                for t in traits
            ],
            "external_url": "https://fractalmesh.net",
        }

        timestamp_ms = int(time.time() * 1000)
        meta_path = METADATA_DIR / f"{timestamp_ms}.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        _json_response(handler, 200, {
            "metadata_path": str(meta_path),
            "metadata": metadata,
        })
    except Exception as exc:
        log.error("generate/metadata error: %s", exc)
        _json_response(handler, 500, {"error": str(exc)})


def _handle_collection_create(handler: BaseHTTPRequestHandler):
    try:
        body = _read_body(handler)
        name             = body.get("name", "")
        symbol           = body.get("symbol", "")
        description      = body.get("description", "")
        chain            = body.get("chain", OPENSEA_CHAIN)
        contract_address = body.get("contract_address", "")

        now = time.time()
        with _get_db() as conn:
            cur = conn.execute(
                """INSERT INTO nft_collections
                   (name, symbol, description, chain, contract_address, status, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (name, symbol, description, chain, contract_address, "active", now),
            )
            collection_id = cur.lastrowid

        log.info("Created collection id=%d name=%s", collection_id, name)
        _json_response(handler, 200, {"collection_id": collection_id})
    except Exception as exc:
        log.error("collection/create error: %s", exc)
        _json_response(handler, 500, {"error": str(exc)})


def _handle_generate_batch(handler: BaseHTTPRequestHandler):
    try:
        body          = _read_body(handler)
        collection_id = int(body.get("collection_id", 0))
        count         = int(body.get("count", 1))
        base_prompt   = body.get("base_prompt", "fractal mesh")
        style         = body.get("style", "digital art")
        name_prefix   = body.get("name_prefix", "FractalMesh")

        tokens = []
        now = time.time()

        for i in range(1, count + 1):
            bg      = random.choice(TRAIT_BACKGROUNDS)
            energy  = random.choice(TRAIT_ENERGIES)
            rarity  = random.choice(TRAIT_RARITIES)
            traits  = [
                {"trait_type": "Background", "value": bg},
                {"trait_type": "Energy",     "value": energy},
                {"trait_type": "Rarity",     "value": rarity},
            ]

            full_prompt = f"{base_prompt}, {style}, high quality, detailed"
            log.info("Batch generate token %d/%d", i, count)

            try:
                image_bytes = _hf_generate_image(full_prompt, 512, 512)
            except Exception as img_exc:
                log.warning("Image generation failed for token %d: %s", i, img_exc)
                image_bytes = b""

            timestamp_ms = int(time.time() * 1000)
            image_path   = IMAGES_DIR / f"{timestamp_ms}.png"
            if image_bytes:
                image_path.write_bytes(image_bytes)

            token_name = f"{name_prefix} #{i:03d}"
            metadata   = {
                "name": token_name,
                "description": f"{token_name} — FractalMesh genesis collection.",
                "image": str(image_path),
                "attributes": [
                    {"trait_type": t["trait_type"], "value": t["value"]} for t in traits
                ],
                "external_url": "https://fractalmesh.net",
            }

            meta_path = METADATA_DIR / f"{timestamp_ms}.json"
            meta_path.write_text(json.dumps(metadata, indent=2))

            token_id_str = str(i)
            with _get_db() as conn:
                cur = conn.execute(
                    """INSERT INTO nft_tokens
                       (collection_id, token_id, name, description, image_url,
                        metadata_uri, traits, floor_price, status, opensea_url, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        collection_id, token_id_str, token_name,
                        metadata["description"], str(image_path),
                        str(meta_path), json.dumps(traits), 0.0,
                        "minted", "", now + i,
                    ),
                )
                db_id = cur.lastrowid

            tokens.append({
                "db_id": db_id,
                "token_id": token_id_str,
                "name": token_name,
                "image_path": str(image_path),
                "metadata_path": str(meta_path),
                "traits": traits,
            })

            if i < count:
                time.sleep(1)

        _json_response(handler, 200, {
            "collection_id": collection_id,
            "generated": len(tokens),
            "tokens": tokens,
        })
    except Exception as exc:
        log.error("generate/batch error: %s", exc)
        _json_response(handler, 500, {"error": str(exc)})


def _handle_listing_create(handler: BaseHTTPRequestHandler):
    try:
        body          = _read_body(handler)
        collection_id = int(body.get("collection_id", 0))
        token_id      = str(body.get("token_id", "1"))
        price_eth     = float(body.get("price_eth", 0.05))
        duration_days = int(body.get("duration_days", 7))

        with _get_db() as conn:
            row = conn.execute(
                "SELECT contract_address FROM nft_collections WHERE id=?",
                (collection_id,)
            ).fetchone()

        if not row:
            _json_response(handler, 404, {"error": "Collection not found"})
            return

        contract = row["contract_address"] or ""
        start_ts = int(time.time())
        end_ts   = start_ts + duration_days * 86400
        price_wei = str(_eth_to_wei(price_eth))

        listing_body = {
            "chain": OPENSEA_CHAIN,
            "protocol_address": SEAPORT_ADDR,
            "parameters": {
                "offerer": OPENSEA_WALLET,
                "offer": [{
                    "itemType": 2,
                    "token": contract,
                    "identifierOrCriteria": token_id,
                }],
                "consideration": [{
                    "amount": price_wei,
                    "recipient": OPENSEA_WALLET,
                }],
                "startTime": str(start_ts),
                "endTime": str(end_ts),
                "salt": "1",
                "conduitKey": CONDUIT_KEY,
            },
        }

        try:
            os_resp = _opensea_post("/listings", listing_body)
            opensea_listing_id = os_resp.get("order", {}).get("order_hash", "") or os_resp.get("order_hash", "")
        except Exception as os_exc:
            log.warning("OpenSea listing failed (continuing locally): %s", os_exc)
            opensea_listing_id = f"local_{int(time.time())}"

        now = time.time()
        with _get_db() as conn:
            cur = conn.execute(
                """INSERT INTO nft_listings
                   (token_id, collection_id, price_eth, currency, start_time, end_time,
                    status, opensea_listing_id, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (token_id, collection_id, price_eth, "ETH", float(start_ts),
                 float(end_ts), "active", opensea_listing_id, now),
            )
            listing_id = cur.lastrowid

        log.info("Listing created id=%d opensea_id=%s", listing_id, opensea_listing_id)
        _json_response(handler, 200, {
            "listing_id": listing_id,
            "opensea_listing_id": opensea_listing_id,
            "price_eth": price_eth,
        })
    except Exception as exc:
        log.error("listing/create error: %s", exc)
        _json_response(handler, 500, {"error": str(exc)})


def _handle_listings_get(handler: BaseHTTPRequestHandler):
    try:
        with _get_db() as conn:
            rows = conn.execute("""
                SELECT l.*, t.name as token_name, t.image_url, t.traits, t.opensea_url
                FROM nft_listings l
                LEFT JOIN nft_tokens t ON l.token_id = t.token_id
                WHERE l.status='active'
                ORDER BY l.created_at DESC
            """).fetchall()

        listings = [dict(r) for r in rows]
        _json_response(handler, 200, {"listings": listings, "count": len(listings)})
    except Exception as exc:
        log.error("listings GET error: %s", exc)
        _json_response(handler, 500, {"error": str(exc)})


def _handle_collection_stats(handler: BaseHTTPRequestHandler, collection_id: int):
    try:
        with _get_db() as conn:
            coll = conn.execute(
                "SELECT * FROM nft_collections WHERE id=?", (collection_id,)
            ).fetchone()
            if not coll:
                _json_response(handler, 404, {"error": "Collection not found"})
                return

            total_tokens = conn.execute(
                "SELECT COUNT(*) as c FROM nft_tokens WHERE collection_id=?",
                (collection_id,)
            ).fetchone()["c"]

            listed_count = conn.execute(
                "SELECT COUNT(*) as c FROM nft_listings WHERE collection_id=? AND status='active'",
                (collection_id,)
            ).fetchone()["c"]

            avg_price_row = conn.execute(
                "SELECT AVG(price_eth) as avg FROM nft_listings WHERE collection_id=? AND status='active'",
                (collection_id,)
            ).fetchone()
            avg_price = avg_price_row["avg"] or 0.0

        slug = coll["name"].lower().replace(" ", "-")
        opensea_stats = {}
        try:
            opensea_stats = _opensea_get(f"/collections/{slug}/stats")
        except Exception as os_exc:
            log.warning("OpenSea stats fetch failed: %s", os_exc)

        _json_response(handler, 200, {
            "collection_id": collection_id,
            "name": coll["name"],
            "symbol": coll["symbol"],
            "chain": coll["chain"],
            "local": {
                "total_tokens": total_tokens,
                "listed_count": listed_count,
                "avg_listing_price_eth": round(avg_price, 6),
            },
            "opensea": opensea_stats,
        })
    except Exception as exc:
        log.error("collection stats error: %s", exc)
        _json_response(handler, 500, {"error": str(exc)})


def _handle_sales_recent(handler: BaseHTTPRequestHandler):
    try:
        with _get_db() as conn:
            coll = conn.execute(
                "SELECT * FROM nft_collections ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

        slug = coll["name"].lower().replace(" ", "-") if coll else "fractalmesh-genesis"
        events = []

        try:
            resp = _opensea_get(f"/events/collection/{slug}?event_type=sale&limit=20")
            events = resp.get("asset_events", [])

            with _get_db() as conn:
                for ev in events:
                    try:
                        price_eth = float(ev.get("payment", {}).get("quantity", 0)) / 1e18
                        conn.execute(
                            """INSERT OR IGNORE INTO nft_sales
                               (token_id, collection_id, price_eth, buyer, seller,
                                tx_hash, opensea_event_id, created_at)
                               VALUES (?,?,?,?,?,?,?,?)""",
                            (
                                str(ev.get("nft", {}).get("identifier", "")),
                                coll["id"] if coll else 0,
                                price_eth,
                                ev.get("buyer", ""),
                                ev.get("seller", ""),
                                ev.get("transaction", ""),
                                ev.get("event_type", "") + "_" + str(ev.get("order_hash", ev.get("id", ""))),
                                time.time(),
                            ),
                        )
                    except Exception as row_exc:
                        log.warning("Sale upsert failed: %s", row_exc)
        except Exception as os_exc:
            log.warning("OpenSea events fetch failed: %s", os_exc)

        _json_response(handler, 200, {"sales": events, "count": len(events)})
    except Exception as exc:
        log.error("sales/recent error: %s", exc)
        _json_response(handler, 500, {"error": str(exc)})


def _handle_pricing_dynamic(handler: BaseHTTPRequestHandler):
    try:
        body          = _read_body(handler)
        collection_id = int(body.get("collection_id", 0))
        strategy      = body.get("strategy", "fixed")
        base_price    = float(body.get("base_price_eth", 0.05))

        with _get_db() as conn:
            coll = conn.execute(
                "SELECT * FROM nft_collections WHERE id=?", (collection_id,)
            ).fetchone()
            if not coll:
                _json_response(handler, 404, {"error": "Collection not found"})
                return

            tokens = conn.execute(
                "SELECT * FROM nft_tokens WHERE collection_id=?", (collection_id,)
            ).fetchall()

        slug = coll["name"].lower().replace(" ", "-")

        if strategy == "floor_plus_10pct":
            floor = base_price
            try:
                stats = _opensea_get(f"/collections/{slug}/stats")
                floor_raw = stats.get("total", {}).get("floor_price", 0) or 0
                floor = float(floor_raw)
            except Exception as os_exc:
                log.warning("Could not fetch floor price: %s", os_exc)
            computed_price = floor * 1.10
            prices_map = {t["id"]: computed_price for t in tokens}

        elif strategy == "rarity_weighted":
            prices_map = {}
            for t in tokens:
                traits_list = json.loads(t["traits"] or "[]")
                rarity_val  = next(
                    (tr["value"] for tr in traits_list if tr.get("trait_type") == "Rarity"),
                    "Common"
                )
                multiplier = RARITY_MULTIPLIERS.get(rarity_val, 1.0)
                prices_map[t["id"]] = round(base_price * multiplier, 6)

        else:
            prices_map = {t["id"]: base_price for t in tokens}

        updated = 0
        with _get_db() as conn:
            for db_id, price in prices_map.items():
                conn.execute(
                    "UPDATE nft_tokens SET floor_price=? WHERE id=?",
                    (price, db_id),
                )
                updated += 1

        log.info("Dynamic pricing updated %d tokens strategy=%s", updated, strategy)
        _json_response(handler, 200, {
            "collection_id": collection_id,
            "prices_updated": updated,
            "strategy": strategy,
        })
    except Exception as exc:
        log.error("pricing/dynamic error: %s", exc)
        _json_response(handler, 500, {"error": str(exc)})


def _handle_tokens_collection(handler: BaseHTTPRequestHandler, collection_id: int):
    try:
        with _get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM nft_tokens WHERE collection_id=? ORDER BY created_at DESC",
                (collection_id,)
            ).fetchall()

        tokens = [dict(r) for r in rows]
        _json_response(handler, 200, {"collection_id": collection_id, "tokens": tokens, "count": len(tokens)})
    except Exception as exc:
        log.error("tokens collection error: %s", exc)
        _json_response(handler, 500, {"error": str(exc)})


def _handle_portfolio(handler: BaseHTTPRequestHandler):
    try:
        resp = _opensea_get(f"/chain/{OPENSEA_CHAIN}/account/{OPENSEA_WALLET}/nfts")
        nfts = resp.get("nfts", [])
        _json_response(handler, 200, {
            "wallet": OPENSEA_WALLET,
            "chain": OPENSEA_CHAIN,
            "nfts": nfts,
            "count": len(nfts),
        })
    except Exception as exc:
        log.error("portfolio error: %s", exc)
        _json_response(handler, 500, {"error": str(exc)})


# ── HTTP handler ──────────────────────────────────────────────────────────────
class NFTEngineHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info(fmt, *args)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/health":
            _handle_health(self)

        elif path == "/listings":
            _handle_listings_get(self)

        elif path == "/sales/recent":
            _handle_sales_recent(self)

        elif path == "/portfolio":
            _handle_portfolio(self)

        elif path.startswith("/collection/") and path.endswith("/stats"):
            parts = path.split("/")
            try:
                coll_id = int(parts[2])
                _handle_collection_stats(self, coll_id)
            except (IndexError, ValueError):
                _json_response(self, 400, {"error": "Invalid collection id"})

        elif path.startswith("/tokens/"):
            parts = path.split("/")
            try:
                coll_id = int(parts[2])
                _handle_tokens_collection(self, coll_id)
            except (IndexError, ValueError):
                _json_response(self, 400, {"error": "Invalid collection id"})

        else:
            _json_response(self, 404, {"error": "Not found", "path": self.path})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/generate/image":
            _handle_generate_image(self)
        elif path == "/generate/metadata":
            _handle_generate_metadata(self)
        elif path == "/generate/batch":
            _handle_generate_batch(self)
        elif path == "/collection/create":
            _handle_collection_create(self)
        elif path == "/listing/create":
            _handle_listing_create(self)
        elif path == "/pricing/dynamic":
            _handle_pricing_dynamic(self)
        else:
            _json_response(self, 404, {"error": "Not found", "path": self.path})


# ── entrypoint ────────────────────────────────────────────────────────────────
def main():
    _ensure_dirs()
    _init_db()

    server = HTTPServer(("0.0.0.0", PORT), NFTEngineHandler)
    log.info("fm-nft-engine listening on port %d", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down fm-nft-engine.")
        server.server_close()


if __name__ == "__main__":
    main()

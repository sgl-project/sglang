#!/usr/bin/env python3
"""
fm_data_api.py — FractalMesh OMEGA Titan Data Monetization REST/GraphQL API (Port 7829)
Sells structured data products via API keys with tier-based rate limiting.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os, json, time, sqlite3, secrets, logging, re, io, csv
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import date

# ---------------------------------------------------------------------------
# Bootstrap vault
# ---------------------------------------------------------------------------
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _l in _vault.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _, _v = _l.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT             = int(os.getenv("DATA_API_PORT", "7829"))
ADMIN_SECRET     = os.getenv("ADMIN_SECRET", "")
STRIPE_SECRET    = os.getenv("STRIPE_SECRET_KEY", "")
ROOT             = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB               = ROOT / "database" / "sovereign.db"
LOG              = ROOT / "logs" / "data_api.log"

for p in (ROOT, LOG.parent, DB.parent):
    p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DATA-API] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("fm_data_api")

# ---------------------------------------------------------------------------
# Tier limits
# ---------------------------------------------------------------------------
TIER_LIMITS = {
    "free":       100,
    "starter":    1000,
    "pro":        10000,
    "enterprise": -1,   # unlimited
}

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB, timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _db_init():
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id              INTEGER PRIMARY KEY,
            key             TEXT UNIQUE,
            owner           TEXT,
            email           TEXT,
            tier            TEXT DEFAULT 'free',
            requests_today  INTEGER DEFAULT 0,
            requests_total  INTEGER DEFAULT 0,
            limit_per_day   INTEGER DEFAULT 100,
            created_at      REAL,
            last_used       REAL
        );
        CREATE TABLE IF NOT EXISTS api_requests (
            id              INTEGER PRIMARY KEY,
            api_key         TEXT,
            endpoint        TEXT,
            method          TEXT,
            response_code   INTEGER,
            latency_ms      REAL,
            created_at      REAL
        );
        CREATE TABLE IF NOT EXISTS data_products (
            id              INTEGER PRIMARY KEY,
            name            TEXT,
            description     TEXT,
            category        TEXT,
            price_usd       REAL,
            format          TEXT,
            record_count    INTEGER,
            last_updated    REAL,
            status          TEXT
        );
        CREATE TABLE IF NOT EXISTS data_exports (
            id              INTEGER PRIMARY KEY,
            api_key         TEXT,
            product_id      INTEGER,
            format          TEXT,
            records_exported INTEGER,
            created_at      REAL
        );
    """)

    # Seed data products if empty
    existing = conn.execute("SELECT COUNT(*) FROM data_products").fetchone()[0]
    if existing == 0:
        now = time.time()
        conn.executemany(
            "INSERT INTO data_products (name, description, category, price_usd, format, record_count, last_updated, status) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [
                ("FractalMesh Leads Database", "B2B leads for AU/NZ businesses",
                 "leads", 49.0, "json/csv", 0, now, "active"),
                ("WiFi Network Intelligence", "Geolocated WiFi networks AU",
                 "wifi_networks", 29.0, "json", 0, now, "active"),
                ("Crypto Market Signals", "Real-time price + sentiment data",
                 "market_data", 19.0, "json", 0, now, "active"),
            ],
        )
    conn.commit()
    conn.close()


def _log_request(api_key: str, endpoint: str, method: str, code: int, latency_ms: float):
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO api_requests (api_key, endpoint, method, response_code, latency_ms, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (api_key, endpoint, method, code, latency_ms, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("_log_request: %s", exc)


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_auth(handler):
    """Return (row, None) on success or (None, error_string) on failure."""
    key = handler.headers.get("X-API-Key", "").strip()
    if not key:
        return None, "401 Unauthorized – missing X-API-Key header"

    conn = _get_db()
    row = conn.execute("SELECT * FROM api_keys WHERE key=?", (key,)).fetchone()
    if not row:
        conn.close()
        return None, "401 Unauthorized – invalid API key"

    row = dict(row)
    today_str = date.today().isoformat()
    last_used_date = ""
    if row["last_used"]:
        last_used_date = date.fromtimestamp(row["last_used"]).isoformat()

    # Reset daily counter if it's a new day
    if last_used_date != today_str:
        conn.execute(
            "UPDATE api_keys SET requests_today=0 WHERE key=?", (key,)
        )
        row["requests_today"] = 0

    limit = row["limit_per_day"]
    if limit != -1 and row["requests_today"] >= limit:
        conn.close()
        return None, f"429 Too Many Requests – daily limit of {limit} reached"

    # Increment counters and update last_used
    conn.execute(
        "UPDATE api_keys SET requests_today=requests_today+1, "
        "requests_total=requests_total+1, last_used=? WHERE key=?",
        (time.time(), key),
    )
    conn.commit()
    conn.close()
    row["requests_today"] += 1
    return row, None


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

def _sample_rows(category: str, limit: int = 3) -> list:
    """Pull sample rows from live tables or return mock data."""
    conn = _get_db()
    rows = []
    try:
        if category == "leads":
            cursor = conn.execute(
                "SELECT id, name, email, company, location, score FROM leads LIMIT ?", (limit,)
            )
            rows = [dict(r) for r in cursor.fetchall()]
        elif category == "wifi_networks":
            cursor = conn.execute(
                "SELECT * FROM osint_reports LIMIT ?", (limit,)
            )
            rows = [dict(r) for r in cursor.fetchall()]
        elif category == "market_data":
            cursor = conn.execute(
                "SELECT * FROM coingecko_queries LIMIT ?", (limit,)
            )
            rows = [dict(r) for r in cursor.fetchall()]
    except sqlite3.OperationalError:
        pass  # table doesn't exist yet — fall through to mock
    conn.close()

    if not rows:
        if category == "leads":
            rows = [
                {"id": 1, "name": "Alice Smith", "email": "alice@acme.com.au", "company": "Acme AU", "location": "Sydney", "score": 0.91},
                {"id": 2, "name": "Bob Jones",   "email": "bob@techco.com.au",  "company": "TechCo",  "location": "Melbourne", "score": 0.85},
                {"id": 3, "name": "Carol Wu",    "email": "carol@startupnz.co.nz","company":"StartupNZ","location":"Auckland","score": 0.78},
            ]
        elif category == "wifi_networks":
            rows = [
                {"id": 1, "ssid": "CafeWifi",    "bssid": "AA:BB:CC:DD:EE:01", "lat": -33.87, "lng": 151.21, "signal": -62},
                {"id": 2, "ssid": "HomeNet_5G",  "bssid": "AA:BB:CC:DD:EE:02", "lat": -37.81, "lng": 144.96, "signal": -71},
                {"id": 3, "ssid": "OfficeAP",    "bssid": "AA:BB:CC:DD:EE:03", "lat": -27.47, "lng": 153.02, "signal": -55},
            ]
        elif category == "market_data":
            rows = [
                {"id": 1, "coin": "bitcoin",  "price_usd": 67000.0, "change_24h": 1.2,  "sentiment": 0.72},
                {"id": 2, "coin": "ethereum", "price_usd": 3500.0,  "change_24h": -0.5, "sentiment": 0.61},
                {"id": 3, "coin": "solana",   "price_usd": 175.0,   "change_24h": 3.1,  "sentiment": 0.80},
            ]
    return rows


# ---------------------------------------------------------------------------
# GraphQL-style entity → SQL mapping
# ---------------------------------------------------------------------------

ENTITY_MAP = {
    "leads":           ("leads",           "SELECT id,name,email,company,location,score FROM leads"),
    "campaigns":       ("campaigns",       "SELECT * FROM campaigns"),
    "nft_collections": ("nft_collections", "SELECT * FROM nft_collections"),
    "nft_tokens":      ("nft_tokens",      "SELECT * FROM nft_tokens"),
    "osint_reports":   ("osint_reports",   "SELECT * FROM osint_reports"),
}

DATASET_MAP = {
    "leads":         "SELECT id,name,email,company,location,score FROM leads",
    "wifi_networks": "SELECT * FROM osint_reports WHERE report_type='wifi'",
    "market_prices": "SELECT * FROM coingecko_queries",
    "nft_sales":     "SELECT * FROM nft_sales",
}

# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class DataAPIHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh-DataAPI/1.0"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, body, content_type: str = "application/json"):
        if isinstance(body, (dict, list)):
            data = json.dumps(body, default=str).encode()
        elif isinstance(body, str):
            data = body.encode()
        else:
            data = body  # bytes
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _admin_auth(self) -> bool:
        if not ADMIN_SECRET:
            return True
        return self.headers.get("X-Admin-Secret", "") == ADMIN_SECRET

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        t0 = time.time()

        if path == "/health":
            self._send(200, {"status": "ok", "service": "fm-data-api", "port": PORT})
            return

        if path == "/docs":
            self._handle_docs()
            return

        if path == "/admin/keys":
            self._handle_admin_list_keys()
            return

        if path.startswith("/admin/keys/"):
            # Only DELETE makes sense here; GET on individual key not specified
            self._send(405, {"error": "Method Not Allowed"})
            return

        if path == "/analytics":
            self._handle_analytics()
            return

        # Auth-required endpoints
        auth_row, err = _check_auth(self)
        if err:
            status_code = int(err.split()[0])
            self._send(status_code, {"error": err})
            return

        latency = (time.time() - t0) * 1000

        if path == "/products":
            self._handle_list_products(auth_row, latency)
            return

        if re.match(r"^/products/\d+$", path):
            product_id = int(path.split("/")[-1])
            self._handle_product_detail(auth_row, product_id, latency)
            return

        if path == "/usage":
            self._handle_usage(auth_row, latency)
            return

        self._send(404, {"error": "Not Found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        t0 = time.time()

        if path == "/admin/keys/create":
            self._handle_create_key()
            return

        if path == "/webhooks/stripe":
            self._handle_stripe_webhook()
            return

        if path == "/graphql":
            auth_row, err = _check_auth(self)
            if err:
                # Allow introspection without auth (returns schema only)
                body = self._read_body()
                query = body.get("query", "")
                if "__schema" in query or "__type" in query:
                    self._handle_graphql_introspection()
                    return
                status_code = int(err.split()[0])
                self._send(status_code, {"error": err})
                return
            self._handle_graphql(auth_row, t0)
            return

        # Auth-required POST endpoints
        auth_row, err = _check_auth(self)
        if err:
            status_code = int(err.split()[0])
            self._send(status_code, {"error": err})
            return

        if path == "/data/query":
            self._handle_data_query(auth_row, t0)
            return

        if path == "/data/export":
            self._handle_data_export(auth_row, t0)
            return

        self._send(404, {"error": "Not Found"})

    def do_DELETE(self):
        path = self.path.split("?")[0].rstrip("/")
        if re.match(r"^/admin/keys/[^/]+$", path):
            self._handle_delete_key(path.split("/")[-1])
            return
        self._send(404, {"error": "Not Found"})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "X-API-Key,X-Admin-Secret,Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ------------------------------------------------------------------
    # /health + /docs
    # ------------------------------------------------------------------

    def _handle_docs(self):
        docs = {
            "service": "fm-data-api",
            "version": "1.0",
            "port": PORT,
            "authentication": "Set X-API-Key header on all protected endpoints",
            "endpoints": [
                {
                    "method": "GET", "path": "/health",
                    "auth": False,
                    "description": "Service health check",
                    "response": {"status": "ok", "service": "fm-data-api", "port": PORT},
                },
                {
                    "method": "GET", "path": "/docs",
                    "auth": False,
                    "description": "This documentation",
                },
                {
                    "method": "POST", "path": "/admin/keys/create",
                    "auth": "X-Admin-Secret header",
                    "body": {"owner": "string", "email": "string", "tier": "free|starter|pro|enterprise"},
                    "response": {"api_key": "fmda_...", "tier": "string", "limit_per_day": "int"},
                },
                {
                    "method": "GET", "path": "/admin/keys",
                    "auth": "X-Admin-Secret header",
                    "description": "List all API keys with usage stats",
                },
                {
                    "method": "DELETE", "path": "/admin/keys/{key}",
                    "auth": "X-Admin-Secret header",
                    "description": "Delete an API key",
                },
                {
                    "method": "GET", "path": "/products",
                    "auth": "X-API-Key",
                    "description": "List all active data products",
                    "response": {"products": [{"id": "int", "name": "str", "category": "str", "price_usd": "float", "record_count": "int", "format": "str"}]},
                },
                {
                    "method": "GET", "path": "/products/{id}",
                    "auth": "X-API-Key",
                    "description": "Full product details with 3-row sample",
                },
                {
                    "method": "POST", "path": "/data/query",
                    "auth": "X-API-Key",
                    "body": {"dataset": "leads|wifi_networks|market_prices|nft_sales", "filters": {}, "limit": 100, "format": "json"},
                    "response": {"dataset": "str", "count": "int", "data": [], "query_id": "int"},
                },
                {
                    "method": "POST", "path": "/graphql",
                    "auth": "X-API-Key",
                    "body": {"query": "{ leads(limit:10, location:\"Albury\") { id name email company score } }"},
                    "description": "Simplified GraphQL-style queries",
                    "entities": list(ENTITY_MAP.keys()),
                },
                {
                    "method": "POST", "path": "/data/export",
                    "auth": "X-API-Key",
                    "body": {"product_id": "int", "format": "csv|json|jsonl"},
                    "description": "Export full product data in chosen format",
                },
                {
                    "method": "GET", "path": "/usage",
                    "auth": "X-API-Key",
                    "description": "Current key usage stats",
                    "response": {"tier": "str", "requests_today": "int", "limit_per_day": "int", "requests_total": "int"},
                },
                {
                    "method": "GET", "path": "/analytics",
                    "auth": "X-Admin-Secret header",
                    "description": "Platform analytics and revenue estimate",
                },
                {
                    "method": "POST", "path": "/webhooks/stripe",
                    "auth": "None (Stripe calls this)",
                    "description": "Stripe webhook for tier upgrades on checkout.session.completed",
                },
            ],
        }
        self._send(200, docs)

    # ------------------------------------------------------------------
    # Admin: key management
    # ------------------------------------------------------------------

    def _handle_create_key(self):
        if not self._admin_auth():
            self._send(401, {"error": "Invalid or missing X-Admin-Secret"})
            return
        body = self._read_body()
        owner = body.get("owner", "").strip()
        email = body.get("email", "").strip()
        tier  = body.get("tier", "free").strip().lower()
        if tier not in TIER_LIMITS:
            self._send(400, {"error": f"Unknown tier. Valid: {list(TIER_LIMITS.keys())}"})
            return
        if not owner:
            self._send(400, {"error": "owner is required"})
            return

        new_key  = f"fmda_{secrets.token_hex(24)}"
        limit    = TIER_LIMITS[tier]
        now      = time.time()
        conn     = _get_db()
        try:
            conn.execute(
                "INSERT INTO api_keys (key, owner, email, tier, requests_today, requests_total, "
                "limit_per_day, created_at, last_used) VALUES (?,?,?,?,0,0,?,?,?)",
                (new_key, owner, email, tier, limit, now, now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            self._send(500, {"error": "Key collision, try again"})
            return
        conn.close()
        self._send(201, {"api_key": new_key, "tier": tier, "limit_per_day": limit,
                         "owner": owner, "email": email})

    def _handle_admin_list_keys(self):
        if not self._admin_auth():
            self._send(401, {"error": "Invalid or missing X-Admin-Secret"})
            return
        conn = _get_db()
        rows = conn.execute(
            "SELECT id, key, owner, email, tier, requests_today, requests_total, "
            "limit_per_day, created_at, last_used FROM api_keys ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        self._send(200, {"keys": [dict(r) for r in rows], "total": len(rows)})

    def _handle_delete_key(self, key: str):
        if not self._admin_auth():
            self._send(401, {"error": "Invalid or missing X-Admin-Secret"})
            return
        conn = _get_db()
        result = conn.execute("DELETE FROM api_keys WHERE key=?", (key,))
        conn.commit()
        affected = result.rowcount
        conn.close()
        if affected == 0:
            self._send(404, {"error": "Key not found"})
        else:
            self._send(200, {"deleted": True, "key": key})

    # ------------------------------------------------------------------
    # Products
    # ------------------------------------------------------------------

    def _handle_list_products(self, auth_row, latency: float):
        conn = _get_db()
        rows = conn.execute(
            "SELECT id, name, category, price_usd, record_count, format, description "
            "FROM data_products WHERE status='active' ORDER BY id"
        ).fetchall()
        conn.close()
        products = [dict(r) for r in rows]
        _log_request(auth_row["key"], "/products", "GET", 200, latency)
        self._send(200, {"products": products})

    def _handle_product_detail(self, auth_row, product_id: int, latency: float):
        conn = _get_db()
        row = conn.execute(
            "SELECT * FROM data_products WHERE id=? AND status='active'", (product_id,)
        ).fetchone()
        conn.close()
        if not row:
            self._send(404, {"error": "Product not found"})
            return
        product = dict(row)
        product["sample"] = _sample_rows(product["category"], limit=3)
        _log_request(auth_row["key"], f"/products/{product_id}", "GET", 200, latency)
        self._send(200, product)

    # ------------------------------------------------------------------
    # /data/query
    # ------------------------------------------------------------------

    def _handle_data_query(self, auth_row, t0: float):
        body    = self._read_body()
        dataset = body.get("dataset", "").strip()
        filters = body.get("filters", {}) or {}
        limit   = min(int(body.get("limit", 100)), 10000)
        fmt     = body.get("format", "json")

        if dataset not in DATASET_MAP:
            self._send(400, {"error": f"Unknown dataset. Valid: {list(DATASET_MAP.keys())}"})
            return

        base_sql = DATASET_MAP[dataset]
        params   = []
        clauses  = []

        # Dataset-specific safe filter mapping
        if dataset == "leads":
            if "location" in filters:
                clauses.append("location LIKE ?")
                params.append(f"%{filters['location']}%")
            if "min_score" in filters:
                try:
                    clauses.append("score >= ?")
                    params.append(float(filters["min_score"]))
                except (ValueError, TypeError):
                    pass
            if "company" in filters:
                clauses.append("company LIKE ?")
                params.append(f"%{filters['company']}%")
        elif dataset == "wifi_networks":
            if "location" in filters:
                clauses.append("location LIKE ?")
                params.append(f"%{filters['location']}%")
            if "ssid" in filters:
                clauses.append("ssid LIKE ?")
                params.append(f"%{filters['ssid']}%")
        elif dataset == "market_prices":
            if "coin" in filters:
                clauses.append("coin LIKE ?")
                params.append(f"%{filters['coin']}%")
        elif dataset == "nft_sales":
            if "collection" in filters:
                clauses.append("collection LIKE ?")
                params.append(f"%{filters['collection']}%")

        if clauses:
            # For datasets that already have a WHERE clause we use AND
            if "WHERE" in base_sql.upper():
                sql = f"{base_sql} AND {' AND '.join(clauses)} LIMIT ?"
            else:
                sql = f"{base_sql} WHERE {' AND '.join(clauses)} LIMIT ?"
        else:
            sql = f"{base_sql} LIMIT ?"
        params.append(limit)

        latency = (time.time() - t0) * 1000
        conn    = _get_db()
        try:
            rows = conn.execute(sql, params).fetchall()
            data = [dict(r) for r in rows]
        except sqlite3.OperationalError as exc:
            conn.close()
            log.warning("data/query SQL error: %s", exc)
            data = []
        conn.close()

        # Log to api_requests and get the inserted id
        _log_request(auth_row["key"], "/data/query", "POST", 200, latency)
        conn2 = _get_db()
        qrow  = conn2.execute(
            "SELECT id FROM api_requests WHERE api_key=? ORDER BY id DESC LIMIT 1",
            (auth_row["key"],),
        ).fetchone()
        conn2.close()
        query_id = qrow["id"] if qrow else 0

        self._send(200, {"dataset": dataset, "count": len(data), "data": data, "query_id": query_id})

    # ------------------------------------------------------------------
    # /graphql
    # ------------------------------------------------------------------

    def _handle_graphql_introspection(self):
        schema = {
            "data": {
                "__schema": {
                    "types": [
                        {"name": entity, "fields": list(ENTITY_MAP.keys())}
                        for entity in ENTITY_MAP
                    ]
                }
            }
        }
        self._send(200, schema)

    def _handle_graphql(self, auth_row, t0: float):
        body  = self._read_body()
        query = body.get("query", "").strip()

        if not query:
            self._send(400, {"errors": [{"message": "query is required"}]})
            return

        # Introspection passthrough
        if "__schema" in query or "__type" in query:
            self._handle_graphql_introspection()
            return

        # Parse: { entity(arg:val, arg2:val2) { field1 field2 } }
        match = re.search(
            r"\{\s*(\w+)\s*(?:\(([^)]*)\))?\s*\{([^}]+)\}\s*\}",
            query,
            re.DOTALL,
        )
        if not match:
            self._send(400, {"errors": [{"message": "Cannot parse query. Expected: { entity(args) { fields } }"}]})
            return

        entity_name = match.group(1).strip()
        args_str    = (match.group(2) or "").strip()
        fields_str  = match.group(3).strip()

        if entity_name not in ENTITY_MAP:
            self._send(400, {"errors": [{"message": f"Unknown entity '{entity_name}'. Valid: {list(ENTITY_MAP.keys())}"}]})
            return

        _, base_sql = ENTITY_MAP[entity_name]

        # Parse args: key:value pairs
        args    = {}
        for arg_match in re.finditer(r'(\w+)\s*:\s*(?:"([^"]*)"|([\d.]+))', args_str):
            k, sv, nv = arg_match.group(1), arg_match.group(2), arg_match.group(3)
            args[k] = sv if sv is not None else (float(nv) if "." in (nv or "") else int(nv or 0))

        limit   = int(args.pop("limit", 100))
        limit   = min(limit, 10000)
        params  = []
        clauses = []

        # Build WHERE from remaining args
        for k, v in args.items():
            clauses.append(f"{k} LIKE ?")
            params.append(f"%{v}%")

        if clauses:
            if "WHERE" in base_sql.upper():
                sql = f"{base_sql} AND {' AND '.join(clauses)} LIMIT ?"
            else:
                sql = f"{base_sql} WHERE {' AND '.join(clauses)} LIMIT ?"
        else:
            sql = f"{base_sql} LIMIT ?"
        params.append(limit)

        # Select only requested fields if they look safe (alphanumeric)
        requested_fields = [f.strip() for f in fields_str.split() if re.match(r"^\w+$", f.strip())]

        latency = (time.time() - t0) * 1000
        conn    = _get_db()
        try:
            rows = conn.execute(sql, params).fetchall()
            all_data = [dict(r) for r in rows]
        except sqlite3.OperationalError as exc:
            conn.close()
            log.warning("graphql SQL error: %s", exc)
            all_data = []
        conn.close()

        # Filter to requested fields
        if requested_fields and all_data:
            available = set(all_data[0].keys())
            keep = [f for f in requested_fields if f in available]
            if keep:
                all_data = [{f: r.get(f) for f in keep} for r in all_data]

        _log_request(auth_row["key"], "/graphql", "POST", 200, latency)
        self._send(200, {"data": {entity_name: all_data}})

    # ------------------------------------------------------------------
    # /data/export
    # ------------------------------------------------------------------

    def _handle_data_export(self, auth_row, t0: float):
        body       = self._read_body()
        product_id = body.get("product_id")
        fmt        = body.get("format", "json").strip().lower()

        if product_id is None:
            self._send(400, {"error": "product_id is required"})
            return
        if fmt not in ("csv", "json", "jsonl"):
            self._send(400, {"error": "format must be csv, json, or jsonl"})
            return

        conn = _get_db()
        product = conn.execute(
            "SELECT * FROM data_products WHERE id=? AND status='active'", (int(product_id),)
        ).fetchone()
        conn.close()
        if not product:
            self._send(404, {"error": "Product not found"})
            return

        product = dict(product)
        # Pull data for the product category (larger sample for export)
        data = _sample_rows(product["category"], limit=1000)

        latency = (time.time() - t0) * 1000

        if fmt == "json":
            body_bytes = json.dumps(data, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Disposition", f'attachment; filename="export_{product_id}.json"')
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        elif fmt == "jsonl":
            lines = "\n".join(json.dumps(r, default=str) for r in data)
            body_bytes = lines.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Disposition", f'attachment; filename="export_{product_id}.jsonl"')
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        elif fmt == "csv":
            buf = io.StringIO()
            if data:
                writer = csv.DictWriter(buf, fieldnames=list(data[0].keys()))
                writer.writeheader()
                writer.writerows(data)
            csv_bytes = buf.getvalue().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", f'attachment; filename="export_{product_id}.csv"')
            self.send_header("Content-Length", str(len(csv_bytes)))
            self.end_headers()
            self.wfile.write(csv_bytes)

        # Log export
        try:
            conn2 = _get_db()
            conn2.execute(
                "INSERT INTO data_exports (api_key, product_id, format, records_exported, created_at) "
                "VALUES (?,?,?,?,?)",
                (auth_row["key"], int(product_id), fmt, len(data), time.time()),
            )
            conn2.commit()
            conn2.close()
        except Exception as exc:
            log.warning("export log error: %s", exc)

        _log_request(auth_row["key"], "/data/export", "POST", 200, latency)

    # ------------------------------------------------------------------
    # /usage
    # ------------------------------------------------------------------

    def _handle_usage(self, auth_row, latency: float):
        _log_request(auth_row["key"], "/usage", "GET", 200, latency)
        self._send(200, {
            "tier":            auth_row["tier"],
            "requests_today":  auth_row["requests_today"],
            "limit_per_day":   auth_row["limit_per_day"],
            "requests_total":  auth_row["requests_total"],
            "key_prefix":      auth_row["key"][:12] + "...",
        })

    # ------------------------------------------------------------------
    # /analytics (admin)
    # ------------------------------------------------------------------

    def _handle_analytics(self):
        if not self._admin_auth():
            self._send(401, {"error": "Invalid or missing X-Admin-Secret"})
            return

        conn = _get_db()

        # Keys by tier
        tier_rows = conn.execute(
            "SELECT tier, COUNT(*) as count FROM api_keys GROUP BY tier"
        ).fetchall()
        keys_by_tier = {r["tier"]: r["count"] for r in tier_rows}

        # Total requests today
        today_start = time.mktime(date.today().timetuple())
        requests_today_total = conn.execute(
            "SELECT COALESCE(SUM(requests_today),0) FROM api_keys"
        ).fetchone()[0]
        requests_all_time = conn.execute(
            "SELECT COALESCE(SUM(requests_total),0) FROM api_keys"
        ).fetchone()[0]

        # Top endpoints
        top_endpoints = conn.execute(
            "SELECT endpoint, COUNT(*) as hits FROM api_requests "
            "GROUP BY endpoint ORDER BY hits DESC LIMIT 10"
        ).fetchall()

        # Revenue estimate: pro keys * $99/mo + starter * $19/mo + enterprise * $499/mo
        pro_count        = keys_by_tier.get("pro", 0)
        starter_count    = keys_by_tier.get("starter", 0)
        enterprise_count = keys_by_tier.get("enterprise", 0)
        revenue_estimate = (pro_count * 99.0) + (starter_count * 19.0) + (enterprise_count * 499.0)

        conn.close()
        self._send(200, {
            "keys_by_tier":        keys_by_tier,
            "total_keys":          sum(keys_by_tier.values()),
            "requests_today":      requests_today_total,
            "requests_all_time":   requests_all_time,
            "top_endpoints":       [dict(r) for r in top_endpoints],
            "revenue_estimate_mo": revenue_estimate,
            "pricing": {"starter": 19, "pro": 99, "enterprise": 499},
        })

    # ------------------------------------------------------------------
    # /webhooks/stripe
    # ------------------------------------------------------------------

    def _handle_stripe_webhook(self):
        body    = self._read_body()
        ev_type = body.get("type", "")
        log.info("Stripe webhook: %s", ev_type)

        if ev_type == "checkout.session.completed":
            session  = body.get("data", {}).get("object", {})
            metadata = session.get("metadata", {})
            api_key  = metadata.get("api_key", "").strip()
            new_tier = metadata.get("tier", "").strip().lower()

            if api_key and new_tier in TIER_LIMITS:
                new_limit = TIER_LIMITS[new_tier]
                conn = _get_db()
                conn.execute(
                    "UPDATE api_keys SET tier=?, limit_per_day=? WHERE key=?",
                    (new_tier, new_limit, api_key),
                )
                conn.commit()
                rows_affected = conn.execute(
                    "SELECT changes()"
                ).fetchone()[0]
                conn.close()
                log.info(
                    "Stripe upgrade: key=%s tier=%s limit=%s rows_affected=%s",
                    api_key[:12] + "...", new_tier, new_limit, rows_affected,
                )
            else:
                log.warning("Stripe webhook missing api_key or unknown tier: %s", metadata)

        self._send(200, {"received": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _db_init()
    log.info("FractalMesh Data API starting on port %d (DB: %s)", PORT, DB)
    server = HTTPServer(("0.0.0.0", PORT), DataAPIHandler)
    log.info("Data API listening on http://0.0.0.0:%d", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutdown requested")
    finally:
        server.server_close()
        log.info("Data API stopped")


if __name__ == "__main__":
    main()

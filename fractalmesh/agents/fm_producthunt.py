#!/usr/bin/env python3
"""
fm_producthunt.py — Product Hunt GraphQL API Agent (Port 7822)
Full Product Hunt v2 GraphQL operations: posts, votes, comments, collections, notifications.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import json
import logging
import os
import signal
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT                   = int(os.getenv("PRODUCTHUNT_PORT", "7822"))
ROOT                   = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB                     = ROOT / "database" / "sovereign.db"
LOG                    = ROOT / "logs" / "producthunt.log"
PRODUCTHUNT_API_KEY    = os.getenv("PRODUCTHUNT_API_KEY", "")
PRODUCTHUNT_API_SECRET = os.getenv("PRODUCTHUNT_API_SECRET", "")
PRODUCTHUNT_DEV_TOKEN  = os.getenv("PRODUCTHUNT_DEV_TOKEN", "")
PH_GQL                 = "https://api.producthunt.com/v2/api/graphql"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PRODUCTHUNT] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("producthunt")

# ── database ──────────────────────────────────────────────────────────────────

def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ph_interactions (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT,
            action  TEXT,
            result  TEXT,
            ts      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def _log_interaction(post_id: str, action: str, result: Any):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO ph_interactions (post_id, action, result) VALUES (?, ?, ?)",
            (str(post_id), action, json.dumps(result)[:4096]),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("db_log error: %s", e)

def _recent_interactions(limit: int = 50) -> list:
    try:
        conn = sqlite3.connect(DB, timeout=5)
        rows = conn.execute(
            "SELECT id, post_id, action, result, ts FROM ph_interactions "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [
            {"id": r[0], "post_id": r[1], "action": r[2],
             "result": json.loads(r[3]) if r[3] else {}, "ts": r[4]}
            for r in rows
        ]
    except Exception:
        return []

# ── Product Hunt GraphQL helper ───────────────────────────────────────────────

def _ph_gql(query: str, variables: Optional[dict] = None) -> dict:
    """POST a GraphQL query/mutation to the Product Hunt v2 API."""
    if not PRODUCTHUNT_DEV_TOKEN:
        return {"error": "PRODUCTHUNT_DEV_TOKEN not configured"}

    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    headers = {
        "Authorization":  f"Bearer {PRODUCTHUNT_DEV_TOKEN}",
        "Content-Type":   "application/json",
        "Accept":         "application/json",
        "User-Agent":     "FractalMesh-ProductHunt/1.0",
    }
    req = urllib.request.Request(PH_GQL, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            if "errors" in data:
                log.warning("graphql errors: %s", data["errors"])
            return data
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        log.warning("ph gql HTTP %d: %s", e.code, detail[:120])
        return {"error": f"http_{e.code}", "detail": detail}
    except Exception as e:
        log.error("ph gql error: %s", e)
        return {"error": str(e)}

# ── query string parser ───────────────────────────────────────────────────────

def _qs(path: str) -> dict:
    if "?" not in path:
        return {}
    return dict(urllib.parse.parse_qsl(path.split("?", 1)[1]))

# ── GraphQL query/mutation definitions ───────────────────────────────────────

_Q_HEALTH = """
query {
  posts(first: 1, order: VOTES) {
    edges { node { id name } }
  }
}
"""

_Q_POSTS = """
query GetPosts($first: Int!) {
  posts(first: $first, order: VOTES) {
    edges {
      node {
        id
        name
        tagline
        votesCount
        commentsCount
        url
        thumbnail { url }
      }
    }
  }
}
"""

_Q_POST = """
query GetPost($id: ID!) {
  post(id: $id) {
    id
    name
    description
    votesCount
    url
    makers { name username }
  }
}
"""

_Q_TRENDING = """
query {
  posts(first: 20, order: VOTES, postedAfter: "%s") {
    edges {
      node {
        id
        name
        tagline
        votesCount
        url
        thumbnail { url }
      }
    }
  }
}
"""

_M_VOTE = """
mutation VotePost($postId: ID!) {
  voteForPost(input: { postId: $postId }) {
    post { id votesCount }
  }
}
"""

_Q_MY_POSTS = """
query {
  viewer {
    user {
      name
      posts(first: 20) {
        edges {
          node {
            id
            name
            tagline
            votesCount
            url
          }
        }
      }
    }
  }
}
"""

_Q_COLLECTIONS = """
query {
  collections(first: 20) {
    edges {
      node {
        id
        name
        postsCount
      }
    }
  }
}
"""

_M_COMMENT = """
mutation CreateComment($body: String!, $postId: ID!) {
  createComment(input: { body: $body, postId: $postId }) {
    comment { id body }
  }
}
"""

_Q_NOTIFICATIONS = """
query {
  viewer {
    notifications(first: 20) {
      edges {
        node {
          id
          ... on VoteNotification { post { id name } }
          ... on CommentNotification { comment { id body } }
        }
      }
    }
  }
}
"""

# ── HTTP handler ──────────────────────────────────────────────────────────────

class ProductHuntHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def _respond(self, code: int, body: Any):
        payload = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _base_path(self) -> str:
        return self.path.split("?")[0]

    # ── GET ───────────────────────────────────────────────────────────────────

    def do_GET(self):
        base = self._base_path()
        qs   = _qs(self.path)

        try:
            if base == "/health":
                result = _ph_gql(_Q_HEALTH)
                ok = "data" in result and "errors" not in result
                self._respond(200, {
                    "status":     "ok" if ok else "error",
                    "configured": bool(PRODUCTHUNT_DEV_TOKEN),
                    "ph_response": result,
                })

            elif base == "/posts":
                first  = int(qs.get("first", 20))
                result = _ph_gql(_Q_POSTS, {"first": first})
                edges  = result.get("data", {}).get("posts", {}).get("edges", [])
                posts  = [e["node"] for e in edges if "node" in e]
                _log_interaction("", "list_posts", {"count": len(posts)})
                self._respond(200, {"count": len(posts), "posts": posts})

            elif base == "/post":
                post_id = qs.get("id", "")
                if not post_id:
                    self._respond(400, {"error": "id param required"})
                    return
                result = _ph_gql(_Q_POST, {"id": post_id})
                post   = result.get("data", {}).get("post")
                _log_interaction(post_id, "get_post", post or result)
                self._respond(200 if post else 404, post or result)

            elif base == "/trending":
                # today's date in ISO format for postedAfter filter
                today  = time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime())
                result = _ph_gql(_Q_TRENDING % today)
                edges  = result.get("data", {}).get("posts", {}).get("edges", [])
                posts  = [e["node"] for e in edges if "node" in e]
                _log_interaction("", "trending", {"count": len(posts)})
                self._respond(200, {"date": today, "count": len(posts), "posts": posts})

            elif base == "/my_posts":
                result = _ph_gql(_Q_MY_POSTS)
                viewer = result.get("data", {}).get("viewer", {})
                user   = viewer.get("user", {})
                edges  = user.get("posts", {}).get("edges", []) if user else []
                posts  = [e["node"] for e in edges if "node" in e]
                _log_interaction("", "my_posts", {"count": len(posts)})
                self._respond(200, {
                    "user":  user.get("name", "unknown"),
                    "count": len(posts),
                    "posts": posts,
                })

            elif base == "/collections":
                result     = _ph_gql(_Q_COLLECTIONS)
                edges      = result.get("data", {}).get("collections", {}).get("edges", [])
                collections = [e["node"] for e in edges if "node" in e]
                _log_interaction("", "list_collections", {"count": len(collections)})
                self._respond(200, {"count": len(collections), "collections": collections})

            elif base == "/notifications":
                result  = _ph_gql(_Q_NOTIFICATIONS)
                viewer  = result.get("data", {}).get("viewer", {})
                edges   = viewer.get("notifications", {}).get("edges", []) if viewer else []
                notifs  = [e["node"] for e in edges if "node" in e]
                _log_interaction("", "notifications", {"count": len(notifs)})
                self._respond(200, {"count": len(notifs), "notifications": notifs})

            elif base == "/interactions":
                rows = _recent_interactions()
                self._respond(200, {"count": len(rows), "interactions": rows})

            else:
                self._respond(404, {"error": "not_found", "path": base})

        except Exception as e:
            log.error("GET %s error: %s", base, e)
            self._respond(500, {"error": str(e)})

    # ── POST ──────────────────────────────────────────────────────────────────

    def do_POST(self):
        base = self._base_path()
        try:
            data = self._read_body()
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid_json"})
            return
        except Exception as e:
            self._respond(400, {"error": str(e)})
            return

        try:
            if base == "/vote":
                self._handle_vote(data)
            elif base == "/comment":
                self._handle_comment(data)
            else:
                self._respond(404, {"error": "unknown_path", "path": base})
        except Exception as e:
            log.error("POST %s error: %s", base, e)
            self._respond(500, {"error": str(e)})

    def _handle_vote(self, data: dict):
        post_id = str(data.get("postId") or data.get("post_id", ""))
        if not post_id:
            self._respond(400, {"error": "postId required"})
            return
        result    = _ph_gql(_M_VOTE, {"postId": post_id})
        vote_data = result.get("data", {}).get("voteForPost")
        _log_interaction(post_id, "vote", vote_data or result)
        self._respond(200 if vote_data else 400, vote_data or result)

    def _handle_comment(self, data: dict):
        body    = data.get("body", "")
        post_id = str(data.get("postId") or data.get("post_id", ""))
        if not body or not post_id:
            self._respond(400, {"error": "body and postId required"})
            return
        result       = _ph_gql(_M_COMMENT, {"body": body, "postId": post_id})
        comment_data = result.get("data", {}).get("createComment")
        _log_interaction(post_id, "comment", comment_data or result)
        self._respond(201 if comment_data else 400, comment_data or result)

# ── shutdown ───────────────────────────────────────────────────────────────────

_running = True

def _shutdown(*_):
    global _running
    log.info("shutdown signal — exiting cleanly")
    _running = False

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    _db_init()
    server = HTTPServer(("0.0.0.0", PORT), ProductHuntHandler)
    log.info("Product Hunt agent listening on port %d", PORT)
    log.info("Dev token: %s | API key: %s",
             "configured" if PRODUCTHUNT_DEV_TOKEN else "NOT SET",
             "configured" if PRODUCTHUNT_API_KEY else "NOT SET")
    try:
        while _running:
            server.handle_request()
    finally:
        server.server_close()
        log.info("Product Hunt agent stopped")

if __name__ == "__main__":
    main()

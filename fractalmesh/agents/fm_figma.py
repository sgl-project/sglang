#!/usr/bin/env python3
"""
FractalMesh Figma — TerraMesh Design Sync
Samuel James Hiotis | ABN 56628117363 | Albury NSW
Integrates: Figma REST API, exports UI components to www/,
            syncs TerraMesh design tokens, publishes previews.
Runs every 30 min. Zero-capital: design assets → product demos → sales.
"""
import os, json, time, logging, urllib.request, urllib.parse, hashlib
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [FIGMA] %(message)s")
log = logging.getLogger("figma")

ROOT    = os.environ.get("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))
VAULT   = os.path.join(ROOT, ".env")
WWW_DIR = os.path.join(ROOT, "www")
DB_PATH = os.path.join(ROOT, "db", "sovereign.db")

def load_env(key, default=""):
    for f in [VAULT, str(Path.home() / ".env"), str(Path.home() / ".secrets/fractal.env")]:
        try:
            for line in Path(f).read_text().splitlines():
                s = line.strip()
                if s.startswith(key + "=") and not s.startswith("#"):
                    val = s.split("=",1)[1].strip().strip('"').strip("'")
                    if val and not val.startswith("YOUR_"):
                        return val
        except Exception:
            pass
    return os.environ.get(key, default)

def figma_get(endpoint):
    """Call Figma REST API."""
    token = load_env("FIGMA_TOKEN")
    if not token:
        raise ValueError("FIGMA_TOKEN not configured")
    req = urllib.request.Request(
        f"https://api.figma.com/v1/{endpoint}",
        headers={"X-Figma-Token": token, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=12) as r:
        return json.loads(r.read())

def get_file_components(file_key):
    """Fetch all components from a Figma file."""
    data = figma_get(f"files/{file_key}/components")
    return data.get("meta",{}).get("components",[])

def export_component_png(file_key, node_id, scale=2):
    """Get PNG export URL for a component."""
    params = urllib.parse.urlencode({"ids":node_id,"format":"png","scale":scale})
    data   = figma_get(f"images/{file_key}?{params}")
    return data.get("images",{}).get(node_id,"")

def get_team_projects(team_id):
    """List projects in a team."""
    return figma_get(f"teams/{team_id}/projects").get("projects",[])

def sync_design_tokens():
    """Pull design tokens (colors, typography) from Figma variables (beta API)."""
    file_key = load_env("FIGMA_FILE_KEY")
    if not file_key:
        log.info("No FIGMA_FILE_KEY — generating FractalMesh default tokens")
        tokens = {
            "colors": {
                "primary": "#00d2ff",
                "secondary": "#7a2ff7",
                "accent": "#ff6b35",
                "bg": "#0a0a1a",
                "surface": "#141428",
                "text": "#e0e0ff",
            },
            "typography": {
                "fontFamily": "Space Grotesk, monospace",
                "scale": [12, 14, 16, 20, 24, 32, 48],
            },
            "spacing": [4, 8, 12, 16, 24, 32, 48, 64],
            "generated_at": datetime.now().isoformat(),
            "source": "FractalMesh default (no Figma key)",
        }
        os.makedirs(WWW_DIR, exist_ok=True)
        Path(os.path.join(WWW_DIR, "design-tokens.json")).write_text(json.dumps(tokens, indent=2))
        log.info("Default design tokens written to www/design-tokens.json")
        return tokens
    try:
        data   = figma_get(f"files/{file_key}/variables/local")
        tokens = {"source": f"Figma file {file_key}", "generated_at": datetime.now().isoformat()}
        variables = data.get("meta",{}).get("variables",{})
        tokens["variables_count"] = len(variables)
        tokens["variables"]       = {k: v.get("name","") for k,v in list(variables.items())[:20]}
        Path(os.path.join(WWW_DIR, "design-tokens.json")).write_text(json.dumps(tokens, indent=2))
        log.info("Design tokens synced from Figma: %d variables", len(variables))
        return tokens
    except Exception as e:
        log.warning("Figma variables sync: %s", e)
        return {}

def generate_terramesh_preview():
    """Generate a TerraMesh component preview HTML for the dashboard."""
    tokens_path = os.path.join(WWW_DIR, "design-tokens.json")
    try:
        tokens = json.loads(Path(tokens_path).read_text())
        colors = tokens.get("colors", {})
        primary   = colors.get("primary","#00d2ff")
        secondary = colors.get("secondary","#7a2ff7")
        bg        = colors.get("bg","#0a0a1a")
        surface   = colors.get("surface","#141428")
        text      = colors.get("text","#e0e0ff")
    except Exception:
        primary,secondary,bg,surface,text = "#00d2ff","#7a2ff7","#0a0a1a","#141428","#e0e0ff"

    html = f"""<!DOCTYPE html>
<html><head><title>TerraMesh Components</title>
<style>
  body {{ background:{bg}; color:{text}; font-family:'Space Grotesk',monospace; padding:24px; }}
  h1 {{ color:{primary}; }}
  .card {{ background:{surface}; border:1px solid {primary}33; border-radius:8px; padding:16px; margin:8px 0; }}
  .btn {{ background:linear-gradient(135deg,{primary},{secondary}); color:#fff; border:none;
          padding:10px 20px; border-radius:6px; cursor:pointer; font-weight:600; }}
  .badge {{ display:inline-block; background:{secondary}; color:#fff; padding:2px 8px;
             border-radius:12px; font-size:11px; }}
  .signal {{ color:{primary}; font-weight:700; }}
</style></head><body>
<h1>⚡ TerraMesh Design System</h1>
<p>Generated by FractalMesh Figma agent — {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<div class="card">
  <h3>Signal Card <span class="badge">LIVE</span></h3>
  <div class="signal">SOL/USDT — BUY 95</div>
  <p>Confidence: 91% | Fractal score: 95</p>
  <button class="btn">Subscribe $35/mo →</button>
</div>
<div class="card">
  <h3>Revenue Card</h3>
  <div class="signal">MRR $217/mo | ARR $2,604/yr</div>
  <button class="btn">View Analytics →</button>
</div>
<div class="card">
  <h3>NFT Card <span class="badge">NEW MINT</span></h3>
  <div class="signal">FM-SYNTHWAVE-{hashlib.md5(datetime.now().isoformat().encode()).hexdigest()[:6].upper()}</div>
  <p>Solana mainnet | 7% royalty | $0.50 SOL</p>
  <button class="btn">Collect Now →</button>
</div>
</body></html>"""
    preview_path = os.path.join(WWW_DIR, "terramesh-preview.html")
    Path(preview_path).write_text(html)
    log.info("TerraMesh preview generated: %s", preview_path)
    return preview_path

def run_sync_cycle():
    tokens = sync_design_tokens()
    generate_terramesh_preview()
    figma_key = load_env("FIGMA_TOKEN")
    if figma_key:
        try:
            team_id = load_env("FIGMA_TEAM_ID")
            if team_id:
                projects = get_team_projects(team_id)
                log.info("Figma team projects: %d", len(projects))
        except Exception as e:
            log.warning("Figma team sync: %s", e)
    else:
        log.info("No FIGMA_TOKEN — running in design-token-only mode")
    log.info("Figma sync cycle complete")

def main():
    log.info("fm-figma started | WWW=%s", WWW_DIR)
    while True:
        try:
            run_sync_cycle()
        except Exception as e:
            log.error("Cycle error: %s", e)
        time.sleep(1800)  # 30 min

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

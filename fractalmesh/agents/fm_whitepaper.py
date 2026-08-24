#!/usr/bin/env python3
"""
FractalMesh Whitepaper Publisher
Markdown → PDF + Zenodo DOI + Dev.to
Runs every 10 min via PM2 cron_restart
Samuel James Hiotis | ABN 56628117363 | Albury NSW
"""
import os, json, requests
from pathlib import Path
from datetime import datetime

ROOT           = os.environ.get("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))
WHITEPAPER_DIR = os.path.join(ROOT, "whitepapers")
os.makedirs(WHITEPAPER_DIR, exist_ok=True)

def load_env(key, default=""):
    for f in [os.path.join(ROOT, ".env"), str(Path.home() / ".env")]:
        try:
            for line in Path(f).read_text().splitlines():
                s = line.strip()
                if s.startswith(key + "=") and not s.startswith("#"):
                    val = s.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and not val.startswith("YOUR_"):
                        return val
        except Exception:
            pass
    return os.environ.get(key, default)

def save_pdf(html_content: str, path: str) -> bool:
    try:
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(path)
        return True
    except ImportError:
        return False

def publish_whitepaper(markdown_content: str, title: str):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path  = os.path.join(WHITEPAPER_DIR, f"{title.replace(' ','_')}_{ts}.md")
    pdf_path = md_path.replace(".md", ".pdf")
    Path(md_path).write_text(markdown_content)
    print(f"[✓] Saved: {md_path}")

    # PDF
    try:
        import markdown2
        html = f"""<html><body style="font-family:Arial;margin:40px;line-height:1.6">
            <h1 style="color:#00d2ff">{title}</h1>
            {markdown2.markdown(markdown_content, extras=['tables','fenced-code-blocks','footnotes'])}
        </body></html>"""
        if save_pdf(html, pdf_path):
            print(f"[✓] PDF: {pdf_path}")
        else:
            print(f"[!] weasyprint not available — markdown saved, install weasyprint for PDF")
    except ImportError:
        print("[!] markdown2 not installed — skipping PDF")

    # Zenodo DOI
    zenodo_token = load_env("ZENODO_TOKEN")
    if zenodo_token:
        meta = {
            "metadata": {
                "title":            title,
                "upload_type":      "publication",
                "publication_type": "article",
                "description":      "FractalMesh Sovereign IP — Edge RL Trading + Sovereign Enclave + Retention Architecture",
                "creators":         [{"name": "Samuel James Hiotis", "affiliation": "Sole Trader ABN 56 628 117 363"}],
                "license":          "CC-BY-4.0",
                "keywords":         ["AI","reinforcement-learning","solana","termux","nft","trading"],
            }
        }
        headers = {"Authorization": f"Bearer {zenodo_token}"}
        try:
            r  = requests.post("https://zenodo.org/api/deposit/depositions", json=meta, headers=headers, timeout=15)
            did = r.json()["id"]
            src = pdf_path if Path(pdf_path).exists() else md_path
            with open(src, "rb") as fh:
                requests.post(f"https://zenodo.org/api/deposit/depositions/{did}/files",
                              files={"file": fh}, headers=headers, timeout=30)
            pub = requests.post(f"https://zenodo.org/api/deposit/depositions/{did}/actions/publish",
                                headers=headers, timeout=15)
            doi = pub.json().get("doi", "pending")
            print(f"[✓] Zenodo DOI: https://doi.org/{doi}")
        except Exception as e:
            print(f"[!] Zenodo failed: {e}")
    else:
        print("[i] ZENODO_TOKEN not set — skipping DOI")

    # Dev.to
    devto_key = load_env("DEVTO_API_KEY")
    if devto_key:
        footer = (
            "\n\n---\n"
            "**Live arbitrage feed $499 AUD** | **Synthwave NFT mint every 10 min** | **Dashboard :8090**\n\n"
            "*Samuel James Hiotis | Sole Trader | ABN 56 628 117 363 | Albury NSW 2640*"
        )
        payload = {
            "article": {
                "title":         title,
                "body_markdown": markdown_content + footer,
                "published":     True,
                "tags":          ["ai","rl","trading","nft","termux"],
            }
        }
        try:
            r = requests.post("https://dev.to/api/articles",
                              headers={"api-key": devto_key, "Content-Type": "application/json"},
                              json=payload, timeout=15)
            url = r.json().get("url", "check dev.to")
            print(f"[✓] Dev.to: {url}")
        except Exception as e:
            print(f"[!] Dev.to failed: {e}")
    else:
        print("[i] DEVTO_API_KEY not set — skipping Dev.to")

    return pdf_path

# ─── default whitepaper content ───────────────────────────────────────────────
DEFAULT_MD = """# FractalMesh Sovereign IP Layer v3.0
## Edge RL Trading + Sovereign Enclave + Retention Architecture

### Abstract
FractalMesh implements a quantized actor-critic reinforcement learning system
deployed at the edge, enabling sub-millisecond trading decisions across
BTC/ETH/SOL/XRP/BNB pairs with fractal signal scoring.

### 1. Edge-Optimized RL

The quantized actor-critic:

    π̂(a|s) = Quantize₄(π_θ(a|s))

4-bit quantization reduces inference latency to <1ms on ARM64 (Termux/proot).

### 2. Sovereign Enclave Architecture

Split-brain RL with isolated execution environments:
- Phone-side: lightweight policy inference (1B–3B LLM)
- Cloud-side: full orchestration via PM2 ecosystem

### 3. Markov Customer Journey (LTV Model)

Transition matrix **P** ∈ ℝ⁶ˣ⁶ models customer states:
Prospect → Trial → Active → Upsell → Champion → Churned

### 4. Trust Dynamics

dT/dt = α·S(t) - β·C(t)

Where S = satisfaction signal, C = churn risk indicator.

### 5. Enochian Gate (Retention Engine)

Multi-armed bandit controlling offer timing. UCB1 exploration:

    a* = argmax[Q(a) + c·√(ln(N)/n(a))]

### Live System
- Dashboard: http://localhost:8090
- NFT mint cycle: every 10 minutes (Solana mainnet)
- Signal feed: 5 pairs, RL-scored in real time

---
**Author:** Samuel James Hiotis | Sole Trader | ABN 56 628 117 363 | Albury NSW 2640
**Ready for implementation.**
"""

if __name__ == "__main__":
    wp_path = os.path.join(WHITEPAPER_DIR, "v3.md")
    if Path(wp_path).exists():
        content = Path(wp_path).read_text()
        print("[i] Using existing v3.md")
    else:
        content = DEFAULT_MD
        Path(wp_path).write_text(content)
        print("[i] Created default v3.md")

    publish_whitepaper(content, "FractalMesh Sovereign IP Layer v3.0")

# Vendor card logo (new brand only)

A new vendor/brand in the cookbook landing grid needs a card logo at
`docs/cards/logos/<org-slug>.png`. **Ask the user for the brand's logo, then generate
the conforming PNG** — never invent, copy, or hallucinate one, and never ship a
non-conforming file. Reference: PR #27400 (added `tencent.png` + `poolside.png`).

If the org already has a card/logo, do nothing here — only update the `<Card href>`.

## Spec (match the existing logos exactly)

| Property | Value |
|---|---|
| Path | `docs/cards/logos/<org-slug>.png` — lowercase, matches the `img=` in the `<Card>` |
| Canvas | **940 × 525** px |
| Mode | **RGBA**, fully **transparent** background |
| Content | **Icon-only** — the brand glyph/mark (the "swirl"), **no wordmark text** |
| Placement | centered; glyph ≈ 0.33 × width and ≈ 0.50 × height of the canvas |

Why icon-only + transparent: cards render on both light and dark backgrounds, so a baked-in
(usually black) wordmark vanishes on dark. `deepseek.png` / `ernie.png` are 940×525 RGBA
exemplars — eyeball your output against them.

## 1. Get the source

Ask the user for the brand logo (SVG preferred → crisp + already transparent; else a high-res
transparent PNG, or a link to the official press/brand asset). Prefer an **icon-only** source;
if they only have a full lockup, ask them to crop the glyph, or crop it yourself.

If the user **pasted** an image inline, it may not be on disk — recover the base64 `image`
block from the session transcript (`~/.claude/projects/<slug>/*.jsonl`) and decode it to a file.

## 2. Generate (Pillow)

There's no system Pillow — use a venv:

```bash
python3 -m venv /tmp/logo-venv && /tmp/logo-venv/bin/pip install -q Pillow
```

```python
from PIL import Image
src = Image.open("SOURCE").convert("RGBA")        # icon-only, already transparent
W, H = 940, 525
target_h = round(H * 0.50)                         # glyph ≈ half the canvas height
scale = target_h / src.height
glyph = src.resize((round(src.width * scale), target_h), Image.LANCZOS)
canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))   # transparent
canvas.paste(glyph, ((W - glyph.width) // 2, (H - glyph.height) // 2), glyph)
canvas.save("docs/cards/logos/<org-slug>.png")
```

Notes:
- **Wordmark present?** Crop to the glyph first (or ask the user for an icon-only asset). Don't
  ship text in the logo.
- **Solid background?** Don't auto-strip it (risky) — ask the user for a transparent source.
- **SVG source?** Rasterize at high res first (`cairosvg` / `rsvg-convert`), then run the above.
- If the glyph is much wider than tall, cap by width instead (≈ 0.33 × W) so it doesn't overflow.

## 3. Verify

```bash
sips -g pixelWidth -g pixelHeight -g hasAlpha docs/cards/logos/<org-slug>.png
# → pixelWidth: 940   pixelHeight: 525   hasAlpha: yes
```

## 4. Wire + track + validate

```bash
# Card in the landing grid (keep card order aligned with the docs.json nav order):
#   <Card title="<NavGroup>" mode="card"
#         href="/cookbook/<category>/<Vendor>/<Model>"
#         img="/cards/logos/<org-slug>.png" />
git add -f docs/cards/logos/<org-slug>.png      # root .gitignore ignores *.png repo-wide
cd docs && mint validate && mint broken-links    # confirms the card href + img resolve
```

#!/usr/bin/env python3
"""Sync SGLang-related LMSYS blog cards into index.mdx."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "index.mdx"
OUTPUT_JSON_PATH = ROOT / "src" / "generated" / "lmsys_sglang_blogs.json"

START_MARKER = "{/* BEGIN_LMSYS_SGLANG_BLOG_CARDS */}"
END_MARKER = "{/* END_LMSYS_SGLANG_BLOG_CARDS */}"

LMSYS_BLOG_API_URL = "https://api.github.com/repos/lm-sys/lm-sys.github.io/contents/blog"
LMSYS_BLOG_BASE_URL = "https://lmsys.org/blog"
LMSYS_BASE_URL = "https://lmsys.org"
DEFAULT_IMAGE_URL = "https://lmsys.org/social.png"

MAX_CARDS = int(os.getenv("LMSYS_SGLANG_MAX_CARDS", "6"))
KEYWORDS = [
    "sglang",
    "sgl-project/sglang",
    "sgl-kernel",
    "sglang-jax",
    "sgl diffusion",
    "sglang diffusion"
]

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", flags=re.DOTALL)
HTML_IMG_RE = re.compile(r"<img[^>]*\ssrc=[\"']([^\"']+)[\"']", flags=re.IGNORECASE)
MD_IMG_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")


@dataclass
class BlogPost:
    slug: str
    title: str
    url: str
    image: str
    date: str


def build_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "sgl-docs-lmsys-blog-sync",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def download_blog_sources() -> list[tuple[str, str]]:
    # Fetch the directory listing for /blog only — no need to download the whole repo.
    request = urllib.request.Request(LMSYS_BLOG_API_URL, headers=build_headers())
    with urllib.request.urlopen(request, timeout=60) as response:
        items: list[dict] = json.loads(response.read())

    sources: list[tuple[str, str]] = []
    for item in items:
        if item.get("type") != "file" or not item.get("name", "").endswith(".md"):
            continue
        download_url = item.get("download_url")
        if not download_url:
            continue
        raw_request = urllib.request.Request(download_url, headers=build_headers())
        with urllib.request.urlopen(raw_request, timeout=30) as raw_response:
            content = raw_response.read().decode("utf-8", errors="replace")
        sources.append((item["name"], content))

    return sources


def split_frontmatter(content: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(content)
    if not match:
        return {}, content

    frontmatter: dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)
        cleaned = value.strip()
        if (
            (cleaned.startswith('"') and cleaned.endswith('"'))
            or (cleaned.startswith("'") and cleaned.endswith("'"))
        ) and len(cleaned) >= 2:
            cleaned = cleaned[1:-1]
        frontmatter[key.strip()] = cleaned

    return frontmatter, content[match.end() :]


def first_image_from_body(body: str) -> str | None:
    markdown_match = MD_IMG_RE.search(body)
    if markdown_match:
        candidate = markdown_match.group(1).strip()
        if candidate.startswith("<") and candidate.endswith(">"):
            candidate = candidate[1:-1]
        if " " in candidate:
            candidate = candidate.split(" ", 1)[0]
        return candidate

    html_match = HTML_IMG_RE.search(body)
    if html_match:
        return html_match.group(1).strip()

    return None


def to_absolute_url(url_or_path: str | None) -> str:
    if not url_or_path:
        return DEFAULT_IMAGE_URL

    value = url_or_path.strip()
    if value.startswith(("http://", "https://")):
        return value
    if value.startswith("//"):
        return f"https:{value}"
    return f"{LMSYS_BASE_URL}/{value.lstrip('/')}"


def is_relevant(slug: str, title: str, body: str) -> bool:
    searchable = f"{slug}\n{title}\n{body}".lower()
    return any(keyword in searchable for keyword in KEYWORDS)


def parse_blog_post(filename: str, content: str) -> BlogPost | None:
    if not filename.endswith(".md"):
        return None

    slug = filename[:-3]
    frontmatter, body = split_frontmatter(content)

    title = frontmatter.get("title", "").strip() or slug.replace("-", " ").title()
    preview_img = frontmatter.get("previewImg") or first_image_from_body(body)
    image = to_absolute_url(preview_img)
    url = f"{LMSYS_BLOG_BASE_URL}/{slug}/"
    date = frontmatter.get("date", "").strip() or slug[:10]

    if not is_relevant(slug=slug, title=title, body=body):
        return None

    return BlogPost(slug=slug, title=title, url=url, image=image, date=date)


def render_cards(posts: list[BlogPost]) -> str:
    if not posts:
        return "No relevant LMSYS blog posts matched the current sync keywords."

    lines = [
        '<div className="not-prose">',
        "  <div",
        "    style={{",
        '      display: "grid",',
        '      gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",',
        '      gap: "1rem",',
        '      alignItems: "stretch",',
        "    }}",
        "  >",
    ]
    for post in posts:
        safe_title = json.dumps(post.title)
        safe_url = json.dumps(post.url)
        safe_image = json.dumps(post.image)
        lines.extend(
            [
                "    <a",
                f"      href={safe_url}",
                '      target="_blank"',
                '      rel="noopener noreferrer"',
                "      style={{",
                '        display: "block",',
                '        border: "1px solid rgba(128, 128, 128, 0.3)",',
                '        borderRadius: "0.75rem",',
                '        overflow: "hidden",',
                '        textDecoration: "none",',
                '        color: "inherit",',
                '        height: "100%",',
                "      }}",
                "    >",
                "      <div",
                "        style={{",
                '          aspectRatio: "16 / 9",',
                '          overflow: "hidden",',
                '          background: "rgba(128, 128, 128, 0.15)",',
                "        }}",
                "      >",
                "        <img",
                f"          src={safe_image}",
                f"          alt={safe_title}",
                "          style={{",
                '            width: "100%",',
                '            height: "100%",',
                '            objectFit: "cover",',
                '            objectPosition: "center",',
                '            display: "block",',
                "          }}",
                "        />",
                "      </div>",
                "      <div style={{ padding: \"0.9rem 1rem 1rem\" }}>",
                "        <p",
                "          style={{",
                '            margin: 0,',
                '            fontWeight: 600,',
                '            lineHeight: 1.35,',
                '            fontSize: "0.98rem",',
                "          }}",
                "        >",
                f"          {{{safe_title}}}",
                "        </p>",
                "        <p",
                "          style={{",
                '            margin: "0.55rem 0 0",',
                '            fontSize: "0.85rem",',
                '            opacity: 0.75,',
                "          }}",
                "        >",
                f"          {{{json.dumps(post.date)}}}",
                "        </p>",
                "      </div>",
                "    </a>",
            ]
        )
    lines.extend(["  </div>", "</div>"])
    return "\n".join(lines)


def replace_generated_block(index_text: str, generated_cards: str) -> str:
    pattern = re.compile(
        rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
        flags=re.DOTALL,
    )
    replacement = f"{START_MARKER}\n{generated_cards}\n{END_MARKER}"
    updated_text, replacements = pattern.subn(replacement, index_text, count=1)
    if replacements != 1:
        raise RuntimeError(
            f"Could not find exactly one marker block in {INDEX_PATH.name}. "
            f"Expected markers: {START_MARKER} ... {END_MARKER}"
        )
    return updated_text


def write_metadata(posts: list[BlogPost], total_blog_files: int) -> None:
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sourceRepo": "https://github.com/lm-sys/lm-sys.github.io/tree/main/blog",
        "keywords": KEYWORDS,
        "maxCards": MAX_CARDS,
        "totalBlogFilesScanned": total_blog_files,
        "cardsPublished": len(posts),
        "posts": [asdict(post) for post in posts],
    }
    OUTPUT_JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    sources = download_blog_sources()
    relevant_posts: list[BlogPost] = []

    for filename, content in sources:
        post = parse_blog_post(filename=filename, content=content)
        if post is not None:
            relevant_posts.append(post)

    relevant_posts.sort(key=lambda post: post.slug, reverse=True)
    selected_posts = relevant_posts[:MAX_CARDS]

    generated_cards = render_cards(selected_posts)
    current_index = INDEX_PATH.read_text(encoding="utf-8")
    updated_index = replace_generated_block(
        index_text=current_index, generated_cards=generated_cards
    )

    if updated_index != current_index:
        INDEX_PATH.write_text(updated_index, encoding="utf-8")

    write_metadata(posts=selected_posts, total_blog_files=len(sources))
    print(
        "Scanned "
        f"{len(sources)} blog files, matched {len(relevant_posts)} posts, "
        f"published {len(selected_posts)} cards."
    )


if __name__ == "__main__":
    main()

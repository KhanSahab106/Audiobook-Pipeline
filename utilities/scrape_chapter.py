"""
scrape_chapter.py

Scrape readable text from a webpage and save it as a chapter file
in a novel's input/ folder. Supports single-chapter and multi-chapter
batch scraping with anti-detection (TLS fingerprint impersonation,
stealth headers, random delays).

Functions:
    _build_stealth_headers(url, referer) — Build realistic Chrome-like request headers.
    _get_session()                     — Get or create a persistent curl_cffi session.
    fetch_html(url, referer)           — Download HTML with 3-tier anti-detection fallback.
    extract_text(html)                 — Parse HTML and return clean readable text.
    _clean_text(text)                  — Normalise whitespace and blank lines.
    _strip_leading_boilerplate(text)   — Remove lines before the first chapter heading.
    _strip_trailing_boilerplate(text)  — Remove common novel-site footer lines.
    next_chapter_number(novel_dir)     — Return the next available chapter number.
    save_chapter(novel_dir, chapter_num, text) — Write chapter text to the input folder.
    find_next_chapter_url(html, current_url)  — Find the "next chapter" link in page HTML.
    build_parser()                     — Build the argparse CLI parser.
    main()                             — CLI entry point: scrape and save chapters.
"""

import os
import re
import sys
import time
import random
import argparse

import requests
import cloudscraper
from curl_cffi import requests as cffi_requests
from bs4 import BeautifulSoup

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novel_manager.novel_utils import get_all_chapters


# ═══════════════════════════════════════════════════════════════════════════
#  ANTI-DETECTION — STEALTH CONFIG
# ═══════════════════════════════════════════════════════════════════════════

# Realistic User-Agent rotation pool (recent Chrome versions on Windows)
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
]

# Chrome-accurate browser impersonation targets for curl_cffi
_IMPERSONATE_TARGETS = ["chrome131", "chrome124", "chrome120", "chrome116"]


def _build_stealth_headers(url: str, referer: str | None = None) -> dict:
    """Build realistic browser headers that match what a real Chrome would send."""
    from urllib.parse import urlparse
    parsed = urlparse(url)

    headers = {
        "User-Agent":                 random.choice(_USER_AGENTS),
        "Accept":                     "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language":            "en-US,en;q=0.9",
        "Accept-Encoding":            "gzip, deflate, br",
        "Cache-Control":              "max-age=0",
        "Sec-Ch-Ua":                  '"Chromium";v="131", "Not_A Brand";v="24"',
        "Sec-Ch-Ua-Mobile":           "?0",
        "Sec-Ch-Ua-Platform":         '"Windows"',
        "Sec-Fetch-Dest":             "document",
        "Sec-Fetch-Mode":             "navigate",
        "Sec-Fetch-Site":             "same-origin" if referer else "none",
        "Sec-Fetch-User":             "?1",
        "Upgrade-Insecure-Requests":  "1",
        "Connection":                 "keep-alive",
    }

    if referer:
        headers["Referer"] = referer

    return headers


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION & FETCHING
# ═══════════════════════════════════════════════════════════════════════════

# Persistent session — reused across requests like a real browser
_session: cffi_requests.Session | None = None


def _get_session() -> cffi_requests.Session:
    """Get or create a persistent curl_cffi session with Chrome TLS impersonation."""
    global _session
    if _session is None:
        target = random.choice(_IMPERSONATE_TARGETS)
        _session = cffi_requests.Session(impersonate=target)
        print(f"  (stealth: TLS profile {target})")
    return _session


def fetch_html(url: str, referer: str | None = None) -> str:
    """
    Download HTML with full anti-detection:
      1. curl_cffi     — Chrome TLS fingerprint + header ordering
      2. cloudscraper   — Cloudflare JS challenge solver (fallback)
      3. requests       — last resort
    """
    headers = _build_stealth_headers(url, referer)

    # ── Primary: curl_cffi (real Chrome TLS fingerprint) ──────────────
    try:
        session = _get_session()
        resp = session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  curl_cffi failed ({e}), trying cloudscraper...")

    # ── Fallback: cloudscraper (handles JS challenges) ────────────────
    try:
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        resp = scraper.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  cloudscraper failed ({e}), trying plain requests...")

    # ── Last resort: plain requests ───────────────────────────────────
    resp = requests.get(url, headers=headers, timeout=30, verify=False)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text


# ═══════════════════════════════════════════════════════════════════════════
#  TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

# Tags that never contain useful reading content
STRIP_TAGS = [
    "script", "style", "noscript", "iframe",
    "nav", "footer", "header", "aside",
    "form", "button", "svg", "canvas",
]

# Common class/id substrings for non-content sections
NOISE_PATTERNS = re.compile(
    r"(comment|sidebar|advert|promo|popup|modal|cookie|footer|header|nav|menu|social|share|related|recommend)",
    re.IGNORECASE,
)


def extract_text(html: str) -> str:
    """
    Parse HTML and return clean, readable text.

    Strips navigation, scripts, ads, and other non-content elements.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags entirely
    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove elements whose class/id looks like noise
    to_remove = []
    for el in soup.find_all(True):
        if el.attrs is None:
            continue
        classes = " ".join(el.get("class", []))
        el_id   = el.get("id", "")
        if NOISE_PATTERNS.search(classes) or NOISE_PATTERNS.search(el_id):
            to_remove.append(el)
    for el in to_remove:
        if el.parent is not None:   # skip already-removed elements
            el.decompose()

    # Try to find the main content area first
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"(content|chapter|story|text|entry|post)", re.I))
        or soup.find("div", id=re.compile(r"(content|chapter|story|text|entry|post)", re.I))
        or soup.body
        or soup
    )

    # Get text with paragraph separation
    raw = main.get_text(separator="\n")

    return _clean_text(raw)


def _clean_text(text: str) -> str:
    """Normalize whitespace and blank lines."""
    # Strip each line
    lines = [line.strip() for line in text.splitlines()]

    # Collapse runs of blank lines into at most two
    cleaned = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 2:
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(line)

    result = "\n".join(cleaned).strip()

    # Strip leading boilerplate (novel title, site name, etc.)
    result = _strip_leading_boilerplate(result)

    # Strip common novel-site trailing boilerplate
    result = _strip_trailing_boilerplate(result)

    return result


# Matches lines like "Chapter 12: An unexpected meeting", "Chapter 1", etc.
_CHAPTER_HEADING = re.compile(
    r"^Chapter\s+\d+",
    re.IGNORECASE,
)


def _strip_leading_boilerplate(text: str) -> str:
    """Remove lines before the first 'Chapter N:' heading (novel title, site name, etc.)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _CHAPTER_HEADING.match(line.strip()):
            return "\n".join(lines[i:]).strip()
    # No chapter heading found — return as-is
    return text


# Lines that commonly appear at the bottom of novel-reading sites
_BOILERPLATE_PATTERNS = re.compile(
    r"^("
    r"Report\s+chapter"
    r"|Tip:\s+You\s+can\s+use"
    r"|Previous\s+Chapter"
    r"|Next\s+Chapter"
    r"|Table\s+of\s+Contents"
    r"|← Prev"
    r"|Next →"
    r"|Share\s+this"
    r"|Rate\s+this"
    r"|Add\s+to\s+library"
    r"|Bookmark"
    r"|Download\s+app"
    r").*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_trailing_boilerplate(text: str) -> str:
    """Remove common novel-site trailing lines."""
    lines = text.rstrip().splitlines()
    # Walk backwards, removing boilerplate or blank lines
    while lines:
        stripped = lines[-1].strip()
        if not stripped or _BOILERPLATE_PATTERNS.match(stripped):
            lines.pop()
        else:
            break
    return "\n".join(lines).rstrip()


# ═══════════════════════════════════════════════════════════════════════════
#  CHAPTER FILE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def next_chapter_number(novel_dir: str) -> int:
    """Return the next available chapter number for the novel."""
    chapters = get_all_chapters(novel_dir)
    if not chapters:
        return 1
    return chapters[-1][0] + 1


def save_chapter(novel_dir: str, chapter_num: int, text: str) -> str:
    """Write the chapter text to the input folder. Returns the file path."""
    input_dir = os.path.join(novel_dir, "input")
    os.makedirs(input_dir, exist_ok=True)

    filename = f"chapter_{chapter_num}.txt"
    filepath = os.path.join(input_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    return filepath


# ═══════════════════════════════════════════════════════════════════════════
#  NEXT CHAPTER DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def find_next_chapter_url(html: str, current_url: str) -> str | None:
    """
    Find the 'next chapter' link in the page HTML.

    Looks for common next-chapter selectors used by novel-reading sites.
    Returns the absolute URL or None if not found.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Try common selectors for next-chapter links
    next_link = (
        soup.select_one("a#next_chap")          # novelbin.com
        or soup.select_one("a.next_chap")
        or soup.select_one("a.btn-next")
        or soup.select_one("a[rel='next']")
        or soup.select_one("a.next-chap")
    )

    if next_link and next_link.get("href"):
        href = next_link["href"]
        # Handle relative URLs
        if href.startswith("/"):
            from urllib.parse import urlparse
            parsed = urlparse(current_url)
            href = f"{parsed.scheme}://{parsed.netloc}{href}"
        return href

    return None


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape novel chapters from a webpage and save them as chapter files.",
        epilog=(
            "Examples:\n"
            "  python scrape_chapter.py shs_and_sws <url>              # single chapter\n"
            "  python scrape_chapter.py shs_and_sws <url> --count 10   # 10 chapters starting from <url>\n"
            "  python scrape_chapter.py shs_and_sws <url> -n 5 -c 12   # 5 chapters, starting as chapter_12"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "novel",
        help="Novel slug (folder name under novels/), e.g. 'shs_and_sws'",
    )
    parser.add_argument(
        "url",
        help="URL of the first chapter to scrape",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=1,
        help="Number of chapters to scrape (default: 1)",
    )
    parser.add_argument(
        "--chapter", "-c",
        type=int,
        default=None,
        help="Override the starting chapter number (default: auto-detect next chapter)",
    )
    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    novel_dir = os.path.join("novels", args.novel)
    if not os.path.isdir(novel_dir):
        print(f"  Error: Novel directory not found: {novel_dir}")
        print(f"  Run first: python setup_novel.py {args.novel}")
        sys.exit(1)

    chapter_num = args.chapter if args.chapter is not None else next_chapter_number(novel_dir)
    total       = args.count
    current_url = args.url
    referer     = None  # first request has no referer (direct navigation)
    saved       = 0
    failed      = 0

    print(f"\n{'=' * 60}")
    print(f"  Scraping {total} chapter(s) into: {novel_dir}/input/")
    print(f"  Starting at chapter {chapter_num}")
    print(f"{'=' * 60}\n")

    for i in range(total):
        label = f"[{i + 1}/{total}]"

        # ── Fetch ─────────────────────────────────────────────────────
        print(f"  {label} Fetching: {current_url}")
        try:
            html = fetch_html(current_url, referer=referer)
        except Exception as e:
            print(f"  {label} [X] Error fetching URL: {e}")
            failed += 1
            break  # can't continue without the page

        # ── Extract ───────────────────────────────────────────────────
        text = extract_text(html)
        if not text.strip():
            print(f"  {label} [X] No readable text found, skipping.")
            failed += 1
        else:
            word_count = len(text.split())
            filepath   = save_chapter(novel_dir, chapter_num, text)
            print(f"  {label} [OK] Saved chapter_{chapter_num}.txt  ({word_count:,} words)")
            saved += 1

        chapter_num += 1

        # ── Find next chapter URL ─────────────────────────────────────
        if i < total - 1:
            next_url = find_next_chapter_url(html, current_url)
            if not next_url:
                print(f"\n  [!] No 'next chapter' link found -- stopping after {i + 1} chapter(s).")
                break
            referer     = current_url  # previous page becomes referer
            current_url = next_url

            # ── 10-minute cooldown every 10 chapters ──────────────────
            if (i + 1) % 10 == 0:
                print(f"  {'':>{len(label)}} 10 chapters done -- cooling down for 10 minutes...\n")
                time.sleep(600)
            else:
                # ── Random delay to avoid rate-limiting ───────────────
                delay = random.uniform(15, 25)
                print(f"  {'':>{len(label)}} Waiting {delay:.0f}s before next request...\n")
                time.sleep(delay)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Done!  [OK] {saved} saved  |  [X] {failed} failed  |  {total} requested")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()


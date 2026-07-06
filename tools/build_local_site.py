#!/usr/bin/env python3
"""Build local-site/ — a fully self-contained, offline-readable copy of the
poisson-solvers report suite.

Open  local-site/index.html  directly in a browser (file://).  No network, no
local server:

  * README.md and reports/*.md are rendered to HTML with python-markdown
    (tables, fenced_code, toc).  Math spans ($$...$$ and $...$) are extracted
    to placeholders BEFORE markdown conversion and the original TeX is
    reinserted afterwards, so underscores/asterisks inside math are never
    mangled.
  * MathJax 3 (tex-svg single-file bundle, SVG output => no font fetches) is
    vendored once into local-site/vendor/mathjax-tex-svg.js.
  * figures/, results/, python/, mathematica/, cg-explorer/ are copied so
    every relative link in the reports resolves inside local-site/.
  * interactive/adi-sweep.html is copied verbatim (it is self-contained and
    its ../reports/*.html back-links resolve against the rendered reports).
  * interactive/hierarchical-solvers.html is patched to preload its 2.6 MB
    JSON via a generated results/hodlr_viz_data.js (window.HODLR_VIZ_DATA)
    instead of fetch(), which does not work from file://.
  * External http(s) links (arXiv, GitHub — citations) are de-linkified into
    visible plain-text URLs so that ZERO http(s) references remain in any
    src=/href= attribute anywhere under local-site/.

The script is idempotent; it ends with a manifest and PASS/FAIL lines.
Run from the repo root:  uv run python tools/build_local_site.py
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parent.parent
SITE = REPO / "local-site"
MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
MATHJAX_FILE = SITE / "vendor" / "mathjax-tex-svg.js"
MATHJAX_MIN_BYTES = 500_000
PAGES_PREFIX = "https://yaroslavvb.github.io/poisson-solvers/"

COPY_DIRS = ["figures", "results", "python", "mathematica", "cg-explorer"]
COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__", ".DS_Store", "*.pyc", ".ipynb_checkpoints"
)

# --------------------------------------------------------------------------
# Math / code protection (mandatory: extract math before markdown conversion)
# --------------------------------------------------------------------------

FENCE_RE = re.compile(r"(?ms)^(`{3,}|~{3,}).*?^\1[ \t]*$")
INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
DISPLAY_MATH_RE = re.compile(r"(?s)\$\$.+?\$\$")
INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?![\s$])((?:\\[^\n]|[^$\n\\])+?)(?<![\s\\])\$")


def protect_math(text: str) -> tuple[str, list[str]]:
    """Replace every math span with an alnum placeholder; return (text, spans).

    Fenced code blocks and inline code spans are shielded first so that a
    stray '$' inside code is never treated as math.  Code is put back before
    markdown conversion (markdown renders it normally); only math stays
    placeheld until after conversion.
    """
    math_spans: list[str] = []
    code_spans: list[str] = []

    def stash_code(m: re.Match) -> str:
        code_spans.append(m.group(0))
        return f"xCODEPHx{len(code_spans) - 1}xENDx"

    def stash_math(m: re.Match) -> str:
        math_spans.append(m.group(0))
        return f"xMATHPHx{len(math_spans) - 1}xENDx"

    text = FENCE_RE.sub(stash_code, text)
    text = INLINE_CODE_RE.sub(stash_code, text)
    text = DISPLAY_MATH_RE.sub(stash_math, text)
    text = INLINE_MATH_RE.sub(stash_math, text)
    # restore code — markdown must still see and render it
    text = re.sub(
        r"xCODEPHx(\d+)xENDx", lambda m: code_spans[int(m.group(1))], text
    )
    return text, math_spans


def restore_math(html_text: str, math_spans: list[str]) -> str:
    """Reinsert the ORIGINAL TeX (html-escaped so <, & survive the DOM)."""
    return re.sub(
        r"xMATHPHx(\d+)xENDx",
        lambda m: html.escape(math_spans[int(m.group(1))], quote=False),
        html_text,
    )


# --------------------------------------------------------------------------
# Markdown conversion
# --------------------------------------------------------------------------

def gh_slug(value: str, separator: str = "-") -> str:
    """GitHub-flavoured heading slug (close match for cross-page #fragments).

    GitHub's slugger keeps unicode letters as-is (e.g. 'ö' stays 'ö'), so we
    must NOT NFKD-fold to ASCII — otherwise anchors written against the
    GitHub-rendered ids miss the locally generated ones.
    """
    value = value.lower()
    value = re.sub(r"[^\w\- ]", "", value)
    return value.replace(" ", separator)


MD = markdown.Markdown(
    extensions=["tables", "fenced_code", "toc"],
    extension_configs={"toc": {"slugify": gh_slug}},
)


def md_to_html(md_text: str) -> str:
    protected, math_spans = protect_math(md_text)
    MD.reset()
    body = MD.convert(protected)
    return restore_math(body, math_spans)


# --------------------------------------------------------------------------
# Link rewriting
# --------------------------------------------------------------------------

ATTR_RE = re.compile(r'\b(href|src)="([^"]*)"')
EXT_A_RE = re.compile(r'<a\s[^>]*href="(https?://[^"]*)"[^>]*>(.*?)</a>', re.S)
TAG_STRIP_RE = re.compile(r"<[^>]+>")


def rewrite_relative_links(page_html: str) -> str:
    """*.md -> *.html for relative links (fragments preserved)."""

    def fix(m: re.Match) -> str:
        attr, url = m.group(1), m.group(2)
        if url.startswith(("http://", "https://", "mailto:", "#", "data:", "javascript:")):
            return m.group(0)
        path, sep, frag = url.partition("#")
        if path.endswith(".md"):
            path = path[:-3] + ".html"
        return f'{attr}="{path}{sep}{frag}"'

    return ATTR_RE.sub(fix, page_html)


def map_pages_url(url: str) -> str | None:
    """Map a GitHub-Pages URL of this repo to a local-site-relative path."""
    if not url.startswith(PAGES_PREFIX) and url.rstrip("/") + "/" != PAGES_PREFIX:
        return None
    rest = url[len(PAGES_PREFIX):] if url.startswith(PAGES_PREFIX) else ""
    path, _, frag = rest.partition("#")
    if path == "" or path.endswith("/"):
        path += "index.html"
    if path.endswith(".md"):
        path = path[:-3] + ".html"
    # reports are generated later in the same run — accept them by stem
    if path.startswith("reports/") and path.endswith(".html"):
        if (REPO / "reports" / (Path(path).stem + ".md")).exists():
            return path + (("#" + frag) if frag else "")
        return None
    if (SITE / path).exists():
        return path + (("#" + frag) if frag else "")
    return None


def delinkify_external(page_html: str, depth: int) -> str:
    """External <a href="http(s)://..."> -> plain-text citation (offline
    guarantee: no http(s) may remain in any href/src attribute).  Known
    GitHub-Pages self-URLs are rewritten to their local equivalents instead.
    """

    def fix(m: re.Match) -> str:
        url, label = m.group(1), m.group(2)
        local = map_pages_url(url)
        if local is not None:
            rel = "../" * depth + local
            return f'<a href="{rel}">{label}</a>'
        plain = TAG_STRIP_RE.sub("", label).strip()
        if plain.rstrip("/") == url.rstrip("/"):
            return f'<span class="citation">{label}</span>'
        return f'{label}<span class="citation"> ({url})</span>'

    return EXT_A_RE.sub(fix, page_html)


# --------------------------------------------------------------------------
# Page template
# --------------------------------------------------------------------------

CSS = """
body { margin: 0; color: #1b1f23; background: #fff;
       font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI",
             Helvetica, Arial, sans-serif; }
.page { max-width: 860px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
nav.site { font-size: .85rem; color: #57606a; margin-bottom: 1.5rem;
           border-bottom: 1px solid #d8dee4; padding-bottom: .5rem; }
nav.site a { color: #0969da; text-decoration: none; }
nav.site a:hover { text-decoration: underline; }
nav.site .sep { margin: 0 .45em; color: #8c959f; }
h1, h2, h3, h4 { line-height: 1.25; margin: 1.6em 0 .6em; }
h1 { font-size: 1.9rem; border-bottom: 1px solid #d8dee4;
     padding-bottom: .3em; margin-top: .2em; }
h2 { font-size: 1.45rem; border-bottom: 1px solid #eaecef;
     padding-bottom: .25em; }
a { color: #0969da; }
img { max-width: 100%; height: auto; }
pre { background: #f6f8fa; padding: .9em 1.1em; overflow-x: auto;
      border-radius: 6px; font-size: .85em; line-height: 1.45; }
code { font-family: "SF Mono", SFMono-Regular, ui-monospace, Menlo,
       Consolas, monospace; background: #f6f8fa; padding: .1em .3em;
       border-radius: 4px; font-size: .9em; }
pre code { background: none; padding: 0; font-size: 1em; }
table { border-collapse: collapse; margin: 1em 0; display: block;
        overflow-x: auto; max-width: 100%; }
th, td { border: 1px solid #d0d7de; padding: .35em .7em; }
th { background: #f6f8fa; }
blockquote { margin: 0 0 1em; padding: 0 1em; color: #57606a;
             border-left: .25em solid #d0d7de; }
hr { border: none; border-top: 1px solid #d8dee4; margin: 2em 0; }
.citation { color: #57606a; word-break: break-all; }
mjx-container[jax="SVG"][display="true"] { overflow-x: auto;
                                           overflow-y: hidden; }
""".strip()

MATHJAX_CONFIG = """
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  },
  svg: { fontCache: 'local' }
};
""".strip()

PAGE_TMPL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
__CSS__
</style>
<script>
__MJCONF__
</script>
<script defer src="__MATHJAX__"></script>
</head>
<body>
<div class="page">
__NAV__
__BODY__
</div>
</body>
</html>
"""


def first_h1(md_text: str) -> str:
    for line in md_text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            title = re.sub(r"[*_`]", "", title)
            title = re.sub(r"\$([^$]+)\$", r"\1", title)  # de-TeX titles
            return title
    return "poisson-solvers"


def build_page(title: str, nav_html: str, body: str, depth: int) -> str:
    rel = "../" * depth + "vendor/mathjax-tex-svg.js"
    return (
        PAGE_TMPL.replace("__TITLE__", html.escape(title))
        .replace("__CSS__", CSS)
        .replace("__MJCONF__", MATHJAX_CONFIG)
        .replace("__MATHJAX__", rel)
        .replace("__NAV__", nav_html)
        .replace("__BODY__", body)
    )


# --------------------------------------------------------------------------
# Build steps
# --------------------------------------------------------------------------

def _sanitize_mathjax(text: str) -> str:
    """The tex-svg bundle contains two <a href="https://www.mathjax.org">
    anchors inside its About/Help context-menu HTML strings.  They are never
    fetched at load time (SVG output makes zero network requests), but we
    de-linkify them anyway so the offline guarantee — no http(s) in any
    src=/href= attribute under local-site/ — holds literally in every file."""
    return text.replace(
        '<a href="https://www.mathjax.org">www.mathjax.org</a>',
        "<span>www.mathjax.org</span>",
    )


def fetch_mathjax() -> None:
    if MATHJAX_FILE.exists() and MATHJAX_FILE.stat().st_size > MATHJAX_MIN_BYTES:
        print(f"  mathjax: cached ({MATHJAX_FILE.stat().st_size:,} bytes) — skipping download")
        cached = MATHJAX_FILE.read_text(encoding="utf-8", errors="ignore")
        sanitized = _sanitize_mathjax(cached)
        if sanitized != cached:
            MATHJAX_FILE.write_text(sanitized, encoding="utf-8")
            print("  mathjax: de-linkified About-dialog anchors in cached copy")
        return
    MATHJAX_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"  mathjax: downloading {MATHJAX_URL} ...")
    with urllib.request.urlopen(MATHJAX_URL, timeout=120) as resp:
        data = resp.read()
    text = _sanitize_mathjax(data.decode("utf-8"))
    MATHJAX_FILE.write_text(text, encoding="utf-8")
    print(f"  mathjax: saved {len(text.encode('utf-8')):,} bytes")


def copy_assets() -> int:
    copied = 0
    for d in COPY_DIRS:
        src = REPO / d
        if not src.is_dir():
            print(f"  WARNING: missing source dir {d}/ — skipped")
            continue
        shutil.copytree(src, SITE / d, ignore=COPY_IGNORE, dirs_exist_ok=True)
        copied += sum(1 for p in (SITE / d).rglob("*") if p.is_file())
    return copied


def render_reports() -> list[Path]:
    report_mds = sorted((REPO / "reports").glob("*.md"))
    stems = [p.stem for p in report_mds]
    out_dir = SITE / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = []

    for i, src in enumerate(report_mds):
        md_text = src.read_text(encoding="utf-8")
        body = md_to_html(md_text)
        body = rewrite_relative_links(body)
        body = delinkify_external(body, depth=1)

        parts = ['<a href="../index.html">index</a>']
        if i > 0:
            parts.append(f'<a href="{stems[i-1]}.html">&larr; {stems[i-1]}</a>')
        else:
            parts.append("<span>&larr; (first report)</span>")
        if i < len(stems) - 1:
            parts.append(f'<a href="{stems[i+1]}.html">{stems[i+1]} &rarr;</a>')
        else:
            parts.append("<span>(last report) &rarr;</span>")
        nav = '<nav class="site">' + '<span class="sep">&middot;</span>'.join(parts) + "</nav>"

        page = build_page(first_h1(md_text), nav, body, depth=1)
        out = out_dir / (src.stem + ".html")
        out.write_text(page, encoding="utf-8")
        pages.append(out)
    return pages


def render_index() -> Path:
    md_text = (REPO / "README.md").read_text(encoding="utf-8")
    body = md_to_html(md_text)
    body = rewrite_relative_links(body)
    body = delinkify_external(body, depth=0)
    nav = ('<nav class="site">poisson-solvers &mdash; offline mirror '
           '(built by tools/build_local_site.py)</nav>')
    page = build_page(first_h1(md_text), nav, body, depth=0)
    out = SITE / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


def transform_interactive() -> None:
    # ---- adi-sweep.html: self-contained; back-links ../reports/13-*.html
    # already resolve against the rendered reports (structure mirrors repo).
    src = (REPO / "interactive" / "adi-sweep.html").read_text(encoding="utf-8")
    (SITE / "interactive").mkdir(parents=True, exist_ok=True)
    (SITE / "interactive" / "adi-sweep.html").write_text(src, encoding="utf-8")

    # ---- results/hodlr_viz_data.js: JSON payload as a plain script global.
    json_text = (REPO / "results" / "hodlr_viz_data.json").read_text(encoding="utf-8")
    json.loads(json_text)  # validate before embedding
    js = ("/* generated by tools/build_local_site.py from "
          "results/hodlr_viz_data.json — offline (file://) replacement for "
          "fetch(). */\nwindow.HODLR_VIZ_DATA =\n" + json_text.strip() + ";\n")
    (SITE / "results").mkdir(parents=True, exist_ok=True)
    (SITE / "results" / "hodlr_viz_data.js").write_text(js, encoding="utf-8")

    # ---- hierarchical-solvers.html: preload data script + drop the fetch().
    page = (REPO / "interactive" / "hierarchical-solvers.html").read_text(encoding="utf-8")

    preload = (
        "<!-- OFFLINE PATCH (tools/build_local_site.py): preload the HODLR\n"
        "     data as a plain script so this page works from file:// with no\n"
        "     network request and no web server. -->\n"
        '<script src="../results/hodlr_viz_data.js"></script>\n'
    )
    assert page.count("<script>") == 1, "expected exactly one <script> in hierarchical-solvers.html"
    page = page.replace("<script>", preload + "<script>", 1)

    offline_boot = (
        "/* OFFLINE PATCH (tools/build_local_site.py): data preloaded by\n"
        "     ../results/hodlr_viz_data.js into window.HODLR_VIZ_DATA —\n"
        "     replaces the runtime JSON request, which browsers block\n"
        "     from file://. */\n"
        "  if (window.HODLR_VIZ_DATA) {\n"
        "    setupData(window.HODLR_VIZ_DATA);\n"
        "    status('', false);\n"
        "    buildUI();\n"
        "  } else {\n"
        "    status('Offline data missing: ../results/hodlr_viz_data.js should "
        "define window.HODLR_VIZ_DATA. Re-run tools/build_local_site.py.', true);\n"
        "  }"
    )
    fetch_re = re.compile(
        r"fetch\('\.\./results/hodlr_viz_data\.json'\)(?s:.)*?\}\);"
    )
    page, n = fetch_re.subn(offline_boot, page)
    assert n == 1, f"expected exactly one fetch() block, patched {n}"
    # also silence the now-momentary 'Loading …' status line
    page = page.replace(
        "  status('Loading ../results/hodlr_viz_data.json …', false);\n\n  ", "  ", 1
    )
    assert "fetch(" not in page, "patched page must have no fetch dependency"
    (SITE / "interactive" / "hierarchical-solvers.html").write_text(page, encoding="utf-8")

    # ---- cg-explorer/index.html (copied with assets): de-linkify its two
    # external anchors and point its "../" home link at ../index.html.
    cg = SITE / "cg-explorer" / "index.html"
    if cg.exists():
        text = cg.read_text(encoding="utf-8")
        text = text.replace('<a href="../">', '<a href="../index.html">')
        text = delinkify_external(text, depth=1)
        cg.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

SKIP_SCHEMES = ("http://", "https://", "mailto:", "data:", "javascript:", "#")


def check_internal_links() -> tuple[int, list[str]]:
    """Every href/src in every HTML page under local-site must resolve."""
    n_checked, failures = 0, []
    for page in sorted(SITE.rglob("*.html")):
        text = page.read_text(encoding="utf-8", errors="ignore")
        for m in ATTR_RE.finditer(text):
            url = m.group(2)
            if url.startswith(SKIP_SCHEMES) or url == "":
                continue
            path = urllib.parse.unquote(url.partition("#")[0].partition("?")[0])
            if not path:
                continue
            target = (page.parent / path).resolve()
            n_checked += 1
            if not target.exists():
                failures.append(f"{page.relative_to(SITE)}: {m.group(1)}=\"{url}\"")
            elif SITE.resolve() not in target.parents and target != SITE.resolve():
                failures.append(f"{page.relative_to(SITE)}: escapes local-site: {url}")
    return n_checked, failures


EXTERNAL_ATTR_RES = [
    re.compile(r"""(?:src|href)\s*=\s*["']https?://""", re.I),
    re.compile(r"""url\(\s*["']?https?://""", re.I),
    re.compile(r"""@import\s+["']https?://""", re.I),
]


def scan_external_refs() -> tuple[int, list[str]]:
    """No http(s) may appear in src=/href=/url()/@import anywhere under
    local-site (plain-text citation URLs in prose are fine)."""
    n_files, hits = 0, []
    for f in sorted(SITE.rglob("*")):
        if not f.is_file() or f.suffix.lower() in {
            ".png", ".gif", ".jpg", ".jpeg", ".pt", ".ico", ".woff", ".woff2"
        }:
            continue
        n_files += 1
        text = f.read_text(encoding="utf-8", errors="ignore")
        for rx in EXTERNAL_ATTR_RES:
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{f.relative_to(SITE)}:{line}: {m.group(0)!r}")
    return n_files, hits


def dir_size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    print(f"Building offline site into {SITE}")
    SITE.mkdir(exist_ok=True)

    fetch_mathjax()
    n_copied = copy_assets()
    report_pages = render_reports()
    index_page = render_index()
    transform_interactive()

    # ---------------- manifest ----------------
    total = dir_size(SITE)
    print("\n=== MANIFEST ===")
    print(f"  pages built     : {len(report_pages) + 1} "
          f"({len(report_pages)} reports + index.html)")
    for p in [index_page] + report_pages:
        print(f"    - {p.relative_to(SITE)}")
    print(f"  asset files     : {n_copied} copied into "
          f"{', '.join(d + '/' for d in COPY_DIRS)}")
    print("  interactive     : adi-sweep.html (verbatim), "
          "hierarchical-solvers.html (fetch -> preloaded "
          "results/hodlr_viz_data.js), cg-explorer/ (de-linkified)")
    print(f"  vendored mathjax: {MATHJAX_FILE.relative_to(SITE)} "
          f"({MATHJAX_FILE.stat().st_size:,} bytes)")
    print(f"  total size      : {total / 1e6:.1f} MB "
          f"({sum(1 for p in SITE.rglob('*') if p.is_file())} files)")

    # ---------------- checks ----------------
    print("\n=== CHECKS ===")
    ok = True

    n_pages = len(report_pages)
    built_ok = n_pages == 16 and index_page.exists()
    ok &= built_ok
    print(f"{'PASS' if built_ok else 'FAIL'}: all report pages + index built "
          f"({n_pages}/16 reports, index={'yes' if index_page.exists() else 'NO'})")

    n_links, link_failures = check_internal_links()
    ok &= not link_failures
    print(f"{'PASS' if not link_failures else 'FAIL'}: internal link check — "
          f"{n_links} href/src references, {len(link_failures)} broken")
    for fchk in link_failures[:20]:
        print(f"    BROKEN: {fchk}")

    n_scanned, ext_hits = scan_external_refs()
    ok &= not ext_hits
    print(f"{'PASS' if not ext_hits else 'FAIL'}: zero external src/href/url() "
          f"references ({n_scanned} text files scanned, {len(ext_hits)} hits)")
    for h in ext_hits[:20]:
        print(f"    EXTERNAL: {h}")

    mj_ok = MATHJAX_FILE.exists() and MATHJAX_FILE.stat().st_size > MATHJAX_MIN_BYTES
    ok &= mj_ok
    print(f"{'PASS' if mj_ok else 'FAIL'}: MathJax vendored "
          f"({MATHJAX_FILE.stat().st_size:,} bytes > {MATHJAX_MIN_BYTES:,})"
          if MATHJAX_FILE.exists() else "FAIL: MathJax file missing")

    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

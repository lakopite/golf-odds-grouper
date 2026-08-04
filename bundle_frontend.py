#!/usr/bin/env python3
"""
bundle_frontend.py -- fold a result file into the scoreboard and export one page.

    python bundle_frontend.py --result build/result.json --out dist/

Produces a single self-contained `.html` -- no build step, no server, nothing fetched
to draw it -- and a `.zip` holding that page, the result JSON it was built from, and a
manifest. Both are artifacts of one run; neither is written back into the repo.

The page does poll ESPN once it is open, and polls from the moment it opens even when
the tournament is days away: that poll is how it learns play has started and turns
itself from a groups sheet into a scoreboard. Everything it DRAWS is baked in, so it
still opens from disk with no network at all.

Two templates ship. `frontend/scoreboard/` is the designed page and the default;
`frontend/template/` is the plain reference that exists to prove the contract below.
Both inline `frontend/lib.js`, which is where the standings rule lives.

THE TEMPLATE CONTRACT
---------------------
A template is a directory with an `index.html` in it. Four things happen to it, and
the last is optional:

  1. The result JSON replaces the marker inside the data script tag:

         <script id="competition-data" type="application/json">
           /*__COMPETITION_JSON__*/
         </script>

     After bundling the page reads its data with one line and never fetches it:

         const DATA = JSON.parse(document.getElementById('competition-data').textContent)

  2. Local `<link rel=stylesheet>`, `<script src>` and `<img src>` are inlined --
     stylesheets and scripts as text, images as data: URIs. Absolute URLs are left
     alone, and so is anything the CSP of a locked-down host would reject anyway. A
     reference may point above the template directory: `../lib.js` is how both shipped
     templates share one copy of the standings rule.

  3. `{{title}}` style tokens are substituted from a small set of fields, so a
     template can name the league in its <title> without running any JavaScript.

  4. The league's art is read and inlined into a second data element, if the template
     asks for one:

         <script id="league-art" type="application/json">
           /*__LEAGUE_ART_JSON__*/
         </script>

     What lands there is `{"logo": "data:image/png;base64,…", "banner": "…"}` -- only
     the keys that resolved, and `{}` for a league with no art.

That is the whole contract. Any HTML that honours it can be dropped in as
`--template`, which is how a designed scoreboard replaces the reference one without
touching this file.

WHY THE ART IS READ HERE AND NOWHERE ELSE
-----------------------------------------
The result file names the league's art with a slug -- `"logo": "wcw"` -- and that name
resolves to `leagues/wcw/logo.png` and `leagues/wcw/banner.png`. The pictures never
enter the result JSON. They enter here, in the last step, into the one artifact whose
whole purpose is to survive being emailed: a page that opens from a USB stick with the
wifi off.

So the two outputs of this program deliberately differ. The `.html` has the images in
it; the `result.json` beside it in the zip carries the slug and stays a document about
a competition. Rebuild from that JSON and the art comes back at the next export, out of
the repository, at whatever the files say today.
"""

import argparse
import base64
import html
import json
import mimetypes
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone

import league as league_mod

_HERE = os.path.dirname(os.path.abspath(__file__))

# The designed page, and what a plain `python bundle_frontend.py --result ...` produces.
DEFAULT_TEMPLATE = os.path.join(_HERE, "frontend", "scoreboard")

# The plain one. It exists to prove the contract rather than to be the design, and it is
# the thing to bundle against when the question is "is the page wrong, or is the data?"
#   python bundle_frontend.py --result build/result.json --template frontend/template
REFERENCE_TEMPLATE = os.path.join(_HERE, "frontend", "template")

JSON_MARKER = "/*__COMPETITION_JSON__*/"

# The league's art, inlined into the page and into nothing else. Optional, unlike the
# data marker: a template that does not draw a masthead never has to mention it.
ART_MARKER = "/*__LEAGUE_ART_JSON__*/"

# An image over this lands in the page as base64, which is a third bigger again. Not a
# refusal -- somebody who wants a 3 MB banner in their scoreboard is entitled to one,
# and a page too heavy to email is a nuisance rather than a wrong answer -- but it is
# the kind of thing to learn from a build rather than from an inbox.
HEAVY_ART_BYTES = 1024 * 1024

_LINK_RE = re.compile(r'<link\b[^>]*?href=["\']([^"\']+)["\'][^>]*?>', re.I)
_SCRIPT_RE = re.compile(r'<script\b[^>]*?src=["\']([^"\']+)["\'][^>]*?>\s*</script>', re.I)
_IMG_RE = re.compile(r'(<img\b[^>]*?src=)["\']([^"\']+)["\']', re.I)
_TOKEN_RE = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")


def is_remote(url):
    return url.startswith(("http://", "https://", "//", "data:", "#", "mailto:"))


def local_asset(base_dir, ref):
    """
    Resolve one asset reference. Returns (path, status).

    status is "file" (path is the file to inline), "skip" (nothing to inline) or
    "missing" (a real reference to a file that is not there; path is where it was
    looked for, and it is worth reporting).

    `os.path.exists` is not the check, and the difference is a crash. A template that
    carries `<img src="">` -- which is what an element whose source arrives from the
    data at runtime looks like -- joins to the template directory itself. That exists,
    so an exists() check sends the bundler off to base64-encode a directory. Empty
    references and directories are both "nothing here", not files and not errors.

    A reference may also point above the template directory: `../lib.js` is how both
    shipped templates share one copy of the standings rule. That is deliberate and it
    inlines like anything else.
    """
    if not ref or not ref.strip() or is_remote(ref):
        return None, "skip"
    path = os.path.join(base_dir, ref)
    if os.path.isfile(path):
        return path, "file"
    if os.path.exists(path):
        return None, "skip"
    return path, "missing"


def json_for_script(payload):
    """
    Serialise so the result can live inside a <script> element safely.

    A `</script>` anywhere in the data -- a team called "</script> FC", a golfer whose
    name a future source HTML-escapes -- ends the element early and silently truncates
    the page's entire dataset. `<!--` does the same to the parser in a different way.
    Both are neutralised here rather than trusted not to appear.
    """
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("</", "<\\/").replace("<!--", "<\\!--")


def inline_assets(markup, base_dir, report):
    """Replace local stylesheet, script and image references with their contents."""

    def resolve(ref):
        """The file to inline, or None -- and a note in the report if it should
        have been one. A missing asset is left in the markup as it was written:
        honest breakage beats a silent drop."""
        path, status = local_asset(base_dir, ref)
        if status == "missing":
            report["missing"].append(ref)
        return path if status == "file" else None

    def do_link(match):
        tag, href = match.group(0), match.group(1)
        if "stylesheet" not in tag.lower():
            return tag
        path = resolve(href)
        if not path:
            return tag
        report["inlined"].append(href)
        with open(path, encoding="utf-8") as f:
            return f"<style>\n{f.read()}\n</style>"

    def do_script(match):
        tag, src = match.group(0), match.group(1)
        path = resolve(src)
        if not path:
            return tag
        report["inlined"].append(src)
        with open(path, encoding="utf-8") as f:
            # A bare </script> in the source would close the tag it is being pasted into.
            return "<script>\n" + f.read().replace("</script>", "<\\/script>") + "\n</script>"

    def do_img(match):
        prefix, src = match.group(1), match.group(2)
        path = resolve(src)
        if not path:
            return match.group(0)
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        report["inlined"].append(src)
        return f'{prefix}"data:{mime};base64,{data}"'

    markup = _LINK_RE.sub(do_link, markup)
    markup = _SCRIPT_RE.sub(do_script, markup)
    markup = _IMG_RE.sub(do_img, markup)
    return markup


def league_art(result, leagues_dir=None, report=None):
    """
    The league's two images as data: URIs, read out of the slug the result file names.

    Returns {"logo": "data:…", "banner": "data:…"} with only the keys that resolved --
    so `{}` for a league with no art, and `{"logo": …}` for one with a badge and no
    banner. Both of those are ordinary and the page draws them.

    A slug that resolves to nothing at all is the one case worth a word: it means the
    build recorded art that this export cannot find, which is a masthead somebody is
    expecting and will not get.
    """
    slug = (result.get("league") or {}).get("logo")
    found = league_mod.art_files(slug, leagues_dir)
    art = {}
    for name, path in found.items():
        if not path:
            continue
        size = os.path.getsize(path)
        if size > HEAVY_ART_BYTES:
            ideal = "256 px square" if name == "logo" else "720 px wide"
            print(f"note: {path} is {size // 1024} KB and lands in the page as about "
                  f"{size * 4 // 3 // 1024} KB of base64. A {ideal} {name} is the right order "
                  "of magnitude; a JPEG beats a PNG on a photograph.")
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            art[name] = f"data:{mime};base64,{base64.b64encode(f.read()).decode('ascii')}"
        if report is not None:
            report["art"].append((name, path, size))

    if slug and not art:
        print(f"!! the league's art slug is {slug!r} and "
              f"{os.path.join(leagues_dir or league_mod.LEAGUES_DIR, slug)} holds no logo or "
              "banner image. The page will show the league's name and no art.")
    return art


def substitute_tokens(markup, result):
    """
    Fill `{{league_name}}` and friends. HTML-escaped, because these come from a
    user-written league file and land in markup rather than in a text node.
    """
    values = {
        "league_name": result["league"]["league_name"],
        "tournament": result["tournament"]["name"],
        "market": result["sources"]["kalshi"]["market_label"],
        "generated_at": result["generated_at"],
        "team_count": str(result["league"]["team_count"]),
        "competition_id": result["competition_id"],
    }
    return _TOKEN_RE.sub(lambda m: html.escape(values.get(m.group(1), m.group(0))), markup)


def require_scoreable(result):
    """
    Refuse a result file the shipped templates cannot render.

    This is the one place a result file re-enters the program from disk, so it is where
    a file from an older schema gets caught. Before 4.0 a build made ahead of the first
    tee time wrote `live: null` and `espn: null` on every golfer, and the pages of that
    era checked for it. The pages of this era do not -- there is no such file any more,
    so they read `DATA.live` at the top and would die on the null before `start()` was
    even defined, leaving an empty skeleton and a manifest describing a working page.

    Failing here instead costs one rebuild and says which one.
    """
    if result.get("live"):
        return
    raise SystemExit(
        f"this result file has no `live` block (schema_version "
        f"{result.get('schema_version', 'unknown')}), so it was built before ESPN "
        "published a field and carries no ESPN athlete id for anybody.\n"
        "Pages built from it could never score, and the current templates cannot render "
        "it at all. ESPN now posts a field about two days before the first round, so "
        "rebuilding fills that half in without touching the draw:\n"
        "    python build_competition.py --from-result <this file> --output <this file>")


def bundle(result, template_dir, out_dir, basename=None, keep_result=True, leagues_dir=None):
    """
    Build the page and the zip. Returns the paths written.

    The zip carries the result JSON alongside the page even though the page already
    contains it, because the page is for reading and the JSON is for re-running: it is
    the input to a rebuild, and digging it back out of an HTML file is nobody's idea of
    a good afternoon.

    The art is the one thing the page has and that JSON does not, and the copy written
    to the zip is the one that was handed in -- untouched, slug and all.
    """
    require_scoreable(result)

    index = os.path.join(template_dir, "index.html")
    if not os.path.exists(index):
        raise SystemExit(f"template {template_dir} has no index.html")

    with open(index, encoding="utf-8") as f:
        markup = f.read()
    if JSON_MARKER not in markup:
        raise SystemExit(
            f"{index} does not contain the data marker {JSON_MARKER}.\n"
            'A template must hold  <script id="competition-data" type="application/json">'
            f'{JSON_MARKER}</script>\n'
            "so the result can be baked in. See the module docstring for the full contract."
        )

    report = {"inlined": [], "missing": [], "art": []}
    markup = inline_assets(markup, template_dir, report)
    markup = substitute_tokens(markup, result)
    # The art marker is replaced whenever it is there, including with `{}`. Leaving it
    # in place would hand the page a JavaScript comment where its JSON should be, and
    # the masthead would take the whole script down with it.
    if ART_MARKER in markup:
        markup = markup.replace(ART_MARKER,
                                json_for_script(league_art(result, leagues_dir, report)))
    markup = markup.replace(JSON_MARKER, json_for_script(result))

    basename = basename or default_basename(result)
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"{basename}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(markup)

    written = [html_path]
    zip_path = os.path.join(out_dir, f"{basename}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(html_path, "index.html")
        if keep_result:
            z.writestr("result.json", json.dumps(result, indent=2, ensure_ascii=False))
        z.writestr("MANIFEST.txt", manifest(result, report))
    written.append(zip_path)

    if report["missing"]:
        print(f"!! {len(report['missing'])} template asset(s) not found and left as-is: "
              + ", ".join(sorted(set(report['missing']))))
    return written, report


def default_basename(result):
    slug = league_mod.slugify
    return f"{slug(result['league']['league_name'])}-{slug(result['tournament']['name'])}"


def manifest(result, report):
    """
    What is in the zip, and -- the part worth getting right -- what the page does.

    The manifest travels with the page, and the thing it has to be right about is when
    the page starts ranking. That is a question about the clock rather than about this
    build: a competition drawn the night before is complete, and it shows the draw
    until the first tee time and then ranks, by itself, with nothing re-run.
    """
    k = result["sources"]["kalshi"]
    g = result["grouping"]
    live = result.get("live") or {}
    started = bool((result["sources"].get("espn") or {}).get("started_at_build"))
    scoring = [f"live scoring    {live.get('espn_leaderboard_url')}"
               + ("" if started else "  (not started yet -- see below)")]
    # The one thing the page holds that the result.json beside it does not, and most of
    # the page's weight. Named file by file, so a heavy export says where the bytes went.
    art = [f"league art      {name}  {path}  ({size // 1024} KB)"
           for name, path, size in report.get("art") or []]
    behaviour = ([
        "ESPN is the only thing it fetches, and it fetches it for the scores.",
    ] if started else [
        "ESPN is the only thing it fetches. Play had not started when this was",
        "built, so the page opens on the draw -- every roster, every price, no",
        "ranking. It polls from the moment it opens and starts ranking on its own",
        "at the first tee time. Nothing needs rebuilding and nobody needs the link",
        "again.",
    ])
    return "\n".join([
        f"{result['league']['league_name']} -- {result['tournament']['name']}",
        f"{k['market_label']} ({k['odds_type']}), priced off the {k['price_mode']}",
        "",
        f"competition_id  {result['competition_id']}",
        f"built           {result['generated_at']}",
        f"tool            {result['generator']['tool']} @ {result['generator']['git_commit']}",
        f"seed            {result['generator']['seed']}",
        "",
        f"teams           {result['league']['team_count']}",
        f"golfers         {len(result['golfers'])} priced, {g['grouped_golfers']} grouped",
        f"partition       {g['summary']}",
        "",
        f"odds source     {k['markets_endpoint']}",
        f"                captured {result['odds_snapshot']['captured_at']}, "
        f"book sums to {result['odds_snapshot']['raw_book_sum']}",
        *scoring,
        "",
        "index.html is self-contained: open it from disk, no server needed.",
        *behaviour,
        "The odds above are baked in as of the capture time -- Kalshi's API",
        "allowlists request origins and returns 403 to a browser, so no page",
        "can read them. Nor is there a newer reading to be had: the odds are",
        "read once, when the groups are drawn, and that is the competition.",
        "",
        f"assets inlined  {len(report['inlined'])}",
        *art,
        f"bundled         {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
    ])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", required=True, help="result JSON from build_competition.py")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE, help=f"template directory (default {DEFAULT_TEMPLATE})")
    ap.add_argument("--out", default="dist", help="where to write the bundle (default dist/)")
    ap.add_argument("--name", help="basename for the .html and .zip")
    ap.add_argument("--no-result-in-zip", action="store_true")
    ap.add_argument("--clean", action="store_true", help="empty the output directory first")
    ap.add_argument("--leagues-dir", default=league_mod.LEAGUES_DIR,
                    help="where the art slug in the result file resolves: <leagues-dir>/<slug>/"
                         f"logo.png and banner.png (default {league_mod.LEAGUES_DIR})")
    args = ap.parse_args(argv)

    with open(args.result, encoding="utf-8") as f:
        result = json.load(f)

    if args.clean and os.path.isdir(args.out):
        shutil.rmtree(args.out)

    written, report = bundle(result, args.template, args.out, args.name,
                             keep_result=not args.no_result_in_zip,
                             leagues_dir=args.leagues_dir)
    for path in written:
        print(f"{path}  ({os.path.getsize(path) // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Render the bundled page in a real browser.

The parity test proves the rules agree. This proves the page built from them actually
works: that it opens from a file:// URL with no server, that the only host it touches
is ESPN -- odds are baked in and never fetched, because Kalshi 403s a browser -- and
that the standings it draws are the ones standings.py computes.

Skipped without Playwright and a browser. Chromium is pre-installed in this
environment; elsewhere, `pip install playwright && playwright install chromium`.
"""

import glob
import json
import os
import re

import pytest

import bundle_frontend as bundler
import espn_leaderboard
import standings

playwright_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESPN_EVENT_ID = "401811960"
LEADERBOARD_GLOB = "**/site.web.api.espn.com/**"


def _chromium_path():
    """
    A browser Playwright did not download itself.

    Environments that pre-install Chromium often pin a different build number than the
    Playwright package expects, and the default launch then fails on a path that does
    not exist. Point at whatever is actually on disk before giving up.

    Two rules, both learned by getting them wrong. Candidates are filtered down to a
    file that is actually EXECUTABLE, because `pw-browsers` holds a directory per build
    (`chromium-1194`, `chromium_headless_shell-1194`) alongside the binary, and handing
    Playwright a directory fails exactly like handing it nothing -- the suite then
    reports "no chromium available" on a machine carrying three of them, which is the
    worst way for a browser test to be wrong: it never runs and never says so. And the
    per-build directories are searched directly, rather than trusting `chromium` to be
    the binary; here it happens to be a symlink to one, but that is this image's
    convention and not a guarantee.
    """
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH"), "/opt/pw-browsers"]
    candidates = [os.environ.get("CHROMIUM_PATH")]
    for root in roots:
        if root:
            candidates += sorted(glob.glob(os.path.join(root, "chromium*", "chrome-linux", "chrome")))
            candidates += sorted(glob.glob(os.path.join(root, "chromium*", "**", "headless_shell"),
                                           recursive=True))
    candidates += ["/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception:                            # noqa: BLE001 -- retried below
            path = _chromium_path()
            if not path:
                pytest.skip("no chromium available")
            try:
                b = p.chromium.launch(executable_path=path)
            except Exception as exc:                 # noqa: BLE001
                pytest.skip(f"no chromium available: {exc}")
        yield b
        b.close()


@pytest.fixture
def competition(espn_final_payload, tmp_path):
    """
    A four-team league over the real finished Rocket Classic field.

    Built here rather than by build_competition.py because that needs a live Kalshi
    pull, and the point of this test is the page rather than the pull. The shape is
    the shape build_competition.py emits -- test_build_competition.py holds that to
    account separately.
    """
    from test_build_competition import golfer_name, live_stage, make_result

    _, players = espn_leaderboard.parse_leaderboard(espn_final_payload)
    names = [golfer_name(i) for i in range(len(players))]
    result = make_result(n_teams=4, n_golfers=len(players), espn=live_stage(names))

    # Re-label the synthetic field with the real one, keeping the odds and the deal.
    # Round-robin by rank, so team 0 holds the winner and the leaderboard order of the
    # teams is known in advance.
    #
    # The athlete id is the point. This fixture used to null it deliberately, to force
    # the page to redo the name join at runtime; there is no runtime name join any more,
    # and the baked id is the only key the page has. So bake the REAL one.
    ordered = sorted(result["golfers"], key=lambda g: -g["odds"]["raw"])
    for golfer, player in zip(ordered, players):
        golfer["name"] = player["name"]
        golfer["espn"] = {"athlete_id": player["athlete_id"], "display_name": player["name"],
                          "headshot": player["headshot"], "country": player["country"],
                          "match": "exact", "in_field": True}
    by_id = {g["golfer_id"]: g for g in result["golfers"]}
    for team in result["teams"]:
        team["golfer_names"] = [by_id[gid]["name"] for gid in team["golfer_ids"]]

    result["sources"]["espn"]["event_id"] = ESPN_EVENT_ID
    url = espn_leaderboard.leaderboard_url(ESPN_EVENT_ID)
    result["sources"]["espn"]["leaderboard_endpoint"] = url
    result["live"]["espn_leaderboard_url"] = url
    result["live"]["espn_event_id"] = ESPN_EVENT_ID
    result["tournament"]["name"] = "Rocket Classic"

    paths, _ = bundler.bundle(result, bundler.DEFAULT_TEMPLATE, str(tmp_path / "dist"))
    return {"result": result, "html": paths[0], "players": players}


@pytest.fixture
def page(browser, competition, espn_final_payload):
    ctx = browser.new_context()
    seen = []
    ctx.on("request", lambda r: seen.append(r.url))

    def serve_espn(route):
        route.fulfill(status=200, content_type="application/json",
                      headers={"access-control-allow-origin": "*"},
                      body=json.dumps(espn_final_payload))

    ctx.route(LEADERBOARD_GLOB, serve_espn)
    p = ctx.new_page()
    errors = []
    p.on("pageerror", lambda e: errors.append(str(e)))
    p.goto("file://" + competition["html"])
    p.wait_for_selector("#standings article.team", timeout=15000)
    yield {"page": p, "requests": seen, "errors": errors}
    ctx.close()


# ---------------------------------------------------------------------------

def test_the_page_renders_with_no_javascript_errors(page):
    assert page["errors"] == []


def test_the_only_host_it_touches_is_espn(page, competition):
    """
    A self-contained page. The file itself, and ESPN. Anything else means an asset did
    not get inlined, and the bundle stops working the moment it leaves this machine.
    """
    external = [u for u in page["requests"] if not u.startswith(("file://", "data:"))]
    assert external, "it should be polling ESPN"
    assert all("espn.com" in u for u in external), external


def test_it_draws_one_card_per_team(page, competition):
    assert page["page"].locator("#standings article.team").count() == len(
        competition["result"]["teams"])


def test_the_order_on_screen_is_the_order_the_rule_computes(page, competition):
    """The page and standings.py must agree, on the real field, through the bundler."""
    result = competition["result"]
    by_team = {}
    for g in result["golfers"]:
        by_team.setdefault(g["team_id"], []).append({"golfer_id": g["golfer_id"], "name": g["name"]})
    teams = [{"team_id": t["team_id"], "golfers": by_team.get(t["team_id"], [])}
             for t in result["teams"]]

    expected = standings.compute(teams, standings.index_players(competition["players"]))
    names = {t["team_id"]: t["team_name"] for t in result["teams"]}

    on_screen = page["page"].locator("#standings article.team header .names strong").all_text_contents()
    assert on_screen == [names[r["team_id"]] for r in expected]

    positions = page["page"].locator("#standings article.team header .pos").all_text_contents()
    assert positions == [r["position"] for r in expected]


def test_the_leader_card_is_marked(page):
    assert page["page"].locator("#standings article.team").first.get_attribute("class") == "team leader"


def test_every_golfer_on_a_card_is_shown(page, competition):
    rows = page["page"].locator("#standings article.team table.golfers tbody tr").count()
    assert rows == sum(t["golfer_count"] for t in competition["result"]["teams"])


def test_cut_golfers_are_marked_out(page, competition):
    """74 of the 147 missed the cut, and the page has to be able to say so."""
    cut = [p for p in competition["players"] if p["position_number"] is None]
    assert page["page"].locator("#standings article.team table.golfers tbody tr.out").count() == len(cut)


def test_the_snapshot_odds_are_stated_with_their_capture_time(page, competition):
    note = page["page"].locator("#odds-note").text_content()
    assert "Odds at creation" in note
    assert competition["result"]["odds_snapshot"]["captured_at"] in note


def test_it_never_asks_kalshi_for_anything(page):
    """
    The simplification, held to account. Kalshi 403s every browser origin, so a page
    that requests it can only ever show an empty panel; the odds are baked in instead
    and the page says so rather than promising a number that cannot arrive.
    """
    remote = [u for u in page["requests"] if not u.startswith(("file://", "data:"))]
    assert not [u for u in remote if "kalshi" in u.lower()], remote
    note = page["page"].locator("#odds-note").text_content()
    assert "Odds at creation" in note
    assert "no odds are ever fetched here" in note
    assert "Live odds" not in note


def test_a_rebuilt_page_shows_the_odds_moving_with_no_network_at_all(browser, competition,
                                                                     espn_final_payload, tmp_path):
    """
    The only way prices move on this page: somebody re-ran the build with
    --refresh-odds and re-sent it. That re-read is baked in exactly as the original
    snapshot is, so it needs no network -- and it must be presented as movement since
    the draw rather than as a second column of levels, because the two columns are on
    different scales and would read as a jump when nothing had moved.
    """
    result = json.loads(json.dumps(competition["result"]))
    for i, golfer in enumerate(result["golfers"]):
        # A third of the field drifts up by a cent, the rest does not move at all.
        golfer["odds"]["current"] = round(golfer["odds"]["raw"] + (0.01 if i % 3 == 0 else 0), 4)
    result["odds_snapshot"]["refreshed"] = {
        "at": "2026-08-06T12:00:00+00:00", "price_mode": "ask", "field_size": len(result["golfers"]),
        "raw_book_sum": 1.31, "matched": len(result["golfers"]),
        "no_longer_priced": [], "priced_since_the_draw": ["Monday Qualifier"],
    }
    paths, _ = bundler.bundle(result, bundler.DEFAULT_TEMPLATE, str(tmp_path / "refreshed"))

    ctx = browser.new_context(viewport={"width": 900, "height": 800})
    seen = []
    ctx.on("request", lambda r: seen.append(r.url))
    ctx.route(LEADERBOARD_GLOB, lambda route: route.fulfill(
        status=200, content_type="application/json",
        headers={"access-control-allow-origin": "*"}, body=json.dumps(espn_final_payload)))
    p = ctx.new_page()
    p.goto("file://" + paths[0])
    p.wait_for_selector("#standings article.team", timeout=15000)
    p.wait_for_function("document.querySelector('td.gmove').textContent !== ''", timeout=15000)

    note = p.locator("#odds-note").text_content()
    assert "re-read" in note and "1.310" in note
    assert "in nobody" in note and "Monday Qualifier" in note

    cells = [c for c in p.locator("td.gmove").all_text_contents() if c]
    assert any(c.startswith("↑ +1.0") for c in cells), cells[:5]
    assert any(c == "→" for c in cells), "an unmoved golfer is not an arrow up"
    assert all("espn.com" in u for u in seen if not u.startswith(("file://", "data:")))
    ctx.close()


def test_the_header_names_the_tournament_and_the_market(page, competition):
    text = page["page"].locator("header.page").text_content()
    assert competition["result"]["tournament"]["name"] in text
    assert competition["result"]["sources"]["kalshi"]["market_label"] in text
    assert "PROVEN OPTIMAL" in text


def test_it_refuses_a_leaderboard_for_a_different_tournament(browser, competition, espn_final_payload):
    """
    The failure worth guarding: ESPN's leaderboard endpoint answers about whatever it
    thinks is current, so next week's tournament arrives on the same URL. Scoring this
    league against it would look completely normal and be completely wrong.
    """
    other = json.loads(json.dumps(espn_final_payload))
    other["events"][0]["id"] = "401811961"
    other["events"][0]["name"] = "Wyndham Championship"

    ctx = browser.new_context()
    ctx.route(LEADERBOARD_GLOB, lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(other)))
    p = ctx.new_page()
    p.goto("file://" + competition["html"])
    p.wait_for_selector("#standings p", timeout=15000)
    text = p.locator("#standings").text_content()
    assert "expected " + ESPN_EVENT_ID in text

    # The rosters and the odds are baked in and stay on screen -- the page is still
    # worth looking at. What must not survive is a POSITION: every one of those would
    # have come from the wrong tournament.
    assert p.locator("#standings article.team").count() == len(competition["result"]["teams"])
    assert set(p.locator("#standings article.team header .pos").all_text_contents()) == {"—"}
    assert set(p.locator("#standings td.gpos").all_text_contents()) == {"—"}
    ctx.close()


def test_it_survives_espn_being_down(browser, competition):
    ctx = browser.new_context()
    ctx.route(LEADERBOARD_GLOB, lambda route: route.abort())
    p = ctx.new_page()
    errors = []
    p.on("pageerror", lambda e: errors.append(str(e)))
    p.goto("file://" + competition["html"])
    p.wait_for_selector("#standings p", timeout=15000)
    assert "ESPN unavailable" in p.locator("#standings").text_content()
    assert errors == []
    # The page is still useful: it knows its own groups and its own odds.
    assert "Odds at creation" in p.locator("#odds-note").text_content()
    ctx.close()


@pytest.fixture
def groups_page(browser, competition, tmp_path):
    """
    The page a build makes before the first tee time: `live` is null.

    Bundled from the same competition with its scoring stripped, exactly as a groups
    build emits it -- no ESPN block on any golfer, and no `live` block at all. The
    context routes ESPN to a hard failure, so if the page asks for anything the request
    is both counted and refused.
    """
    result = json.loads(json.dumps(competition["result"]))
    result["build_mode"] = "groups"
    result["live"] = None
    result["sources"]["espn"]["field_size_at_build"] = 0
    result["sources"]["espn"]["match_report"] = None
    for golfer in result["golfers"]:
        golfer["espn"] = None
    paths, _ = bundler.bundle(result, bundler.DEFAULT_TEMPLATE, str(tmp_path / "groups"))

    ctx = browser.new_context()
    seen = []
    ctx.on("request", lambda r: seen.append(r.url))
    # Anything over the wire fails hard. file:// must still load, so this is
    # scoped by scheme rather than by a glob that would swallow the navigation.
    ctx.route(re.compile(r"^https?://"), lambda route: route.abort())
    p = ctx.new_page()
    errors = []
    p.on("pageerror", lambda e: errors.append(str(e)))
    p.goto("file://" + paths[0])
    p.wait_for_selector("#standings article.team", timeout=15000)
    yield {"page": p, "errors": errors, "requests": seen, "result": result}
    ctx.close()


def test_a_groups_page_makes_no_request_at_all(groups_page):
    """
    The claim the whole split rests on, and only a browser can check it. Not "it
    degrades gracefully when the fetch fails" -- there is no fetch. `live` is null,
    which is the build saying there was no field to score against, and a page that
    polled anyway would be asking a question whose answer it could not use.
    """
    external = [u for u in groups_page["requests"] if not u.startswith(("file://", "data:"))]
    assert external == []
    assert groups_page["errors"] == []


def test_a_groups_page_calls_itself_the_groups(groups_page):
    """
    A heading that says Standings over a board that ranks nobody reads as a scoreboard
    that is broken, rather than as the thing that exists before a tournament starts.
    """
    page = groups_page["page"]
    assert page.locator("#standings-heading").text_content() == "Groups"
    text = page.locator("#standings").text_content()
    assert "had not started when this page was made" in text
    assert "rebuilt page" in text, "it has to say how to get one that scores"
    assert "nothing is fetched by this page" in page.locator("#sources").text_content()


def test_a_groups_page_still_shows_every_roster(groups_page):
    """
    ESPN publishes no competitors until play starts, and everything the pool actually
    cares about at that point -- who holds whom, what each golfer was worth, that the
    draw came out even -- was decided on Wednesday and is baked into the file. An empty
    div throws all of it away.
    """
    page, result = groups_page["page"], groups_page["result"]
    assert page.locator("#standings article.team").count() == len(result["teams"])
    assert page.locator("#standings article.team table.golfers tbody tr").count() == sum(
        t["golfer_count"] for t in result["teams"])

    text = page.locator("#standings").text_content()
    for team in result["teams"]:
        assert team["team_name"] in text
    assert "%" in text, "the odds at creation are the point of the groups board"


def test_a_groups_page_ranks_nobody(groups_page):
    """
    Nothing to rank on, so nothing is ranked. Running the standings rule over an empty
    leaderboard would order the teams by roster size -- every golfer tier 2, the longer
    vector winning on padding -- and print it as a leaderboard, which is a worse answer
    than saying the tournament has not started.
    """
    page = groups_page["page"]
    assert set(page.locator("#standings article.team header .pos").all_text_contents()) == {"\u2014"}
    assert page.locator("#standings article.team.leader").count() == 0
    assert page.locator("#standings tr.out").count() == 0


def test_a_live_page_whose_field_is_empty_still_refuses_to_rank(browser, competition):
    """
    The other empty board, and it still exists. A live page polls, and ESPN can answer
    with a payload carrying no competitors -- between rounds, or on a bad response. The
    never-rank-an-empty-board guard is independent of the build mode and has to survive.
    """
    ctx = browser.new_context()
    empty = {"events": [{"id": ESPN_EVENT_ID, "name": "Rocket Classic",
                         "status": {"type": {"state": "pre"}},
                         "competitions": [{"status": {"period": 0}, "competitors": []}]}]}
    ctx.route(LEADERBOARD_GLOB, lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(empty)))
    p = ctx.new_page()
    errors = []
    p.on("pageerror", lambda e: errors.append(str(e)))
    p.goto("file://" + competition["html"])
    p.wait_for_selector("#standings article.team", timeout=15000)

    assert "Waiting for ESPN" in p.locator("#standings").text_content()
    assert p.locator("#standings article.team.leader").count() == 0
    assert set(p.locator("#standings article.team header .pos").all_text_contents()) == {"\u2014"}
    assert errors == []
    ctx.close()

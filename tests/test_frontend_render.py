"""
Render the bundled page in a real browser.

The parity test proves the rules agree. This proves the page built from them actually
works: that it opens from a file:// URL with no server, that the only host it touches
is ESPN, that the standings it draws are the ones standings.py computes, and that when
live odds are unavailable it says so instead of showing a blank panel.

Skipped without Playwright and a browser. Chromium is pre-installed in this
environment; elsewhere, `pip install playwright && playwright install chromium`.
"""

import json
import os

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
    """
    for candidate in (os.environ.get("CHROMIUM_PATH"),
                      os.path.join(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""), "chromium"),
                      "/opt/pw-browsers/chromium"):
        if candidate and os.path.exists(candidate):
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
    from test_build_competition import make_result

    _, players = espn_leaderboard.parse_leaderboard(espn_final_payload)
    result = make_result(n_teams=4, n_golfers=len(players))

    # Re-label the synthetic field with the real one, keeping the odds and the deal.
    # Round-robin by rank, so team 0 holds the winner and the leaderboard order of the
    # teams is known in advance.
    ordered = sorted(result["golfers"], key=lambda g: -g["odds"]["raw"])
    for golfer, player in zip(ordered, players):
        golfer["name"] = player["name"]
        golfer["espn"]["athlete_id"] = None          # force the runtime name join
    by_id = {g["golfer_id"]: g for g in result["golfers"]}
    for team in result["teams"]:
        team["golfer_names"] = [by_id[gid]["name"] for gid in team["golfer_ids"]]

    result["sources"]["espn"]["event_id"] = ESPN_EVENT_ID
    url = espn_leaderboard.leaderboard_url(ESPN_EVENT_ID)
    result["sources"]["espn"]["leaderboard_endpoint"] = url
    result["live"]["espn_leaderboard_url"] = url
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


def test_it_says_why_live_odds_are_missing_rather_than_showing_nothing(page):
    note = page["page"].locator("#odds-note").text_content()
    assert "Live odds unavailable" in note
    assert "allowlists request origins" in note


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
    assert p.locator("#standings article.team").count() == 0
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


def test_a_pre_tournament_field_says_so_rather_than_showing_an_empty_board(browser, competition):
    ctx = browser.new_context()
    empty = {"events": [{"id": ESPN_EVENT_ID, "name": "Rocket Classic",
                         "status": {"type": {"state": "pre"}},
                         "competitions": [{"status": {"period": 0}, "competitors": []}]}]}
    ctx.route(LEADERBOARD_GLOB, lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(empty)))
    p = ctx.new_page()
    p.goto("file://" + competition["html"])
    p.wait_for_selector("#standings p", timeout=15000)
    assert "not posted yet" in p.locator("#standings").text_content()
    ctx.close()

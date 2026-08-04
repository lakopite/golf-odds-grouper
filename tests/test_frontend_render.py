"""
Render the bundled reference page in a real browser.

`frontend/template/` is the plain implementation: no design, no views, one long list
of cards. It exists to prove the contract in docs/FRONTEND-SPEC.md rather than to be
the page anybody is handed, so this suite is where the contract itself gets checked --
that the bundle opens from a file:// URL with no server, that the only host it touches
is ESPN, and that the standings it draws are the ones standings.py computes.

The designed page ships by default and has its own suite next door,
tests/test_scoreboard_render.py, making the same claims against its own markup. The
browser, the competition and the ESPN stub are shared from conftest.py.

Skipped without Playwright and a browser. Chromium is pre-installed in this
environment; elsewhere, `pip install playwright && playwright install chromium`.
"""

import json
import os
import re

import pytest

import bundle_frontend as bundler
import standings
from conftest import ESPN_EVENT_ID, LEADERBOARD_GLOB

pytest.importorskip("playwright.sync_api", reason="playwright not installed")

TEMPLATE = bundler.REFERENCE_TEMPLATE


@pytest.fixture
def competition(rocket_classic, tmp_path):
    paths, _ = bundler.bundle(rocket_classic["result"], TEMPLATE, str(tmp_path / "dist"))
    return {"result": rocket_classic["result"], "html": paths[0],
            "players": rocket_classic["players"]}


@pytest.fixture
def page(browser, competition, serve_espn):
    ctx = browser.new_context()
    seen = []
    ctx.on("request", lambda r: seen.append(r.url))
    ctx.route(LEADERBOARD_GLOB, serve_espn())
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


def test_the_reference_page_has_no_column_a_price_could_move_in(page, competition):
    """
    A rebuild used to be able to re-read Kalshi and bake a second price in beside the
    drawn one, which this page rendered as an arrow. Two prices for one golfer -- one
    he was dealt on, one he was not -- is the shape of a draw being adjusted after the
    fact, so there is one price per golfer now and no cell that could hold another.
    """
    p = page["page"]
    assert p.locator("td.gmove").count() == 0
    rows = p.locator("#standings article.team table.golfers tbody tr").first
    assert rows.locator("td").count() == 5


def test_a_result_carrying_a_second_reading_renders_no_movement_anyway(browser, competition,
                                                                       serve_espn, tmp_path):
    """A file built by an older tool, or hand-edited. The extra fields are ignored."""
    result = json.loads(json.dumps(competition["result"]))
    for golfer in result["golfers"]:
        golfer["odds"]["current"] = round(golfer["odds"]["raw"] + 0.01, 4)
    result["odds_snapshot"]["refreshed"] = {
        "at": "2026-08-06T12:00:00+00:00", "price_mode": "ask",
        "field_size": len(result["golfers"]), "raw_book_sum": 1.31,
        "matched": len(result["golfers"]),
        "no_longer_priced": [], "priced_since_the_draw": ["Monday Qualifier"],
    }
    paths, _ = bundler.bundle(result, TEMPLATE, str(tmp_path / "stale-schema"))

    ctx = browser.new_context(viewport={"width": 900, "height": 800})
    ctx.route(LEADERBOARD_GLOB, serve_espn())
    p = ctx.new_page()
    p.goto("file://" + paths[0])
    p.wait_for_selector("#standings article.team", timeout=15000)

    assert p.locator("td.gmove").count() == 0
    note = p.locator("#odds-note").text_content()
    assert "re-read" not in note
    assert "1.310" not in note
    assert "Monday Qualifier" not in note
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
def groups_page(browser, competition, espn_not_started_payload, tmp_path):
    """
    The page for the first day or two of its life: built, complete, and answered by an
    ESPN that has posted the field and has no positions in it yet.

    Requests are counted, because this page is expected to poll -- that poll is how it
    finds out play has started.
    """
    ctx = browser.new_context()
    seen = []
    ctx.on("request", lambda r: seen.append(r.url))
    ctx.route(LEADERBOARD_GLOB, lambda route: route.fulfill(
        status=200, content_type="application/json",
        headers={"access-control-allow-origin": "*"},
        body=json.dumps(espn_not_started_payload)))
    p = ctx.new_page()
    errors = []
    p.on("pageerror", lambda e: errors.append(str(e)))
    p.goto("file://" + competition["html"])
    p.wait_for_selector("#standings article.team", timeout=15000)
    yield {"page": p, "errors": errors, "requests": seen, "result": competition["result"]}
    ctx.close()


def test_the_page_polls_from_the_moment_it_opens_even_days_early(groups_page):
    """
    The mechanism the whole simplification runs on, and only a browser can check it. The
    page has nothing to show but the draw and it polls anyway, because that poll is how
    it learns the tournament has started. A page that waited for a reason to poll would
    sit on the draw all weekend.
    """
    external = [u for u in groups_page["requests"] if not u.startswith(("file://", "data:"))]
    assert any("site.web.api.espn.com" in u for u in external), "it has to be watching"
    assert groups_page["errors"] == []


def test_it_still_opens_from_disk_with_no_network_at_all(browser, competition):
    """
    The property a null `live` block used to guarantee, now that every page has one: a
    failed poll costs the ranking and nothing else. The draw is baked in, so the page
    opened on a plane is the page opened anywhere.
    """
    ctx = browser.new_context()
    # Anything over the wire fails hard. file:// must still load, so this is
    # scoped by scheme rather than by a glob that would swallow the navigation.
    ctx.route(re.compile(r"^https?://"), lambda route: route.abort())
    p = ctx.new_page()
    errors = []
    p.on("pageerror", lambda e: errors.append(str(e)))
    p.goto("file://" + competition["html"])
    p.wait_for_selector("#standings article.team", timeout=15000)

    assert p.locator("#standings article.team").count() == len(competition["result"]["teams"])
    assert set(p.locator("#standings article.team header .pos").all_text_contents()) == {"\u2014"}
    assert errors == []
    ctx.close()


def test_a_page_before_the_first_tee_time_calls_itself_the_groups(groups_page):
    """
    A heading that says Standings over a board that ranks nobody reads as a scoreboard
    that is broken, rather than as the thing that exists before a tournament starts.
    """
    page = groups_page["page"]
    assert page.locator("#standings-heading").text_content() == "Groups"
    text = page.locator("#standings").text_content()
    assert "The tournament has not started" in text
    assert "ranking begins on its own" in text, "it must not imply a second build"
    assert "scores polled from" in page.locator("#sources").text_content()


def test_a_page_before_the_first_tee_time_still_shows_every_roster(groups_page):
    """
    ESPN publishes no positions until play starts, and everything the pool actually
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


def test_a_page_before_the_first_tee_time_ranks_nobody(groups_page):
    """
    The assertion the change created, and the most important one here.

    ESPN has answered with a COMPLETE field: every athlete id matches a golfer on
    somebody's team and every one of them resolves. Only the positions are missing. Rank
    on that and the standings rule falls through to tier 1 and orders every team by
    ESPN's pre-tournament sortOrder -- a full league table with a leader and tie-breaks,
    entirely invented, and much more convincing than an empty board.
    """
    page = groups_page["page"]
    assert set(page.locator("#standings article.team header .pos").all_text_contents()) == {"\u2014"}
    assert page.locator("#standings article.team.leader").count() == 0
    assert page.locator("#standings tr.out").count() == 0


def test_the_page_crosses_from_the_draw_to_the_scoreboard_on_its_own(
        browser, competition, espn_not_started_payload, espn_final_payload):
    """
    The contract this template exists to prove, and the whole justification for one
    build instead of two: the page changes over by itself, from the poll it was already
    making, with nobody rebuilding or reloading anything.
    """
    payload = {"body": json.dumps(espn_not_started_payload)}
    ctx = browser.new_context()
    ctx.route(LEADERBOARD_GLOB, lambda route: route.fulfill(
        status=200, content_type="application/json",
        headers={"access-control-allow-origin": "*"}, body=payload["body"]))
    p = ctx.new_page()
    errors = []
    p.on("pageerror", lambda e: errors.append(str(e)))
    p.goto("file://" + competition["html"])
    p.wait_for_selector("#standings article.team", timeout=15000)

    assert p.locator("#standings-heading").text_content() == "Groups"
    assert set(p.locator("#standings article.team header .pos").all_text_contents()) == {"\u2014"}

    payload["body"] = json.dumps(espn_final_payload)
    p.evaluate("poll()")

    p.wait_for_function(
        "document.getElementById('standings-heading').textContent === 'Standings'",
        timeout=15000)
    positions = p.locator("#standings article.team header .pos").all_text_contents()
    assert positions != ["\u2014"] * len(positions), "it has to actually rank"
    assert p.locator("#standings article.team.leader").count() == 1
    assert errors == []
    ctx.close()


def test_a_page_whose_field_comes_back_empty_still_refuses_to_rank(browser, competition):
    """
    The other empty board, and it still exists. ESPN can answer mid-tournament with a
    payload carrying no competitors -- between rounds, or on a bad response. The
    never-rank-an-empty-board guard is a separate thing from the not-started gate and
    has to survive on its own.
    """
    ctx = browser.new_context()
    empty = {"events": [{"id": ESPN_EVENT_ID, "name": "Rocket Classic",
                         "status": {"type": {"state": "in"}},
                         "competitions": [{"status": {"period": 2}, "competitors": []}]}]}
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

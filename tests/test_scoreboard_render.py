"""
Render the bundled scoreboard in a real browser.

`frontend/scoreboard/` is the designed page and the one `bundle_frontend.py` produces
by default, so this is the suite that says whether what the pool actually opens on a
Sunday works. It makes the same claims as the reference suite next door -- one host,
no Kalshi, the standings rule's answer and nobody else's, a groups sheet that fetches
nothing -- against the markup the design ships, plus the things only this page has:
two views, expandable groups, a status pill and a league masthead.

The rule itself is not retested here. `frontend/lib.js` is inlined verbatim into both
pages and `tests/test_frontend_parity.py` runs it against standings.py; what is checked
below is that this page renders that answer rather than a different one.

Skipped without Playwright and a browser. Chromium is pre-installed in this
environment; elsewhere, `pip install playwright && playwright install chromium`.
"""

import json
import re

import pytest

import bundle_frontend as bundler
import standings
from conftest import ESPN_EVENT_ID, LEADERBOARD_GLOB

pytest.importorskip("playwright.sync_api", reason="playwright not installed")

TEMPLATE = bundler.DEFAULT_TEMPLATE

# A 1x1 PNG. Enough to prove the masthead wires an image through; a real crest would
# only make the fixture bigger.
PIXEL = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
         "DUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def open_page(browser, html, route=None, viewport=None):
    """A context, a page, and the two lists every test here wants: requests and errors."""
    ctx = browser.new_context(viewport=viewport or {"width": 1280, "height": 900})
    seen = []
    ctx.on("request", lambda r: seen.append(r.url))
    if route is not None:
        ctx.route(LEADERBOARD_GLOB, route)
    p = ctx.new_page()
    errors = []
    p.on("pageerror", lambda e: errors.append(str(e)))
    p.goto("file://" + html)
    p.wait_for_selector("#board tbody.team", timeout=15000)
    return ctx, {"page": p, "requests": seen, "errors": errors}


@pytest.fixture
def competition(rocket_classic, tmp_path):
    paths, _ = bundler.bundle(rocket_classic["result"], TEMPLATE, str(tmp_path / "dist"))
    return {"result": rocket_classic["result"], "html": paths[0],
            "players": rocket_classic["players"]}


@pytest.fixture
def page(browser, competition, serve_espn):
    ctx, out = open_page(browser, competition["html"], serve_espn())
    yield out
    ctx.close()


# ---------------------------------------------------------------------------
# It is a self-contained page and it talks to exactly one host
# ---------------------------------------------------------------------------

def test_the_page_renders_with_no_javascript_errors(page):
    assert page["errors"] == []


def test_the_only_host_it_touches_is_espn(page):
    """
    The file itself, and ESPN. Anything else means an asset did not get inlined, and
    the bundle stops working the moment it leaves this machine -- which for a page whose
    whole point is opening from a USB stick is the failure that matters.
    """
    external = [u for u in page["requests"] if not u.startswith(("file://", "data:"))]
    assert external, "it should be polling ESPN"
    assert all("espn.com" in u for u in external), external


def test_it_never_asks_kalshi_for_anything(page):
    """
    Kalshi 403s every browser origin, so a page that requests it can only ever show an
    empty panel. The odds are baked in instead and the page says so rather than
    promising a number that cannot arrive.
    """
    remote = [u for u in page["requests"] if not u.startswith(("file://", "data:"))]
    assert not [u for u in remote if "kalshi" in u.lower()], remote

    page["page"].locator("#tab-odds").click()
    text = page["page"].locator("#odds-cards").text_content()
    assert "no odds are ever fetched here" in text
    assert "Live odds" not in text


def test_no_webfont_or_other_cdn_is_referenced(competition):
    """
    The design was drawn with Google Fonts. An absolute URL is left alone by the
    bundler, so it would survive into the export and be requested on every open --
    and fall back to something unplanned the first time the page is opened offline,
    which for this page is most of the time.
    """
    with open(competition["html"], encoding="utf-8") as f:
        markup = f.read()
    assert "fonts.googleapis.com" not in markup
    assert "fonts.gstatic.com" not in markup
    assert not re.search(r'<(?:link|script)[^>]+(?:href|src)=["\']https?://', markup)


# ---------------------------------------------------------------------------
# The standings it draws are the ones the rule computes
# ---------------------------------------------------------------------------

def test_it_draws_one_block_per_team(page, competition):
    assert page["page"].locator("#board tbody.team").count() == len(
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

    on_screen = page["page"].locator("#board tbody.team .tname").all_text_contents()
    assert on_screen == [names[r["team_id"]] for r in expected]

    positions = page["page"].locator("#board tbody.team td.c-rk").all_text_contents()
    assert positions == [r["position"] for r in expected]


def test_the_leader_is_marked_and_is_first(page):
    board = page["page"].locator("#board tbody.team")
    assert "is-leader" in (board.first.get_attribute("class") or "")
    # Ties for the lead are real and all of them are marked; nobody below them is.
    marked = page["page"].locator("#board tbody.team.is-leader").count()
    assert 1 <= marked <= board.count()


def test_every_golfer_a_team_holds_has_a_row(page, competition):
    rows = page["page"].locator("#board tr.golfer").count()
    assert rows == sum(t["golfer_count"] for t in competition["result"]["teams"])


def test_cut_golfers_are_listed_and_visibly_out(page, competition):
    """74 of the 147 missed the cut. A roster does not shrink; it greys."""
    cut = [p for p in competition["players"] if p["position_number"] is None]
    assert page["page"].locator("#board tr.golfer.out").count() == len(cut)
    # Dimming alone is not a marker on a page read across a room, so every one of them
    # also carries a word.
    tags = page["page"].locator("#board tr.golfer.out td.g-tag").all_text_contents()
    assert all(t.strip() for t in tags)


def test_the_rule_the_page_is_running_is_on_the_page(page):
    tiers = page["page"].locator("#tiers")
    assert tiers.is_visible()
    text = tiers.text_content()
    assert "padding" in text and "cut" in text


# ---------------------------------------------------------------------------
# The margin is the content
# ---------------------------------------------------------------------------

def test_the_golfer_that_decided_a_position_is_shown_without_expanding(page):
    """
    `decided_at` is the story of a close afternoon, and a group that is collapsed above
    it hides the answer to the only interesting question on the page. Every team the
    rule separated shows its group down to and including the golfer that did it.
    """
    marks = page["page"].locator("#board .decided-here")
    assert marks.count() > 0, "the fixture should separate at least one pair of teams"
    for i in range(marks.count()):
        assert marks.nth(i).is_visible()


def test_a_team_expands_to_its_whole_group(page, competition):
    board = page["page"].locator("#board tbody.team").first
    before = board.locator("tr.golfer:visible").count()
    total = board.locator("tr.golfer").count()
    assert before < total, "the fixture's groups should be longer than the preview"

    board.locator(".team-row").click()
    page["page"].wait_for_timeout(50)
    board = page["page"].locator("#board tbody.team").first
    assert board.locator("tr.golfer:visible").count() == total
    assert board.locator(".team-row").get_attribute("aria-expanded") == "true"


def test_a_keyboard_can_expand_a_team(page):
    board = page["page"].locator("#board tbody.team").first
    row = board.locator(".team-row")
    row.focus()
    row.press("Enter")
    page["page"].wait_for_timeout(50)
    assert page["page"].locator("#board tbody.team").first.locator(
        ".team-row").get_attribute("aria-expanded") == "true"
    # Rebuilding the board must not drop the keyboard on the floor.
    assert page["page"].evaluate(
        "() => document.activeElement.classList.contains('team-row')")


# ---------------------------------------------------------------------------
# Odds and the draw
# ---------------------------------------------------------------------------

def test_the_odds_view_states_the_snapshot_and_when_it_was_taken(page, competition):
    snap = competition["result"]["odds_snapshot"]
    page["page"].locator("#tab-odds").click()
    tiles = page["page"].locator("#odds-tiles").text_content()
    assert str(snap["field_size"]) in tiles
    assert str(snap["raw_book_sum"]) in tiles
    assert snap["price_mode"] in tiles
    assert "never moves" in tiles


def test_the_optimality_certificate_is_shown(page, competition):
    grouping = competition["result"]["grouping"]
    page["page"].locator("#tab-odds").click()
    text = page["page"].locator("#odds-cards").text_content()
    assert "Proven optimal" in text
    assert f"{grouping['delta_ticks']} tick" in text

    # A delta is a difference between two probabilities and arrives as a raw float, so
    # printing it verbatim gives 9.192866335723479e-05 in a 30px display face on the one
    # number that is supposed to read as "these groups are worth the same".
    assert f"{grouping['delta']:.6f}" in text
    assert not re.search(r"\d[eE][-+]\d", text), "no float noise in a display face"


def test_the_provenance_names_the_build(page, competition):
    result = competition["result"]
    text = page["page"].locator("footer.prov").text_content()
    assert result["competition_id"] in text
    assert result["generator"]["tool"] in text
    assert "PROVEN OPTIMAL" in text


def test_the_masthead_names_the_tournament(page, competition):
    text = page["page"].locator(".topbar").text_content()
    assert competition["result"]["tournament"]["name"] in text


def test_a_page_that_was_never_rebuilt_has_no_movement_column(page):
    """Not a blank column, not a spinner, not an apology. Absent."""
    assert not page["page"].locator("#board .group th.g-move").first.is_visible()
    assert page["page"].locator("#board td.g-move").first.text_content() == ""


def test_a_rebuilt_page_shows_the_odds_moving_with_no_network_at_all(browser, competition,
                                                                     serve_espn, tmp_path):
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
    paths, _ = bundler.bundle(result, TEMPLATE, str(tmp_path / "refreshed"))

    ctx, out = open_page(browser, paths[0], serve_espn())
    p = out["page"]
    p.wait_for_function("document.querySelector('td.g-move').textContent !== ''", timeout=15000)

    cells = [c for c in p.locator("#board td.g-move").all_text_contents() if c]
    assert any(c.startswith("↑ +1.0") for c in cells), cells[:5]
    assert any(c == "→" for c in cells), "an unmoved golfer is not an arrow up"

    p.locator("#tab-odds").click()
    text = p.locator("#odds-cards").text_content()
    assert "re-read" in text and "1.310" in text
    assert "not a feed" in text
    assert "Monday Qualifier" in text and "in nobody" in text

    assert all("espn.com" in u for u in out["requests"]
               if not u.startswith(("file://", "data:")))
    assert out["errors"] == []
    ctx.close()


# ---------------------------------------------------------------------------
# The league's own identity
# ---------------------------------------------------------------------------

def test_a_league_with_no_art_still_looks_finished(page):
    assert not page["page"].locator("#league-crest").is_visible()
    assert not page["page"].locator("#bannerwrap").is_visible()
    assert not page["page"].locator("#league-tagline").is_visible()
    # The badge is always there; only its contents change.
    assert page["page"].locator("#board tbody.team .mono.is-empty").count() > 0


def test_a_crest_a_banner_and_a_tagline_are_wired_through(browser, rocket_classic,
                                                          serve_espn, tmp_path):
    result = json.loads(json.dumps(rocket_classic["result"]))
    result["league"]["crest"] = PIXEL
    result["league"]["banner"] = PIXEL
    result["league"]["tagline"] = "10th Anniversary"
    paths, _ = bundler.bundle(result, TEMPLATE, str(tmp_path / "branded"))

    ctx, out = open_page(browser, paths[0], serve_espn())
    p = out["page"]
    assert p.locator("#league-crest").is_visible()
    assert p.locator("#league-crest").get_attribute("src") == PIXEL
    assert p.locator("#bannerwrap").is_visible()
    assert p.locator("#league-tagline").text_content() == "10th Anniversary"
    assert out["errors"] == []
    ctx.close()


# ---------------------------------------------------------------------------
# Every way there is nothing to rank
# ---------------------------------------------------------------------------

def test_it_refuses_a_leaderboard_for_a_different_tournament(browser, competition,
                                                             espn_final_payload):
    """
    ESPN's leaderboard endpoint answers about whatever it thinks is current, so next
    week's tournament arrives on the same URL. Scoring this league against it would look
    completely normal and be completely wrong.
    """
    other = json.loads(json.dumps(espn_final_payload))
    other["events"][0]["id"] = "401811961"
    other["events"][0]["name"] = "Wyndham Championship"

    ctx, out = open_page(browser, competition["html"], lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(other)))
    p = out["page"]
    assert "expected " + ESPN_EVENT_ID in p.locator("#not-started").text_content()

    # The rosters and the odds are baked in and stay on screen -- the page is still
    # worth looking at. What must not survive is a POSITION: every one of those would
    # have come from the wrong tournament.
    assert p.locator("#board tbody.team").count() == len(competition["result"]["teams"])
    assert set(p.locator("#board tbody.team td.c-rk").all_text_contents()) == {"—"}
    assert set(p.locator("#board td.g-pos").all_text_contents()) == {"—"}
    assert out["errors"] == []
    ctx.close()


def test_it_survives_espn_being_down(browser, competition):
    ctx, out = open_page(browser, competition["html"], lambda route: route.abort())
    p = out["page"]
    assert p.locator("#status-label").text_content() == "ESPN unreachable"
    assert "is-down" in (p.locator("#status-pill").get_attribute("class") or "")
    assert "ESPN unavailable" in p.locator("#not-started").text_content()
    assert out["errors"] == []

    # The page is still useful: it knows its own groups and its own odds.
    assert p.locator("#board tbody.team").count() == len(competition["result"]["teams"])
    p.locator("#tab-odds").click()
    assert "Captured" in p.locator("#odds-tiles").text_content()
    ctx.close()


def test_a_live_page_whose_field_is_empty_still_refuses_to_rank(browser, competition):
    """
    A live page polls, and ESPN can answer with a payload carrying no competitors --
    between rounds, or on a bad response. Running the standings rule over an empty
    leaderboard puts every golfer in tier 2, which orders the teams by roster size and
    prints it as a leaderboard. The guard is independent of the build mode.
    """
    empty = {"events": [{"id": ESPN_EVENT_ID, "name": "Rocket Classic",
                         "status": {"type": {"state": "pre"}},
                         "competitions": [{"status": {"period": 0}, "competitors": []}]}]}
    ctx, out = open_page(browser, competition["html"], lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(empty)))
    p = out["page"]
    assert "Waiting for ESPN" in p.locator("#not-started").text_content()
    # ESPN answered, so the page is working -- but a green "Live" dot over a board with
    # no positions on it reads as broken rather than early.
    assert p.locator("#status-label").text_content() == "Not started"
    assert "is-live" not in (p.locator("#status-pill").get_attribute("class") or "")
    assert p.locator("#board tbody.team.is-leader").count() == 0
    assert set(p.locator("#board tbody.team td.c-rk").all_text_contents()) == {"—"}
    assert p.locator("#board tr.golfer.out").count() == 0
    assert out["errors"] == []
    ctx.close()


# ---------------------------------------------------------------------------
# The groups sheet -- half of all the pages this repo produces
# ---------------------------------------------------------------------------

@pytest.fixture
def groups_page(browser, groups_result, tmp_path):
    paths, _ = bundler.bundle(groups_result, TEMPLATE, str(tmp_path / "groups"))
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    seen = []
    ctx.on("request", lambda r: seen.append(r.url))
    # Anything over the wire fails hard. file:// must still load, so this is scoped by
    # scheme rather than by a glob that would swallow the navigation.
    ctx.route(re.compile(r"^https?://"), lambda route: route.abort())
    p = ctx.new_page()
    errors = []
    p.on("pageerror", lambda e: errors.append(str(e)))
    p.goto("file://" + paths[0])
    p.wait_for_selector("#board tbody.team", timeout=15000)
    yield {"page": p, "errors": errors, "requests": seen, "result": groups_result}
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
    p = groups_page["page"]
    assert p.locator("#standings-heading").text_content() == "Groups"
    assert p.locator("#tab-standings").text_content().strip() == "Groups"
    assert p.locator("#status-label").text_content() == "Not started"
    text = p.locator("#not-started").text_content()
    assert "had not started when this page was made" in text
    assert "rebuilt page" in text, "it has to say how to get one that scores"

    p.locator("#tab-odds").click()
    assert "nothing is fetched by this page" in p.locator("#odds-cards").text_content()


def test_a_groups_page_still_shows_every_roster(groups_page):
    """
    Everything the pool cares about before the first tee time -- who holds whom, what
    each golfer was worth, that the draw came out even -- was decided on Wednesday and
    is baked into the file. An empty div throws all of it away.
    """
    p, result = groups_page["page"], groups_page["result"]
    assert p.locator("#board tbody.team").count() == len(result["teams"])
    assert p.locator("#board tr.golfer").count() == sum(
        t["golfer_count"] for t in result["teams"])

    text = p.locator("#board").text_content()
    for team in result["teams"]:
        assert team["team_name"] in text
    assert "%" in text, "the odds at creation are the point of the groups board"


def test_a_groups_page_ranks_nobody(groups_page):
    """
    Nothing to rank on, so nothing is ranked. Running the rule over an empty leaderboard
    would order the teams by roster size -- every golfer tier 2, the longer vector
    winning on padding -- and print it as a leaderboard.
    """
    p = groups_page["page"]
    assert set(p.locator("#board tbody.team td.c-rk").all_text_contents()) == {"—"}
    assert p.locator("#board tbody.team.is-leader").count() == 0
    assert p.locator("#board tr.golfer.out").count() == 0
    # No positions, no scores, no thru: those columns are gone rather than full of
    # dashes, and the tier legend is meaningless until something has been ranked.
    assert not p.locator("#board .group th.g-pos").first.is_visible()
    assert not p.locator("#tiers").is_visible()


def test_a_groups_page_says_there_was_no_join_to_report_on(groups_page):
    p = groups_page["page"]
    p.locator("#tab-odds").click()
    assert "No join was possible" in p.locator("#odds-cards").text_content()


def test_a_groups_page_orders_the_teams_by_what_they_were_drawn_at(groups_page):
    """
    Not a ranking -- the positions all read "—" -- but not arbitrary either. Roster
    order would present itself as a leaderboard just as loudly.
    """
    p, result = groups_page["page"], groups_page["result"]
    names = p.locator("#board tbody.team .tname").all_text_contents()
    expected = [t["team_name"] for t in sorted(result["teams"],
                                               key=lambda t: -t["total_odds"])]
    assert names == expected


# ---------------------------------------------------------------------------
# A phone held one-handed in front of a television
# ---------------------------------------------------------------------------

def test_the_team_name_is_readable_on_a_phone(browser, competition, serve_espn):
    """
    The regression this exists for: `table-layout: fixed` takes its columns from the
    first row, the group row under every team carries colspan="7", and a media query
    that removes three columns leaves the table inventing three auto-width phantoms.
    They split the slack with the team name, which measured four pixels wide at 390px
    and rendered as a lone ellipsis -- on the one column nobody can do without.
    """
    ctx, out = open_page(browser, competition["html"], serve_espn(),
                         viewport={"width": 390, "height": 900})
    p = out["page"]
    assert p.locator("#board tbody.team td.c-team").first.bounding_box()["width"] > 150
    # Not clipped: the whole name fits in the space it was given rather than ending in
    # an ellipsis, which is what the four-pixel column produced.
    assert p.evaluate("""() => {
      const n = document.querySelector('#board tbody.team .tname');
      return n.scrollWidth <= n.clientWidth + 1;
    }""")
    # The columns a thumb cannot use are gone, and the group underneath still has the
    # golfer, the score and the price.
    assert not p.locator("#board thead th.c-lead").first.is_visible()
    assert p.locator("#board .group th.g-score").first.is_visible()
    assert out["errors"] == []
    ctx.close()

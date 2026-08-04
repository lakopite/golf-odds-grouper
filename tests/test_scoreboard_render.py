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

    p = page["page"]
    p.locator("#tab-odds").click()
    text = p.locator("#view-odds").text_content()
    assert "no odds are ever fetched here" in text
    assert "Live odds" not in text
    # That sentence is static markup, so on its own it would hold however badly the view
    # were broken. Pair it with the view having actually rendered: a JS error that left
    # the odds panel empty must not read as "the page correctly says it fetches nothing".
    assert p.locator("#odds-tiles .tile").count() == 4
    assert p.locator("#odds-cards .card").count() == 2


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
    assert "the odds never change after this" in tiles


def test_the_tiles_explain_themselves_without_the_jargon(page):
    """
    These four numbers are read by people who have never priced a book. "ask prices ·
    probability basis" is accurate and it is also the sentence that gets a scoreboard
    accused of hiding something.
    """
    page["page"].locator("#tab-odds").click()
    tiles = page["page"].locator("#odds-tiles").text_content()
    for jargon in ("probability basis", "share of the slots", "auto-exclude", "1/"):
        assert jargon not in tiles, jargon
    assert "went into the groups" in tiles
    assert "is aiming for" in tiles


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


def test_the_odds_view_carries_only_the_draw_and_nothing_that_needs_explaining(page):
    """
    The odds view is read by the league, not by whoever built it. Three cards were
    removed from it -- a prices/movement card, a name-join report and a list of API
    endpoints -- because each one raised a question the reader could not act on, and
    two of them described machinery rather than the pool.
    """
    page["page"].locator("#tab-odds").click()
    cards = page["page"].locator("#odds-cards")
    assert cards.locator(".card").count() == 2
    text = cards.text_content()
    assert "Excluded from the draw" in text
    assert "Grouping certificate" in text
    for gone in ("Name join", "Where the numbers came from", "Odds re-read",
                 "Prices do not move here", "unresolved", "matched"):
        assert gone not in text, gone


def test_the_odds_view_lists_the_full_draw_group_by_group(page, competition):
    """
    The certificate one card up says the groups came out within a tick of each other.
    This is the working: every group, every golfer in it, and what each was worth when
    the groups were drawn.
    """
    result = competition["result"]
    p = page["page"]
    p.locator("#tab-odds").click()

    groups = p.locator("#odds-draw .card")
    assert groups.count() == len(result["teams"])
    assert p.locator("#odds-draw .draw-row").count() == sum(
        t["golfer_count"] for t in result["teams"])

    text = p.locator("#odds-draw").text_content()
    for team in result["teams"]:
        assert team["team_name"] in text
    # Every grouped golfer, by name, with a price beside them.
    for golfer in result["golfers"]:
        if golfer["team_id"]:
            assert golfer["name"] in text
    assert text.count("%") >= len(result["golfers"])


def test_the_full_draw_is_in_dealt_order_and_each_group_states_its_total(page, competition):
    result = competition["result"]
    p = page["page"]
    p.locator("#tab-odds").click()

    heads = p.locator("#odds-draw .card-head").all_text_contents()
    expected = [t["team_name"] for t in sorted(result["teams"],
                                               key=lambda t: t["group_index"])]
    assert [h.strip() for h in heads] == expected

    totals = p.locator("#odds-draw .draw-total b").all_text_contents()
    assert totals == [f"{t['total_odds'] * 100:.2f}%" for t in sorted(
        result["teams"], key=lambda t: t["group_index"])]


def test_the_provenance_names_the_build(page, competition):
    result = competition["result"]
    text = page["page"].locator("footer.prov").text_content()
    assert result["competition_id"] in text
    assert result["generator"]["tool"] in text
    assert "PROVEN OPTIMAL" in text


def test_the_provenance_still_says_where_the_numbers_came_from(page, competition):
    """
    The "Where the numbers came from" card was the only place on the page naming the
    Kalshi event and the ESPN event, and it was removed as clutter. Odds nobody can
    trace back to a market are a number somebody typed, so both ids moved into the
    footer rather than off the page -- docs/FRONTEND-SPEC.md §5.4 requires it.
    """
    result = competition["result"]
    text = page["page"].locator("footer.prov").text_content()
    assert result["sources"]["kalshi"]["event_ticker"] in text
    assert result["sources"]["kalshi"]["price_mode"] in text
    assert str(result["sources"]["espn"]["event_id"]) in text


def test_the_masthead_names_the_tournament(page, competition):
    text = page["page"].locator(".topbar").text_content()
    assert competition["result"]["tournament"]["name"] in text


def test_no_group_row_has_anywhere_to_put_a_price_moving(page, competition):
    """
    There was a Move column here, fed by a second Kalshi reading a rebuild could bake
    in beside the first. Two prices for one golfer -- one he was dealt on, one he was
    not -- read as the draw being adjusted after the fact, so there is now exactly one
    price per golfer and no column that could hold a second.
    """
    p = page["page"]
    assert p.locator("#board .group th.g-move").count() == 0
    assert p.locator("#board td.g-move").count() == 0

    # Seven columns, and the same seven in the header and every body row -- deleting a
    # <th> without its <td> shifts every right-aligned cell one column over and lands
    # the draw percentages under the wrong heading. Nothing else here catches that.
    headers = p.locator("#board .group").first.locator("thead th").count()
    assert headers == 7
    cells = p.locator("#board .group").first.locator("tbody tr").first.locator("td").count()
    assert cells == headers


def test_a_result_carrying_a_second_reading_still_renders_one_price_per_golfer(
        browser, competition, serve_espn, tmp_path):
    """
    Belt and braces for a file built by an older tool, or hand-edited. The page must
    ignore the extra fields rather than grow a column back off them.
    """
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

    ctx, out = open_page(browser, paths[0], serve_espn())
    p = out["page"]
    assert p.locator("#board td.g-move").count() == 0
    p.locator("#tab-odds").click()
    text = p.locator("#view-odds").text_content()
    assert "re-read" not in text
    assert "Monday Qualifier" not in text
    assert "1.310" not in text
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
    assert "expected " + ESPN_EVENT_ID in p.locator("#status-note").text_content()

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
    assert "with no field yet" in p.locator("#status-note").text_content()
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
    # Two words in the pill and a short caption, where there used to be a paragraph
    # explaining which of four reasons there was nothing to rank. The paragraph was
    # read as an apology for a page that was working exactly as built.
    assert "Built before the field posted" in p.locator("#status-note").text_content()
    assert p.locator("#standings-sub").text_content() == "The draw. Nothing is ranked yet."

    # A groups page fetches nothing at all, so the odds view has to stand on its own.
    p.locator("#tab-odds").click()
    assert p.locator("#odds-tiles .tile").count() == 4
    assert p.locator("#odds-cards .card").count() == 2


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
    # dashes.
    assert not p.locator("#board .group th.g-pos").first.is_visible()


def test_a_groups_page_still_shows_the_full_draw(groups_page):
    """
    The odds view is the same on both halves of the split. A groups page has no join to
    report on and never had a leaderboard, but the draw is exactly as decided as it will
    ever be, and it is the whole of what there is to look at on a Wednesday.
    """
    p, result = groups_page["page"], groups_page["result"]
    p.locator("#tab-odds").click()
    assert p.locator("#odds-draw .card").count() == len(result["teams"])
    assert p.locator("#odds-draw .draw-row").count() == sum(
        t["golfer_count"] for t in result["teams"])


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

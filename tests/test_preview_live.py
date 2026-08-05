"""
tools/preview_live.py -- the preview of a scoreboard nobody can see yet.

The page ranks itself at the first tee time, which makes the ranked page impossible to
look at until the tournament exists. The preview tool deals a played tournament's
leaderboard out over this week's field so it can be looked at on a Wednesday.

Two claims are worth holding to account, and they pull in opposite directions:

  * the preview is REAL ESPN DATA -- the field is this week's, the scores are a real
    leaderboard's, and nothing about either is invented. What is invented is only which
    of this week's golfers wears which score;
  * the preview is NOT this week's scoreboard, and every file it writes says so.

Everything below is one of those two. The last test is the one that matters most: the
bundled preview page ranks in a real browser, with no network at all, through the same
poll loop and the same standings rule the real page uses. Nothing in `frontend/` was
changed to make that work, and a test that stopped passing would mean it had been.
"""

import base64
import copy
import json
import os
import random
import sys

import pytest

import bundle_frontend as bundler
import espn_leaderboard
from conftest import load_fixture

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import preview_live                                                # noqa: E402

WYNDHAM = "401811961"


@pytest.fixture
def wyndham_result(espn_pre_payload):
    """
    A five-team competition over the real Wyndham field, as `build_competition.py`
    writes one the night before: every golfer carrying a real ESPN athlete id, and not
    one position between them.
    """
    from test_build_competition import espn_stage, golfer_name, make_result

    _, players = espn_leaderboard.parse_leaderboard(espn_pre_payload)
    names = [golfer_name(i) for i in range(len(players))]
    result = make_result(n_teams=5, n_golfers=len(players), espn=espn_stage(names))

    ordered = sorted(result["golfers"], key=lambda g: -g["odds"]["raw"])
    for golfer, player in zip(ordered, players):
        golfer["name"] = player["name"]
        golfer["espn"] = {"athlete_id": player["athlete_id"], "display_name": player["name"],
                          "headshot": player["headshot"], "country": player["country"],
                          "match": "exact", "in_field": True}
    by_id = {g["golfer_id"]: g for g in result["golfers"]}
    for team in result["teams"]:
        team["golfer_names"] = [by_id[gid]["name"] for gid in team["golfer_ids"]]
    return result


@pytest.fixture
def donor():
    """
    The finished donor: 73 made the cut, 74 did not.

    The harder of the two shipped stages to transplant, and so the one to test with.
    Half the field holds no position at all, which is the state a naive re-ranking would
    quietly repair -- and repairing it would invent a cut line rather than carry a real
    one across.
    """
    return load_fixture("espn_final_with_cut.json")


def order_for(result, payload, seed=7):
    competitors = payload["events"][0]["competitions"][0]["competitors"]
    weight = {g["espn"]["athlete_id"]: g["odds"]["grouping_weight"] for g in result["golfers"]}
    weights = [weight.get(c["athlete"]["id"], 0.0) for c in competitors]
    return preview_live.market_order(weights, random.Random(seed))


def preview_of(result, field, donor_payload, seed=7):
    return preview_live.transplant(field, donor_payload, order_for(result, field, seed))


# ---------------------------------------------------------------------------
# It is this week's field
# ---------------------------------------------------------------------------

def test_the_field_is_this_weeks_field_untouched(wyndham_result, espn_pre_payload, donor):
    """Names, athlete ids and headshots are the ones ESPN published for this event."""
    preview = preview_of(wyndham_result, espn_pre_payload, donor)

    was = {c["athlete"]["id"]: c["athlete"]["displayName"]
           for c in espn_pre_payload["events"][0]["competitions"][0]["competitors"]}
    now = {c["athlete"]["id"]: c["athlete"]["displayName"]
           for c in preview["events"][0]["competitions"][0]["competitors"]}
    assert now == was
    assert preview["events"][0]["id"] == WYNDHAM


def test_the_event_stays_this_weeks_event(wyndham_result, espn_pre_payload, donor):
    """
    The page refuses a payload whose event id is not the one it was built for, so a
    preview that carried the donor's id would render nothing and blame ESPN for it.
    """
    preview = preview_of(wyndham_result, espn_pre_payload, donor)
    meta, _ = espn_leaderboard.parse_leaderboard(preview)
    assert meta["event_id"] == wyndham_result["live"]["espn_event_id"] == WYNDHAM
    assert meta["event"] == "Wyndham Championship"


# ---------------------------------------------------------------------------
# The scores are a real leaderboard's
# ---------------------------------------------------------------------------

def test_the_leaderboard_is_the_donors_verbatim(wyndham_result, espn_pre_payload, donor):
    """
    Every position, tie and to-par on the preview is one the donor published. Nothing is
    recomputed, so the cut falls exactly where it fell and the ties are the real ties.
    """
    preview = preview_of(wyndham_result, espn_pre_payload, donor)
    _, players = espn_leaderboard.parse_leaderboard(preview)
    _, donors = espn_leaderboard.parse_leaderboard(donor)

    assert sorted(p["position"] or "" for p in players) == sorted(d["position"] or "" for d in donors)
    assert sorted(p["to_par"] or 0 for p in players) == sorted(d["to_par"] or 0 for d in donors)
    assert sorted(p["status"] or "" for p in players) == sorted(d["status"] or "" for d in donors)


def test_the_field_that_could_not_rank_now_ranks(wyndham_result, espn_pre_payload, donor):
    """
    The whole purpose, in one assertion. A `pre` payload is a complete field with no
    leaderboard in it and `has_started` is what says so; the preview is the same field
    with a leaderboard on it.
    """
    before_meta, before = espn_leaderboard.parse_leaderboard(espn_pre_payload)
    assert len(before) == 147
    assert not espn_leaderboard.has_started(before_meta["state"], before)

    preview = preview_of(wyndham_result, espn_pre_payload, donor)
    meta, players = espn_leaderboard.parse_leaderboard(preview)
    assert espn_leaderboard.has_started(meta["state"], players)
    assert sum(1 for p in players if p["position_number"] is not None) > 0


def test_tee_times_move_onto_this_weeks_dates(wyndham_result, espn_pre_payload, donor):
    """
    The donor is a month old. Its rounds keep their shape and their clock and land on
    this tournament's days, so the payload beside the page does not read as last month's.
    """
    preview = preview_of(wyndham_result, espn_pre_payload, donor)
    shift = preview_live.day_shift(espn_pre_payload, donor)
    assert shift != 0

    donor_rows = sorted(donor["events"][0]["competitions"][0]["competitors"],
                        key=lambda c: c.get("sortOrder", 9999))
    tees = [ls.get("teeTime") for ls in donor_rows[0]["linescores"] if ls.get("teeTime")]
    moved = [ls.get("teeTime")
             for ls in preview["events"][0]["competitions"][0]["competitors"][0]["linescores"]
             if ls.get("teeTime")]
    assert tees and len(moved) == len(tees)
    for old, new in zip(tees, moved):
        assert new[11:] == old[11:]                      # same time of day
        assert new[:10] > old[:10]                       # a later date


# ---------------------------------------------------------------------------
# Who ends up where
# ---------------------------------------------------------------------------

def test_the_draw_is_reproducible(wyndham_result, espn_pre_payload):
    assert order_for(wyndham_result, espn_pre_payload, seed=7) == \
           order_for(wyndham_result, espn_pre_payload, seed=7)
    assert order_for(wyndham_result, espn_pre_payload, seed=7) != \
           order_for(wyndham_result, espn_pre_payload, seed=8)


def test_the_market_draw_favours_the_favourite():
    """
    A shortest price should finish near the top more often than a 150-1 shot -- not
    always, which is what makes it a draw rather than a forecast. Over enough seeds the
    means separate, and if they ever stop separating the draw has quietly become a
    shuffle.
    """
    weights = [0.30] + [0.01] * 70
    favourite, longshot = [], []
    for seed in range(60):
        order = preview_live.market_order(weights, random.Random(seed))
        favourite.append(order.index(0))
        longshot.append(order.index(1))
    assert sum(favourite) / len(favourite) < sum(longshot) / len(longshot) / 3
    assert max(favourite) > 0                            # and not always the winner


def test_a_golfer_with_no_price_still_plays():
    """ESPN's field is not always Kalshi's, and a zero weight is not a scratching."""
    order = preview_live.market_order([0.5, 0.5, 0.0], random.Random(3))
    assert sorted(order) == [0, 1, 2]


def test_uniform_ignores_the_prices():
    order = preview_live.uniform_order(40, random.Random(1))
    assert sorted(order) == list(range(40))


# ---------------------------------------------------------------------------
# The file it writes says what it is
# ---------------------------------------------------------------------------

def test_only_the_poll_target_changes(wyndham_result, espn_pre_payload, donor):
    """The competition is not re-dealt, re-priced or re-grouped to preview it."""
    payload = preview_of(wyndham_result, espn_pre_payload, donor)
    before = copy.deepcopy(wyndham_result)
    out = preview_live.preview_result(wyndham_result, payload, "final",
                                      preview_live.STAGES["final"], 7, "market")

    assert wyndham_result == before                       # the input is not mutated
    for key in ("teams", "golfers", "grouping", "odds_snapshot", "tournament", "sources"):
        assert out[key] == before[key]
    assert out["live"]["espn_event_id"] == before["live"]["espn_event_id"]
    assert out["live"]["espn_leaderboard_url"] != before["live"]["espn_leaderboard_url"]


def test_the_page_polls_the_payload_itself(wyndham_result, espn_pre_payload, donor):
    """The baked URL decodes back to the leaderboard, byte for byte."""
    payload = preview_of(wyndham_result, espn_pre_payload, donor)
    out = preview_live.preview_result(wyndham_result, payload, "final",
                                      preview_live.STAGES["final"], 7, "market")

    url = out["live"]["espn_leaderboard_url"]
    assert url.startswith("data:application/json;base64,")
    assert json.loads(base64.b64decode(url.split(",", 1)[1])) == payload
    assert "</script>" not in url


def test_it_says_it_is_a_preview(wyndham_result, espn_pre_payload, donor):
    payload = preview_of(wyndham_result, espn_pre_payload, donor)
    out = preview_live.preview_result(wyndham_result, payload, "final",
                                      preview_live.STAGES["final"], 7, "market")

    assert out["preview"]["simulated"] is True
    assert out["preview"]["stage"] == "final"
    assert out["preview"]["seed"] == 7
    assert "PREVIEW" in out["league"]["tagline"]
    assert wyndham_result["league"].get("tagline") in (None, "") or \
        wyndham_result["league"]["tagline"] in out["league"]["tagline"]

    plain = preview_live.preview_result(wyndham_result, payload, "final",
                                        preview_live.STAGES["final"], 7, "market", label=False)
    assert plain["league"].get("tagline") == wyndham_result["league"].get("tagline")
    assert plain["preview"]["simulated"] is True          # the file still says so


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def test_the_command_writes_both_halves(wyndham_result, tmp_path, capsys):
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(wyndham_result))

    preview_live.main(["--result", str(result_path), "--stage", "final",
                       "--field", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "fixtures", "espn_pre_tournament.json"),
                       "--out", str(tmp_path)])

    payload = json.loads((tmp_path / "preview-final-leaderboard.json").read_text())
    out = json.loads((tmp_path / "preview-final.json").read_text())
    meta, players = espn_leaderboard.parse_leaderboard(payload)
    assert espn_leaderboard.has_started(meta["state"], players)
    assert out["preview"]["stage"] == "final"
    # It prints the standings the page will show, so the run itself says whether the
    # preview came out rankable.
    assert "STANDINGS THE PAGE WILL SHOW" in capsys.readouterr().out


def test_a_field_for_another_event_is_refused(wyndham_result, tmp_path):
    """
    The same refusal the page makes, made where the message can explain itself. A
    preview built on the wrong field would look like ESPN failing rather than like a
    mistake somebody made two commands ago.
    """
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(wyndham_result))
    wrong = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fixtures", "espn_final_with_cut.json")

    with pytest.raises(SystemExit) as exc:
        preview_live.main(["--result", str(result_path), "--field", wrong,
                           "--out", str(tmp_path)])
    assert "401811960" in str(exc.value) and WYNDHAM in str(exc.value)


# ---------------------------------------------------------------------------
# And then it is just the page
# ---------------------------------------------------------------------------

def test_the_bundled_preview_ranks_in_a_browser(browser, wyndham_result, espn_pre_payload,
                                                donor, tmp_path):
    """
    The claim the whole tool rests on: the exported preview is the real page. It runs the
    shipped `app.js` and the shipped standings rule, it reaches the ranked view through
    the ordinary poll loop rather than through a flag, and it asks the network for
    nothing at all.
    """
    payload = preview_of(wyndham_result, espn_pre_payload, donor)
    out = preview_live.preview_result(wyndham_result, payload, "final",
                                      preview_live.STAGES["final"], 7, "market")
    paths, _ = bundler.bundle(out, bundler.DEFAULT_TEMPLATE, str(tmp_path / "dist"))

    ctx = browser.new_context()
    seen = []
    ctx.on("request", lambda r: seen.append(r.url))
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("file://" + paths[0])
    page.wait_for_selector("#board tbody.team", timeout=15000)
    page.wait_for_function("document.getElementById('status-label').textContent === 'Final'",
                           timeout=15000)

    assert errors == []
    assert page.inner_text("#standings-heading").strip() == "Standings"
    assert [u for u in seen if not u.startswith("file://")] == []
    positions = page.eval_on_selector_all(
        "#board tbody.team .c-rk", "els => els.map(e => e.textContent.trim())")
    assert positions[0] in ("1", "T1") and len(positions) == 5
    ctx.close()

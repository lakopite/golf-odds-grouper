"""
Tests for espn_leaderboard.py, against the checked-in live payload.

espn-api/lb.json is a real Rocket Classic Round 2 leaderboard, captured mid-round with
147 competitors. That matters: every ESPN trap this module exists to handle only shows
up mid-round. A final payload agrees with itself and proves nothing.
"""

import pytest

import espn_leaderboard as espn


# ---------------------------------------------------------------------------
# The three fields that lie, and the one that does not
# ---------------------------------------------------------------------------

def test_the_payload_is_not_in_rank_order(espn_payload):
    """If it were, nothing downstream would need sortOrder at all."""
    raw = espn_payload["events"][0]["competitions"][0]["competitors"]
    assert [c["sortOrder"] for c in raw] != sorted(c["sortOrder"] for c in raw)


def test_parse_returns_players_in_rank_order(espn_players):
    orders = [p["sort_order"] for p in espn_players]
    assert orders == sorted(orders)
    assert orders[0] == 1


def test_score_display_value_is_stale_mid_round(espn_players):
    """
    The measurement the module is built on: 42 of 147 players disagree with the
    score field mid-round. If this ever drops to zero ESPN has changed the field's
    meaning and the live sum is no longer needed -- but until then, never rank on it.
    """
    stale = [p for p in espn_players if p["to_par"] != espn.to_par(p["stale_to_par"])]
    assert len(stale) == 42
    assert len([p for p in stale if p["to_par"] is not None]) == 41


def test_sort_order_tracks_the_live_total_not_the_stale_one(espn_players):
    """
    Zero inversions against the live total; the stale field inverts 29 times. This is
    what licenses sortOrder as the tie-break underneath the displayed position.
    """
    live = [p["to_par"] for p in espn_players if p["to_par"] is not None]
    stale = [espn.to_par(p["stale_to_par"]) for p in espn_players
             if espn.to_par(p["stale_to_par"]) is not None]
    assert sum(1 for a, b in zip(live, live[1:]) if a > b) == 0
    assert sum(1 for a, b in zip(stale, stale[1:]) if a > b) == 29


def test_future_round_stubs_are_dropped(espn_players):
    """R3 and R4 stubs carry no displayValue at all; summing them would throw."""
    leader = espn_players[0]
    assert [r["round"] for r in leader["rounds"]] == [1, 2]
    assert leader["to_par"] == sum(r["to_par"] for r in leader["rounds"])


def test_live_total_is_the_sum_of_the_rounds(espn_players):
    for p in espn_players:
        if p["rounds"]:
            assert p["to_par"] == sum(r["to_par"] for r in p["rounds"])
        else:
            assert p["to_par"] is None


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

def test_meta_reads_the_round_from_the_competition_not_the_event(espn_payload):
    meta, _ = espn.parse_leaderboard(espn_payload)
    assert meta["round"] == 2                       # competitions[0].status.period
    assert meta["state"] == "in"
    assert meta["event_id"] == "401811960"
    assert meta["par"] == 70


def test_empty_payload_is_not_a_crash():
    assert espn.parse_leaderboard({"events": []}) == (None, [])
    assert espn.parse_leaderboard({}) == (None, [])


# ---------------------------------------------------------------------------
# to_par / position_number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("E", 0), ("-2", -2), ("+3", 3), ("-", None), ("", None), (None, None), ("WD", None),
])
def test_to_par(raw, expected):
    assert espn.to_par(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("1", 1), ("T12", 12), ("T1", 1), ("-", None), (None, None), ("", None), ("CUT", None),
])
def test_position_number(raw, expected):
    assert espn.position_number(raw) == expected


def test_a_cut_player_has_no_position_number(espn_players):
    """
    All 74 cut players share the position "-". Anything that ranks on the position
    number has to have an answer for them, which is why the rank key has tiers.
    """
    cut = [p for p in espn_players if p["status"] == "STATUS_CUT"]
    assert cut == [] or all(p["position_number"] is None for p in cut)


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    ("Rasmus Højgaard", "Rasmus Hojgaard"),
    ("Erik Van Rooyen", "Erik van Rooyen"),
    ("Thorbjørn Olesen", "Thorbjorn Olesen"),
    ("Davis Love III", "Davis Love"),
    ("C.T. Pan", "CT Pan"),
    ("Adrien Dumont De Chassart", "Adrien Dumont de Chassart"),
    ("Séamus Power", "Seamus Power"),
])
def test_normalization_folds_the_same_golfer_together(a, b):
    assert espn.normalize_name(a) == espn.normalize_name(b)


def test_normalization_keeps_different_golfers_apart():
    assert espn.normalize_name("Jordan Smith") != espn.normalize_name("Jordan Spieth")


@pytest.mark.parametrize("a,b", [
    ("Zachary Bauchou", "Zach Bauchou"),
    ("Cameron Davis", "Cam Davis"),
    ("Kris Ventura", "Kristoffer Ventura"),
    ("Nicolas Echavarria", "Nico Echavarria"),
    ("Matthew McCarty", "Matt McCarty"),
    ("Benjamin James", "Ben James"),
    ("Jordan L. Smith", "Jordan Smith"),
    ("Hao-Tong Li", "Haotong Li"),
])
def test_initial_last_key_bridges_formal_and_familiar_names(a, b):
    """Every one of these is a real Kalshi/ESPN disagreement from a measured field."""
    assert espn.initial_last_key(a) == espn.initial_last_key(b)


def test_initial_last_key_does_not_bridge_different_people():
    assert espn.initial_last_key("Jordan Smith") != espn.initial_last_key("Cameron Smith")


def test_initial_last_key_needs_two_parts():
    assert espn.initial_last_key("Cher") is None


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------

def test_field_has_no_ambiguous_initial_last_keys(espn_players):
    """
    The licence for tier 2. If a field ever does contain two J. Smiths the key is
    dropped rather than guessed, and the golfer falls through to unresolved.
    """
    assert espn.build_index(espn_players)["ambiguous"] == []


def test_exact_match(espn_players):
    hit, how = espn.match_golfer("Cameron Young", espn.build_index(espn_players))
    assert how == "exact" and hit["name"] == "Cameron Young"


def test_initial_last_match(espn_players):
    hit, how = espn.match_golfer("Zachary Bauchou", espn.build_index(espn_players))
    assert how == "initial_last" and hit["name"] == "Zach Bauchou"


def test_alias_beats_everything(espn_players):
    index = espn.build_index(espn_players)
    hit, how = espn.match_golfer("Zachary Bauchou", index, {"Zachary Bauchou": "Cameron Young"})
    assert how == "alias" and hit["name"] == "Cameron Young"


def test_a_golfer_who_is_not_playing_stays_unresolved(espn_players):
    hit, how = espn.match_golfer("Tiger Woods", espn.build_index(espn_players))
    assert hit is None and how == "unresolved"


def test_ambiguous_key_is_refused_rather_than_guessed():
    players = [
        {"name": "Jordan Smith", "athlete_id": "1", "sort_order": 1},
        {"name": "Jamie Smith", "athlete_id": "2", "sort_order": 2},
    ]
    index = espn.build_index(players)
    assert index["ambiguous"] == ["j|smith"]
    assert espn.match_golfer("Jonathan Smith", index) == (None, "unresolved")


def test_match_field_reports_every_tier(espn_players):
    names = ["Cameron Young", "Zachary Bauchou", "Tiger Woods", "Rickie Fowler"]
    matches, report = espn.match_field(names, espn_players)
    assert report["matched"] == 3
    assert report["matched_exact"] == 2
    assert report["matched_initial_last"] == 1
    assert report["unresolved"] == ["Tiger Woods"]
    assert matches["Zachary Bauchou"]["player"]["name"] == "Zach Bauchou"


def test_match_field_on_an_empty_field_resolves_nothing(espn_players):
    """A pre-tournament event returns no competitors. Nothing should match, quietly."""
    matches, report = espn.match_field(["Cameron Young"], [])
    assert matches == {} and report["unresolved"] == ["Cameron Young"]


# ---------------------------------------------------------------------------
# Event resolution
# ---------------------------------------------------------------------------

SEASON = [
    {"event_id": "1", "name": "Rocket Classic", "start": "2026-07-30", "state": "post"},
    {"event_id": "2", "name": "Wyndham Championship", "start": "2026-08-06", "state": "pre"},
    {"event_id": "3", "name": "The Open Championship", "start": "2026-07-16", "state": "post"},
    {"event_id": "4", "name": "Sony Open in Hawaii", "start": "2026-01-15", "state": "post"},
]


@pytest.mark.parametrize("query,expected", [
    ("Wyndham Championship", "2"),
    ("wyndham", "2"),
    ("2026 Wyndham Championship", "2"),
    ("Rocket Classic", "1"),
    ("rocket", "1"),
    ("Sony Open", "4"),
    ("The Open", "3"),
])
def test_resolve_event(query, expected):
    best, _ = espn.resolve_event(query, 2026, events=SEASON)
    assert best["event_id"] == expected


def test_resolve_event_returns_runners_up_so_a_guess_can_be_shown():
    best, ranked = espn.resolve_event("open", 2026, events=SEASON)
    assert best is not None
    assert len(ranked) >= 2                       # both "Open" events scored


def test_resolve_event_with_no_match_returns_none():
    best, ranked = espn.resolve_event("Ryder Cup", 2026, events=SEASON)
    assert best is None and ranked == []


def test_generic_golf_words_alone_never_score_confidently():
    """
    "championship" is in half the calendar. It still matches -- refusing to answer is
    not obviously better than answering badly -- but it must never reach the score a
    distinctive word reaches, and it must leave the result flagged ambiguous so the
    caller stops and asks rather than grouping the wrong tournament.
    """
    best, ranked = espn.resolve_event("championship", 2026, events=SEASON)
    assert best["score"] < 0.5
    assert len(ranked) >= 2
    assert espn.ambiguous(ranked) or ranked[1]["score"] < ranked[0]["score"]

    sharp, sharp_ranked = espn.resolve_event("wyndham", 2026, events=SEASON)
    assert sharp["event_id"] == "2" and sharp["score"] == 1.0
    assert not espn.ambiguous(sharp_ranked)


def test_a_year_in_the_query_does_not_hurt_the_match():
    """Kalshi titles its events "2026 Wyndham Championship"; ESPN does not."""
    assert espn.score_name("2026 Wyndham Championship", "Wyndham Championship") == 1.0


def test_resolve_event_raises_on_an_empty_season():
    with pytest.raises(RuntimeError, match="no .* events"):
        espn.resolve_event("Wyndham", 2026, events=[])


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

def test_leaderboard_url_pins_the_event():
    url = espn.leaderboard_url("401811961")
    assert "event=401811961" in url and "league=pga" in url


def test_leaderboard_url_without_an_event_is_the_current_one():
    assert "event=" not in espn.leaderboard_url(None)

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


# ---------------------------------------------------------------------------
# The season calendar
# ---------------------------------------------------------------------------

CALENDAR_PAYLOAD = {
    "leagues": [{
        "calendar": [
            {"id": "1", "label": "January Open", "startDate": "2026-01-08T08:00Z",
             "endDate": "2026-01-11T08:00Z"},
            {"id": "2", "label": "June Classic", "startDate": "2026-06-04T08:00Z",
             "endDate": "2026-06-07T08:00Z"},
            {"id": "3", "label": "Last Week", "startDate": "2026-07-30T07:00Z",
             "endDate": "2026-08-02T07:00Z"},
            {"id": "4", "label": "This Week", "startDate": "2026-08-06T07:00Z",
             "endDate": "2026-08-09T07:00Z"},
            {"id": "5", "label": "Next Month", "startDate": "2026-09-03T07:00Z",
             "endDate": "2026-09-06T07:00Z"},
        ],
    }],
    "events": [],
}


@pytest.fixture
def calendar(monkeypatch):
    """The season calendar, without the 35 MB request that carries it."""
    calls = []

    def fake_get(url, **params):
        calls.append((url, params))
        return CALENDAR_PAYLOAD

    monkeypatch.setattr(espn, "_get", fake_get)
    return calls


def test_the_calendar_is_read_from_a_single_day(calendar):
    """
    The measurement this exists for: dates=<year> embeds every competitor of every
    event played so far (35 MB in August), and dates=<one day> returns the same whole
    season calendar in 12 KB. Asking for a day is the entire optimisation, so the day
    has to actually be in the request.
    """
    rows = espn.season_calendar(2026)
    assert calendar[0][1]["dates"] == "20260701"
    assert [r["event_id"] for r in rows] == ["1", "2", "3", "4", "5"]
    assert rows[0]["name"] == "January Open"


def calendar_rows():
    return [{"event_id": e["id"], "name": e["label"], "start": e["startDate"], "end": e["endDate"]}
            for e in CALENDAR_PAYLOAD["leagues"][0]["calendar"]]


def test_finished_before_takes_the_ones_that_are_over_newest_first():
    rows = espn.finished_before(calendar_rows(), cutoff="2026-08-03T21:00:00+00:00")
    assert [r["event_id"] for r in rows] == ["3", "2", "1"]


def test_finished_before_can_skip_the_tournament_being_built():
    assert [r["event_id"] for r in
            espn.finished_before(calendar_rows(), "2026-09-10", exclude_ids=["3", "4"])] \
        == ["5", "2", "1"]


def test_finished_before_compares_on_the_date_alone():
    """
    ESPN writes `2026-08-02T07:00Z` and this project writes `2026-08-03T21:00:00+00:00`.
    Only the first ten characters of either mean the same thing, and tournaments are
    days apart, so only the first ten are compared.
    """
    rows = [{"event_id": "a", "name": "A", "start": "2026-08-02T07:00Z", "end": "2026-08-02T07:00Z"}]
    assert espn.finished_before(rows, "2026-08-03T21:00:00+00:00")
    assert not espn.finished_before(rows, "2026-08-02T23:59:59+00:00")


# ---------------------------------------------------------------------------
# Matching against earlier tournaments
# ---------------------------------------------------------------------------

def leaderboard(names, event_id="1", event="An Earlier Tournament"):
    """A minimal but genuinely shaped ESPN payload, so parse_leaderboard does the work."""
    return {"events": [{
        "id": event_id, "name": event, "date": "2026-07-30T07:00Z",
        "status": {"type": {"state": "post", "completed": True}},
        "courses": [{"name": "Somewhere CC", "shotsToPar": 70}],
        "competitions": [{
            "status": {"period": 4, "type": {"detail": "Final"}},
            "competitors": [{
                "athlete": {"id": f"{event_id}{i:03d}", "displayName": name,
                            "shortName": name, "headshot": {"href": f"https://x/{i}.png"},
                            "flag": {"alt": "USA", "href": "https://x/usa.png"}},
                "sortOrder": i + 1,
                "status": {"position": {"displayName": str(i + 1), "isTie": False},
                           "type": {"name": "STATUS_PLAY_COMPLETE"}},
                "linescores": [{"period": 1, "displayValue": "-2", "value": 68}],
            } for i, name in enumerate(names)],
        }],
    }]}


@pytest.fixture
def history(monkeypatch):
    """
    Three finished tournaments, newest first, each with a field of its own.

    Returns the list of event ids actually fetched, because "it stopped once everything
    was resolved" is a behaviour rather than an implementation detail: every extra event
    is a 276 KB payload nobody needed.
    """
    fields = {
        "3": ["Cameron Young", "Zach Bauchou", "Hideki Matsuyama"],
        "2": ["Cam Davis", "Tom Kim"],
        "1": ["Carson Young", "Somebody Else"],
    }
    fetched = []

    def fake_fetch(event_id, league=espn.DEFAULT_LEAGUE):
        fetched.append(str(event_id))
        return leaderboard(fields[str(event_id)], event_id=str(event_id))

    monkeypatch.setattr(espn, "fetch_leaderboard", fake_fetch)
    monkeypatch.setattr(espn, "season_calendar", lambda season, league=espn.DEFAULT_LEAGUE, on=None: [
        {"event_id": e["id"], "name": e["label"], "start": e["startDate"], "end": e["endDate"]}
        for e in CALENDAR_PAYLOAD["leagues"][0]["calendar"]])
    return fetched


def test_identity_keeps_the_person_and_drops_the_week(espn_players):
    """
    The whole safety argument for looking backwards. A player object from a finished
    tournament carries that tournament's position, sortOrder and to-par, and those are
    exactly the fields the standings rule ranks on -- so a golfer who won in July would
    show T1 on Thursday morning of a tournament nobody has teed off in.
    """
    ident = espn.identity(espn_players[0], {"event_id": "9", "name": "Last Week", "end": "x"})
    assert ident["athlete_id"] and ident["name"] and ident["headshot"]
    for scoring in ("position", "position_number", "sort_order", "to_par", "thru", "rounds",
                    "status", "tied", "tee_time", "round"):
        assert scoring not in ident
    assert ident["from_event"]["name"] == "Last Week"


def test_history_resolves_names_with_no_field_at_all(history):
    matches, report = espn.match_history(
        ["Cameron Young", "Hideki Matsuyama"], 2026, cutoff="2026-08-04", exclude_ids=["4"])
    assert report["matched"] == 2
    assert matches["Cameron Young"]["player"]["athlete_id"] == "3000"
    assert matches["Cameron Young"]["source"] == "history"
    assert matches["Cameron Young"]["event"]["name"] == "Last Week"


def test_history_stops_as_soon_as_everything_is_resolved(history):
    """Every extra tournament is a payload nobody needed."""
    espn.match_history(["Cameron Young"], 2026, cutoff="2026-08-04", exclude_ids=["4"])
    assert history == ["3"]


def test_history_walks_back_until_it_finds_them(history):
    matches, report = espn.match_history(
        ["Cameron Young", "Tom Kim"], 2026, cutoff="2026-08-04", exclude_ids=["4"])
    assert history == ["3", "2"]
    assert report["matched"] == 2
    assert matches["Tom Kim"]["event"]["name"] == "June Classic"


def test_history_is_bounded_and_says_what_it_did_not_read(history):
    matches, report = espn.match_history(
        ["Somebody Else"], 2026, max_events=1, cutoff="2026-08-04", exclude_ids=["4"])
    assert history == ["3"]
    assert report["unresolved"] == ["Somebody Else"]
    assert report["unscanned_events"] == 2, "two tournaments were left unread; say so"


def test_history_can_be_switched_off(history):
    matches, report = espn.match_history(["Cameron Young"], 2026, max_events=0)
    assert matches == {} and report["unresolved"] == ["Cameron Young"] and history == []


def test_history_matches_formal_names_across_tournaments(history):
    """Kalshi's "Zachary Bauchou" and "Cameron Davis" against ESPN's Zach and Cam."""
    matches, _ = espn.match_history(["Zachary Bauchou", "Cameron Davis"], 2026,
                                    cutoff="2026-08-04", exclude_ids=["4"])
    assert matches["Zachary Bauchou"]["match"] == "initial_last"
    assert matches["Cameron Davis"]["player"]["name"] == "Cam Davis"


def test_history_refuses_a_key_two_athletes_share(history):
    """
    Tier 2 is measured collision-free inside one field; a whole season is not one field.
    Carson Young joins the union three tournaments back and `c|young` stops meaning
    anything, so "Cam Young" comes back unresolved rather than bound to the wrong man --
    while "Cameron Young", who matches exactly, is unaffected.
    """
    matches, report = espn.match_history(["Cam Young", "Cameron Young"], 2026,
                                         cutoff="2026-08-04", exclude_ids=["4"])
    assert report["unresolved"] == ["Cam Young"]
    assert matches["Cameron Young"]["match"] == "exact"
    assert "c|young" in report["ambiguous_keys"]


def test_history_survives_a_leaderboard_that_will_not_load(history, monkeypatch):
    """One unreadable tournament is not a reason to abandon the join."""
    real = espn.fetch_leaderboard

    def flaky(event_id, league=espn.DEFAULT_LEAGUE):
        if str(event_id) == "3":
            raise RuntimeError("ESPN returned HTTP 500")
        return real(event_id, league)

    monkeypatch.setattr(espn, "fetch_leaderboard", flaky)
    matches, report = espn.match_history(["Tom Kim"], 2026, cutoff="2026-08-04", exclude_ids=["4"])
    assert matches["Tom Kim"]["player"]["athlete_id"] == "2001"
    assert any(e.get("error") for e in report["scanned"])


# ---------------------------------------------------------------------------
# The two halves together
# ---------------------------------------------------------------------------

def test_this_weeks_field_answers_first_and_history_only_covers_the_rest(history):
    """
    A golfer in this week's field must resolve against THIS week, scoring and all. Only
    the ones it cannot answer for are looked up in tournaments that are over.
    """
    _, players = espn.parse_leaderboard(leaderboard(["Hideki Matsuyama"], event_id="4"))
    matches, report = espn.match_field_and_history(
        ["Hideki Matsuyama", "Tom Kim"], players, 2026, cutoff="2026-08-04", exclude_ids=["4"])

    assert matches["Hideki Matsuyama"]["source"] == "field"
    assert matches["Hideki Matsuyama"]["player"]["position_number"] == 1
    assert matches["Tom Kim"]["source"] == "history"
    assert "position_number" not in matches["Tom Kim"]["player"]
    assert report["from_field"] == 1 and report["from_history"] == 1
    assert report["matched"] == 2


def test_a_golfer_missing_from_a_posted_field_is_reported_as_not_playing(history):
    """
    Two different facts, and the result file needs both. History can hand back Tom Kim's
    headshot; it cannot make him a starter. "Not in this week's field" is what the pool
    wants to know, and it survives being identified.
    """
    _, players = espn.parse_leaderboard(leaderboard(["Hideki Matsuyama"], event_id="4"))
    matches, report = espn.match_field_and_history(
        ["Hideki Matsuyama", "Tom Kim"], players, 2026, cutoff="2026-08-04", exclude_ids=["4"])
    assert report["not_in_field"] == ["Tom Kim"]
    assert report["unresolved"] == []


def test_nobody_is_not_in_a_field_that_does_not_exist_yet(history):
    _, report = espn.match_field_and_history(["Cameron Young"], [], 2026,
                                             cutoff="2026-08-04", exclude_ids=["4"])
    assert report["not_in_field"] == []
    assert report["from_history"] == 1


def test_the_combined_report_counts_both_sources(history):
    _, players = espn.parse_leaderboard(leaderboard(["Hideki Matsuyama"], event_id="4"))
    _, report = espn.match_field_and_history(
        ["Hideki Matsuyama", "Zachary Bauchou"], players, 2026,
        cutoff="2026-08-04", exclude_ids=["4"])
    assert report["matched_exact"] == 1
    assert report["matched_initial_last"] == 1
    assert report["matched_alias"] == 0
    assert report["history"]["scanned"][0]["name"] == "Last Week"


def test_an_alias_still_wins_when_the_match_comes_out_of_history(history):
    matches, _ = espn.match_field_and_history(
        ["The Big Cat"], [], 2026, aliases={"The Big Cat": "Hideki Matsuyama"},
        cutoff="2026-08-04", exclude_ids=["4"])
    assert matches["The Big Cat"]["match"] == "alias"
    assert matches["The Big Cat"]["player"]["name"] == "Hideki Matsuyama"

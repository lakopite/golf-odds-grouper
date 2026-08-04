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
    """
    Every one of these is a real Kalshi/ESPN disagreement from a measured field.

    The key no longer MATCHES them -- it suggests them, and somebody confirms. What it
    still has to do is bring the pair together at all, because that is what puts the
    right athlete at the top of the review file rather than three streets away. See
    test_every_formal_name_suggests_its_familiar_one_first.
    """
    assert espn.initial_last_key(a) == espn.initial_last_key(b)


def test_initial_last_key_does_not_bridge_different_people():
    assert espn.initial_last_key("Jordan Smith") != espn.initial_last_key("Cameron Smith")


def test_initial_last_key_needs_two_parts():
    assert espn.initial_last_key("Cher") is None


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------

def test_a_real_field_holds_no_two_athletes_with_the_same_name(espn_players):
    """
    The licence for the exact tier. Normalisation is aggressive -- it folds accents,
    punctuation, hyphens and generational suffixes -- so it is worth checking that it
    does not fold two DIFFERENT people together in a real 147-player field. It does not.
    """
    assert espn.build_index(espn_players)["ambiguous"] == []


def test_the_index_reaches_an_athlete_by_id_as_well_as_by_name(espn_players):
    """
    The id half is what a reviewed decision binds through, and what the exported page
    joins on. A name is how a person finds an athlete; an id is how the software does.
    """
    index = espn.build_index(espn_players)
    young = index["exact"][espn.normalize_name("Cameron Young")]
    assert index["by_id"][str(young["athlete_id"])] is young


def test_exact_match(espn_players):
    hit, how = espn.match_golfer("Cameron Young", espn.build_index(espn_players))
    assert how == "exact" and hit["name"] == "Cameron Young"


def test_a_formal_first_name_is_not_matched_on_its_own(espn_players):
    """
    The demotion, stated. "Zachary Bauchou" IS Zach Bauchou and the old first-initial
    tier bound them silently. It would have bound just as silently the week a field held
    two J. Smiths, and nothing downstream could have told the two cases apart.

    So the join refuses, and the name goes to review carrying the answer -- see
    test_every_formal_name_suggests_its_familiar_one_first. Nothing is lost except the
    part where nobody was asked.
    """
    hit, how = espn.match_golfer("Zachary Bauchou", espn.build_index(espn_players))
    assert hit is None and how == "unresolved"


def test_an_alias_beats_an_exact_match_of_the_same_name(espn_players):
    """
    Precedence has to be tested against a name that WOULD otherwise resolve, or it
    proves only that an alias beats nothing. "Cameron Young" is in this field under
    exactly that spelling; the alias still wins.

    This is the escape hatch for the day a field contains two people a source spells the
    same way, and it only works if it outranks the tier below it.
    """
    index = espn.build_index(espn_players)
    hit, how = espn.match_golfer("Cameron Young", index, {"Cameron Young": "Rickie Fowler"})
    assert how == "alias" and hit["name"] == "Rickie Fowler"


def test_a_golfer_who_is_not_playing_stays_unresolved(espn_players):
    hit, how = espn.match_golfer("Tiger Woods", espn.build_index(espn_players))
    assert hit is None and how == "unresolved"


def test_two_athletes_with_the_same_name_are_both_refused_rather_than_one_guessed():
    """
    A coin flip decides which of two real people is on somebody's team, so there is no
    coin flip. Both names leave the index and both golfers go to review, where a person
    settles it against the athlete ids.
    """
    players = [
        {"name": "Jordan Smith", "athlete_id": "1", "sort_order": 1},
        {"name": "Jordan Smith", "athlete_id": "2", "sort_order": 2},
    ]
    index = espn.build_index(players)
    assert index["ambiguous"] == ["jordan smith"]
    assert espn.match_golfer("Jordan Smith", index) == (None, "unresolved")


def test_match_field_reports_every_tier(espn_players):
    names = ["Cameron Young", "Rickie Fowler", "Zach Bauchou", "Tiger Woods"]
    matches, report = espn.match_field(
        names, espn_players, aliases={"Rickie Fowler": "Rickie Fowler"})
    assert report["matched"] == 3
    assert report["matched_exact"] == 2 and report["matched_alias"] == 1
    assert report["matched_decision"] == 0
    assert report["unresolved"] == ["Tiger Woods"]
    assert report["absent"] == [] and report["problems"] == []


def test_match_field_on_an_empty_field_resolves_nothing(espn_players):
    """
    Nothing should match, quietly, rather than dividing by a field size of zero.

    The build never gets here with an empty field -- it stops instead, because ESPN
    posts a field about two days out and finding none means the read failed. But this
    function is also called directly from the CLI, and it has to behave.
    """
    matches, report = espn.match_field(["Cameron Young"], [])
    assert matches == {} and report["unresolved"] == ["Cameron Young"]
    assert report["espn_field_size"] == 0


# ---------------------------------------------------------------------------
# Has anybody teed off -- the gate everything that ranks is behind
# ---------------------------------------------------------------------------

def test_a_posted_field_with_no_positions_has_not_started(espn_pre_payload):
    """
    The measurement the whole program was re-shaped around, held in place.

    The real 2026 Wyndham leaderboard, captured 2026-08-04 for a tournament starting on
    the 6th: a complete field, with athlete ids and headshots and tee times, and not one
    position in it. So "there are players in this payload" stopped meaning "this
    tournament is under way", and everything that ranks is gated on this instead.
    """
    meta, players = espn.parse_leaderboard(espn_pre_payload)
    assert meta["state"] == "pre"
    assert len(players) == 147
    assert all(p["athlete_id"] and p["name"] for p in players), "joinable"
    assert all(p["position_number"] is None for p in players), "not rankable"
    assert meta["started"] is False


def test_play_under_way_and_a_finished_tournament_have_both_started(espn_payload,
                                                                    espn_final_payload):
    assert espn.parse_leaderboard(espn_payload)[0]["started"] is True
    assert espn.parse_leaderboard(espn_final_payload)[0]["started"] is True


def test_either_signal_is_enough_and_neither_is_required_of_the_other():
    """
    Two signals, either sufficient, because they fail in opposite directions.

    `state` is ESPN's own answer and is read off the event envelope, so a payload with a
    missing or stale envelope would otherwise blank a board that is plainly live. A
    golfer holding a real position is proof from the field itself. Requiring both would
    lose a good board on a bad envelope; requiring neither is what this exists to
    prevent.
    """
    positionless = [{"position_number": None}, {"position_number": None}]
    assert espn.has_started("in", positionless) is True
    assert espn.has_started("post", positionless) is True
    assert espn.has_started("pre", positionless) is False
    assert espn.has_started(None, positionless) is False
    # One real position is enough on its own, whatever the envelope says.
    assert espn.has_started(None, [{"position_number": None}, {"position_number": 4}]) is True
    assert espn.has_started("pre", [{"position_number": 1}]) is True
    # An empty field has not started and cannot be ranked either way.
    assert espn.has_started("pre", []) is False


# ---------------------------------------------------------------------------
# Reviewed decisions
# ---------------------------------------------------------------------------

def test_a_decision_binds_a_name_to_an_athlete_id(espn_players):
    matches, report = espn.match_field(
        ["Zachary Bauchou"], espn_players,
        decisions={"Zachary Bauchou": {"athlete_id": _athlete_id(espn_players, "Zach Bauchou")}})
    assert matches["Zachary Bauchou"]["match"] == "decision"
    assert matches["Zachary Bauchou"]["player"]["name"] == "Zach Bauchou"
    assert report["matched_decision"] == 1


def test_a_decision_can_name_the_espn_player_instead_of_the_id(espn_players):
    """
    A person editing the file by hand has the name in front of them and not the id, and
    refusing that would be pedantry. The id is still what the file records afterwards.
    """
    matches, _ = espn.match_field(
        ["Zachary Bauchou"], espn_players,
        decisions={"Zachary Bauchou": {"espn_name": "Zach Bauchou"}})
    assert matches["Zachary Bauchou"]["player"]["name"] == "Zach Bauchou"


def test_a_confirmed_absence_is_not_the_same_as_an_unsettled_name(espn_players):
    """
    Both score nothing, so the standings cannot tell them apart. A person deciding
    whether the build is finished can: `absent` was checked, `unresolved` was not.
    """
    matches, report = espn.match_field(
        ["Brooks Koepka", "Jason Day"], espn_players,
        decisions={"Jason Day": {"absent": True, "note": "withdrew before the first round"}})
    assert matches == {}
    assert report["absent"] == ["Jason Day"]
    assert report["unresolved"] == ["Brooks Koepka"]


def test_a_decision_naming_an_athlete_who_is_not_here_is_reported_and_then_ignored(espn_players):
    """
    A typo in a review file must not cost a golfer their whole week. The decision is
    refused loudly, and the name then goes through the automatic tiers exactly as if
    nobody had written anything -- which for an exactly-spelled name resolves it.
    """
    matches, report = espn.match_field(
        ["Cameron Young"], espn_players,
        decisions={"Cameron Young": {"athlete_id": "99999999"}})
    assert matches["Cameron Young"]["match"] == "exact"
    assert len(report["problems"]) == 1
    assert "99999999" in report["problems"][0] and "not in this field" in report["problems"][0]


def test_a_decision_that_decides_nothing_is_reported(espn_players):
    _, report = espn.match_field(["Tiger Woods"], espn_players,
                                 decisions={"Tiger Woods": {"note": "no idea"}})
    assert report["unresolved"] == ["Tiger Woods"]
    assert "decides nothing" in report["problems"][0]


def test_one_athlete_cannot_be_held_by_two_golfers(espn_players):
    """
    Two Kalshi names reaching one person means one of them is wrong. Scoring both would
    hand a team a golfer it does not have, so the second is refused and named.
    """
    young = _athlete_id(espn_players, "Cameron Young")
    _, report = espn.match_field(
        ["Cameron Young", "C Young"], espn_players,
        decisions={"C Young": {"athlete_id": young}})
    # The decision wins and the exact match is the one refused: somebody looked at one
    # of these two, and it was not the one that merely spells the same.
    assert report["unresolved"] == ["Cameron Young"]
    assert "already held by" in report["problems"][0]


@pytest.mark.parametrize("order", [
    ["Cameron Young", "C Young"],
    ["C Young", "Cameron Young"],
])
def test_which_of_the_two_is_refused_does_not_depend_on_the_kalshi_ordering(espn_players, order):
    """
    Kalshi returns its markets in whatever order it likes, and that must not decide
    which of two competing golfers keeps the athlete. Decisions are settled in a first
    pass and the automatic tiers fill in around them, so the answer is the same either
    way round.
    """
    matches, _ = espn.match_field(
        order, espn_players,
        decisions={"C Young": {"athlete_id": _athlete_id(espn_players, "Cameron Young")}})
    assert set(matches) == {"C Young"}


def _athlete_id(players, name):
    return next(p["athlete_id"] for p in players if p["name"] == name)


# ---------------------------------------------------------------------------
# Suggestions -- what replaced the first-initial match tier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kalshi,espn_name", [
    ("Zachary Bauchou", "Zach Bauchou"),
    ("Cameron Davis", "Cam Davis"),
    ("Kris Ventura", "Kristoffer Ventura"),
    ("Nicolas Echavarria", "Nico Echavarria"),
    ("Matthew McCarty", "Matt McCarty"),
    ("Benjamin James", "Ben James"),
    ("Jordan L. Smith", "Jordan Smith"),
    ("Hao-Tong Li", "Haotong Li"),
])
def test_every_formal_name_suggests_its_familiar_one_first(espn_players, kalshi, espn_name):
    """
    The regression fixture for the demotion. These eight pairs are the entire population
    the deleted tier used to swallow silently, measured on a real field -- so if the
    suggestion generator stops putting the right athlete at the top, the refactor has
    cost eight golfers a week rather than made them visible.
    """
    top = espn.suggest_matches(kalshi, espn_players)
    assert top, f"{kalshi} produced no suggestion at all"
    assert top[0]["espn_name"] == espn_name
    assert top[0]["why"] == "same first initial and last name"


def test_a_colliding_key_does_not_produce_one_confident_answer():
    """
    Why the tier was demoted, kept as evidence. Widen the population past one field and
    `c|young` stops being unique -- four tournaments back from the 2026 Wyndham it holds
    Cameron Young and Carson Young at once.

    As a MATCH that was a coin flip between two real people. As a SUGGESTION it is two
    rows in a file with a person reading them, which is the entire difference.
    """
    field = [{"name": "Cameron Young", "athlete_id": "1"},
             {"name": "Carson Young", "athlete_id": "2"}]
    top = espn.suggest_matches("Cam Young", field)
    assert [s["espn_name"] for s in top] == ["Cameron Young", "Carson Young"]
    assert top[0]["confidence"] == top[1]["confidence"], (
        "two candidates that tie must not be presented as one confident answer")


def test_a_suggestion_carries_the_reason_it_was_offered(espn_players):
    """A reviewer confirming a binding wants the reason, not a number."""
    reasons = {s["why"] for name in ("Zachary Bauchou", "Cameron Smith")
               for s in espn.suggest_matches(name, espn_players)}
    assert reasons <= {"same first initial and last name", "same last name",
                       "same first name", "similar spelling"} | {
                           r for r in reasons if r.startswith("shares ")}


def test_a_name_nobody_in_the_field_resembles_gets_no_suggestions(espn_players):
    """
    Which is itself the answer. An ESPN field and a Kalshi field for one tournament are
    very nearly the same people, so a name with nothing close to it is a withdrawal --
    and an empty suggestion list says so more clearly than a bad guess would.
    """
    assert espn.suggest_matches("Xavier Quetzalcoatl", espn_players) == []


def test_suggestions_are_capped_and_ordered_by_confidence(espn_players):
    top = espn.suggest_matches("Matthew McCarty", espn_players, limit=3)
    assert len(top) <= 3
    assert [s["confidence"] for s in top] == sorted((s["confidence"] for s in top), reverse=True)
    assert all(s["confidence"] >= espn.SUGGESTION_FLOOR for s in top)


def test_unclaimed_is_the_other_half_of_a_review(espn_players):
    """
    A reviewer given only the unmatched Kalshi names is guessing. Given both lists, the
    answer is usually obvious -- and a SHORT unclaimed list is itself the evidence that
    nothing was missed.
    """
    names = [p["name"] for p in espn_players[:5]]
    matches, _ = espn.match_field(names, espn_players)
    free = espn.unclaimed(espn_players, matches)
    assert len(free) == len(espn_players) - 5
    assert not ({p["name"] for p in free} & set(names))


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

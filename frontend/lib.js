/*
 * lib.js -- the parts of the scoreboard that are rules rather than presentation.
 *
 * ESPN leaderboard parsing and the standings rule. Kept separate from app.js and
 * free of the DOM so it can be run under Node against the same fixtures as the
 * Python side: `tests/test_frontend_parity.py` feeds both this file and standings.py
 * the checked-in ESPN payload and fails if they disagree about who is winning.
 *
 * That parity test is the reason this file exists. Two implementations of a rule is
 * one implementation and one rumour unless something checks them against each other.
 *
 * THERE IS NO NAME MATCHING HERE, AND THAT IS THE POINT
 * -----------------------------------------------------
 * This file used to carry a transliteration table, a name normaliser and a
 * first-initial-and-last-name fallback, mirrored character for character against
 * espn_leaderboard.py. It needed them because a page could be built before ESPN
 * published a field, and then the golfer->athlete join had to happen in the browser,
 * against a leaderboard the build had never seen.
 *
 * A page is no longer built that way. A build made before the field exists produces
 * a groups sheet with `live: null` and does not poll at all; a build made after it
 * exists has already resolved every golfer it could to an ESPN athlete id, and that
 * id is baked into the page. So the runtime join is a Map lookup on an integer, it
 * cannot pick the wrong Smith, and the second copy of a subtle string algorithm that
 * had to be kept in step with the first is simply gone.
 *
 * A golfer with no athlete id resolves to null, scores nothing and is not counted.
 * Which is correct: they either withdrew or nobody has settled their name yet, and
 * the result file says which. See match_review.py.
 *
 * ONE COPY, ABOVE THE TEMPLATES
 * -----------------------------
 * It lives in `frontend/` rather than inside a template directory because more than
 * one template ships: `frontend/scoreboard/` is the designed page and
 * `frontend/template/` is the plain reference. Both pull this file in from ../lib.js
 * with a script tag, which the bundler inlines like any other local asset, so the rule
 * that decides the pool exists once and the parity test covers every page built from
 * it. A template that copied it would be a second rumour.
 *
 * A new design should do the same: reuse this file verbatim and write only its own
 * app.js. See docs/FRONTEND-SPEC.md.
 */
'use strict';

var GolfPool = (function () {

  /* ---------------------------------------------------------------- *
   * Finding a golfer's ESPN row.
   * ---------------------------------------------------------------- */

  /* athlete id -> parsed player, keyed as a string because the result file writes
   * ESPN's ids as strings and the leaderboard payload is not consistent about it. */
  function indexByAthleteId(players) {
    var byId = new Map();
    players.forEach(function (p) {
      if (p.athleteId != null) byId.set(String(p.athleteId), p);
    });
    return byId;
  }

  /* -> the ESPN player, or null. Null is a real answer here and means "not scoring
   * this week": either the build confirmed this golfer is not in the field, or it
   * could not settle their name and said so. Both are in the result file. */
  function resolveGolfer(golfer, byId) {
    var id = golfer.espn && golfer.espn.athlete_id;
    if (id == null) return null;
    return byId.get(String(id)) || null;
  }

  /* ---------------------------------------------------------------- *
   * ESPN leaderboard parsing -- mirrors espn_leaderboard.parse_leaderboard.
   *
   * `sortOrder` is the live rank and a total order over the whole field.
   * `score.displayValue` counts COMPLETED ROUNDS ONLY and was wrong for 42 of 147
   * players in the measured mid-round payload, so the live total is summed from
   * the linescores -- which carry stub entries for rounds not yet played.
   *
   * ESPN posts the FIELD about two days before the first round and posts no
   * POSITIONS with it: measured 2026-08-04 on the Wyndham, 147 competitors with
   * athlete ids and tee times, every one of them at position "-". So a payload
   * full of players is not a tournament in progress, and `meta.started` is what
   * tells the two apart. Nothing ranks until it is true.
   * ---------------------------------------------------------------- */

  function toPar(value) {
    var s = String(value == null ? '' : value).trim();
    if (s === 'E') return 0;
    var n = parseInt(s.replace('+', ''), 10);
    return isNaN(n) ? null : n;
  }

  function fmtPar(n) {
    if (n === null || n === undefined) return '—';
    return n === 0 ? 'E' : (n > 0 ? '+' + n : String(n));
  }

  /* 'T12' -> 12, '3' -> 3, '-' -> null. null is the honest answer for a cut,
   * withdrawn or disqualified player: ESPN gives them no position. */
  function positionNumber(display) {
    var m = /^T?(\d+)$/.exec(String(display == null ? '' : display).trim());
    return m ? parseInt(m[1], 10) : null;
  }

  /* Has anybody teed off? Mirrors espn_leaderboard.has_started, and the two are
   * checked against each other by tests/test_frontend_parity.py.
   *
   * Two signals, either sufficient, because they fail in opposite directions.
   * `state` is ESPN's own answer, read off the event envelope; a golfer holding a
   * real position is proof from the field itself. Requiring both would blank the
   * board on a good payload with an odd envelope. Requiring neither is what this
   * exists to prevent: a pre-tournament field ranks perfectly well on sortOrder
   * and means nothing at all. */
  function hasStarted(state, players) {
    if (state === 'in' || state === 'post') return true;
    return players.some(function (p) {
      return p.positionNumber !== null && p.positionNumber !== undefined;
    });
  }

  function parseLeaderboard(payload) {
    var events = (payload && payload.events) || [];
    if (!events.length) return { meta: null, players: [] };
    var ev = events[0];
    var comp = (ev.competitions || [{}])[0];
    var course = (ev.courses || [{}])[0];

    var meta = {
      eventId: ev.id,
      event: ev.name,
      course: course.name,
      par: course.shotsToPar,
      round: (comp.status || {}).period,
      detail: ((comp.status || {}).type || {}).detail,
      state: ((ev.status || {}).type || {}).state,
      completed: !!((ev.status || {}).type || {}).completed,
      start: ev.date,
      end: ev.endDate
    };

    var players = (comp.competitors || []).map(function (c) {
      var a = c.athlete || {};
      var st = c.status || {};
      var pos = st.position || {};
      var rounds = (c.linescores || [])
        .filter(function (ls) { return Object.prototype.hasOwnProperty.call(ls, 'displayValue'); })
        .map(function (ls) { return { round: ls.period, toPar: toPar(ls.displayValue) }; })
        .filter(function (r) { return r.toPar !== null; });
      var live = rounds.length
        ? rounds.reduce(function (s, r) { return s + r.toPar; }, 0)
        : null;
      return {
        athleteId: a.id,
        name: a.displayName,
        headshot: (a.headshot || {}).href,
        country: (a.flag || {}).alt,
        amateur: !!c.amateur,
        sortOrder: c.sortOrder == null ? 9999 : c.sortOrder,
        position: pos.displayName,
        positionNumber: positionNumber(pos.displayName),
        tied: !!pos.isTie,
        thru: st.displayThru || st.thru,
        teeTime: st.teeTime,
        status: (st.type || {}).name,
        statusShort: st.displayValue,
        toPar: live,
        rounds: rounds
      };
    });

    players.sort(function (a, b) { return a.sortOrder - b.sortOrder; });
    /* Derived here so there is one answer to it. A field exists from about two days
     * out; a leaderboard does not exist until somebody hits a ball. */
    meta.started = hasStarted(meta.state, players);
    return { meta: meta, players: players };
  }

  /* ---------------------------------------------------------------- *
   * The standings rule -- mirrors standings.py. See docs/FRONTEND-SPEC.md.
   *
   * A team ranks on the best leaderboard position it holds; ties break on the
   * next-best golfer, and so on. A team that runs out of golfers loses to one
   * that has not. That is a lexicographic comparison of each team's golfer ranks
   * in ascending order, with the shorter roster padded by something worse than
   * anything real.
   * ---------------------------------------------------------------- */

  var PADDING = [3, 0];

  /* (tier, value). Never a bare number: a cut golfer is in a different class
   * from everyone still playing, and collapsing the two lets a team of cut
   * golfers outrank a team holding someone in contention. */
  function golferRank(player) {
    if (!player) return [2, 0];
    if (player.positionNumber !== null && player.positionNumber !== undefined) {
      return [0, player.positionNumber];
    }
    return [1, player.sortOrder == null ? 9999 : player.sortOrder];
  }

  function cmpRank(a, b) { return a[0] - b[0] || a[1] - b[1]; }

  /* -1 / 0 / +1 plus the 1-based depth that decided it (null when fully tied).
   * The depth is the whole story of a close pool day and is free here. */
  function compareVectors(va, vb) {
    var depth = Math.max(va.length, vb.length);
    for (var i = 0; i < depth; i++) {
      var c = cmpRank(va[i] || PADDING, vb[i] || PADDING);
      if (c !== 0) return { result: c < 0 ? -1 : 1, at: i + 1 };
    }
    return { result: 0, at: null };
  }

  /* teams: [{team_id, ...}]; golfersByTeam: Map(team_id -> [golfer]);
   * resolve: golfer -> ESPN player | null. */
  function computeStandings(teams, golfersByTeam, resolve) {
    var rows = teams.map(function (team) {
      var detail = (golfersByTeam.get(team.team_id) || []).map(function (g) {
        var player = resolve(g);
        return {
          golfer: g,
          player: player,
          rank: golferRank(player),
          inField: !!player,
          madeCut: !!(player && player.positionNumber !== null && player.positionNumber !== undefined)
        };
      });
      detail.sort(function (a, b) { return cmpRank(a.rank, b.rank); });
      var scores = detail
        .filter(function (d) { return d.player && d.player.toPar !== null; })
        .map(function (d) { return d.player.toPar; });
      return {
        team: team,
        golfers: detail,
        vector: detail.map(function (d) { return d.rank; }),
        best: detail[0] || null,
        counting: detail.filter(function (d) { return d.rank[0] === 0; }).length,
        inField: detail.filter(function (d) { return d.inField; }).length,
        roster: detail.length,
        toPar: scores.length ? scores.reduce(function (s, v) { return s + v; }, 0) : null
      };
    });

    // Total and stable: the team_id tie-break decides nothing about who is
    // winning -- rows that reach it equal are marked tied and share a rank.
    rows.sort(function (a, b) {
      return compareVectors(a.vector, b.vector).result ||
             String(a.team.team_id).localeCompare(String(b.team.team_id));
    });

    var rank = 1;
    rows.forEach(function (row, i) {
      row.tied = false;
      row.decidedAt = null;
      if (i > 0) {
        var c = compareVectors(rows[i - 1].vector, row.vector);
        if (c.result === 0) {
          row.tied = true;
          rows[i - 1].tied = true;
        } else {
          rank = i + 1;
          row.decidedAt = c.at;
        }
      }
      row.rank = rank;
    });
    // A row can be marked tied by the row after it, so positions are only correct
    // once the whole pass is done.
    rows.forEach(function (r) { r.position = r.tied ? 'T' + r.rank : String(r.rank); });
    return rows;
  }

  return {
    indexByAthleteId: indexByAthleteId,
    resolveGolfer: resolveGolfer,
    toPar: toPar,
    fmtPar: fmtPar,
    positionNumber: positionNumber,
    hasStarted: hasStarted,
    parseLeaderboard: parseLeaderboard,
    golferRank: golferRank,
    compareVectors: compareVectors,
    computeStandings: computeStandings,
    PADDING: PADDING
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = GolfPool;

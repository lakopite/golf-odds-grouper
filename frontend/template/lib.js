/*
 * lib.js -- the parts of the scoreboard that are rules rather than presentation.
 *
 * Name matching, ESPN leaderboard parsing, and the standings rule. Kept separate
 * from app.js and free of the DOM so it can be run under Node against the same
 * fixtures as the Python side: `tests/test_frontend_parity.py` feeds both this file
 * and standings.py the checked-in ESPN payload and fails if they disagree about who
 * is winning.
 *
 * That parity test is the reason this file exists. Two implementations of a rule is
 * one implementation and one rumour unless something checks them against each other.
 *
 * A designed scoreboard should reuse this file verbatim and write only its own
 * app.js. See docs/FRONTEND-SPEC.md.
 */
'use strict';

var GolfPool = (function () {

  /* ---------------------------------------------------------------- *
   * Name matching -- mirrors espn_leaderboard.py.
   *
   * Needed at runtime as well as at build time because a pre-tournament ESPN
   * event returns ZERO competitors: the field does not exist until play starts,
   * so a Wednesday-night build has nothing to join against.
   * ---------------------------------------------------------------- */

  var SUFFIXES = /\b(jr|sr|ii|iii|iv|v)\b\.?/g;

  /* Letters NFKD does not decompose, because they are letters in their own right
   * rather than a base plus an accent. A golf field is full of them -- Rasmus
   * Højgaard and Thorbjørn Olesen are both in the measured Wyndham field -- and one
   * source writing "Hojgaard" while the other writes "Højgaard" is exactly the miss
   * this table prevents. Must stay in step with _TRANSLITERATE in
   * espn_leaderboard.py; tests/test_frontend_parity.py fails if it drifts. */
  var TRANSLITERATE = {
    'ø': 'o', 'Ø': 'o', 'æ': 'ae', 'Æ': 'ae', 'å': 'a', 'Å': 'a',
    'ð': 'd', 'Ð': 'd', 'þ': 'th', 'Þ': 'th', 'ł': 'l', 'Ł': 'l',
    'đ': 'd', 'Đ': 'd', 'ß': 'ss', 'œ': 'oe', 'Œ': 'oe', 'ı': 'i'
  };

  function transliterate(text) {
    var out = '';
    for (var i = 0; i < text.length; i++) {
      var ch = text[i];
      out += Object.prototype.hasOwnProperty.call(TRANSLITERATE, ch) ? TRANSLITERATE[ch] : ch;
    }
    return out;
  }

  /* Join runs of CONSECUTIVE single letters, so "C.T. Pan" and "CT Pan" agree, as do
   * "J.J. Spaun"/"JJ Spaun". Only consecutive ones: "Jordan L. Smith" keeps its
   * middle initial as its own token, which is what lets initialLastKey drop it and
   * reach "Jordan Smith". */
  function joinInitials(parts) {
    var out = [], prevSingle = false;
    parts.forEach(function (part) {
      var single = part.length === 1;
      if (single && prevSingle) out[out.length - 1] += part;
      else out.push(part);
      prevSingle = single;
    });
    return out;
  }

  function normalizeName(name) {
    var s = transliterate(String(name == null ? '' : name))
      .normalize('NFKD')
      .replace(/\p{M}/gu, '')          // combining marks NFKD split off
      .toLowerCase()
      .replace(/-/g, ' ')
      .replace(/['’]/g, '')
      .replace(SUFFIXES, ' ')
      .replace(/[^a-z ]/g, ' ');
    return joinInitials(s.split(/\s+/).filter(Boolean)).join(' ');
  }

  /* (first initial, last name). Resolves Zachary/Zach, Cameron/Cam, Matthew/Matt
   * and stray middle initials. Measured collisions inside a real 147-player
   * field: zero. */
  function initialLastKey(name) {
    var parts = normalizeName(name).split(' ').filter(Boolean);
    if (parts.length < 2) return null;
    return parts[0][0] + '|' + parts[parts.length - 1];
  }

  function buildIndex(players) {
    var exact = new Map(), initial = new Map(), byId = new Map();
    function bump(map, key, player) {
      if (!key) return;
      if (map.has(key)) map.set(key, null);   // ambiguous: refuse rather than guess
      else map.set(key, player);
    }
    players.forEach(function (p) {
      if (p.athleteId != null) byId.set(String(p.athleteId), p);
      if (!p.name) return;
      bump(exact, normalizeName(p.name), p);
      bump(initial, initialLastKey(p.name), p);
    });
    return { exact: exact, initial: initial, byId: byId };
  }

  /* -> {player, how}. how is athlete_id / alias / exact / initial_last / unresolved. */
  function matchGolfer(golfer, index, aliases) {
    aliases = aliases || {};
    var espnId = golfer.espn && golfer.espn.athlete_id;
    if (espnId && index.byId.get(String(espnId))) {
      return { player: index.byId.get(String(espnId)), how: 'athlete_id' };
    }
    var alias = aliases[golfer.name] || aliases[normalizeName(golfer.name)];
    if (alias) {
      var viaAlias = index.exact.get(normalizeName(alias));
      if (viaAlias) return { player: viaAlias, how: 'alias' };
    }
    var exact = index.exact.get(normalizeName(golfer.name));
    if (exact) return { player: exact, how: 'exact' };

    var key = initialLastKey(golfer.name);
    if (key) {
      var viaInitial = index.initial.get(key);
      if (viaInitial) return { player: viaInitial, how: 'initial_last' };
    }
    return { player: null, how: 'unresolved' };
  }

  /* ---------------------------------------------------------------- *
   * ESPN leaderboard parsing -- mirrors espn_leaderboard.parse_leaderboard.
   *
   * `sortOrder` is the live rank and a total order over the whole field.
   * `score.displayValue` counts COMPLETED ROUNDS ONLY and was wrong for 42 of 147
   * players in the measured mid-round payload, so the live total is summed from
   * the linescores -- which carry stub entries for rounds not yet played.
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
    normalizeName: normalizeName,
    initialLastKey: initialLastKey,
    buildIndex: buildIndex,
    matchGolfer: matchGolfer,
    toPar: toPar,
    fmtPar: fmtPar,
    positionNumber: positionNumber,
    parseLeaderboard: parseLeaderboard,
    golferRank: golferRank,
    compareVectors: compareVectors,
    computeStandings: computeStandings,
    PADDING: PADDING
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = GolfPool;

/*
 * app.js -- the designed scoreboard's presentation and network layer.
 *
 * The rules are not in here. Ranking, leaderboard parsing and score arithmetic all live
 * in ../lib.js, which `tests/test_frontend_parity.py` runs against standings.py on the
 * same payloads; this file decides what the page looks like and nothing else. That
 * split is the reason a design can be replaced without anybody re-deriving the rule
 * that decides who won.
 *
 * No framework, no build step, no network on load. The competition is baked into the
 * page and ESPN is the only host it ever asks for anything -- the only one that will
 * answer a browser at all (see docs/FRONTEND-SPEC.md §4 on Kalshi's origin allowlist).
 *
 * TWO PAGES, ONE FILE
 * -------------------
 * `DATA.live` is null when the competition was built before ESPN published a field.
 * That page is a groups sheet: teams, rosters, the odds the groups were drawn on, and
 * no network of any kind. It is not a scoreboard waiting for a fetch to succeed --
 * there is nothing to fetch, because the field does not exist yet.
 *
 * `DATA.live` non-null means the field existed at build time and every golfer the join
 * settled carries an ESPN athlete id. That page polls, joins on the id, and ranks.
 *
 * AND ONE MORE STATE THAT IS NEITHER
 * ----------------------------------
 * A live page can be holding an empty board any afternoon of the week: ESPN down, a
 * payload refused for the wrong event, a first poll still in flight. It then renders
 * exactly like the groups sheet -- every roster, every price, no ranking -- and the
 * status pill says which of the reasons it is, in four words rather than a paragraph.
 * `ranked()` below is that distinction, and it is the only thing in this file that
 * decides whether a position ever appears on screen.
 *
 * ONE PRICE PER GOLFER
 * --------------------
 * The odds were read once, when the groups were drawn, and that reading is baked in.
 * There is no second reading and no way to ask for one, so nothing on this page shows
 * a price moving -- not a feed, not an arrow, not a stale number waiting to be
 * refreshed. What a golfer was worth on draw day is what this page says he is worth.
 */
'use strict';

var DATA = JSON.parse(document.getElementById('competition-data').textContent);

/* The whole of the difference between the two pages above. */
var LIVE = DATA.live || null;

var GOLFERS_BY_TEAM = new Map();
DATA.golfers.forEach(function (g) {
  if (!g.team_id) return;
  if (!GOLFERS_BY_TEAM.has(g.team_id)) GOLFERS_BY_TEAM.set(g.team_id, []);
  GOLFERS_BY_TEAM.get(g.team_id).push(g);
});

var STATE = {
  view: 'standings',
  open: Object.create(null),
  players: [],
  meta: null,
  index: null,
  error: null,
  lastGood: null
};

var POLL_SECONDS = (LIVE && LIVE.poll_interval_seconds) || 60;

/* Older than three polls with nothing new is stale rather than live, and saying so is
 * cheaper than being quietly wrong on a Sunday. */
var STALE_AFTER_MS = POLL_SECONDS * 3000;

function $(id) { return document.getElementById(id); }

function el(tag, cls, text) {
  var n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

/* By ESPN athlete id and nothing else. A golfer without one is not in the field, or has
 * not been settled, and either way has no row on the leaderboard to show. */
function resolvePlayer(golfer) {
  if (!STATE.index) return null;
  return GolfPool.resolveGolfer(golfer, STATE.index);
}

/* ------------------------------------------------------------------ *
 * Formatting
 * ------------------------------------------------------------------ */

function pct(v, places) { return v == null ? '—' : (v * 100).toFixed(places === undefined ? 2 : places) + '%'; }

/* The strongest golfer in the field, for the weight bars on a groups sheet. That page
 * has no scores to show, and a column of percentages is a column of percentages; the
 * bar is what makes "these two groups are worth the same" visible at a glance, which is
 * the only claim the page is making before anybody tees off. */
var MAX_WEIGHT = DATA.golfers.reduce(function (m, g) {
  return Math.max(m, g.odds.grouping_weight || 0);
}, 0);

function stamp(iso) {
  if (!iso) return null;
  var d = new Date(iso);
  return isNaN(d.getTime()) ? null : d;
}

function shortDate(d) {
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
}

function dateRange(startIso, endIso) {
  var a = stamp(startIso), b = stamp(endIso);
  if (!a) return null;
  var year = (b || a).getFullYear();
  return (b && b.getTime() !== a.getTime() ? shortDate(a) + '–' + shortDate(b) : shortDate(a))
    + ' ' + year;
}

/* The partition's delta, at a size somebody can read.
 *
 * It is a difference between two probabilities and arrives as a raw float, so a good
 * partition produces something like 9.192866335723479e-05 -- twenty characters of
 * spurious precision set in a 30px display face, on the one number that is supposed to
 * say "these groups are worth the same". Six places is well below the tick grid the
 * odds live on; anything smaller than that is a rounding artefact and says so. */
function fmtDelta(v) {
  if (v === null || v === undefined) return '—';
  var n = Number(v);
  if (!isFinite(n)) return '—';
  if (n === 0) return '0';
  return n >= 0.000001 ? n.toFixed(6) : n.toExponential(1);
}

function since(d) {
  var s = Math.max(0, Math.round((Date.now() - d.getTime()) / 1000));
  if (s < 60) return s + 's ago';
  var m = Math.floor(s / 60);
  if (m < 60) return m + 'm ' + (s % 60) + 's ago';
  return Math.floor(m / 60) + 'h ' + (m % 60) + 'm ago';
}

/* Two initials, for a team with no logo. Not a fallback so much as the other half of
 * the design: the badge is always there, and only its contents change. */
function initials(name) {
  var words = String(name || '').split(/\s+/).filter(Boolean);
  if (!words.length) return '??';
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[words.length - 1][0]).toUpperCase();
}

/* ------------------------------------------------------------------ *
 * The one thing that decides whether a position appears
 * ------------------------------------------------------------------ */

function ranked() { return !!LIVE && STATE.players.length > 0; }

/* ------------------------------------------------------------------ *
 * Masthead. Baked data only, so it is drawn once.
 * ------------------------------------------------------------------ */

function renderBrand() {
  var league = DATA.league;
  $('league-name').textContent = league.league_name;

  if (league.tagline) {
    $('league-tagline').textContent = league.tagline;
    $('league-tagline').hidden = false;
  }
  // A crest or banner is a data: URI by the time it gets here, or an absolute URL, or
  // null. Null is the common case and gets no element at all rather than a broken one.
  if (league.crest) {
    var crest = $('league-crest');
    crest.src = league.crest;
    crest.alt = league.league_name + ' crest';
    crest.hidden = false;
  }
  if (league.banner) {
    $('league-banner').src = league.banner;
    $('league-banner').alt = league.league_name + (league.tagline ? ' — ' + league.tagline : '');
    $('bannerwrap').hidden = false;
  }

  $('tournament').textContent = DATA.tournament.name;

  var course = DATA.tournament.course || {};
  var bits = [];
  if (course.name) bits.push(course.name);
  if (course.par) bits.push('par ' + course.par);
  var when = dateRange(DATA.tournament.start, DATA.tournament.end);
  if (when) bits.push(when);
  $('event-sub').textContent = bits.join(' · ');

  $('nav-standings-label').textContent = LIVE ? 'Standings' : 'Groups';
  $('standings-heading').textContent = LIVE ? 'Standings' : 'Groups';
}

/* ------------------------------------------------------------------ *
 * Status pill. Redrawn on a one-second tick as well as on every poll, so "updated 14s
 * ago" is true when it is read rather than when it was written -- and so a poll that
 * silently stops answering turns the pill amber without waiting for a reply that is
 * not coming.
 * ------------------------------------------------------------------ */

function statusView() {
  if (!LIVE) {
    return { cls: '', label: 'Not started',
             note: 'Built before the field posted · this page fetches nothing' };
  }
  var good = STATE.lastGood;
  if (STATE.error) {
    return { cls: 'is-down', label: 'ESPN unreachable',
             note: STATE.error + (good ? ' · showing the board from ' + good.toLocaleTimeString()
                                       : ' · nothing has arrived yet') };
  }
  if (!good) {
    return { cls: '', label: 'Waiting', note: 'asking ESPN for the leaderboard' };
  }
  var meta = STATE.meta || {};
  if (meta.completed) {
    return { cls: 'is-live', label: 'Final',
             note: (meta.detail || 'Final') + ' · this will not change again' };
  }
  if (Date.now() - good.getTime() > STALE_AFTER_MS) {
    return { cls: 'is-stale', label: 'Stale',
             note: 'ESPN last answered ' + since(good) + ' · retrying every ' + POLL_SECONDS + 's' };
  }
  // ESPN answering with an empty field is a working page and a tournament that has not
  // begun, not a live board. A green dot over no positions reads as a page that has
  // broken rather than one that is early.
  if (!STATE.players.length) {
    return { cls: '', label: 'Not started',
             note: 'ESPN answered ' + since(good) + ' with no field yet · retrying every '
                   + POLL_SECONDS + 's' };
  }
  return { cls: 'is-live', label: 'Live',
           note: 'Leaderboard updated ' + since(good) + ' · polling ESPN every ' + POLL_SECONDS + 's' };
}

function renderStatus() {
  var s = statusView();
  $('status-pill').className = 'pill ' + s.cls;
  $('status-label').textContent = s.label;
  $('status-note').textContent = s.note;
}

/* ------------------------------------------------------------------ *
 * The league table
 * ------------------------------------------------------------------ */

/* One golfer, ready to draw. `player` is null before the first tee time and for anyone
 * who never teed off, and the row still has to say who they are and what they were
 * worth -- a roster does not shrink because the leaderboard has not opened.
 *
 * Those two nulls are different, which is why "out" is conditioned on the board
 * existing: once it does, a golfer with no player is a golfer who is not playing and
 * belongs greyed out AND labelled. Before it does, nobody is out of anything. */
function golferView(golfer, player, isRanked) {
  var madeCut = !!(player && player.positionNumber !== null && player.positionNumber !== undefined);
  var espn = golfer.espn || null;
  var tag = '';
  if (isRanked && !madeCut) {
    if (player) tag = player.statusShort || 'CUT';
    else if (!espn) tag = 'no join';
    else if (espn.match === 'absent') tag = 'not in field';
    else if (espn.match === 'unresolved') tag = 'unresolved';
    else tag = 'not on board';
  }
  return {
    out: isRanked && !madeCut,
    // A cut golfer's ESPN position is the string "-", and a golfer with no row has no
    // position at all. Both read as an em dash; what happened to them is the tag's job,
    // and printing "CUT" in three columns is not three facts.
    pos: madeCut ? player.position : '—',
    name: golfer.name,
    score: player ? GolfPool.fmtPar(player.toPar) : '—',
    thru: madeCut ? String(player.thru || '') : '',
    odds: pct(golfer.odds.grouping_weight),
    bar: MAX_WEIGHT ? ((golfer.odds.grouping_weight || 0) / MAX_WEIGHT * 100).toFixed(1) + '%' : '0%',
    tag: tag,
    unresolved: !!(espn && espn.match === 'unresolved')
  };
}

/* Ranked. The rule is lib.js's, verbatim, and everything here is a label on it. */
function liveRows() {
  return GolfPool.computeStandings(DATA.teams, GOLFERS_BY_TEAM, resolvePlayer).map(function (row) {
    var best = row.best;
    var bestPlayer = best && best.player;
    return {
      team: row.team,
      position: row.position,
      leader: row.rank === 1,
      tied: row.tied,
      decidedAt: row.decidedAt,
      leadName: best ? best.golfer.name : 'no golfers',
      leadLine: bestPlayer
        ? (bestPlayer.position || bestPlayer.statusShort || 'CUT') + ' · '
          + GolfPool.fmtPar(bestPlayer.toPar)
          + (bestPlayer.thru ? ' · thru ' + bestPlayer.thru : '')
        : (best ? 'not on the board' : '—'),
      leadLive: !!(best && best.rank[0] === 0),
      colB: row.counting + '/' + row.roster,
      colC: row.decidedAt ? 'golfer #' + row.decidedAt : (row.tied ? 'tied — unbroken' : '—'),
      colCDecided: !!row.decidedAt,
      odds: pct(row.team.total_odds),
      // Enough of the group to show the golfer that decided the position, and never
      // fewer than three -- a one-line group reads as an error.
      keep: Math.min(row.golfers.length, Math.max(3, row.decidedAt || 0)),
      golfers: row.golfers.map(function (d) { return golferView(d.golfer, d.player, true); })
    };
  });
}

/* Not ranked. Everything about the pool is already decided and is worth showing: who
 * holds whom, what each golfer was worth when the groups were drawn, and that the draw
 * came out even. Ranking anyway would order teams by roster size and present it as a
 * leaderboard, which is worse than saying "not started". */
function drawRows() {
  return DATA.teams.slice()
    .sort(function (a, b) { return (b.total_odds || 0) - (a.total_odds || 0); })
    .map(function (team) {
      var golfers = (GOLFERS_BY_TEAM.get(team.team_id) || []).slice()
        .sort(function (a, b) {
          return (b.odds.grouping_weight || 0) - (a.odds.grouping_weight || 0);
        });
      return {
        team: team,
        position: '—',
        leader: false,
        tied: false,
        decidedAt: null,
        leadName: golfers.length ? golfers[0].name : 'no golfers',
        leadLine: golfers.length
          ? pct(golfers[0].odds.grouping_weight) + ' · no position yet' : '—',
        leadLive: false,
        colB: String(team.golfer_count),
        colC: 'group ' + (team.group_index + 1) + ' of ' + DATA.league.team_count,
        colCDecided: false,
        odds: pct(team.total_odds),
        keep: Math.min(golfers.length, 3),
        golfers: golfers.map(function (g) { return golferView(g, null, false); })
      };
    });
}

function badge(team) {
  var node = el('div', 'mono');
  if (team.team_logo) {
    var img = document.createElement('img');
    img.src = team.team_logo;
    img.alt = '';
    node.append(img);
  } else {
    node.className = 'mono is-empty';
    node.textContent = initials(team.team_name);
  }
  return node;
}

function groupTable(row, isRanked) {
  var table = el('table', 'group' + (isRanked ? '' : ' is-draw'));

  var head = document.createElement('thead');
  var hr = document.createElement('tr');
  [['g-pos', 'Pos'], ['g-name', 'Golfer'], ['g-score', 'Score'], ['g-thru', 'Thru'],
   ['g-bar', ''], ['g-odds', 'Draw'], ['g-tag', '']].forEach(function (pair) {
    var th = el('th', pair[0], pair[1]);
    th.scope = 'col';
    hr.append(th);
  });
  head.append(hr);
  table.append(head);

  var body = document.createElement('tbody');
  row.golfers.forEach(function (g, i) {
    var tr = el('tr', 'golfer' + (g.out ? ' out' : '') + (i >= row.keep ? ' is-rest' : ''));
    tr.append(el('td', 'g-pos', g.pos));

    var name = el('td', 'g-name');
    var inner = el('div', 'gname-inner');
    inner.append(el('span', null, g.name));
    if (row.decidedAt === i + 1) inner.append(el('span', 'decided-here', 'decided here'));
    name.append(inner);
    tr.append(name);

    tr.append(el('td', 'g-score', g.score));
    tr.append(el('td', 'g-thru', g.thru));

    var barTd = el('td', 'g-bar');
    var bar = el('span', 'bar');
    var fill = el('i');
    fill.style.width = g.bar;
    bar.append(fill);
    barTd.append(bar);
    tr.append(barTd);

    tr.append(el('td', 'g-odds', g.odds));
    tr.append(el('td', 'g-tag' + (g.unresolved ? ' is-unresolved' : ''), g.tag));
    body.append(tr);
  });
  table.append(body);
  return table;
}

function teamBlock(row, isRanked) {
  var open = !!STATE.open[row.team.team_id];
  var hidden = row.golfers.length - row.keep;

  var tbody = el('tbody', 'team' + (row.leader ? ' is-leader' : '')
                          + (row.tied ? ' is-tied' : '') + (open ? ' is-open' : ''));
  tbody.dataset.teamId = row.team.team_id;

  var tr = el('tr', 'team-row');
  tr.tabIndex = 0;
  tr.setAttribute('role', 'button');
  tr.setAttribute('aria-expanded', open ? 'true' : 'false');
  tr.setAttribute('aria-label', row.team.team_name + ', ' + row.golfers.length + ' golfers');

  tr.append(el('td', 'c-rk pos', row.position));

  var teamTd = el('td', 'c-team');
  var cell = el('div', 'teamcell');
  cell.append(badge(row.team));
  var names = el('div', 'teamnames');
  names.append(el('div', 'tname', row.team.team_name));
  names.append(el('div', 'towner', row.team.player_name));
  cell.append(names);
  teamTd.append(cell);
  tr.append(teamTd);

  var lead = el('td', 'c-lead');
  lead.append(el('div', 'leadname', row.leadName));
  lead.append(el('div', 'leadline' + (row.leadLive ? ' is-live' : ''), row.leadLine));
  tr.append(lead);

  tr.append(el('td', 'c-b', row.colB));
  tr.append(el('td', 'c-c' + (row.colCDecided ? ' is-decided' : ''), row.colC));
  tr.append(el('td', 'c-odds', row.odds));

  var chev = el('td', 'c-chev');
  chev.append(el('span', 'chev', open ? '−' : '+'));
  tr.append(chev);
  tbody.append(tr);

  var detail = el('tr', 'team-detail');
  var cellTd = document.createElement('td');
  cellTd.colSpan = 7;
  cellTd.append(groupTable(row, isRanked));
  if (hidden > 0) {
    var more = el('button', 'more', open
      ? 'Hide the rest of the group'
      : 'Show all ' + row.golfers.length + ' golfers (' + hidden + ' more)');
    more.type = 'button';
    more.addEventListener('click', function () { toggle(row.team.team_id); });
    cellTd.append(more);
  }
  detail.append(cellTd);
  tbody.append(detail);

  function fire() { toggle(row.team.team_id); }
  tr.addEventListener('click', fire);
  tr.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') { e.preventDefault(); fire(); }
  });
  return tbody;
}

function toggle(teamId) {
  STATE.open[teamId] = !STATE.open[teamId];
  renderStandings();
  // The board is rebuilt from scratch, which throws away the element the keyboard was
  // on. Put it back, or Enter on a team row silently drops focus to the document and
  // the next Tab starts again from the top of the page.
  var blocks = document.querySelectorAll('#board tbody.team');
  for (var i = 0; i < blocks.length; i++) {
    if (blocks[i].dataset.teamId === teamId) {
      blocks[i].querySelector('.team-row').focus();
      return;
    }
  }
}

function renderStandings() {
  var isRanked = ranked();
  var rows = isRanked ? liveRows() : drawRows();
  var meta = STATE.meta || {};

  /* A few words, not a paragraph. Which of the several ways there is nothing to rank
   * yet -- ESPN down, wrong event, no field posted, built before the tournament -- is
   * the status pill's job, and it says it in two words at the top of the page. Saying
   * it twice, the second time at length, read as an apology for a page that was
   * working exactly as built. */
  $('standings-sub').textContent = isRanked
    ? (meta.detail || (meta.round ? 'Round ' + meta.round : 'In play'))
      + ' · best position held wins'
    : 'The draw. Nothing is ranked yet.';

  $('board-title').textContent = isRanked ? 'League table' : 'League table · the draw';
  $('board-note').textContent = DATA.league.team_count + ' teams · '
    + DATA.grouping.grouped_golfers + ' golfers grouped · tap a team for its full group';
  $('head-lead').textContent = isRanked ? 'Leading golfer' : 'Shortest price held';
  $('head-b').textContent = isRanked ? 'In it' : 'Golfers';
  $('head-c').textContent = isRanked ? 'Decided on' : 'Group';

  var board = $('board');
  board.className = 'board' + (isRanked ? '' : ' is-draw');
  while (board.tBodies.length) board.removeChild(board.tBodies[0]);
  rows.forEach(function (row) { board.append(teamBlock(row, isRanked)); });

  renderChips(isRanked);
}

function renderChips(isRanked) {
  var chips = [];
  if (DATA.rebuilt_from) chips.push('rebuild #' + (DATA.rebuilt_from.rebuild_count || 1));
  var report = (DATA.sources.espn || {}).match_report;
  if (isRanked && report && report.unresolved && report.unresolved.length) {
    chips.push(report.unresolved.length + ' golfer(s) unmatched');
  }
  var host = $('standings-chips');
  clear(host);
  chips.forEach(function (text) { host.append(el('span', 'chip', text)); });
}

/* ------------------------------------------------------------------ *
 * Odds and the draw
 * ------------------------------------------------------------------ */

function tile(label, value, note, gold) {
  var node = el('div', 'tile' + (gold ? ' is-gold' : ''));
  node.append(el('div', 'tile-label', label));
  node.append(el('div', 'tile-value', value));
  node.append(el('div', 'tile-note', note));
  return node;
}

function card(title, cls) {
  var node = el('div', 'card' + (cls ? ' ' + cls : ''));
  node.append(el('div', 'card-head', title));
  var body = el('div', 'card-body');
  node.append(body);
  node.body = body;
  return node;
}

/* The four numbers at the top, and their notes in the words somebody who has never
 * priced a book would use. "ask prices · probability basis" is precise and it is also
 * the sentence that made a league mate ask whether the draw had been rigged. */
function renderOdds() {
  var snap = DATA.odds_snapshot;
  var grouping = DATA.grouping;

  var tiles = $('odds-tiles');
  clear(tiles);
  tiles.append(tile('Field priced', String(snap.field_size),
    grouping.grouped_golfers + ' went into the groups · ' + snap.excluded.length
      + ' left out'));
  tiles.append(tile('Book sum', String(snap.raw_book_sum),
    'what the raw prices added up to, before they were levelled out'));
  var captured = stamp(snap.captured_at);
  tiles.append(tile('Captured', captured ? shortDate(captured) : '—',
    (captured ? captured.toLocaleTimeString() + ' · ' : '')
      + 'the odds never change after this'));
  var fair = 1 / DATA.league.team_count;
  tiles.append(tile('Fair share', pct(fair),
    'what each of the ' + DATA.league.team_count + ' groups is aiming for', true));

  var cards = $('odds-cards');
  clear(cards);
  cards.append(excludedCard(snap));
  cards.append(certificateCard(grouping));
}

function excludedCard(snap) {
  var node = card('Excluded from the draw');
  if (!snap.excluded.length) {
    node.body.append(el('p', null, snap.auto_exclude
      ? 'Nobody. Every priced golfer was inside a group’s fair share of the book, so '
        + 'the whole field went into the partition.'
      : 'Nobody, and the fair-share rule was switched off for this build — so nothing was '
        + 'dropped and nothing was checked.'));
    return node;
  }
  snap.excluded.forEach(function (e) {
    var row = el('div', 'excl');
    var left = el('div');
    left.append(el('div', 'excl-name', e.golfer_name));
    left.append(el('div', 'excl-why', e.reason === 'over_fair_share'
      ? 'over_fair_share — above 1/' + DATA.league.team_count + ' of the book'
      : e.reason.replace(/_/g, ' ')));
    row.append(left);
    row.append(el('span', 'excl-raw', pct(e.raw_odds, 1) + ' raw'));
    row.append(el('span', 'excl-devig', pct(e.devigged_odds, 2)));
    node.body.append(row);
  });
  node.body.append(el('p', 'small', 'A golfer worth more than a whole group’s fair share '
    + 'cannot be balanced around, so they were dropped before the partition ran.'));
  return node;
}

function certificateCard(g) {
  var node = card('Grouping certificate', 'is-cert');
  var head = el('div', 'cert-delta');
  head.append(el('b', null, fmtDelta(g.delta)));
  head.append(el('span', null, 'delta · ' + g.delta_ticks + ' tick'
    + (g.delta_ticks === 1 ? '' : 's')));
  node.body.append(head);
  node.body.append(el('span', 'stamp', g.optimal ? 'Proven optimal' : 'Best found'));
  node.body.append(el('p', null, g.optimal
    ? 'No partition of this field does better. Group sizes are uneven on purpose — forcing '
      + 'them equal costs real balance, and "more golfers" is a real tie-break.'
    : 'The search did not prove this optimal. Usually that means one golfer is worth more '
      + 'than a whole group’s fair share and no partition can balance around them; '
      + 'excluding that golfer fixes it, searching harder does not.'));
  if (g.dominant_golfers && g.dominant_golfers.length) {
    node.body.append(el('p', 'small', 'Dominant: ' + g.dominant_golfers.join(', ') + '.'));
  }
  node.body.append(el('div', 'cert-sizes', 'group_sizes [' + (g.group_sizes || []).join(', ') + ']'));
  return node;
}

/* ------------------------------------------------------------------ *
 * The full draw
 *
 * Every group as it was dealt, and what each golfer in it was worth at the moment it
 * was dealt. The league table upstairs holds the same rosters, but it holds them
 * ranked, one team at a time, behind a chevron -- which is the right shape for "who is
 * winning" and the wrong shape for "what did we all get". This is the flat answer, and
 * it is the thing the certificate above is a certificate ABOUT: the four group totals
 * sitting within a tick of each other is the whole claim, and it is easier to believe
 * when you can see the golfers it is made of.
 *
 * Its own container rather than a third row of `#odds-cards`, so the two cards above
 * keep the width the design drew them at.
 * ------------------------------------------------------------------ */

function drawCard(team) {
  var golfers = (GOLFERS_BY_TEAM.get(team.team_id) || []).slice()
    .sort(function (a, b) {
      return (b.odds.grouping_weight || 0) - (a.odds.grouping_weight || 0);
    });

  var node = card(team.team_name, 'is-draw-group');

  var head = el('div', 'draw-total');
  head.append(el('b', null, pct(team.total_odds)));
  head.append(el('span', null, 'group ' + (team.group_index + 1) + ' of '
    + DATA.league.team_count + ' · ' + golfers.length + ' golfers'
    + (team.player_name ? ' · ' + team.player_name : '')));
  node.body.append(head);

  golfers.forEach(function (g) {
    var row = el('div', 'draw-row');
    row.append(el('span', 'draw-name', g.name));

    var barCell = el('span', 'draw-bar');
    var bar = el('span', 'bar');
    var fill = el('i');
    fill.style.width = MAX_WEIGHT
      ? ((g.odds.grouping_weight || 0) / MAX_WEIGHT * 100).toFixed(1) + '%'
      : '0%';
    bar.append(fill);
    barCell.append(bar);
    row.append(barCell);

    row.append(el('span', 'draw-odds', pct(g.odds.grouping_weight)));
    node.body.append(row);
  });
  return node;
}

/* In dealt order, 1..n. The league table sorts by what each group is worth, because
 * that is the nearest thing to a standing before anybody tees off; here the group
 * number is the thing somebody is looking up. */
function renderDraw() {
  var host = $('odds-draw');
  clear(host);
  DATA.teams.slice()
    .sort(function (a, b) { return a.group_index - b.group_index; })
    .forEach(function (team) { host.append(drawCard(team)); });
}

/* ------------------------------------------------------------------ *
 * Provenance
 * ------------------------------------------------------------------ */

/* Where every number on the page came from, in one line at the bottom.
 *
 * The event tickers used to live in a card on the odds view, which is the wrong place
 * for them: nobody checks a market ticker on a Sunday, and a card that exists to be
 * ignored is a card in the way of the ones that do not. They still have to be ON the
 * page though -- odds nobody can trace back to a market are just a number somebody
 * typed -- so they are here, where provenance goes. */
function renderProvenance() {
  var gen = DATA.generator || {};
  var kalshi = DATA.sources.kalshi;
  var espn = DATA.sources.espn || {};
  var bits = ['built ' + DATA.generated_at];
  if (gen.tool) bits.push(gen.tool + (gen.git_commit ? ' @ ' + gen.git_commit : ''));
  if (gen.seed !== undefined && gen.seed !== null) bits.push('seed ' + gen.seed);
  if (DATA.rebuilt_from) {
    bits.push('rebuilt ' + DATA.rebuilt_from.mode + ' from a file first built '
      + DATA.rebuilt_from.first_built_at);
  }
  bits.push('Kalshi ' + kalshi.event_ticker + ' · ' + kalshi.market_label
    + ' · ' + kalshi.price_mode);
  bits.push('ESPN ' + (espn.league || 'pga') + '/' + (espn.event_id || '—'));
  bits.push(DATA.grouping.summary);
  $('provenance').textContent = bits.join(' · ');
}

/* ------------------------------------------------------------------ *
 * Views
 * ------------------------------------------------------------------ */

function showView(name) {
  STATE.view = name;
  ['standings', 'odds'].forEach(function (v) {
    var on = v === name;
    $('view-' + v).hidden = !on;
    var tab = $('tab-' + v);
    tab.className = 'nav-item' + (on ? ' is-on' : '');
    tab.setAttribute('aria-selected', on ? 'true' : 'false');
  });
}

/* What a poll can change, and nothing else.
 *
 * The odds view and the footer are pure functions of the baked data: not one number in
 * either of them is waiting on ESPN. They are drawn once, in start(). Rebuilding the
 * full draw -- a hundred and fifty rows, every one of them the same as it was -- on a
 * sixty-second timer would be a page doing fourteen hundred pointless things a day
 * behind a tab nobody has open. */
function render() {
  renderStatus();
  renderStandings();
}

/* ------------------------------------------------------------------ *
 * Poll loop
 * ------------------------------------------------------------------ */

function poll() {
  return fetch(LIVE.espn_leaderboard_url, { headers: { Accept: 'application/json' } })
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(function (payload) {
      var parsed = GolfPool.parseLeaderboard(payload);
      // ESPN's leaderboard answers about whatever it thinks is current. If the result
      // file pinned an event id, refuse a payload for a different one rather than
      // quietly scoring the league against the wrong tournament.
      var want = LIVE.espn_event_id;
      if (want && parsed.meta && String(parsed.meta.eventId) !== String(want)) {
        throw new Error('ESPN returned event ' + parsed.meta.eventId + ', expected ' + want);
      }
      STATE.meta = parsed.meta;
      STATE.players = parsed.players;
      STATE.index = GolfPool.indexByAthleteId(parsed.players);
      STATE.error = null;
      STATE.lastGood = new Date();
    })
    .catch(function (err) { STATE.error = String(err.message || err); })
    .then(render);
}

/* One loop, one endpoint -- or no loop at all. A groups page has `live: null`, which is
 * not a missing setting to work around but the build saying there was no field to score
 * against. It renders once, from data it already has, and stops. */
function start() {
  renderBrand();
  showView('standings');
  renderOdds();
  renderDraw();
  renderProvenance();
  render();

  $('nav').addEventListener('click', function (e) {
    var button = e.target.closest ? e.target.closest('.nav-item') : null;
    if (button) showView(button.dataset.view);
  });

  if (!LIVE) return;
  poll();
  setInterval(poll, POLL_SECONDS * 1000);
  // The pill's wording is relative to now, so it has to be redrawn by the clock rather
  // than by the network. Two text nodes; it costs nothing.
  setInterval(renderStatus, 1000);
}

start();

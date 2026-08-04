/*
 * app.js -- the reference scoreboard's presentation and network layer.
 *
 * DELIBERATELY PLAIN. This is the working proof of the data contract in
 * docs/FRONTEND-SPEC.md, not the design. The rules it runs on live in lib.js and are
 * checked against the Python implementation by tests/test_frontend_parity.py.
 *
 * No framework, no build step, no network on load. The competition data is baked into
 * the page, and ESPN is the ONLY host it ever fetches -- the only one that will answer
 * a browser at all.
 *
 * TWO PAGES, ONE FILE
 * -------------------
 * `DATA.live` is null when the competition was built before ESPN published a field.
 * That page is a groups sheet: teams, rosters, the odds the groups were drawn on, and
 * no network of any kind. It is not a degraded scoreboard waiting for a fetch to
 * succeed -- there is nothing to fetch, because the field does not exist yet, and a
 * page that polled would be asking a question whose answer it could not use.
 *
 * `DATA.live` non-null means the field existed at build time and every golfer that
 * could be resolved carries an ESPN athlete id. That page polls, joins on the id, and
 * ranks. See build_competition.py's `build_mode`.
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

var STATE = { players: [], meta: null, index: null, error: null, lastPoll: null };

function $(sel) { return document.querySelector(sel); }

function el(tag, cls, text) {
  var n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

/* By ESPN athlete id and nothing else. A golfer without one is not in the field, or
 * has not been settled, and either way has no row on the leaderboard to show. */
function resolvePlayer(golfer) {
  if (!STATE.index) return null;
  return GolfPool.resolveGolfer(golfer, STATE.index);
}

/* ------------------------------------------------------------------ *
 * Render
 *
 * The page never fetches odds, and there is no state in which it might. Kalshi
 * allowlists request origins: a GET carrying Origin: https://kalshi.com returns 200,
 * and every other origin -- localhost, GitHub Pages, file:// -- gets a 403 with no
 * CORS headers at all, preflight included (measured 2026-08-03). The odds here are the
 * ones baked into the file, they were read when the groups were drawn, and there is
 * exactly one of them per golfer: nothing in this repo produces a second reading, so
 * nothing on this page shows a price moving.
 * ------------------------------------------------------------------ */

function pct(v) { return v == null ? '—' : (v * 100).toFixed(2) + '%'; }

/* One golfer's row. `player` is null before the tournament starts and for anyone who
 * never teed off, and the row still has to say who they are and what they were worth --
 * a roster does not shrink because the leaderboard has not opened.
 *
 * Those two nulls are different, which is why the "out" marker is conditioned on the
 * field existing: once it does, a golfer with no player is a golfer who is not playing
 * and belongs greyed out. Before it does, nobody is out of anything. */
function golferRow(golfer, player) {
  var started = STATE.players.length > 0;
  var tr = document.createElement('tr');
  var madeCut = !!(player && player.positionNumber !== null && player.positionNumber !== undefined);
  if (started && !madeCut) tr.className = 'out';
  tr.append(el('td', 'gpos', player ? (player.position || player.statusShort || 'CUT')
                                    : (started ? 'n/a' : '—')));
  tr.append(el('td', 'gname', golfer.name));
  tr.append(el('td', 'gscore', player ? GolfPool.fmtPar(player.toPar) : '—'));
  tr.append(el('td', 'gthru', player ? String(player.thru || player.statusShort || '') : ''));
  tr.append(el('td', 'godds', pct(golfer.odds.grouping_weight)));
  return tr;
}

/* The card. Built from a view rather than from a standings row, because the page has
 * to draw the same card in two situations: ranked, once ESPN has a leaderboard, and
 * unranked, before the first tee time. Only the header and the note differ. */
function teamCard(view) {
  var card = el('article', 'team' + (view.leader ? ' leader' : ''));

  var head = el('header', 'team-head');
  head.append(el('span', 'pos', view.position));
  if (view.team.team_logo) {
    var img = document.createElement('img');
    img.className = 'logo';
    img.src = view.team.team_logo;
    img.alt = '';
    head.append(img);
  }
  var names = el('div', 'names');
  names.append(el('strong', null, view.team.team_name));
  names.append(el('span', 'muted', view.team.player_name));
  head.append(names);

  var stat = el('div', 'stat');
  stat.append(el('span', 'big', view.statValue));
  stat.append(el('span', 'muted small', view.statLabel));
  head.append(stat);
  card.append(head);
  card.append(el('p', 'muted small', view.bits.join(' · ')));

  var table = el('table', 'golfers');
  var tbody = document.createElement('tbody');
  view.golfers.forEach(function (d) { tbody.append(golferRow(d.golfer, d.player)); });
  table.append(tbody);
  card.append(table);
  return card;
}

/* Before the first tee time ESPN publishes no competitors at all, so there is nothing
 * to rank on -- but everything else about the pool is already decided and is worth
 * showing: who holds whom, what each golfer was worth when the groups were drawn, and
 * that the draw came out even. Ranking anyway would order teams by roster size and
 * present it as a leaderboard, which is worse than saying "not started". */
function renderRosters(host) {
  DATA.teams.forEach(function (team) {
    var golfers = (GOLFERS_BY_TEAM.get(team.team_id) || []).slice()
      .sort(function (a, b) { return (b.odds.grouping_weight || 0) - (a.odds.grouping_weight || 0); });
    host.append(teamCard({
      team: team,
      position: '—',
      leader: false,
      statValue: pct(team.total_odds),
      statLabel: 'odds at creation',
      bits: [team.golfer_count + ' golfers', 'group ' + (team.group_index + 1) + ' of '
             + DATA.league.team_count],
      golfers: golfers.map(function (g) { return { golfer: g, player: null }; })
    }));
  });
}

function renderHeader() {
  $('#league-name').textContent = DATA.league.league_name;
  $('#tournament').textContent = DATA.tournament.name;
  var k = DATA.sources.kalshi;
  $('#market').textContent = k.market_label + ' · ' + k.price_mode + ' · ' + k.event_ticker;
  var m = STATE.meta;
  var course = DATA.tournament.course || {};
  $('#round').textContent = m
    ? (m.detail || m.state) + ' — ' + (m.course || '') + (m.par ? ' (par ' + m.par + ')' : '')
    : (LIVE ? 'waiting for ESPN…'
            : 'Groups' + (course.name ? ' — ' + course.name : ''));
  $('#standings-heading').textContent = LIVE ? 'Standings' : 'Groups';
  $('#built').textContent = 'built ' + DATA.generated_at + ' · ' + DATA.grouping.summary;
}

/* Why there is no ranking. Three different reasons, and saying the wrong one is how a
 * page gets accused of being broken when it is working exactly as built. */
function notStarted() {
  var start = DATA.tournament.start ? new Date(DATA.tournament.start).toLocaleString() : null;
  if (!LIVE) {
    return 'The tournament had not started when this page was made, so ESPN had published '
      + 'no field and there is nothing to rank yet' + (start ? ' — first round ' + start : '')
      + '. This page shows the draw: it is final, and it was drawn on the odds below. It '
      + 'fetches nothing and will not change. Ask for a rebuilt page once play begins and '
      + 'that one will carry live scoring.';
  }
  if (STATE.error) return 'ESPN unavailable: ' + STATE.error + '. Groups and odds below are '
    + 'baked into this page and are unaffected.';
  return 'Waiting for ESPN to publish positions' + (start ? ' — first round ' + start : '')
    + '. The groups below are final and were drawn on the odds shown.';
}

function renderStandings() {
  var host = $('#standings');
  host.textContent = '';

  if (!STATE.players.length) {
    host.append(el('p', 'muted', notStarted()));
    renderRosters(host);
    return;
  }

  GolfPool.computeStandings(DATA.teams, GOLFERS_BY_TEAM, resolvePlayer).forEach(function (row) {
    var bits = [row.counting + '/' + row.roster + ' still in'];
    if (row.toPar !== null) bits.push('aggregate ' + GolfPool.fmtPar(row.toPar));
    if (row.decidedAt) bits.push('separated on golfer #' + row.decidedAt);
    if (row.tied) bits.push('tied');
    bits.push('odds at creation ' + (row.team.total_odds * 100).toFixed(2) + '%');

    host.append(teamCard({
      team: row.team,
      position: row.position,
      leader: row.rank === 1,
      statValue: row.best && row.best.player
        ? (row.best.player.position || row.best.player.statusShort || 'CUT') : '—',
      statLabel: row.best ? row.best.golfer.name : 'no golfers',
      bits: bits,
      golfers: row.golfers
    }));
  });
}

function renderOdds() {
  var snap = DATA.odds_snapshot;
  var parts = ['Odds at creation: ' + snap.field_size + ' golfers, ' + snap.price_mode
    + ' book summing to ' + snap.raw_book_sum + ', captured ' + snap.captured_at + '.'];
  if (snap.excluded.length) {
    parts.push('Excluded: ' + snap.excluded.map(function (e) {
      return e.golfer_name + ' (' + e.reason.replace(/_/g, ' ') + ')';
    }).join(', ') + '.');
  }
  // Not an apology for a missing feature -- a statement of what these numbers are.
  // There is no fetch that could make them newer, and no rebuild that would either:
  // the odds are read once, when the groups are drawn.
  parts.push('These are the prices the groups were drawn on. They do not move while '
    + 'this page is open: Kalshi’s API returns 403 to a browser, so no odds are ever '
    + 'fetched here. They do not move between two copies of this page either — there '
    + 'is one reading per competition and this is it.');
  $('#odds-note').textContent = parts.join(' ');
}

/* Provenance, not a list of what gets fetched. On a live page exactly one of these two
 * is ever requested; on a groups page neither is, and the wording has to say so rather
 * than name an endpoint nothing calls. */
function renderFooter() {
  $('#sources').textContent = (LIVE
      ? 'scores polled from ' + LIVE.espn_leaderboard_url + ' · odds captured from '
      : 'nothing is fetched by this page · odds captured from ')
    + DATA.sources.kalshi.markets_endpoint;
  $('#poll').textContent = STATE.lastPoll ? 'last poll ' + STATE.lastPoll.toLocaleTimeString() : '';
}

function render() {
  renderHeader();
  renderStandings();
  renderOdds();
  renderFooter();
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
    })
    .catch(function (err) { STATE.error = String(err.message || err); })
    .then(function () {
      STATE.lastPoll = new Date();
      render();
    });
}

/* One loop, one endpoint -- or no loop at all. A groups page has `live: null`, which
 * is not a missing setting to work around but the build saying there was no field to
 * score against. It renders once, from data it already has, and stops. */
function start() {
  render();
  if (!LIVE) return;
  poll();
  setInterval(poll, (LIVE.poll_interval_seconds || 60) * 1000);
}

start();

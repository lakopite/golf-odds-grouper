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
 * TWO PAGES, ONE FILE, AND THE CLOCK DECIDES WHICH
 * ------------------------------------------------
 * A competition is built the night before, when ESPN has posted the field and nobody
 * has teed off. So this page opens as the draw -- teams, rosters, the odds the groups
 * were drawn on, nothing ranked -- and turns into a scoreboard by itself at the first
 * tee time, while it is sitting open. There is no second build and no new link.
 *
 * A field is joinable long before it is rankable: ESPN posts the competitors about two
 * days out and gives them no positions until play starts, so every golfer's athlete id
 * is baked in already and only the scores are missing. `meta.started` off each poll is
 * what says whether they have arrived. A template that ignores it and ranks anything
 * ESPN answers with will order the league by a pre-tournament sort and call it a
 * leaderboard. See GolfPool.hasStarted and docs/FRONTEND-SPEC.md §2.
 */
'use strict';

var DATA = JSON.parse(document.getElementById('competition-data').textContent);

/* The league's art: {logo: 'data:…', banner: 'data:…'}, either key or both absent.
 * Separate from DATA because DATA is the result file verbatim, and the result file
 * names the art with a slug rather than carrying the images themselves. The bundler
 * resolves the slug and fills this element in. */
var ART = (function () {
  try { return JSON.parse(document.getElementById('league-art').textContent) || {}; }
  catch (e) { return {}; }
})();

/* Where the page polls and what it will accept back. Always present since schema 4.0:
 * a competition cannot be built without a published ESPN field. */
var LIVE = DATA.live;

/* The poll loop's handle, so it can be stopped once the tournament is final. */
var TIMER = null;

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
 * board being open: once it is, a golfer with no player is a golfer who is not playing
 * and belongs greyed out. Before it is, nobody is out of anything.
 *
 * It asks ranked() rather than `STATE.players.length`, and the difference is two days
 * long. ESPN posts the field early, so a page opened on Wednesday has 147 players and
 * an unopened board; keying on the count greys out every golfer in the pool and labels
 * them "n/a" on the one view that is supposed to be showing the draw. */
function golferRow(golfer, player) {
  var open = ranked();
  var tr = document.createElement('tr');
  var madeCut = !!(player && player.positionNumber !== null && player.positionNumber !== undefined);
  if (open && !madeCut) tr.className = 'out';
  tr.append(el('td', 'gpos', player ? (player.position || player.statusShort || 'CUT')
                                    : (open ? 'n/a' : '—')));
  tr.append(el('td', 'gname', golfer.name));
  tr.append(el('td', 'gscore', player ? GolfPool.fmtPar(player.toPar) : '—'));
  // `thru` only, never statusShort: before a golfer tees off ESPN puts a raw ISO
  // timestamp in there, and a cell that reads "2026-08-06T18:00:00Z" under a column
  // headed "Thru" is worse than an empty one.
  tr.append(el('td', 'gthru', player && player.thru ? String(player.thru) : ''));
  tr.append(el('td', 'godds', pct(golfer.odds.grouping_weight)));
  return tr;
}

/* The card. Built from a view rather than from a standings row, because the page has
 * to draw the same card in two situations: ranked, once play is under way, and
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

/* Before the first tee time ESPN publishes the field but no positions, so there is
 * nothing to rank on -- but everything else about the pool is already decided and is
 * worth showing: who holds whom, what each golfer was worth when the groups were
 * drawn, and that the draw came out even. Ranking anyway would order the teams by
 * ESPN's pre-tournament sort and present that as a leaderboard, which is worse than
 * saying "not started". */
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
  ['logo', 'banner'].forEach(function (name) {
    if (!ART[name]) return;
    var img = $('#league-' + name);
    img.src = ART[name];
    img.alt = DATA.league.league_name + ' ' + name;
    img.hidden = false;
  });
  $('#tournament').textContent = DATA.tournament.name;
  var k = DATA.sources.kalshi;
  $('#market').textContent = k.market_label + ' · ' + k.price_mode + ' · ' + k.event_ticker;
  var m = STATE.meta;
  var course = DATA.tournament.course || {};
  $('#round').textContent = m
    ? (m.detail || m.state) + ' — ' + (m.course || '') + (m.par ? ' (par ' + m.par + ')' : '')
    : 'waiting for ESPN…';
  // Relabelled on every render rather than once at load, because the page crosses from
  // one to the other on its own the moment somebody tees off.
  $('#standings-heading').textContent = ranked() ? 'Standings' : 'Groups';
  $('#built').textContent = 'built ' + DATA.generated_at + ' · ' + DATA.grouping.summary;
}

/* The one question this file asks about whether to rank, and both halves matter.
 * `players.length` is "has a poll come back at all"; `meta.started` is "has anybody
 * teed off". ESPN answers with a full field about two days early and gives every
 * player position "-", so the second does not follow from the first. */
function ranked() {
  return STATE.players.length > 0 && !!(STATE.meta && STATE.meta.started);
}

/* Why there is no ranking. Three different reasons, and saying the wrong one is how a
 * page gets accused of being broken when it is working exactly as built. */
function notStarted() {
  var start = DATA.tournament.start ? new Date(DATA.tournament.start).toLocaleString() : null;
  if (STATE.error) return 'ESPN unavailable: ' + STATE.error + '. Groups and odds below are '
    + 'baked into this page and are unaffected.';
  if (STATE.meta && !STATE.meta.started) {
    return 'The tournament has not started' + (start ? ' — first round ' + start : '')
      + '. ESPN has posted the field and this page has already matched every golfer to it, '
      + 'so ranking begins on its own at the first tee time. Nothing needs rebuilding and '
      + 'nobody needs to reload. The groups below are final and were drawn on the odds shown.';
  }
  return 'Waiting for ESPN to publish positions' + (start ? ' — first round ' + start : '')
    + '. The groups below are final and were drawn on the odds shown.';
}

function renderStandings() {
  var host = $('#standings');
  host.textContent = '';

  if (!ranked()) {
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

/* Provenance, not a list of what gets fetched. Exactly one of these two is ever
 * requested: ESPN. The Kalshi endpoint is named because it is where the odds came
 * from, not because anything asks it -- it would 403 a browser if anything did. */
function renderFooter() {
  var espn = DATA.sources.espn || {};
  $('#sources').textContent = 'scores polled from ' + LIVE.espn_leaderboard_url
    + ' · odds captured from ' + DATA.sources.kalshi.markets_endpoint
    // Both events named, because the two URLs above identify one apiece. §5.4.
    + ' · ESPN ' + (espn.league || 'pga') + '/' + (espn.event_id || '—');
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
      // A finished tournament does not change again, so the loop stops. Every page
      // polls from the moment it opens now, which means an archived one reopened
      // months later would otherwise hit ESPN once a minute for as long as the tab is
      // up, to be told the same final scores every time.
      if (parsed.meta && parsed.meta.completed && TIMER) {
        clearInterval(TIMER);
        TIMER = null;
      }
      STATE.error = null;
    })
    .catch(function (err) { STATE.error = String(err.message || err); })
    .then(function () {
      STATE.lastPoll = new Date();
      render();
    });
}

/* One loop, one endpoint, always running. The page polls even when the tournament is
 * days away, because the poll is how it finds out that it is not any more -- that is
 * the whole mechanism by which the draw becomes a scoreboard with nobody rebuilding
 * anything. */
function start() {
  render();
  poll();
  TIMER = setInterval(poll, (LIVE.poll_interval_seconds || 60) * 1000);
}

start();

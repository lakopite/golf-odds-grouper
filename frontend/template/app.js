/*
 * app.js -- the reference scoreboard's presentation and network layer.
 *
 * DELIBERATELY PLAIN. This is the working proof of the data contract in
 * docs/FRONTEND-SPEC.md, not the design. The rules it runs on live in lib.js and are
 * checked against the Python implementation by tests/test_frontend_parity.py.
 *
 * No framework, no build step, no network on load. The competition data is baked into
 * the page; the only fetch is ESPN, which is the only host that will answer a browser.
 */
'use strict';

var DATA = JSON.parse(document.getElementById('competition-data').textContent);

var GOLFERS_BY_TEAM = new Map();
DATA.golfers.forEach(function (g) {
  if (!g.team_id) return;
  if (!GOLFERS_BY_TEAM.has(g.team_id)) GOLFERS_BY_TEAM.set(g.team_id, []);
  GOLFERS_BY_TEAM.get(g.team_id).push(g);
});

var STATE = { players: [], meta: null, index: null, live: null, error: null, lastPoll: null };

function $(sel) { return document.querySelector(sel); }

function el(tag, cls, text) {
  var n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function resolvePlayer(golfer) {
  if (!STATE.index) return null;
  return GolfPool.matchGolfer(golfer, STATE.index, DATA.live.name_match.aliases || {}).player;
}

/* ------------------------------------------------------------------ *
 * Live odds.
 *
 * Kalshi allowlists request origins: a GET carrying Origin: https://kalshi.com
 * returns 200, and every other origin -- localhost, GitHub Pages, file:// -- gets a
 * 403 with no CORS headers at all, preflight included (measured 2026-08-03). So a
 * static page cannot read live odds unless the result file names a relay. Without
 * one this reports why and the page shows the snapshot, which is always present and
 * always correct as of the time it states.
 * ------------------------------------------------------------------ */

function fetchLiveOdds() {
  var template = DATA.live.kalshi_proxy_url_template;
  if (!template) return Promise.resolve({ ok: false, reason: 'no-proxy' });
  var url = template.replace('{url}', encodeURIComponent(DATA.live.kalshi_markets_url));
  return fetch(url, { headers: { Accept: 'application/json' } })
    .then(function (r) {
      if (!r.ok) throw new Error('http-' + r.status);
      return r.json();
    })
    .then(function (body) {
      // A relay may hand back the payload verbatim or wrapped in a string field.
      var markets = body.markets ||
        (typeof body.contents === 'string' ? (JSON.parse(body.contents).markets) : null);
      if (!markets) return { ok: false, reason: 'unrecognised relay response shape' };
      var byId = new Map(), sum = 0;
      markets.forEach(function (m) {
        if (m.status !== 'active') return;
        var ask = parseFloat(m.yes_ask_dollars || '0');
        sum += ask;
        var id = (m.custom_strike || {}).golf_competitor;
        if (id) byId.set(id, { ask: ask, bid: parseFloat(m.yes_bid_dollars || '0') });
      });
      return { ok: true, byId: byId, sum: sum, at: new Date() };
    })
    .catch(function (err) { return { ok: false, reason: String(err.message || err) }; });
}

/* ------------------------------------------------------------------ *
 * Render
 * ------------------------------------------------------------------ */

function pct(v) { return v == null ? '—' : (v * 100).toFixed(2) + '%'; }

function liveCell(golfer) {
  if (!STATE.live || !STATE.live.ok) return '';
  var hit = STATE.live.byId.get(golfer.golfer_id);
  if (!hit) return '';
  var delta = hit.ask - (golfer.kalshi.ask || 0);
  var arrow = Math.abs(delta) < 1e-9 ? '→' : (delta > 0 ? '↑' : '↓');
  return (hit.ask * 100).toFixed(1) + '% ' + arrow;
}

function renderHeader() {
  $('#league-name').textContent = DATA.league.league_name;
  $('#tournament').textContent = DATA.tournament.name;
  var k = DATA.sources.kalshi;
  $('#market').textContent = k.market_label + ' · ' + k.price_mode + ' · ' + k.event_ticker;
  var m = STATE.meta;
  $('#round').textContent = m
    ? (m.detail || m.state) + ' — ' + (m.course || '') + (m.par ? ' (par ' + m.par + ')' : '')
    : 'waiting for ESPN…';
  $('#built').textContent = 'built ' + DATA.generated_at + ' · ' + DATA.grouping.summary;
}

function renderStandings() {
  var host = $('#standings');
  host.textContent = '';

  if (!STATE.players.length) {
    host.append(el('p', 'muted', STATE.error
      ? 'ESPN unavailable: ' + STATE.error
      : 'The field is not posted yet. ESPN publishes no competitors until play starts.'));
    return;
  }

  GolfPool.computeStandings(DATA.teams, GOLFERS_BY_TEAM, resolvePlayer).forEach(function (row) {
    var card = el('article', 'team' + (row.rank === 1 ? ' leader' : ''));

    var head = el('header', 'team-head');
    head.append(el('span', 'pos', row.position));
    if (row.team.team_logo) {
      var img = document.createElement('img');
      img.className = 'logo';
      img.src = row.team.team_logo;
      img.alt = '';
      head.append(img);
    }
    var names = el('div', 'names');
    names.append(el('strong', null, row.team.team_name));
    names.append(el('span', 'muted', row.team.player_name));
    head.append(names);

    var stat = el('div', 'stat');
    stat.append(el('span', 'big', row.best && row.best.player
      ? (row.best.player.position || row.best.player.statusShort || 'CUT') : '—'));
    stat.append(el('span', 'muted small', row.best ? row.best.golfer.name : 'no golfers'));
    head.append(stat);
    card.append(head);

    var bits = [row.counting + '/' + row.roster + ' still in'];
    if (row.toPar !== null) bits.push('aggregate ' + GolfPool.fmtPar(row.toPar));
    if (row.decidedAt) bits.push('separated on golfer #' + row.decidedAt);
    if (row.tied) bits.push('tied');
    bits.push('odds at creation ' + (row.team.total_odds * 100).toFixed(2) + '%');
    card.append(el('p', 'muted small', bits.join(' · ')));

    var table = el('table', 'golfers');
    var tbody = document.createElement('tbody');
    row.golfers.forEach(function (d) {
      var tr = document.createElement('tr');
      if (!d.madeCut) tr.className = 'out';
      tr.append(el('td', 'gpos', d.player ? (d.player.position || d.player.statusShort || 'CUT') : 'n/a'));
      tr.append(el('td', 'gname', d.golfer.name));
      tr.append(el('td', 'gscore', d.player ? GolfPool.fmtPar(d.player.toPar) : '—'));
      tr.append(el('td', 'gthru', d.player ? String(d.player.thru || d.player.statusShort || '') : ''));
      tr.append(el('td', 'godds', pct(d.golfer.odds.grouping_weight)));
      tr.append(el('td', 'glive', liveCell(d.golfer)));
      tbody.append(tr);
    });
    table.append(tbody);
    card.append(table);
    host.append(card);
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
  if (!STATE.live) parts.push('Live odds: checking…');
  else if (STATE.live.ok) {
    parts.push('Live odds updated ' + STATE.live.at.toLocaleTimeString()
      + ', book now sums to ' + STATE.live.sum.toFixed(3) + '.');
  } else if (STATE.live.reason === 'no-proxy') {
    parts.push('Live odds unavailable: Kalshi’s API allowlists request origins and returns 403 '
      + 'to a browser. Set live.kalshi_proxy_url_template in the result file to enable them.');
  } else {
    parts.push('Live odds unavailable (' + STATE.live.reason + '). The snapshot above still holds.');
  }
  $('#odds-note').textContent = parts.join(' ');
}

function renderFooter() {
  $('#sources').textContent = 'scores ' + DATA.sources.espn.leaderboard_endpoint
    + ' · odds ' + DATA.sources.kalshi.markets_endpoint;
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
  return fetch(DATA.live.espn_leaderboard_url, { headers: { Accept: 'application/json' } })
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(function (payload) {
      var parsed = GolfPool.parseLeaderboard(payload);
      // ESPN's leaderboard answers about whatever it thinks is current. If the result
      // file pinned an event id, refuse a payload for a different one rather than
      // quietly scoring the league against the wrong tournament.
      var want = DATA.sources.espn.event_id;
      if (want && parsed.meta && String(parsed.meta.eventId) !== String(want)) {
        throw new Error('ESPN returned event ' + parsed.meta.eventId + ', expected ' + want);
      }
      STATE.meta = parsed.meta;
      STATE.players = parsed.players;
      STATE.index = GolfPool.buildIndex(parsed.players);
      STATE.error = null;
    })
    .catch(function (err) { STATE.error = String(err.message || err); })
    .then(function () {
      STATE.lastPoll = new Date();
      render();
    });
}

function start() {
  render();
  poll().then(function () {
    return fetchLiveOdds().then(function (live) { STATE.live = live; render(); });
  });

  var interval = (DATA.live.poll_interval_seconds || 60) * 1000;
  setInterval(poll, interval);
  if (DATA.live.kalshi_proxy_url_template) {
    setInterval(function () {
      fetchLiveOdds().then(function (live) { STATE.live = live; render(); });
    }, Math.max(interval, 120000));
  }
}

start();

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
 *
 * There is a second way for prices to move without a relay: somebody re-ran the build
 * with --refresh-odds and re-sent the page. That re-read is baked in the same way the
 * original snapshot is, so it needs no network at all -- and it is compared against
 * the price the groups were DRAWN on rather than against the raw ask, because those
 * differ on any price mode but the default.
 * ------------------------------------------------------------------ */

function bakedOdds() {
  var refreshed = DATA.odds_snapshot.refreshed;
  if (!refreshed) return null;
  var byId = new Map();
  DATA.golfers.forEach(function (g) {
    if (g.golfer_id && g.odds.current !== null && g.odds.current !== undefined) {
      byId.set(g.golfer_id, { ask: g.odds.current });
    }
  });
  return {
    ok: true, byId: byId, sum: refreshed.raw_book_sum,
    at: new Date(refreshed.at), basis: 'drawn_price', source: 'rebuild'
  };
}

function fetchLiveOdds() {
  var template = DATA.live.kalshi_proxy_url_template;
  if (!template) {
    return Promise.resolve(bakedOdds() || { ok: false, reason: 'no-proxy' });
  }
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
      return { ok: true, byId: byId, sum: sum, at: new Date(), basis: 'kalshi_ask',
               source: 'relay' };
    })
    .catch(function (err) {
      // A relay that is down does not make a baked re-read wrong; it just makes it
      // the freshest thing left.
      return bakedOdds() || { ok: false, reason: String(err.message || err) };
    });
}

/* ------------------------------------------------------------------ *
 * Render
 * ------------------------------------------------------------------ */

function pct(v) { return v == null ? '—' : (v * 100).toFixed(2) + '%'; }

/* Movement, not a second column of levels. The number beside it is what the golfer was
 * worth when the groups were drawn; repeating today's price next to it reads as a jump
 * even when nothing has moved, because one is de-vigged and the other is not. What is
 * actually interesting is "shorter than when you drafted him", in points of the same
 * price. Under half a point is not a move worth an arrow. */
function liveCell(golfer) {
  if (!STATE.live || !STATE.live.ok) return '';
  var hit = STATE.live.byId.get(golfer.golfer_id);
  if (!hit) return '';
  var base = STATE.live.basis === 'drawn_price' ? golfer.odds.raw : (golfer.kalshi.ask || 0);
  var delta = (hit.ask - (base || 0)) * 100;
  if (Math.abs(delta) < 0.05) return '→';
  return (delta > 0 ? '↑ +' : '↓ −') + Math.abs(delta).toFixed(1);
}

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
  tr.append(el('td', 'glive', liveCell(golfer)));
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
  $('#round').textContent = m
    ? (m.detail || m.state) + ' — ' + (m.course || '') + (m.par ? ' (par ' + m.par + ')' : '')
    : 'waiting for ESPN…';
  $('#built').textContent = 'built ' + DATA.generated_at + ' · ' + DATA.grouping.summary;
}

function notStarted() {
  if (STATE.error) return 'ESPN unavailable: ' + STATE.error + '. Groups and odds below are '
    + 'baked into this page and are unaffected.';
  var start = DATA.tournament.start ? new Date(DATA.tournament.start).toLocaleString() : null;
  return 'Not started' + (start ? ' — first round ' + start : '') + '. ESPN publishes no '
    + 'competitors until play begins, so there are no positions yet. The groups below are '
    + 'final and were drawn on the odds shown.';
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
  if (!STATE.live) parts.push('Live odds: checking…');
  else if (STATE.live.ok && STATE.live.source === 'rebuild') {
    // Not a feed. Somebody re-ran the build against Kalshi and re-sent this page, and
    // that re-read is baked in exactly as the snapshot is. Saying "live" would promise
    // a number that will not change until the next rebuild.
    parts.push('Odds re-read ' + STATE.live.at.toLocaleString() + ' when this page was '
      + 'rebuilt, book then summing to ' + STATE.live.sum.toFixed(3)
      + '. The arrows are movement since the draw; they will not change again until the '
      + 'next rebuild.');
    if (snap.refreshed && snap.refreshed.priced_since_the_draw.length) {
      parts.push(snap.refreshed.priced_since_the_draw.length + ' golfer(s) were added to the '
        + 'market after the draw and are in nobody’s group: '
        + snap.refreshed.priced_since_the_draw.join(', ') + '.');
    }
  } else if (STATE.live.ok) {
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

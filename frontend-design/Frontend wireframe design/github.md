repo: lakopite/golf-odds-grouper
branch: main
path: docs/FRONTEND-SPEC.md, frontend/template/

## Last sync
date: 2026-08-04T00:35:00Z

### Updated in this project
- Built the hi-fi frontend wireframe from docs/FRONTEND-SPEC.md (live scoreboard, groups sheet, odds snapshot, phone layouts).
- Standings rule ported from frontend/template/lib.js verbatim (tiered golfer ranks, lexicographic vector compare, ties as ties).
- Fantrax-style dense chrome with the WCW navy/gold crest and 10th-anniversary banner.
- Sample data: 12 teams, 105 grouped golfers, Wyndham Championship, round 3 in progress.

## Screen map
| Screen (view in WCW Scoreboard.dc.html) | Built from |
|---|---|
| Standings (live) | docs/FRONTEND-SPEC.md §5.1–5.2, §6; frontend/template/lib.js (computeStandings, golferRank, fmtPar); frontend/template/app.js (teamCard, golferRow) |
| Groups & the draw (build_mode "groups") | FRONTEND-SPEC §2, §3 "states", §10; app.js (renderRosters, notStarted) |
| Odds snapshot | FRONTEND-SPEC §4, §5.3, §7 odds_snapshot/grouping; app.js (renderOdds, MOVEMENT) |
| Name-join / review panel | FRONTEND-SPEC §8; sources.espn.match_report shape in §7 |
| Provenance footer | FRONTEND-SPEC §5.4, §7 sources/generator; app.js (renderFooter) |
| Phone layouts | FRONTEND-SPEC §1, §10 ("a phone held one-handed") |
| Status pill / stale + down states | FRONTEND-SPEC §3 states table, §10 "refresh honestly"; app.js poll() error path |

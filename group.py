"""
group.py -- partition a golf field into equal-weighted groups and deal them out.

Odds come from Kalshi (see kalshi_odds.py). DraftKings was the original source and
its parser is still here, so an archived dk_data.json still runs, but nothing new
should be written against it: DK moved its endpoint every season, which is the whole
reason for the migration.
"""

import argparse
import json
import os
import random
import time

import kalshi_odds
from groupers import (
    calculate_total_odds,
    backtracking_generate_groups,
    dp_generate_groups,
    sa_generate_groups,
    ga_generate_groups,
    greedy_redistribute_groups,
)

# ---------------------------------------------------------------------------
# Configuration. These are the defaults; every one has a CLI flag.
# ---------------------------------------------------------------------------

# The Kalshi series to price off. This is the direct analogue of the old
# DraftKings ODDS_TYPE knob -- the market type is nothing more than a prefix on
# the event ticker, and every series returns the identical shape.
#   KXPGATOUR      outright Winner   <- the default, and the best signal
#   KXPGATOP5      Top 5 Finishers
#   KXPGATOP10     Top 10 Finishers
#   KXPGAMAKECUT   To Make the Cut
# Winner ticks at $0.001 below 10c; the others are flat $0.01, which is 10x
# coarser exactly where a golf field sits. Prefer Winner.
MARKET_SERIES = "KXPGATOUR"

# "ask" is the only price mode that gives every golfer a real quoted probability.
# See kalshi_odds.to_golfers() for the measurements behind that.
PRICE_MODE = "ask"

EXCLUDE_LIST = ["Scottie Scheffler", "Rory McIlroy"]

# Read in order; the first that exists wins. If neither is present the grouper
# fetches live from Kalshi.
DATA_FILES = ("kalshi_data.json", "dk_data.json")
PARTICIPANTS_FILE = "participants.json"
OUTPUT_DIR = "output"

# Legacy DraftKings market label, used only when reading an archived dk_data.json.
DK_ODDS_TYPE = "Winner"


# ---------------------------------------------------------------------------
# Odds parsing
# ---------------------------------------------------------------------------

def fractional_odds_to_implied_probability(fraction_str):
    """DraftKings only. Kalshi prices are already probabilities."""
    numerator, denominator = fraction_str.split('/')
    numerator = float(numerator)
    denominator = float(denominator)
    implied_probability = denominator / (denominator + numerator)
    return implied_probability


def print_completion_time(start_time, end_time):
    print(f"Completed in {int(end_time - start_time) // 60} minutes and {int(end_time - start_time) % 60} seconds")


def list_dk_golf_odds(d, odds_type=DK_ODDS_TYPE):
    """
    Parse the DraftKings eventgroup envelope: markets[] matched on name,
    selections[] filtered on marketId.

    Legacy path. Kept so an archived dk_data.json still runs, and so a DK payload
    can be compared side by side against Kalshi if one is ever captured by hand.
    """
    result = []

    market_section = next((i for i in d['markets'] if i.get('name') == odds_type), None)
    if market_section is None:
        available = sorted({i.get('name') for i in d['markets'] if i.get('name')})
        raise ValueError(
            f"no DraftKings market named {odds_type!r}. Available: {available}"
        )
    odds_type_marketID = market_section.get('id')

    odds_section = [s for s in d['selections'] if s.get('marketId') == odds_type_marketID]

    for o in odds_section:
        result.append({
            "golfer_name": o['participants'][0]['name'],
            "odds": fractional_odds_to_implied_probability(o['displayOdds']['fractional'])
        })
    return result


def list_kalshi_golf_odds(d, price=PRICE_MODE):
    """
    Parse a Kalshi payload into the grouper's [{golfer_name, odds}, ...] shape.

    Accepts either of the two things kalshi_odds.py writes:
      * the raw /markets envelope   {"markets": [...]}     -- one market per golfer
      * the grouper odds file       {"golfers": [...]}     -- already converted

    Kalshi prices ARE implied probabilities (a YES contract settles at $1.00), so
    there is no fractional decode on this path. Settled markets quote
    bid=0.0000/ask=1.0000 and are dropped by the status filter in to_golfers().
    """
    if "golfers" in d:
        golfers = d["golfers"]
        if not golfers:
            raise ValueError("Kalshi odds file contains an empty golfers list")
        out = [
            {
                "golfer_name": g["golfer_name"],
                "odds": float(g["odds"]),
                "golfer_id": g.get("golfer_id"),
            }
            for g in golfers
        ]
        # This branch reads odds that were converted earlier, so none of to_golfers()'s
        # guards ran here. Re-apply the one that matters.
        check_book(out, price_mode=d.get("price_mode"), requested_price=price)
        return out

    markets = d["markets"]
    golfers = kalshi_odds.to_golfers(markets, price=price)
    if not golfers:
        raise ValueError(
            f"{len(markets)} Kalshi markets parsed to zero golfers -- every market is "
            "settled or finalized. Re-pull during tournament week."
        )
    out = kalshi_odds.clean_golfers(golfers)
    check_book(out)
    return out


def check_book(_golfers, price_mode=None, requested_price=None):
    """
    Sanity-check a book that arrived already converted, i.e. one that never passed
    through to_golfers() and so never met its filters.

    A settled Kalshi market quotes bid=0.0000 / ask=1.0000. to_golfers() drops those on
    the status filter, but `kalshi_odds.py --include-closed` can write them into an odds
    file, and reading that file back produces a field where every golfer is a certainty.
    The book sums to the field size, normalizes cleanly, and yields a confident grouping
    of a tournament that finished months ago. Raise instead.
    """
    certainties = [g["golfer_name"] for g in _golfers if g["odds"] >= 1.0]
    if certainties:
        raise ValueError(
            f"{len(certainties)} golfer(s) are priced at or above 1.0 "
            f"({certainties[:3]}...): this file holds settled markets, which quote "
            "ask=1.0000. It was almost certainly written with --include-closed. "
            "Re-pull during tournament week."
        )

    if price_mode and requested_price and price_mode != requested_price:
        print(
            f"!! WARNING: this odds file was written with --price {price_mode}, but the "
            f"run asked for {requested_price}. A converted file cannot be re-priced -- "
            f"{price_mode} is what will be grouped. Re-pull to change it."
        )

    total = sum(g["odds"] for g in _golfers)
    if total > 1.6:
        print(
            f"!! WARNING: this book sums to {total:.3f}. A Winner book runs ~1.3. A Top 5 "
            "book runs toward 5 and a Top 10 toward 10 -- those outcomes are NOT mutually "
            "exclusive, so the de-vig gives share-of-N-slots rather than probability, and "
            "the 1/participants threshold does not mean what it means on a Winner book. "
            "Groups will still balance; read the numbers as weights, not odds."
        )
    elif total < 0.9:
        print(
            f"!! WARNING: this book sums to {total:.3f}, below 1.0. A live Winner ask book "
            "carries an overround and sums above 1.0. This field is stale, thin, or was "
            "priced off the bid."
        )


def _looks_like_kalshi_markets(markets):
    """A Kalshi market carries yes_sub_title / event_ticker; a DraftKings one does not."""
    if not markets:
        return False
    head = markets[0]
    return isinstance(head, dict) and ("yes_sub_title" in head or "event_ticker" in head)


def load_golfers(d, price=PRICE_MODE, dk_odds_type=DK_ODDS_TYPE):
    """
    Dispatch on the shape of a loaded odds file.

    Both DraftKings and Kalshi envelopes have a top-level "markets" key, so the
    shapes are told apart by their contents rather than by that key alone. An
    unrecognised shape raises: guessing here would produce a confident wrong
    grouping, which is the failure this project has already paid for once.
    """
    if isinstance(d, list):
        # A bare [{golfer_name, odds}, ...] list -- already in grouper shape.
        if not d:
            raise ValueError("odds file is an empty list")
        if not all(isinstance(g, dict) and "golfer_name" in g and "odds" in g for g in d):
            raise ValueError("list-shaped odds file must hold {golfer_name, odds} objects")
        out = [
            {"golfer_name": g["golfer_name"], "odds": float(g["odds"]), "golfer_id": g.get("golfer_id")}
            for g in d
        ]
        check_book(out)
        return out

    if not isinstance(d, dict):
        raise ValueError(f"unrecognised odds file: expected an object or a list, got {type(d).__name__}")

    if "golfers" in d:
        return list_kalshi_golf_odds(d, price=price)

    if "selections" in d and "markets" in d:
        return list_dk_golf_odds(d, odds_type=dk_odds_type)

    if "markets" in d:
        if _looks_like_kalshi_markets(d["markets"]):
            return list_kalshi_golf_odds(d, price=price)
        raise ValueError(
            "file has markets[] but no selections[] and no Kalshi fields. If this is a "
            "DraftKings payload it is truncated; if it is Kalshi it holds no markets."
        )

    raise ValueError(
        f"unrecognised odds file shape (top-level keys: {sorted(d)}). Expected a Kalshi "
        "markets/golfers envelope or a DraftKings markets+selections envelope."
    )


def read_odds_file(path, price=PRICE_MODE, dk_odds_type=DK_ODDS_TYPE):
    with open(path) as f:
        return load_golfers(json.load(f), price=price, dk_odds_type=dk_odds_type)


def find_odds_file(candidates=DATA_FILES):
    return next((p for p in candidates if os.path.exists(p)), None)


def read_participants(path):
    """
    Load the participant list, failing with a sentence rather than a stack trace.

    A JSON object here used to be the nastiest input in the tool: it has a length, so
    everything downstream accepted it, and the run died with a KeyError inside stdlib
    random.py -- after the grouping had already run.
    """
    try:
        with open(path) as f:
            names = json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"{path} not found. It should hold a JSON list of participant names, "
            'e.g. ["Mo", "Diogo", "Luis", "Cody", "Darwin"] -- one group is built per name.'
        )
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}")

    if not isinstance(names, list):
        raise SystemExit(
            f"{path} holds a JSON {type(names).__name__}, not a list. Expected "
            '["Mo", "Diogo", ...] -- one group is built per name.'
        )
    if not names:
        raise SystemExit(f"{path} holds no participants")
    if not all(isinstance(n, str) for n in names):
        bad = [n for n in names if not isinstance(n, str)][:3]
        raise SystemExit(f"{path} must hold only name strings; found {bad}")
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"{path} has duplicate names {dupes}; each participant gets one group")
    return names


# ---------------------------------------------------------------------------
# De-vig and exclusions
# ---------------------------------------------------------------------------

def normalize_probabilities(_golfers):
    """
    De-vig: divide every golfer by the observed field sum so the book becomes a
    distribution summing to 1.0.

    Normalizing by the OBSERVED sum matters. A Kalshi Winner ask book runs ~1.30,
    not "just above 1.0" -- and a Top 5 book runs toward 5.0, a Top 10 book toward
    10.0. Any de-vig that assumes a target of ~1.0 is wrong on every one of them.
    """
    total = sum(g["odds"] for g in _golfers)
    if total <= 0:
        raise ValueError(
            "field probabilities sum to zero -- the odds never loaded. Check the source "
            "before treating this as a real field."
        )
    return [{**g, "odds": g["odds"] / total} for g in _golfers]


def odds_to_conditional(_golfers, _excluded_golfers=(), _verbose=False):
    """
    De-vig the field, drop the excluded golfers, and re-scale the rest so the
    remaining probabilities sum to 1.0 again.

    The result is "probability of winning GIVEN that none of the excluded golfers
    wins" -- which is what the pool actually plays for once the favourites are
    taken off the board.
    """
    excluded = set(_excluded_golfers)
    total_prob = sum(g["odds"] for g in _golfers)
    if total_prob <= 0:
        raise ValueError("field probabilities sum to zero -- the odds never loaded")

    fair = normalize_probabilities(_golfers)

    missing = excluded - {g["golfer_name"] for g in fair}
    if missing:
        print(f"!! WARNING: excluded golfers not in this field, so they excluded nothing: {sorted(missing)}")

    survivors = [g for g in fair if g["golfer_name"] not in excluded]
    if not survivors:
        raise ValueError("the exclusion list covers the whole field -- nothing left to group")

    # Re-scale by what SURVIVES rather than by 1 - sum(excluded). The two are equal in
    # exact arithmetic, but a sum of normalized floats lands on 0.9999999999999999 often
    # enough that a "did we exclude everyone?" test against 1.0 quietly fails and returns
    # an empty field instead of raising.
    excluded_fair_prob = sum(g["odds"] for g in fair if g["golfer_name"] in excluded)
    surviving_prob = sum(g["odds"] for g in survivors)
    scale_factor = 1 / surviving_prob

    if _verbose:
        print("Conditional Odds Mode.. calculating VIG")
        print(f"Total Market Probability: {total_prob}")
        print(f"Excluded Fair Probability: {excluded_fair_prob}")
        print(f"Conditional Scale Factor: {scale_factor}")

    return [{**g, "odds": g["odds"] * scale_factor} for g in survivors]


def golfers_over_threshold(_golfers, n_participants):
    """
    Golfers whose de-vigged probability exceeds 1/participants.

    Such a golfer is worth more than a whole group's fair share on their own, so no
    partition can balance around them. The threshold is compared against DE-VIGGED
    probabilities -- comparing against the raw ask book would flag golfers ~30% too
    eagerly, because that book sums to ~1.30 rather than 1.0.
    """
    if n_participants <= 0:
        raise ValueError("n_participants must be positive")
    threshold = 1.0 / n_participants
    return [g for g in normalize_probabilities(_golfers) if g["odds"] > threshold]


def auto_exclusions(_golfers, n_participants):
    """
    Resolve the exclusion set for the 1/participants rule, iterating to a fixed point.

    Removing a golfer redistributes their probability over everyone left, which can
    push the next-best golfer over the line. One pass is not enough.

    The cascade stops before it can leave fewer golfers than there are groups to fill.
    A field that short is not groupable at all, and the DP partitioner does not fail
    gracefully on one -- it fails to terminate.
    """
    excluded, remaining = [], list(_golfers)
    while len(remaining) > n_participants:
        over = golfers_over_threshold(remaining, n_participants)
        if not over:
            break
        names = {g["golfer_name"] for g in over}
        if len(remaining) - len(names) < n_participants:
            # Removing the whole batch would over-shrink the field. Take them in
            # descending order and stop at the floor.
            room = len(remaining) - n_participants
            names = {
                g["golfer_name"]
                for g in sorted(over, key=lambda g: g["odds"], reverse=True)[:room]
            }
            if not names:
                break
        excluded.extend(sorted(names))
        remaining = [g for g in remaining if g["golfer_name"] not in names]
    return excluded


# ---------------------------------------------------------------------------
# Grouping and reporting
# ---------------------------------------------------------------------------

def validate_groups(groups, golfers):
    valid_groups = True
    seen_golfers = set()
    for group in groups.values():
        for golfer in group:
            if golfer["golfer_name"] in seen_golfers:
                valid_groups = False
                break
            seen_golfers.add(golfer["golfer_name"])
    if valid_groups:
        valid_groups = len(seen_golfers) == len(golfers)
    return valid_groups


# Function to print results
def print_results(method_name, assigned_groups, group_totals):
    print(f"\n{method_name} Results: {max([v for v in group_totals.values()]) - min([v for v in group_totals.values()])} delta")
    for name, group in assigned_groups.items():
        print(f"{name}'s group: {[golfer['golfer_name'] for golfer in group]}, Total Odds: {group_totals[name]}")
    print(json.dumps(assigned_groups, indent=2))


def percentage_difference(value1, value2):
    """Calculate the percentage difference between two decimal values."""
    try:
        difference = abs(value1 - value2)
        avg = (value1 + value2) / 2
        percent_diff = (difference / avg) * 100
        return percent_diff
    except ZeroDivisionError:
        return float('inf')


def confirm_group(method_name, assigned_groups, group_totals, golfers, output_dir=OUTPUT_DIR):
    # print_results(method_name, assigned_groups, group_totals)
    group_info = {
        "method": method_name,
        "groups": assigned_groups,
        "totals": group_totals,
        "valid": validate_groups(assigned_groups, golfers),
        "delta": max([v for v in group_totals.values()]) - min([v for v in group_totals.values()]),
        "delta_percentage": percentage_difference(max([v for v in group_totals.values()]), min([v for v in group_totals.values()]))
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{method_name}.json'), 'w') as f:
        json.dump(group_info, f, indent=4)
    if group_info['valid']:
        return group_info
    return None


METHODS = ("backtracking", "dp", "sa", "ga", "greedy")

METHOD_LABELS = {
    "backtracking": "Backtracking",
    "dp": "Dynamic Programming",
    "sa": "Simulated Annealing",
    "ga": "Genetic Algorithm",
    "greedy": "Greedy Algorithm",
}


def run_methods(golfers, n_groups, methods=METHODS, sa_iter=100000, ga_pop=500,
                ga_generations=10000, ga_mutation=0.1):
    """
    Run each requested partitioner and return {label: groups}.

    Every method needs at least n_groups golfers. dp_generate_groups in particular
    does not merely return a bad answer on a short field -- its reconstruction loop
    never terminates. main() checks the field size after exclusions for that reason.
    """
    if len(golfers) < n_groups:
        raise ValueError(f"{len(golfers)} golfers cannot fill {n_groups} groups")
    out = {}
    for m in methods:
        # Each method gets its own list: greedy_redistribute_groups sorts in place,
        # so sharing one would make a run depend on the order the methods were asked for.
        field = list(golfers)
        if m == "backtracking":
            out[METHOD_LABELS[m]] = backtracking_generate_groups(field, n_groups)
        elif m == "dp":
            out[METHOD_LABELS[m]] = dp_generate_groups(field, n_groups)
        elif m == "sa":
            out[METHOD_LABELS[m]] = sa_generate_groups(field, n_groups, max_iter=sa_iter)
        elif m == "ga":
            out[METHOD_LABELS[m]] = ga_generate_groups(
                field, n_groups, pop_size=ga_pop, generations=ga_generations, mutation_rate=ga_mutation
            )
        elif m == "greedy":
            out[METHOD_LABELS[m]] = greedy_redistribute_groups(field, n_groups)
        else:
            raise ValueError(f"unknown method: {m}")
    return out


def load_odds(args):
    """
    Resolve the odds for this run and report where they came from.

    Order: an explicit --data-file, then --event (live pull), then a local
    kalshi_data.json / dk_data.json, then a live pull of the newest event that has
    active markets. Returns (golfers, description).

    Whatever the route, the field comes back in one canonical order. The partitioners
    are order-sensitive -- feeding the DP the API's own ordering rather than a sorted
    one measurably worsens the partition -- so the sort belongs here, once, rather than
    in whichever caller happens to remember it.
    """
    path = args.data_file or (None if args.event else find_odds_file())
    if path:
        golfers = read_odds_file(path, price=args.price, dk_odds_type=args.dk_odds_type)
        return sort_field(golfers), f"{path}{_file_provenance(path)}"

    event_ticker, golfers, report = kalshi_odds.fetch_golfers(
        event_ticker=args.event, price=args.price, series=args.series
    )
    print(f"Kalshi liquidity: {report['two_sided_quotes']}/{report['golfers']} two-sided, "
          f"{report['spreads_over_5c']} spreads over 5c, book sums to {report['probability_sum']}")
    return sort_field(kalshi_odds.clean_golfers(golfers)), f"Kalshi {event_ticker} ({args.price})"


def sort_field(_golfers):
    """
    Canonical order: probability descending, then name.

    Sorting on probability alone is not a total order here. The bottom third of a golf
    field is functionally tied -- a dozen golfers sit on the 0.1c floor together -- and
    Python's sort is stable, so tied golfers keep whatever order the file happened to
    list them in. The partitioners are order-sensitive, so that alone is enough to make
    the same odds produce different groups depending on how the file was written.
    """
    return sorted(_golfers, key=lambda g: (-g["odds"], g["golfer_name"]))


def _file_provenance(path):
    """When and what an odds file captured, so a stale file cannot pass unnoticed."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return ""
    if not isinstance(d, dict):
        return ""
    bits = [str(d[k]) for k in ("event", "price_mode", "fetched_at") if d.get(k)]
    return f" [{', '.join(bits)}]" if bits else ""


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", help="Kalshi event ticker, e.g. KXPGATOUR-WYC26. Pulls live.")
    ap.add_argument("--series", default=MARKET_SERIES,
                    help=f"Kalshi series to resolve the current event from (default {MARKET_SERIES})")
    ap.add_argument("--price", default=PRICE_MODE, choices=list(kalshi_odds.PRICE_MODES),
                    help=f"Kalshi price mode (default {PRICE_MODE})")
    ap.add_argument("--data-file", help=f"read odds from a file instead of the API (default: first of {DATA_FILES})")
    ap.add_argument("--dk-odds-type", default=DK_ODDS_TYPE,
                    help="market label to read from an archived DraftKings payload")
    ap.add_argument("--participants", default=PARTICIPANTS_FILE)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--exclude", action="append", metavar="NAME",
                    help="golfer to exclude; repeatable. Overrides the built-in list.")
    ap.add_argument("--no-exclude", action="store_true", help="exclude nobody")
    ap.add_argument("--auto-exclude", action="store_true",
                    help="also exclude every golfer whose de-vigged probability exceeds 1/participants")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help=f"comma-separated subset of {','.join(METHODS)}")
    ap.add_argument("--sa-iter", type=int, default=100000)
    ap.add_argument("--ga-pop", type=int, default=500)
    ap.add_argument("--ga-generations", type=int, default=10000)
    ap.add_argument("--ga-mutation", type=float, default=0.1)
    ap.add_argument("--seed", type=int, help="seed the RNG so a run is reproducible")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.seed is not None:
        random.seed(args.seed)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown method(s) {unknown}; choose from {list(METHODS)}")

    start_time = time.time()

    # Read the participants BEFORE fetching odds. It is the cheapest thing that can be
    # wrong, and finding out after a live pull and a five-minute grouping run is no way
    # to learn that the file is a JSON object rather than a list.
    participant_real_names = read_participants(args.participants)

    golfers, source = load_odds(args)

    participant_names = [f"Group {i}" for i in range(len(participant_real_names))]
    n_groups = len(participant_names)
    if len(golfers) < n_groups:
        raise SystemExit(f"only {len(golfers)} golfers for {n_groups} groups")

    print(f"Odds source: {source} -- {len(golfers)} golfers, book sums to "
          f"{sum(g['odds'] for g in golfers):.4f}")

    if args.no_exclude:
        exclude_list = []
    else:
        exclude_list = list(args.exclude) if args.exclude else list(EXCLUDE_LIST)

    over = golfers_over_threshold(golfers, n_groups)
    if over:
        print(f"Golfers above the 1/{n_groups} fair-share threshold "
              f"({1 / n_groups:.4f}): " +
              ", ".join(f"{g['golfer_name']} {g['odds']:.4f}" for g in over))
    else:
        print(f"No golfer exceeds the 1/{n_groups} fair-share threshold ({1 / n_groups:.4f}).")

    if args.auto_exclude:
        # Resolve the threshold against the field the pool will actually play, i.e.
        # after the named exclusions are gone -- their weight redistributes too.
        named = set(exclude_list)
        remaining = [g for g in golfers if g["golfer_name"] not in named]
        auto = [n for n in auto_exclusions(remaining, n_groups) if n not in named]
        if auto:
            print(f"Auto-excluding: {auto}")
            exclude_list.extend(auto)

    if exclude_list:
        golfers = odds_to_conditional(golfers, exclude_list, True)
    else:
        # De-vig anyway, so group totals are always read on the same 1.0 scale.
        golfers = normalize_probabilities(golfers)

    # Re-check AFTER exclusions. The check above ran on the full field, and an
    # exclusion list -- named or automatic -- can drop it below the group count.
    # dp_generate_groups does not fail gracefully on a short field; it hangs.
    if len(golfers) < n_groups:
        raise SystemExit(
            f"only {len(golfers)} golfers left after excluding {len(exclude_list)}, "
            f"which cannot fill {n_groups} groups. Exclude fewer golfers or run fewer groups."
        )

    print(f"Grouping {len(golfers)} golfers into {n_groups} groups...")

    groups_by_method = run_methods(
        golfers, n_groups, methods=methods, sa_iter=args.sa_iter, ga_pop=args.ga_pop,
        ga_generations=args.ga_generations, ga_mutation=args.ga_mutation,
    )

    result = {}
    for label, groups in groups_by_method.items():
        assigned = {name: group for name, group in zip(participant_names, groups)}
        totals = {name: calculate_total_odds(group) for name, group in assigned.items()}
        result[label] = confirm_group(label, assigned, totals, golfers, output_dir=args.output_dir)

    best_groups = None
    for k, v in result.items():
        if v is not None:
            if best_groups is None:
                best_groups = v
            elif v.get('delta_percentage') < best_groups.get('delta_percentage'):
                best_groups = v

    if best_groups is None:
        print("NO VALID GROUPS FOUND")
        return 1

    print(f"Odds source was {source}...")
    print(f"Best Grouping Method was {best_groups.get('method')} with a delta percentage of {best_groups.get('delta_percentage')}%")
    print("Assigning names to group...")
    # Shuffle participant names and assign groups
    random.shuffle(participant_real_names)
    print(f"Randomized Order: {participant_real_names}")
    final_groups = {"groups": {}, "totals": {}}
    for index, i in enumerate(participant_real_names):
        final_groups['groups'][i] = best_groups['groups'][f"Group {index}"]
        final_groups['totals'][i] = best_groups['totals'][f"Group {index}"]
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'BESTGROUPS.json'), 'w') as f:
        json.dump(final_groups, f, indent=4)
    print("-------------FINAL GROUPS--------------")
    for k, v in final_groups['groups'].items():
        print(f"Group {k} - Total Odds {final_groups['totals'][k]}: {', '.join([i['golfer_name'] for i in v])}")
        print("------------------------------------")
    print_completion_time(start_time, time.time())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

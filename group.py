import json
import random
import time
from groupers import calculate_total_odds, backtracking_generate_groups, dp_generate_groups, sa_generate_groups, ga_generate_groups, greedy_redistribute_groups

def fraction_to_decimal(fraction_str):
    numerator, denominator = fraction_str.split('/')
    numerator = float(numerator)
    denominator = float(denominator)
    decimal_value = numerator / denominator
    return decimal_value

def print_completion_time(start_time,end_time):
    print(f"Completed in {int(end_time - start_time) // 60} minutes and {int(end_time - start_time) % 60} seconds")

def list_dk_golf_odds(d, odds_type="Winner"):
    result = []
    odds_section = next((i for i in d['eventGroup']['offerCategories'][0]['offerSubcategoryDescriptors'][0]['offerSubcategory']['offers'][0] if i.get('label') == odds_type), None)
    sublist = odds_section.get('outcomes')
    for o in sublist:
        result.append({
            "golfer_name": o['participants'][0]['name'],
            "odds": 1 / fraction_to_decimal(o['oddsFractional'])
        })
    return result

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

def confirm_group(method_name, assigned_groups, group_totals, golfers):
    # print_results(method_name, assigned_groups, group_totals)
    group_info = {
        "method": method_name,
        "groups": assigned_groups,
        "totals": group_totals,
        "valid": validate_groups(assigned_groups, golfers),
        "delta": max([v for v in group_totals.values()]) - min([v for v in group_totals.values()]),
        "delta_percentage": percentage_difference(max([v for v in group_totals.values()]), min([v for v in group_totals.values()]))
    }
    with open(f'output/{method_name}.json','w') as f:
        json.dump(group_info, f, indent=4)
    if group_info['valid']:
        return group_info
    return None

if __name__ == '__main__':
    # ODDS_TYPE = "Winner"
    ODDS_TYPE = "Top 10 (Including Ties)"
    start_time = time.time()
    with open('dk_data.json') as dkdf:
        dkd = json.load(dkdf)
    with open ('participants.json') as pf:
        participant_real_names = json.load(pf)
    participant_names = [f"Group {i}" for i in range(len(participant_real_names))]
    golfers = list_dk_golf_odds(dkd, odds_type=ODDS_TYPE)
    n_groups = len(participant_names)

    backtracking_groups = backtracking_generate_groups(golfers, n_groups)
    dp_groups = dp_generate_groups(golfers, n_groups)
    sa_groups = sa_generate_groups(golfers, n_groups, max_iter=100000)
    ga_groups = ga_generate_groups(golfers, n_groups, pop_size=500, generations=10000, mutation_rate=0.1)
    greedy_groups = greedy_redistribute_groups(golfers, n_groups)   

    backtracking_assigned_groups = {name: group for name, group in zip(participant_names, backtracking_groups)}
    dp_assigned_groups = {name: group for name, group in zip(participant_names, dp_groups)}
    sa_assigned_groups = {name: group for name, group in zip(participant_names, sa_groups)}
    ga_assigned_groups = {name: group for name, group in zip(participant_names, ga_groups)}
    greedy_assigned_groups = {name: group for name, group in zip(participant_names, greedy_groups)}

    # Calculate total odds for each group
    backtracking_group_totals = {name: calculate_total_odds(group) for name, group in backtracking_assigned_groups.items()}
    dp_group_totals = {name: calculate_total_odds(group) for name, group in dp_assigned_groups.items()}
    sa_group_totals = {name: calculate_total_odds(group) for name, group in sa_assigned_groups.items()}
    ga_group_totals = {name: calculate_total_odds(group) for name, group in ga_assigned_groups.items()}
    greedy_group_totals = {name: calculate_total_odds(group) for name, group in greedy_assigned_groups.items()}

    # Print results for each method
    result = {}
    result["Backtracking"]=confirm_group("Backtracking", backtracking_assigned_groups, backtracking_group_totals, golfers)
    result["Dynamic Programming"]=confirm_group("Dynamic Programming", dp_assigned_groups, dp_group_totals, golfers)
    result["Simulated Annealing"]=confirm_group("Simulated Annealing", sa_assigned_groups, sa_group_totals, golfers)
    result["Genetic Algorithm"]=confirm_group("Genetic Algorithm", ga_assigned_groups, ga_group_totals, golfers)
    result["Greedy Algorithm"]=confirm_group("Greedy Algorithm", greedy_assigned_groups, greedy_group_totals, golfers)

    min_delta = math.infinity
    best_groups = None
    for k,v in result.items():
        if v is not None:
            if best_groups is None:
                best_groups = v
            elif v.get('delta_percentage') < best_groups.get('delta_percentage'):
                best_groups = v
    
    if best_groups is None:
        print("NO VALID GROUPS FOUND")
    else:
        print(f"Best Grouping Method was {best_groups.get('method')} with a delta percentage of {best_groups.get('method')}%")
        print("Assigning names to group...")
        # Shuffle participant names and assign groups
        random.shuffle(participant_real_names)
        print(f"Randomized Order: {participant_real_names}")
        final_groups = {"groups": {}, "totals": {}}
        for index, i in enumerate(participant_real_names):
            final_groups['groups'][i] = best_groups['groups'][f"Group {index}"]
            final_groups['totals'][i] = best_groups['totals'][f"Group {index}"]
        with open(f'output/BESTGROUPS.json','w') as f:
            json.dump(final_groups, f, indent=4)
        print("FINAL GROUPS:")
        print(json.dumps(final_groups, indent=4))

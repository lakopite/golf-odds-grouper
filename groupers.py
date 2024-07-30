import random
import math
import numpy as np
from tqdm import tqdm
from functools import lru_cache

# Function to calculate the total odds of a group
def calculate_total_odds(group):
    return sum(golfer["odds"] for golfer in group)

# Backtracking Algorithm
def backtracking_generate_groups(golfers, n_groups):
    # Extract odds as a separate list for easier handling
    odds = [golfer["odds"] for golfer in golfers]
    n = len(odds)

    # Memoization cache
    @lru_cache(None)
    def dp(i, g):
        if g == 0:
            return 0 if i == 0 else float('inf')
        if i == 0:
            return float('inf')

        include_current = dp(i - 1, g - 1) + odds[i - 1]
        exclude_current = dp(i - 1, g)

        return min(include_current, exclude_current)

    # Initialize progress bar
    total_steps = (n + 1) * (n_groups + 1)
    progress_bar = tqdm(total=total_steps, desc="Memoized Partitioning Progress")

    # Populate the DP table and update progress bar
    for i in range(1, n + 1):
        for g in range(1, n_groups + 1):
            dp(i, g)
            progress_bar.update(1)

    progress_bar.close()

    # Reconstruct groups
    groups = [[] for _ in range(n_groups)]
    used = [False] * n  # Track used golfers
    i, j = n, n_groups

    # Backtracking to find which items were included
    while j > 0 and i > 0:
        if dp(i, j) == dp(i - 1, j):
            i -= 1
        else:
            groups[j - 1].append(golfers[i - 1])
            used[i - 1] = True
            i -= 1
            j -= 1

    # Assign remaining unassigned golfers to the best group
    for idx, golfer in enumerate(golfers):
        if not used[idx]:
            # Find the group with the smallest current total odds
            min_group_index = min(range(n_groups), key=lambda k: sum(g["odds"] for g in groups[k]))
            groups[min_group_index].append(golfer)

    # Calculate the total odds for each group and the difference
    group_totals = [sum(golfer["odds"] for golfer in group) for group in groups]
    max_total = max(group_totals)
    min_total = min(group_totals)
    min_diff = max_total - min_total

    return groups

# Dynamic Programming Approach
def dp_generate_groups(golfers, n_groups):
    n = len(golfers)
    max_sum = sum(g["odds"] for g in golfers)
    
    # Initialize DP table and a table for tracking the group distribution
    dp = np.full((n + 1, n_groups + 1, max_sum + 1), float('inf'))
    dp[0][0][0] = 0
    
    # Track the selection of golfers
    selected = np.zeros((n + 1, n_groups + 1, max_sum + 1), dtype=bool)
    
    # Progress bar setup
    total_steps = (n + 1) * (n_groups + 1)
    progress_bar = tqdm(total=total_steps, desc="DP Progress")
    
    for i in range(1, n + 1):
        golfer_odds = golfers[i - 1]["odds"]
        for j in range(1, n_groups + 1):
            for k in range(max_sum + 1):
                # Skip if previous group sum is invalid
                if dp[i - 1][j - 1][k] == float('inf'):
                    continue
                
                # Including the current golfer
                if k + golfer_odds <= max_sum:
                    include = dp[i - 1][j - 1][k] + golfer_odds
                    if include < dp[i][j][k + golfer_odds]:
                        dp[i][j][k + golfer_odds] = include
                        selected[i][j][k + golfer_odds] = True
                
                # Excluding the current golfer
                exclude = dp[i - 1][j][k]
                if exclude < dp[i][j][k]:
                    dp[i][j][k] = exclude
                    selected[i][j][k] = False
                
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Finding the minimal difference
    min_diff = float('inf')
    best_groups = []
    
    for k in range(max_sum + 1):
        if dp[n][n_groups][k] != float('inf'):
            max_total = k
            min_total = dp[n][n_groups][k]
            diff = max_total - min_total
            if diff < min_diff:
                min_diff = diff
                best_groups = []
                group_sums = [0] * n_groups
                
                # Reconstruct groups
                i, j, current_sum = n, n_groups, max_total
                while j > 0 and i > 0:
                    if selected[i][j][current_sum]:
                        best_groups.append(golfers[i - 1])
                        group_sums[j - 1] += golfers[i - 1]["odds"]
                        current_sum -= golfers[i - 1]["odds"]
                        j -= 1
                    i -= 1
                
                # Assign the remaining golfers to the groups
                unassigned_golfers = set(golfers) - set(best_groups)
                for golfer in unassigned_golfers:
                    min_group_index = min(range(n_groups), key=lambda x: group_sums[x])
                    group_sums[min_group_index] += golfer["odds"]
                    best_groups[min_group_index].append(golfer)
    
    # Ensuring all golfers are included in the final groups
    groups = [[] for _ in range(n_groups)]
    for group in best_groups:
        min_group_index = min(range(n_groups), key=lambda x: sum(g["odds"] for g in groups[x]))
        groups[min_group_index].append(group)
    
    return groups

# Simulated Annealing
def sa_generate_groups(golfers, n_groups, max_iter=1000, temp=1000, cooling_rate=0.99):
    def calculate_cost(groups):
        totals = [calculate_total_odds(group) for group in groups]
        return max(totals) - min(totals)

    def swap_golfers(groups):
        new_groups = [group[:] for group in groups]
        g1, g2 = random.sample(range(n_groups), 2)
        if new_groups[g1] and new_groups[g2]:
            i1, i2 = random.randint(0, len(new_groups[g1]) - 1), random.randint(0, len(new_groups[g2]) - 1)
            new_groups[g1][i1], new_groups[g2][i2] = new_groups[g2][i2], new_groups[g1][i1]
        return new_groups

    golfers = golfers[:]
    random.shuffle(golfers)
    groups = [[] for _ in range(n_groups)]
    for i, golfer in enumerate(golfers):
        groups[i % n_groups].append(golfer)

    current_cost = calculate_cost(groups)
    best_groups, best_cost = groups, current_cost

    progress_bar = tqdm(total=max_iter, desc="SA Progress")

    for _ in range(max_iter):
        new_groups = swap_golfers(groups)
        new_cost = calculate_cost(new_groups)
        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temp):
            groups, current_cost = new_groups, new_cost
            if new_cost < best_cost:
                best_groups, best_cost = new_groups, new_cost
        temp *= cooling_rate
        progress_bar.update(1)

    progress_bar.close()
    
    return best_groups

# Genetic Algorithm
def ga_generate_groups(golfers, n_groups, pop_size=100, generations=1000, mutation_rate=0.1):
    def initialize_population():
        """Initialize the population with random solutions."""
        population = []
        for _ in range(pop_size):
            shuffled = golfers[:]
            random.shuffle(shuffled)
            groups = [shuffled[i::n_groups] for i in range(n_groups)]
            population.append(groups)
        return population

    def calculate_fitness(groups):
        """Calculate the fitness based on the difference between the max and min group odds."""
        group_totals = [sum(g["odds"] for g in group) for group in groups]
        return max(group_totals) - min(group_totals)

    def crossover(parent1, parent2):
        """Crossover two parents to produce two offspring."""
        child1, child2 = [], []
        crossover_point = random.randint(1, len(golfers) - 2)
        for i in range(n_groups):
            child1.append(parent1[i][:crossover_point] + parent2[i][crossover_point:])
            child2.append(parent2[i][:crossover_point] + parent1[i][crossover_point:])
        
        return repair(child1), repair(child2)

    def mutate(groups):
        """Mutate the groups by swapping golfers between groups."""
        for group in groups:
            if random.random() < mutation_rate:
                g1, g2 = random.sample(range(n_groups), 2)
                if groups[g1] and groups[g2]:
                    i1 = random.randint(0, len(groups[g1]) - 1)
                    i2 = random.randint(0, len(groups[g2]) - 1)
                    groups[g1][i1], groups[g2][i2] = groups[g2][i2], groups[g1][i1]
        return groups

    def repair(groups):
        """Repair groups to ensure all golfers are unique and included."""
        all_golfers = {golfer['golfer_name']: golfer for golfer in golfers}
        used_golfers = set()
        repaired_groups = [[] for _ in range(n_groups)]

        for group in groups:
            for golfer in group:
                if golfer["golfer_name"] not in used_golfers:
                    repaired_groups[groups.index(group)].append(golfer)
                    used_golfers.add(golfer["golfer_name"])

        # Add missing golfers
        missing_golfers = [g for g in golfers if g["golfer_name"] not in used_golfers]
        for golfer in missing_golfers:
            min_group_index = min(range(n_groups), key=lambda k: sum(g["odds"] for g in repaired_groups[k]))
            repaired_groups[min_group_index].append(golfer)

        return repaired_groups

    def select_parents(population):
        """Select parents based on their fitness scores using a tournament selection."""
        tournament_size = 5
        selected = random.sample(population, tournament_size)
        selected.sort(key=lambda groups: calculate_fitness(groups))
        return selected[0], selected[1]

    population = initialize_population()
    progress_bar = tqdm(total=generations, desc="Genetic Algorithm Progress")

    best_solution = None
    best_fitness = float('inf')

    for generation in range(generations):
        new_population = []

        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = sorted(new_population, key=lambda groups: calculate_fitness(groups))[:pop_size]

        current_best = population[0]
        current_fitness = calculate_fitness(current_best)

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = current_best

        progress_bar.update(1)

    progress_bar.close()

    return best_solution

# Greedy Algorithm with Redistribution
def greedy_redistribute_groups(golfers, n_groups):
    groups = [[] for _ in range(n_groups)]
    golfers.sort(key=lambda x: x["odds"], reverse=True)

    for i, golfer in enumerate(golfers):
        groups[i % n_groups].append(golfer)

    progress_bar = tqdm(total=len(golfers), desc="Greedy Redistribution Progress")

    def redistribute(groups):
        for _ in range(len(golfers)):
            max_group = max(groups, key=calculate_total_odds)
            min_group = min(groups, key=calculate_total_odds)
            if calculate_total_odds(max_group) - calculate_total_odds(min_group) < 0.01:
                break
            max_golfer = max(max_group, key=lambda x: x["odds"])
            min_golfer = min(min_group, key=lambda x: x["odds"])
            if max_golfer["odds"] > min_golfer["odds"]:
                max_group.remove(max_golfer)
                min_group.append(max_golfer)
                min_group.remove(min_golfer)
                max_group.append(min_golfer)
            progress_bar.update(1)

    redistribute(groups)
    progress_bar.close()
    
    return groups
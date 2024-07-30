# Golf Odds Grouper

## Pre-requisites
- Requires a valid json response from DraftKings Sportsbook e.g (`https://sportsbook-nash.draftkings.com/sites/US-SB/api/v5/eventgroups/205797?format=json`) to exist in the root of the project `/path/to/repo/dk_data.json`
- Requires a json list of participant name strings to exist at the root of the project `/path/to/repo/participants.json` (example `["Mo", "Diogo", "Luis", "Cody", "Darwin"]`)

## Purpose
Using the documented algorithms below, this tool aims to determine the optimal groups aiming to minimize the difference between the groups with the highest and lowest cumulative odds. Once the best groups are identified, they are randomly assigned to the participants.

## Golfers Grouping Algorithms

This project provides five different methods to partition a list of golfers into a specified number of balanced groups based on their odds of winning. The methods aim to minimize the difference between the group with the highest cumulative odds and the group with the lowest cumulative odds.

## Methods

1. **Backtracking**
2. **Dynamic Programming**
3. **Simulated Annealing**
4. **Genetic Algorithm**
5. **Greedy Algorithm with Redistribution**

### 1. Backtracking

**Function:** `backtracking_generate_groups`

**Parameters:**
- `golfers`: List of dictionaries, where each dictionary contains `golfer_name` and `odds` of each golfer.
- `n_groups`: Integer, the number of groups to divide the golfers into.

**Description:**
Uses a backtracking approach to generate balanced groups of golfers. It recursively tries to assign golfers to groups to minimize the difference in total odds.

### 2. Dynamic Programming

**Function:** `dp_generate_groups`

**Parameters:**
- `golfers`: List of dictionaries, where each dictionary contains `golfer_name` and `odds` of each golfer.
- `n_groups`: Integer, the number of groups to divide the golfers into.

**Description:**
Uses a dynamic programming approach to generate balanced groups of golfers. It iteratively calculates the minimum difference in total odds between groups and assigns golfers accordingly.

### 3. Simulated Annealing

**Function:** `sa_generate_groups`

**Parameters:**
- `golfers`: List of dictionaries, where each dictionary contains `golfer_name` and `odds` of each golfer.
- `n_groups`: Integer, the number of groups to divide the golfers into.
- `max_iter`: Integer (default=1000), the maximum number of iterations to perform.
- `temp`: Float (default=1000), the initial temperature for the simulated annealing process. Think of temp as the "energy" or "excitement" level at the start of the process. A high temperature means the algorithm is more willing to accept changes, even if they make things worse temporarily, to explore more potential solutions.
- `cooling_rate`: Float (default=0.99), the rate at which the temperature decreases. The cooling_rate controls how quickly the temperature drops over time. A high cooling rate means the temperature drops quickly, and the algorithm becomes more selective faster. A low cooling rate means the temperature drops slowly, allowing the algorithm to explore a broader range of solutions for a longer time.

**Description:**
Uses a simulated annealing approach to generate balanced groups of golfers. It starts with a random initial solution and iteratively tries to improve it by making random swaps and accepting changes based on the temperature.

### 4. Genetic Algorithm

**Function:** `ga_generate_groups`

**Parameters:**
- `golfers`: List of dictionaries, where each dictionary contains `golfer_name` and `odds` of each golfer.
- `n_groups`: Integer, the number of groups to divide the golfers into.
- `pop_size`: Integer (default=100), the size of the population. This is the number of possible solutions (groups of golfers) the algorithm starts with. A larger population size means more potential solutions to choose from and evolve, but it also requires more computational resources.
- `generations`: Integer (default=1000), the number of generations to evolve. This is how many times the algorithm will improve the population. In each generation, the algorithm selects the best solutions, combines them to create new solutions, and possibly mutates them to explore new possibilities. More generations mean more chances to find a good solution.
- `mutation_rate`: Float (default=0.1), the probability of mutation. This is the chance that a small random change will be made to a solution. Mutations help the algorithm explore new solutions that might not be found through selection and crossover alone. A higher mutation rate means more randomness and diversity in the solutions, while a lower rate means more stability and refinement of existing solutions.

**Description:**
Uses a genetic algorithm to generate balanced groups of golfers. It initializes a population of random solutions, evaluates their fitness, and iteratively applies crossover and mutation operations to evolve better solutions.

### 5. Greedy Algorithm with Redistribution

**Function:** `greedy_redistribute_groups`

**Parameters:**
- `golfers`: List of dictionaries, where each dictionary contains `golfer_name` and `odds` of each golfer.
- `n_groups`: Integer, the number of groups to divide the golfers into.

**Description:**
Uses a greedy algorithm to initially distribute golfers into groups and then redistributes them to balance the total odds. It sorts golfers by their odds and assigns them to groups in a round-robin fashion, followed by a redistribution step to minimize the difference in total odds between groups.

import sys

def tsp(bitmask, pos, n, dist, dp):
    if bitmask == (1 << n) - 1:
        return dist[pos][0], [pos]  # Return distance and route to the starting city

    if dp[bitmask][pos] != -1:
        return dp[bitmask][pos]

    min_distance = sys.maxsize
    optimal_route = []

    for i in range(n):
        if bitmask & (1 << i) == 0:  # Check if city i has not been visited
            new_bitmask = bitmask | (1 << i)
            new_distance, new_route = tsp(new_bitmask, i, n, dist, dp)
            new_distance += dist[pos][i]

            if new_distance < min_distance:
                min_distance = new_distance
                optimal_route = [pos] + new_route

    dp[bitmask][pos] = min_distance, optimal_route
    return min_distance, optimal_route

def bitmasking_tsp(distance_matrix):
    n = len(distance_matrix)
    # Initialize dp array with -1
    dp = [[-1] * n for _ in range(1 << n)]

    # Start from city 0 with bitmask 1 (indicating city 0 is visited)
    min_cost, optimal_path = tsp(1, 0, n, distance_matrix, dp)
    return optimal_path, min_cost

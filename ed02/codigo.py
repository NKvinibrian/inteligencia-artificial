import heapq
from collections import deque
import time
import tracemalloc


goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]


def manhattan_distance(state):
    distance = 0
    for i, tile in enumerate(state):
        if tile == 0:
            continue
        goal_pos = goal_state.index(tile)
        distance += abs(i % 3 - goal_pos % 3) + abs(i // 3 - goal_pos // 3)
    return distance


def get_neighbors(state):
    neighbors = []
    zero_index = state.index(0)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  

    for dx, dy in moves:
        x, y = zero_index % 3, zero_index // 3
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < 3 and 0 <= new_y < 3:
            new_index = new_y * 3 + new_x
            new_state = state[:]
            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
            neighbors.append(new_state)
    return neighbors


def measure_performance(func, initial_state):
    tracemalloc.start()
    start_time = time.time()
    result = func(initial_state)
    end_time = time.time()
    memory_used = tracemalloc.get_traced_memory()[1] / (1024 ** 2)  
    tracemalloc.stop()
    return result, end_time - start_time, memory_used


def bfs(initial_state):
    visited = set()
    queue = deque([(initial_state, [])])

    while queue:
        current_state, path = queue.popleft()
        if current_state == goal_state:
            return path + [current_state]
        visited.add(tuple(current_state))

        for neighbor in get_neighbors(current_state):
            if tuple(neighbor) not in visited:
                queue.append((neighbor, path + [current_state]))


def dfs(initial_state, depth=1000):
    visited = set()
    stack = [(initial_state, [])]

    while stack:
        current_state, path = stack.pop()
        if current_state == goal_state:
            return path + [current_state]
        if len(path) >= depth:
            continue
        visited.add(tuple(current_state))

        for neighbor in get_neighbors(current_state):
            if tuple(neighbor) not in visited:
                stack.append((neighbor, path + [current_state]))


def greedy(initial_state):
    visited = set()
    heap = [(manhattan_distance(initial_state), initial_state, [])]

    while heap:
        _, current_state, path = heapq.heappop(heap)
        if current_state == goal_state:
            return path + [current_state]
        visited.add(tuple(current_state))

        for neighbor in get_neighbors(current_state):
            if tuple(neighbor) not in visited:
                heapq.heappush(heap, (manhattan_distance(neighbor), neighbor, path + [current_state]))


def a_star(initial_state):
    visited = set()
    heap = [(manhattan_distance(initial_state), 0, initial_state, [])]

    while heap:
        f, g, current_state, path = heapq.heappop(heap)
        if current_state == goal_state:
            return path + [current_state]
        visited.add(tuple(current_state))

        for neighbor in get_neighbors(current_state):
            if tuple(neighbor) not in visited:
                new_g = g + 1
                new_f = new_g + manhattan_distance(neighbor)
                heapq.heappush(heap, (new_f, new_g, neighbor, path + [current_state]))


puzzle_states = [
    [1,2,0,4,5,3,7,8,6],
    [1,3,0,4,2,6,7,5,8],
    [1,3,5,4,0,2,7,8,6],
    [1,3,6,4,5,2,0,7,8],
    [0,2,3,1,8,5,4,7,6],
    [1,2,3,4,0,8,7,6,5],
    [0,2,3,1,4,8,7,6,5],
    [1,0,3,4,2,5,7,8,6],
    [7,3,6,2,1,0,5,4,8],
    [7,1,3,0,5,6,4,2,8]
]


for index, initial in enumerate(puzzle_states):
    print(f"\nEstado inicial {index+1}: {initial}")

    for algorithm in [bfs, dfs, greedy, a_star]:
        solution, exec_time, memory = measure_performance(algorithm, initial)
        print(f"{algorithm.__name__.upper()} - Tempo: {exec_time:.4f}s, Memória: {memory:.4f}MB, Passos da solução: {len(solution)-1}")

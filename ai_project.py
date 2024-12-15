import numpy as np
import heapq

class Board:
    def __init__(self, board, cost, heuristic, x, y, parent, current_tile):
        self.board = board
        self.cost = cost
        self.heuristic = heuristic
        self.x = x
        self.y = y
        self.parent = parent
        self.current_tile = current_tile

    @staticmethod
    def calculate_manhattan(current, goal):
        distance = 0
        for i in range(3):
            for j in range(3):
                if current[i][j] != 0:
                    for x in range(3):
                        for y in range(3):
                            if current[i][j] == goal[x][y]:
                                distance += abs(i - x) + abs(j - y)
        return distance

    def __eq__(self, other):
        return np.array_equal(self.board, other.board)

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def __hash__(self):
        return hash(self.board.tobytes())

def perform_a_star_search(initial_state, goal_state, initial_x, initial_y):
    open_set = []
    closed_set = set()

    start = Board(initial_state, 0, Board.calculate_manhattan(initial_state, goal_state), initial_x, initial_y, None, 1)
    heapq.heappush(open_set, start)

    nodes_expanded = 0
    while open_set and nodes_expanded < 10:
        current = heapq.heappop(open_set)

        nodes_expanded += 1
        print(f"Expanded State (Node {nodes_expanded}):")
        print_board(current.board)

        if np.array_equal(current.board, goal_state):
            print(f"Goal reached at node {nodes_expanded}!")
            display_solution_path(current)
            print(f"Total cost: {current.cost}")
            return

        closed_set.add(current)

        neighbors = find_adjacent_tiles(current, goal_state)
        for neighbor in neighbors:
            if neighbor not in closed_set:
                heapq.heappush(open_set, neighbor)

    print(f"Search terminated after {nodes_expanded} expansions.")
    if nodes_expanded == 10:
        print("Expanded up to the 10th node without finding the goal state.")

def find_adjacent_tiles(current, goal_state):
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    costs = [1, 1, 2, 2]  # Costs for up, down, left, right

    tile_x, tile_y = np.where(current.board == current.current_tile)
    if len(tile_x) == 0 or len(tile_y) == 0:
        return neighbors

    tile_x, tile_y = tile_x[0], tile_y[0]

    for i, (dx, dy) in enumerate(directions):
        new_x, new_y = tile_x + dx, tile_y + dy

        if 0 <= new_x < 3 and 0 <= new_y < 3 and current.board[new_x][new_y] == 0:
            new_board = current.board.copy()
            new_board[new_x][new_y], new_board[tile_x][tile_y] = new_board[tile_x][tile_y], new_board[new_x][new_y]

            new_cost = current.cost + costs[i]
            new_heuristic = Board.calculate_manhattan(new_board, goal_state)
            next_tile = (current.current_tile % 3) + 1  # Cycle tiles: 1 -> 2 -> 3

            neighbor = Board(new_board, new_cost, new_heuristic, new_x, new_y, current, next_tile)
            neighbors.append(neighbor)

    return neighbors

def print_board(board):
    for row in board:
        print(row)
    print()

def display_solution_path(state):
    if state is None:
        return
    display_solution_path(state.parent)
    print_board(state.board)


initial_state = np.zeros((3, 3), dtype=int)
goal_state = np.zeros((3, 3), dtype=int)

print("Enter the initial state (3x3 matrix, row by row):")
initial_x, initial_y = 0, 0
for i in range(3):
    row = list(map(int, input().split()))
    initial_state[i] = row
    if 0 in row:
        initial_x, initial_y = i, row.index(0)

print("Enter the goal state (3x3 matrix, row by row):")
for i in range(3):
    goal_state[i] = list(map(int, input().split()))

perform_a_star_search(initial_state, goal_state, initial_x, initial_y)
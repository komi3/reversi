import numpy as np
import multiprocessing
import os
import random

# ===================================================================
# 1. CONSTANTS
# ===================================================================
BOARD_SIZE = 8
BLACK = -1
WHITE = 1
EMPTY = 0
DIRECTIONS = [(-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, 1), (1, 0), (0, -1)]


# ===================================================================
# 2. HEADLESS GAME LOGIC (NO PYGAME)
# ===================================================================
class HeadlessBoard:
    """A Reversi board class with NO graphics. For data generation only."""

    def __init__(self):
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.grid[3, 3] = WHITE
        self.grid[3, 4] = BLACK
        self.grid[4, 3] = BLACK
        self.grid[4, 4] = WHITE

    def clone_board(self):
        clone = HeadlessBoard()
        clone.grid = self.grid.copy()
        return clone

    def check_if_on_board(self, row, col):
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def check_for_valid_show(self, current_turn):
        valid_moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.grid[row, col] != EMPTY:
                    continue
                for dr, dc in DIRECTIONS:
                    r, c = row + dr, col + dc
                    to_flip = []
                    while self.check_if_on_board(r, c) and self.grid[r, c] == -current_turn:
                        to_flip.append((r, c))
                        r, c = r + dr, c + dc
                    if self.check_if_on_board(r, c) and self.grid[r, c] == current_turn and to_flip:
                        valid_moves.append((row, col))
                        break
        return valid_moves

    def flip(self, col, row, current_turn):
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            to_flip = []
            while self.check_if_on_board(r, c) and self.grid[r, c] == -current_turn:
                to_flip.append((r, c))
                r, c = r + dr, c + dc
            if self.check_if_on_board(r, c) and self.grid[r, c] == current_turn:
                for fr, fc in to_flip:
                    self.grid[fr, fc] = current_turn

    def end_of_match(self):
        if np.count_nonzero(self.grid == EMPTY) == 0:
            return True
        if not self.check_for_valid_show(WHITE) and not self.check_for_valid_show(BLACK):
            return True
        return False

    def get_winner(self):
        black_points = np.sum(self.grid == BLACK)
        white_points = np.sum(self.grid == WHITE)
        if black_points > white_points: return "black"
        if white_points > black_points: return "white"
        return "draw"


# ===================================================================
# 3. DATA GENERATION FUNCTIONS
# ===================================================================
def minimax_data_gathering(games_to_generate):
    """Plays a set number of games and returns the collected data."""
    all_board_states, all_game_outcomes, all_turns = [], [], []

    for game_num in range(games_to_generate):
        board = HeadlessBoard()
        current_turn = BLACK
        current_game_states, current_turn_list = [], []

        while True:
            valid_moves = board.check_for_valid_show(current_turn)

            current_game_states.append(board.grid.copy())
            current_turn_list.append(current_turn)

            if not valid_moves:
                current_turn = -current_turn
                if not board.check_for_valid_show(current_turn):
                    break  # Both players have no moves, game over
                continue

            # Simplified AI move choice for data generation
            move_to_make = random.choice(valid_moves)
            row, col = move_to_make
            board.grid[row, col] = current_turn
            board.flip(col, row, current_turn)
            current_turn = -current_turn

            if board.end_of_match():
                break

        winner = board.get_winner()
        print(f"Process {os.getpid()} finished game {game_num + 1}/{games_to_generate}. Winner: {winner}")

        if winner == "white":
            outcome = 1.0
        elif winner == "black":
            outcome = -1.0
        else:
            outcome = 0.0

        for i in range(len(current_game_states)):
            all_board_states.append(current_game_states[i])
            all_turns.append(current_turn_list[i])
            all_game_outcomes.append(outcome)

    return all_board_states, all_game_outcomes, all_turns


def worker_process(games_to_generate, result_queue):
    """The function that each process will run."""
    states, outcomes, turns = minimax_data_gathering(games_to_generate)
    result_queue.put({'states': states, 'outcomes': outcomes, 'turns': turns})


def run_data_generation(dataset_path, total_games, num_processes):
    """Manages the creation and execution of data generation processes."""
    if total_games < num_processes:
        num_processes = total_games

    processes = []
    result_queue = multiprocessing.Queue()
    games_per_process = total_games // num_processes

    print(f"Starting {num_processes} processes, each generating ~{games_per_process} games...")

    for _ in range(num_processes):
        process = multiprocessing.Process(target=worker_process, args=(games_per_process, result_queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("All processes have finished collecting data.")

    new_states, new_outcomes, new_turns = [], [], []
    while not result_queue.empty():
        result = result_queue.get()
        new_states.extend(result['states'])
        new_outcomes.extend(result['outcomes'])
        new_turns.extend(result['turns'])

    if os.path.exists(dataset_path) and os.path.getsize(dataset_path) > 0:
        print("Existing dataset found. Appending new data...")
        with np.load(dataset_path) as old_data:
            final_states = np.concatenate([old_data['boards'], np.array(new_states)])
            final_outcomes = np.concatenate([old_data['outcomes'], np.array(new_outcomes)])
            final_turns = np.concatenate([old_data['turns'], np.array(new_turns)])
    else:
        print("No existing dataset found. Creating a new one...")
        final_states = np.array(new_states)
        final_outcomes = np.array(new_outcomes)
        final_turns = np.array(new_turns)

    np.savez_compressed(dataset_path, boards=final_states, outcomes=final_outcomes, turns=final_turns)
    print(f"Dataset with {len(final_states)} states saved successfully to {dataset_path}")


# ===================================================================
# 4. MAIN EXECUTION GUARD
# ===================================================================
if __name__ == '__main__':
    dataset_path = "reversi_dataset_minimax.npz"
    games_to_generate = 100
    num_processes = 4  # Use a number <= your number of CPU cores

    run_data_generation(dataset_path, games_to_generate, num_processes)
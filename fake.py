import pygame
import sys
import numpy as np
import random
import pickle
import threading
import os
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from multiprocessing import Queue

# --- Constants and Configuration ---
HEADER = 30
SCREEN_SIZE = 800
BOARD_SIZE = 8
SQUARE_SIZE = SCREEN_SIZE // BOARD_SIZE
WHITE, BLACK, EMPTY = 1, -1, 0
WINDOW_SIZE = SCREEN_SIZE + HEADER
DIRECTIONS_TO_CHECK = [(-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, 1), (1, 0), (0, -1)]

# --- AI and Game Data Configuration ---
DATASET_PATH = "C:/Users/micha/reversi_cursor/reversi_dataset_minimax.npz"
SAVE_PATH = "C:/Users/micha/reversi_cursor/reversi_model_nn.pth"

# --- Board Evaluation Weights ---
EARLY_GAME_BOARD = np.array([
    [120, -20, 20, 5, 5, 20, -20, 120],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [120, -20, 20, 5, 5, 20, -20, 120]
])
MID_GAME_BOARD = np.array([
    [100, -10, 8, 6, 6, 8, -10, 100],
    [-10, -20, 1, 1, 1, 1, -20, -10],
    [8, 1, 5, 4, 4, 5, 1, 8],
    [6, 1, 4, 2, 2, 4, 1, 6],
    [6, 1, 4, 2, 2, 4, 1, 6],
    [8, 1, 5, 4, 4, 5, 1, 8],
    [-10, -20, 1, 1, 1, 1, -20, -10],
    [100, -10, 8, 6, 6, 8, -10, 100]
])
LATE_GAME_BOARD = np.array([
    [70, 20, 20, 20, 20, 20, 20, 70],
    [20, 10, 10, 10, 10, 10, 10, 20],
    [20, 10, 5, 5, 5, 5, 10, 20],
    [20, 10, 5, 2, 2, 5, 10, 20],
    [20, 10, 5, 2, 2, 5, 10, 20],
    [20, 10, 5, 5, 5, 5, 10, 20],
    [20, 10, 10, 10, 10, 10, 10, 20],
    [70, 20, 20, 20, 20, 20, 20, 70]
])


# --- Helper Functions ---
def get_borders(board_size):
    borders = []
    for col in range(board_size):
        borders.append([0, col])
        borders.append([board_size - 1, col])
    for row in range(1, board_size - 1):
        borders.append([row, 0])
        borders.append([row, board_size - 1])
    return borders


def update_weighted_board(board):
    occupied_squares = np.count_nonzero(board.grid)
    if occupied_squares <= 20:
        return EARLY_GAME_BOARD
    elif occupied_squares <= 40:
        return MID_GAME_BOARD
    else:
        return LATE_GAME_BOARD


def update_depth(board):
    occupied_squares = np.count_nonzero(board.grid)
    if occupied_squares <= 20:
        return 4
    elif occupied_squares <= 40:
        return 3
    else:
        return 5


# --- Board Class (Logic Only) ---
class Board:
    def __init__(self):
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.grid[3, 3] = WHITE
        self.grid[3, 4] = BLACK
        self.grid[4, 3] = BLACK
        self.grid[4, 4] = WHITE
        self.WEIGHTED_BOARD = EARLY_GAME_BOARD  # Default initialization

    def check_if_on_board(self, row, col):
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def end_of_match(self):
        white_points = np.sum(self.grid == WHITE)
        black_points = np.sum(self.grid == BLACK)
        end_of_the_match = (white_points + black_points == BOARD_SIZE * BOARD_SIZE)

        winner = "draw"
        if white_points > black_points:
            winner = "white"
        elif black_points > white_points:
            winner = "black"

        return end_of_the_match, white_points, black_points, winner

    def check_for_valid_show(self, current_turn):
        valid_moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.grid[row, col] != EMPTY:
                    continue
                for dx, dy in DIRECTIONS_TO_CHECK:
                    px, py = row + dx, col + dy
                    to_flip = []
                    while self.check_if_on_board(px, py) and self.grid[px, py] == -current_turn:
                        to_flip.append([px, py])
                        px += dx
                        py += dy
                    if self.check_if_on_board(px, py) and self.grid[px, py] == current_turn and to_flip:
                        valid_moves.append([row, col])
                        break
        return valid_moves

    def flip(self, col, row, current_turn):
        self.grid[row, col] = current_turn
        for x, y in DIRECTIONS_TO_CHECK:
            px, py = row + y, col + x
            to_flip = []
            while self.check_if_on_board(px, py) and self.grid[px, py] == -current_turn:
                to_flip.append([px, py])
                px += y
                py += x
            if self.check_if_on_board(px, py) and self.grid[px, py] == current_turn:
                for flip_row, flip_col in to_flip:
                    self.grid[flip_row, flip_col] = current_turn

    def evaluate_player(self, current_turn):
        self.WEIGHTED_BOARD = update_weighted_board(self)
        white_points = np.sum(np.where(self.grid == WHITE, self.WEIGHTED_BOARD, 0))
        black_points = np.sum(np.where(self.grid == BLACK, self.WEIGHTED_BOARD, 0))

        if current_turn == WHITE:
            return white_points - black_points
        else:
            return black_points - white_points

    def minimax(self, depth, current_turn, no_valid_move_counter, alfa, beta, borders):
        is_full, _, _, _ = self.end_of_match()
        if is_full or depth == 0:
            return self.evaluate_player(current_turn)

        valid_moves = self.check_for_valid_show(current_turn)
        if not valid_moves:
            if no_valid_move_counter == 1:
                return self.evaluate_player(current_turn)
            return self.minimax(depth - 1, -current_turn, 1, alfa, beta, borders)

        if current_turn == WHITE:  # Maximizing Player
            max_eval = float('-inf')
            for move in valid_moves:
                row, col = move
                cloned_board = self.__class__()
                cloned_board.grid = np.copy(self.grid)
                cloned_board.flip(col, row, current_turn)
                eval_score = cloned_board.minimax(depth - 1, -current_turn, 0, alfa, beta, borders)
                max_eval = max(max_eval, eval_score)
                alfa = max(alfa, eval_score)
                if beta <= alfa:
                    break
            return max_eval
        else:  # Minimizing Player (BLACK)
            min_eval = float('inf')
            for move in valid_moves:
                row, col = move
                cloned_board = self.__class__()
                cloned_board.grid = np.copy(self.grid)
                cloned_board.flip(col, row, current_turn)
                eval_score = cloned_board.minimax(depth - 1, -current_turn, 0, alfa, beta, borders)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alfa:
                    break
            return min_eval


# --- Data Generation Logic ---
def minimax_data_gathering(games_to_generate):
    all_board_states = []
    all_game_outcomes = []
    all_turns = []
    games_played = 0
    borders = get_borders(BOARD_SIZE)

    while games_played < games_to_generate:
        board = Board()
        current_turn = BLACK
        no_valid_move_counter = 0
        current_game_states = []
        current_turn_list = []

        game_over = False
        while not game_over:
            valid_moves = board.check_for_valid_show(current_turn)

            if not valid_moves:
                no_valid_move_counter += 1
                if no_valid_move_counter >= 2:
                    game_over = True
                else:
                    current_turn = -current_turn
                    continue
            else:
                no_valid_move_counter = 0
                current_game_states.append(board.grid.copy())
                current_turn_list.append(current_turn)

                best_move = None
                best_score = float('-inf') if current_turn == WHITE else float('inf')
                depth = update_depth(board)

                for move in valid_moves:
                    row, col = move
                    cloned_board = board.__class__()
                    cloned_board.grid = np.copy(board.grid)
                    cloned_board.flip(col, row, current_turn)
                    score = cloned_board.minimax(depth, -current_turn, 0, float('-inf'), float('inf'), borders)

                    if current_turn == WHITE:
                        if score > best_score:
                            best_score = score
                            best_move = move
                    else:  # BLACK
                        if score < best_score:
                            best_score = score
                            best_move = move

                if best_move:
                    row, col = best_move
                    board.flip(col, row, current_turn)
                    current_turn = -current_turn
                else:
                    game_over = True

            is_full, _, _, _ = board.end_of_match()
            if is_full:
                game_over = True

        # --- Game Finished: Process and store the data ---
        _, _, _, winner = board.end_of_match()
        if winner == "white":
            outcome = 1.0
        elif winner == "black":
            outcome = -1.0
        else:  # Draw
            outcome = 0.0

        all_board_states.extend(current_game_states)
        all_turns.extend(current_turn_list)
        all_game_outcomes.extend([outcome] * len(current_game_states))

        games_played += 1
        print(f"Worker {os.getpid()}: Game {games_played}/{games_to_generate} finished. Winner: {winner}")

    # --- Worker finished all its games: Return the collected data ---
    states_array = np.array(all_board_states)
    outcomes_array = np.array(all_game_outcomes)
    turn_array = np.array(all_turns)

    return states_array, outcomes_array, turn_array


def worker_process(games_to_generate, result_queue):
    states, outcomes, turns = minimax_data_gathering(games_to_generate)
    result_queue.put({'states': states, 'outcomes': outcomes, 'turns': turns})


def run_data_generation(dataset_path, total_games, num_processes):
    if total_games < num_processes:
        num_processes = total_games

    games_per_process = total_games // num_processes
    print(f"Starting {num_processes} processes, each generating {games_per_process} games...")

    processes = []
    result_queue = multiprocessing.Queue()

    for i in range(num_processes):
        process = multiprocessing.Process(target=worker_process, args=(games_per_process, result_queue))
        processes.append(process)
        process.start()

    print("--- [Main Process] Now waiting for all workers to finish...")
    for process in processes:
        process.join()
    print("--- [Main Process] All workers have finished!")

    # --- Collect all data from the queue ---
    all_new_states = []
    all_new_outcomes = []
    all_new_turns = []
    while not result_queue.empty():
        result = result_queue.get()
        all_new_states.extend(result['states'])
        all_new_outcomes.extend(result['outcomes'])
        all_new_turns.extend(result['turns'])

    print(f"--- [Main Process] Collected a total of {len(all_new_states)} new game states.")

    # --- Load old data and combine with new data ---
    if not all_new_states:
        print("No new data was generated. Nothing to save.")
        return

    final_states = np.array(all_new_states)
    final_outcomes = np.array(all_new_outcomes)
    final_turns = np.array(all_new_turns)

    if os.path.exists(dataset_path):
        try:
            print("Existing dataset found. Appending new data...")
            with np.load(dataset_path) as old_data:
                old_states = old_data['boards']
                old_outcomes = old_data['outcomes']
                old_turns = old_data['turns']
                final_states = np.concatenate([old_states, final_states])
                final_outcomes = np.concatenate([old_outcomes, final_outcomes])
                final_turns = np.concatenate([old_turns, final_turns])
        except Exception as e:
            print(f"Error loading old dataset: {e}. Overwriting with new data.")

    # --- Save the final combined data ---
    try:
        np.savez_compressed(dataset_path, boards=final_states, outcomes=final_outcomes, turns=final_turns)
        print(f"\n[SUCCESS] Dataset with {len(final_states)} states saved to {dataset_path}")
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not save data to file: {e}")


# --- Main Execution Block ---
if __name__ == '__main__':
    total_games_to_generate = 40
    num_processes_to_use = 4  # Should be <= number of CPU cores for best performance

    run_data_generation(DATASET_PATH, total_games_to_generate, num_processes_to_use)

    print("\nData generation script has finished.")
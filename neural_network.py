from math import inf
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

header = 30
SCREEN_SIZE = 800
BOARD_SIZE = 8
SQUARE_SIZE = SCREEN_SIZE // BOARD_SIZE
WHITE, BLACK, EMPTY = 1, -1, 0
window_size = SCREEN_SIZE + header
filename = "C:/Users/micha/Documents/reversi_q_table.pkl"
directions_to_check = [(-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, 1), (1, 0), (0, -1)]

# pygame.init()
# screen = pygame.display.set_mode((SCREEN_SIZE, window_size))
# pygame.display.set_caption('Reversi')
# font = pygame.font.Font(None, 48)

# jak jsou hodnoceny pozice
WEIGHTED_BOARD = np.array([
    [100, -10, 10, 5, 5, 10, -10, 100],
    [-10, -20, 1, 1, 1, 1, -20, -10],
    [10, 1, 5, 5, 5, 5, 1, 10],
    [5, 1, 5, 0, 0, 5, 1, 5],
    [5, 1, 5, 0, 0, 5, 1, 5],
    [10, 1, 5, 5, 5, 5, 1, 10],
    [-10, -20, 1, 1, 1, 1, -20, -10],
    [100, -10, 10, 5, 5, 10, -10, 100]
])
# na začátku hry (0..20 [tahů])
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
# uprostřed hry(20..40 [tahů])
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

# na konci hry(40...  [tahů])
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

white_points = 0
black_points = 0
game_mode = "playing"
current_turn = - 1

# inicializace základních proměnných pro minimax algoritmus
maximizing_player = True
depth = 1

alfa = float('-inf')
beta = float('inf')

# Jak moc meni nove informace stare informace
alpha = 0.5
# Jak moc dulezite jsou dlouhodobe odmeny
gamma = 0.95
# Jak moc bude nas agent delat nahodne tahy
epsilon = 0.01
# Jak moc se epsilon zmensuje casem
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 50000

save_path = "C:/Users/micha/reversi_cursor/reversi_model_nn.pth"
# all ready trained on 1000 games
save_path_depth_1 = "C:/Users/micha/reversi_cursor/reversi_model_nn_depth_2.pth"
save_path_depth_2_low = "C:/Users/micha/reversi_cursor/reversi_model_nn_depth_1.pth"
save_path_depth_2 = "C:/Users/micha/reversi_cursor/reversi_model_nn_depth_2_normal.pth"
save_path_depth_3 = "C:/Users/micha/reversi_cursor/reversi_model_nn_depth_3_new.pth"

save_path_quality_1 = "C:/Users/micha/reversi_cursor/reversi_model_nn_quality_1.pth"
save_path_quality_2 = "C:/Users/micha/reversi_cursor/reversi_model_nn_quality_2_20.pth"
save_path_quality_3 = "C:/Users/micha/reversi_cursor/reversi_model_nn_quality_3.pth"

save_path_quality_1_1 = "C:/Users/micha/reversi_cursor/reversi_model_nn_quality_1_1.pth"
save_path_quality_2_1 = "C:/Users/micha/reversi_cursor/reversi_model_nn_quality_2_1.pth"
save_path_quality_3_1 = "C:/Users/micha/reversi_cursor/reversi_model_nn_quality_3_1.pth"

dataset_path_test = "C:/Users/micha/reversi_cursor/reversi_dataset_minimax_test.npz"
dataset_path_depth_1 = "C:/Users/micha/reversi_cursor/reversi_dataset_minimax_depth_1.npz"
dataset_path_depth_2 = "C:/Users/micha/reversi_cursor/reversi_dataset_minimax_depth_2.npz"
dataset_path_depth_3 = "C:/Users/micha/reversi_cursor/reversi_dataset_minimax_depth_3.npz"


def create_a_board(board_state):
    board_instance = Board()
    board_instance.grid = np.array(board_state, dtype=int, copy=True)
    return board_instance


def get_borders(BOARD_SIZE):
    borders = []
    for col in range(0, BOARD_SIZE):
        borders.append([0, col])
        borders.append([BOARD_SIZE - 1, col])
    for row in range(1, BOARD_SIZE - 1):
        borders.append([row, 0])
        borders.append([row, BOARD_SIZE - 1])

    return borders


def update_weighted_board(board):
    WEIGHTED_BOARD = np.array([
        [100, -10, 10, 5, 5, 10, -10, 100],
        [-10, -20, 1, 1, 1, 1, -20, -10],
        [10, 1, 5, 5, 5, 5, 1, 10],
        [5, 1, 5, 0, 0, 5, 1, 5],
        [5, 1, 5, 0, 0, 5, 1, 5],
        [10, 1, 5, 5, 5, 5, 1, 10],
        [-10, -20, 1, 1, 1, 1, -20, -10],
        [100, -10, 10, 5, 5, 10, -10, 100]
    ])
    # na začátku hry (0..20 [tahů])
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
    # uprostřed hry(20..40 [tahů])
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

    # na konci hry(40...  [tahů])
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


def convert_to_tensor(board_state, current_player):
    board = create_a_board(board_state)

    # makes a board that is composed of 0 and 1  the ones are the possible moves a player can make``
    possible_moves = board.valid_moves_board(current_player, directions_to_check).astype(np.float32)
    # findes and converts the player in to True False and then in to numbers---flaout 32 because we dont need that big of a number
    my_pieces_plane = (board.grid == current_player).astype(np.float32)
    opponent_pieces_plane = (board.grid == -current_player).astype(np.float32)
    # layer the arrays on to each other to make one
    stacked_planes = np.stack([my_pieces_plane, opponent_pieces_plane, possible_moves], axis=0)
    # flatten the array to 1 dimention so it can be prossesed by the neural network
    flat = stacked_planes.reshape(-1)
    # make a array for the turn so the neural network always knows whos turn it is
    turn = np.array([current_player], dtype=np.float32)
    # conect it with the other flatend array
    input_batch = np.concatenate([flat, turn], axis=0)
    # converts the np array into a tenser for better work by pytorch and the neural network
    board_tensor = torch.from_numpy(input_batch)
    # makes a tenser that can be used in the neural network when for example we are inputing multiple boardes for training
    return board_tensor.unsqueeze(0)


def convert_to_tensor_board(board, current_player):
    # makes a board that is composed of 0 and 1  the ones are the possible moves a player can make``
    possible_moves = board.valid_moves_board(current_player, directions_to_check).astype(np.float32)
    # findes and converts the player in to True False and then in to numbers---flaout 32 because we dont need that big of a number
    my_pieces_plane = (board.grid == current_player).astype(np.float32)
    opponent_pieces_plane = (board.grid == -current_player).astype(np.float32)
    # layer the arrays on to each other to make one
    stacked_planes = np.stack([my_pieces_plane, opponent_pieces_plane, possible_moves], axis=0)
    # flatten the array to 1 dimention so it can be prossesed by the neural network
    flat = stacked_planes.reshape(-1)
    # make a array for the turn so the neural network always knows whos turn it is
    turn = np.array([current_player], dtype=np.float32)
    # conect it with the other flatend array
    input_batch = np.concatenate([flat, turn], axis=0)
    # converts the np array into a tenser for better work by pytorch and the neural network
    board_tensor = torch.from_numpy(input_batch)
    # makes a tenser that can be used in the neural network when for example we are inputing multiple boardes for training
    return board_tensor.unsqueeze(0)


def minimax_data_gathering(games_to_generate, depth):
    # Initialize all necessary variables
    board = Board()
    current_turn = random.choice([1, -1])
    no_valid_move_counter = 0
    current_game_states = []
    current_turn_list = []
    all_board_states = []
    all_game_outcomes = []
    all_turns = []
    games_played = 0
    borders = get_borders(BOARD_SIZE)
    white_wins_in_training = 0
    black_wins_in_training = 0

    while games_played < games_to_generate:
        valid_moves = board.check_for_valid_show(current_turn, directions_to_check)
        current_game_states.append(board.grid.copy())
        current_turn_list.append(current_turn)

        end_of_the_match = False

        if not valid_moves:
            no_valid_move_counter += 1
            if no_valid_move_counter >= 2:
                print(f"Worker {os.getpid()}: Game {games_played + 1} ended in a stalemate.")
                end_of_the_match = True
            else:
                current_turn = -current_turn
                continue
        else:
            no_valid_move_counter = 0

        if not end_of_the_match:
            # AI move calculation
            best_move = None
            second_best_move = None
            third_best_move = None
            best_score = float('-inf') if current_turn == WHITE else float('inf')

            for move in valid_moves:
                row, col = move
                cloned_board = board.clone_board()
                cloned_board.grid[row, col] = current_turn
                cloned_board.flip(col, row, current_turn, directions_to_check)

                score = cloned_board.minimax(depth, current_turn, no_valid_move_counter, alfa, beta, borders,
                                             maximizing_player)

                if current_turn == WHITE:
                    if score > best_score:
                        third_best_move = second_best_move
                        second_best_move = best_move
                        best_move = move
                        best_score = score
                else:  # BLACK
                    if score < best_score:
                        third_best_move = second_best_move
                        second_best_move = best_move
                        best_move = move
                        best_score = score

            if best_move:
                list_of_moves = [best_move, second_best_move, third_best_move]
                list_of_moves_2 = [move for move in list_of_moves if move is not None]

                if list_of_moves_2:
                    row, col = random.choice(list_of_moves_2)
                    board.grid[row, col] = current_turn
                    board.flip(col, row, current_turn, directions_to_check)
                    current_turn = -current_turn
                    end_of_the_match, white_points, black_points, winner = board.end_of_match()

        if end_of_the_match:
            games_played += 1
            print(f"Game{games_played} winner:{winner}")

            if winner == "white":
                outcome = 1.0
            elif winner == "black":
                outcome = -1.0
            else:
                outcome = 0.0

            for i in range(len(current_game_states)):
                state = current_game_states[i]
                turn = current_turn_list[i]
                all_board_states.append(state)
                all_game_outcomes.append(outcome)
                all_turns.append(turn)

            current_game_states = []
            current_turn_list = []
            board = Board()
            current_turn = random.choice([1, -1])
            no_valid_move_counter = 0

    states_array = np.array(all_board_states)
    outcomes_array = np.array(all_game_outcomes)
    turn_array = np.array(all_turns)

    for outcome in all_game_outcomes:
        if outcome == 1.0:
            white_wins_in_training += 1
        elif outcome == -1.0:
            black_wins_in_training += 1

    print(f"Training data stats: WHITE wins: {white_wins_in_training}, BLACK wins: {black_wins_in_training}")

    return states_array, outcomes_array, turn_array


def training_on_data(model, dataset_path, num_epochs, learning_rate, save_path):
    print("Loading dataset...")
    data = np.load(dataset_path)
    board_states = data['boards']
    game_outcomes = data['outcomes']
    turns = data['turns']
    # panalizes errors quadraticli big mistakes big changes... can be worst when thre are outliers or uncurton outcomes
    # nn.MSELoss()
    # uses quadratic and linear penalization more safe and stabel
    loss_function = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # puts the model in training mode deactivets some neurons that  could cose overfitting
    model.train()
    print("starting training")
    print(f"dataset size: {len(board_states)} states")

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i in range(len(board_states)):
            board = board_states[i]
            outcome = game_outcomes[i]
            turn = turns[i]
            # odstanime hodnoty z minuleho kroku
            optimizer.zero_grad()
            # hru převedu do tenseru
            tenser = convert_to_tensor(board, turn)
            # výsledek hry převedu do tenseru
            winner = outcome * turn
            target_tensor = torch.tensor([[winner]], dtype=torch.float32)
            # model mi da jeho predikci
            prediction = model(tenser)
            # zjistim jak moc dobrej nebo spatnej to byl tah
            loss = loss_function(prediction, target_tensor)
            # vypocita upravu weights a bias
            loss.backward()
            # upravo weights a bias
            optimizer.step()
            # celkovy ukazatel zpravnosti odpovedi podle ktereho lze poznat jestli se model zpravne uci
            total_loss += loss.item()

            avg_loss = total_loss / len(board_states)

        print(f"Epoch {epoch + 1}/{num_epochs} complete | Average Loss: {avg_loss}")

    torch.save(model.state_dict(), save_path)
    print(f"--- Training Finished. Model saved to {save_path} ---")


def run_data_generation(dataset_path, games_per_process, num_processes, depth):
    processes = []
    result_queue = multiprocessing.Queue()

    print(f"Starting {num_processes} worker processes, each generating {games_per_process} games...")

    for i in range(num_processes):
        process = multiprocessing.Process(target=worker_process, args=(games_per_process, result_queue, depth))
        processes.append(process)
        process.start()

    # Collect results as they come in
    all_new_states = []
    all_new_outcomes = []
    all_new_turns = []
    completed_workers = 0

    while completed_workers < num_processes:
        # try:
        # Wait for a result with timeout
        result = result_queue.get()
        print(f"  - Received result from worker. Contains {len(result.get('states', []))} states.")

        all_new_states.extend(result['states'])
        all_new_outcomes.extend(result['outcomes'])
        all_new_turns.extend(result['turns'])
        completed_workers += 1

        print(f"  - Progress: {completed_workers}/{num_processes} workers completed")

    # Wait for all processes to finish
    print("\n--- [Main Process] Waiting for all processes to terminate...")
    for i, process in enumerate(processes):
        process.join(timeout=10)
        if process.is_alive():
            print(f"Process {i} is still alive, terminating...")
            process.terminate()
            process.join()  # 10 second timeout

    print(f"Finished collecting data. Total new game states collected: {len(all_new_states)}")

    final_states = np.array(all_new_states)
    final_outcomes = np.array(all_new_outcomes)
    final_turns = np.array(all_new_turns)

    if os.path.exists(dataset_path):
        try:
            print(f"Existing dataset found at {dataset_path}. Loading and appending...")
            with np.load(dataset_path) as old_data:
                old_states = old_data['boards']
                old_outcomes = old_data['outcomes']
                old_turns = old_data['turns']
                print(f"  - Old dataset has {len(old_states)} states.")

                final_states = np.concatenate([old_states, final_states])
                final_outcomes = np.concatenate([old_outcomes, final_outcomes])
                final_turns = np.concatenate([old_turns, final_turns])
                print("  - Concatenation complete.")
        except Exception as e:
            print(f"  - [ERROR] Could not load or append to old dataset: {e}")
            print("  - Saving new data as a new file instead.")
    else:
        print("No existing dataset found. Using new data only.")

    print(f"The final dataset to be saved has {len(final_states)} total states.")
    try:
        np.savez_compressed(dataset_path, boards=final_states, outcomes=final_outcomes, turns=final_turns)
        print(f"\n[SUCCESS] Dataset with {len(final_states)} states saved to {dataset_path}")
    except Exception as e:
        print(f"\n[FATAL ERROR] FAILED to save the file: {e}")


def worker_process(games_to_generate, result_queue, depth):
    # upravovat promenou muze jenom jedno vlakno ktere ma lock
    states, outcomes, turns = minimax_data_gathering(games_to_generate, depth)
    print("worker_process")
    result_queue.put({'states': states, 'outcomes': outcomes, 'turns': turns})


def AI_plays(model, current_turn, board, epsilon=0.1):
    valid_moves = board.check_for_valid_show(current_turn, directions_to_check)
    if not valid_moves:
        return None

    # S pravděpodobností epsilon zahraj náhodný tah
    if random.random() < epsilon:
        return random.choice(valid_moves)

    # Jinak zahraj nejlepší tah
    best_move = None
    # Hledáme tah, který maximalizuje *naše* skóre
    # Naše nejlepší skóre začíná na nejnižší možné hodnotě
    best_score = float('-inf')

    for move in valid_moves:
        row, col = move
        board_copy = board.clone_board()
        board_copy.grid[row, col] = current_turn
        board_copy.flip(col, row, current_turn, directions_to_check)

        # Nyní je na tahu soupeř (-current_turn)
        # Vytvoříme tenzor z perspektivy SOUPEŘE
        tensor = convert_to_tensor_board(board_copy, -current_turn)

        with torch.no_grad():
            # Získáme skóre, které model předpovídá PRO SOUPEŘE
            opponent_score = model(tensor).item()

        # Naše skóre je negativní hodnota skóre soupeře
        # (Pokud soupeř vyhrává (1.0), my prohráváme (-1.0))
        my_score = -opponent_score

        # Chceme maximalizovat *naše* skóre
        if my_score > best_score:
            best_score = my_score
            best_move = [row, col]

    return best_move


def AI_self_play_data_generation(model, games_to_generate):
    # Initialize all necessary variables
    move = None
    model.eval()
    board = Board()
    current_turn = random.choice([1, -1])
    no_valid_move_counter = 0
    current_game_states = []
    current_turn_list = []
    all_board_states = []
    all_game_outcomes = []
    all_turns = []
    games_played = 0
    borders = get_borders(BOARD_SIZE)

    while games_played < games_to_generate:
        valid_moves = board.check_for_valid_show(current_turn, directions_to_check)
        current_game_states.append(board.grid.copy())
        current_turn_list.append(current_turn)

        end_of_the_match = False

        if not valid_moves:
            no_valid_move_counter += 1
            if no_valid_move_counter >= 2:
                end_of_the_match = True
            else:
                current_turn = -current_turn
                continue
        else:
            no_valid_move_counter = 0

        if not end_of_the_match:
            move = AI_plays(model, current_turn, board)
            print(move)
            if move is None or move not in valid_moves:
                row, col = random.choice(valid_moves)
                board.grid[row, col] = current_turn
                board.flip(col, row, current_turn, directions_to_check)
                current_turn = -current_turn
                end_of_the_match, white_points, black_points, winner = board.end_of_match()
            else:
                row, col = move
                board.grid[row, col] = current_turn
                board.flip(col, row, current_turn, directions_to_check)
                current_turn = -current_turn
                end_of_the_match, white_points, black_points, winner = board.end_of_match()

        if end_of_the_match:
            games_played += 1
            print(f"Game{games_played} winner:{winner}")

            if winner == "white":
                outcome = 1.0
            elif winner == "black":
                outcome = -1.0
            else:
                outcome = 0.0

            for i in range(len(current_game_states)):
                state = current_game_states[i]
                turn = current_turn_list[i]
                all_board_states.append(state)
                all_game_outcomes.append(outcome)
                all_turns.append(turn)

            current_game_states = []
            current_turn_list = []
            board = Board()
            current_turn = random.choice([1, -1])
            no_valid_move_counter = 0

    states_array = np.array(all_board_states)
    outcomes_array = np.array(all_game_outcomes)
    turn_array = np.array(all_turns)

    return states_array, outcomes_array, turn_array


def training_AI(model, board_states, game_outcomes, turns):
    model.load(save_path)
    # panalizes errors quadraticli big mistakes big changes... can be worst when thre are outliers or uncurton outcomes
    # nn.MSELoss()
    # uses quadratic and linear penalization more safe and stabel
    loss_function = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # puts the model in training mode deactivets some neurons that  could cose overfitting
    model.train()
    print("starting training")
    print(f"dataset size: {len(board_states)} states")

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i in range(len(board_states)):
            board = board_states[i]
            outcome = game_outcomes[i]
            turn = turns[i]
            # odstanime hodnoty z minuleho kroku
            optimizer.zero_grad()
            # hru převedu do tenseru
            tenser = convert_to_tensor(board, turn)
            # výsledek hry převedu do tenseru
            winner = outcome * turn
            target_tensor = torch.tensor([[winner]], dtype=torch.float32)
            # model mi da jeho predikci
            prediction = model(tenser)
            # zjistim jak moc dobrej nebo spatnej to byl tah
            loss = loss_function(prediction, target_tensor)
            # vypocita upravu weights a bias
            loss.backward()
            # upravo weights a bias
            optimizer.step()
            # celkovy ukazatel zpravnosti odpovedi podle ktereho lze poznat jestli se model zpravne uci
            total_loss += loss.item()

            avg_loss = total_loss / len(board_states)

        print(f"Epoch {epoch + 1}/{num_epochs} complete | Average Loss: {avg_loss}")

    torch.save(model.state_dict(), save_path)
    print(f"--- Training Finished. Model saved to {save_path} ---")


def AI_self_trainig_full_function(model, epochs):
    for epoch in range(epochs):
        model.load(save_path)
        board_states, game_outcomes, turns = AI_self_play_data_generation(model, 50)
        training_AI(model, board_states, game_outcomes, turns)


def turnament(save_path, save_path_2, games_to_generate):
    print("turnament has started")

    model_t = Neural_agent()
    model_2_t = Neural_agent()

    model_t.load(save_path)
    model_2_t.load(save_path_2)

    white_wins = 0
    black_wins = 0
    draws = 0

    model_t.eval()
    model_2_t.eval()

    board = Board()
    current_turn = random.choice([1, -1])

    no_valid_move_counter = 0
    games_played = 0

    while games_played < games_to_generate:
        end_of_the_match = False
        valid_moves = board.check_for_valid_show(current_turn, directions_to_check)

        # Kontrola platných tahů
        if not valid_moves:
            no_valid_move_counter += 1
            if no_valid_move_counter >= 2:
                # Oba hráči nemohou táhnout - konec hry
                end_of_the_match, white_points, black_points, winner = board.end_of_match()
                games_played += 1
                if winner == "white":
                    white_wins += 1
                elif winner == "black":
                    black_wins += 1
                else:
                    draws += 1

                # Reset hry
                board = Board()
                current_turn = random.choice([1, -1])
                no_valid_move_counter = 0
                continue
            else:
                # Pouze změna hráče
                current_turn = -current_turn
                continue

        # Existují platné tahy - reset počítadla

        no_valid_move_counter = 0

        if current_turn == 1:
            move = AI_plays(model_t, current_turn, board)

            if move is None or move not in valid_moves:
                print("random move was played")
                row, col = random.choice(valid_moves)
                board.grid[row, col] = current_turn
                board.flip(col, row, current_turn, directions_to_check)
                current_turn = -current_turn
                end_of_the_match, white_points, black_points, winner = board.end_of_match()
            else:
                row, col = move
                board.grid[row, col] = current_turn
                board.flip(col, row, current_turn, directions_to_check)
                current_turn = -current_turn
                end_of_the_match, white_points, black_points, winner = board.end_of_match()

        elif current_turn == - 1:
            move = AI_plays(model_2_t, current_turn, board)

            if move is None or move not in valid_moves:
                row, col = random.choice(valid_moves)
                board.grid[row, col] = current_turn
                board.flip(col, row, current_turn, directions_to_check)
                current_turn = -current_turn
                end_of_the_match, white_points, black_points, winner = board.end_of_match()
            else:
                row, col = move
                board.grid[row, col] = current_turn
                board.flip(col, row, current_turn, directions_to_check)
                current_turn = -current_turn
                end_of_the_match, white_points, black_points, winner = board.end_of_match()

        if end_of_the_match:
            games_played += 1

            if winner == "white":
                white_wins = white_wins + 1
            elif winner == "black":
                black_wins = black_wins + 1
            else:
                draws = draws + 1

            board = Board()
            current_turn = random.choice([1, -1])
            no_valid_move_counter = 0
    return white_wins, black_wins, draws


class Board:
    def __init__(self):
        directions_to_check = [(-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, 1), (1, 0), (0, -1)]

        # self.font = pygame.font.Font(None, 48)
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

        self.grid[3, 3] = WHITE
        self.grid[3, 4] = BLACK
        self.grid[4, 3] = BLACK
        self.grid[4, 4] = WHITE

    def check_if_on_board(self, row, col):

        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return True
        else:
            return False

    def end_of_match(self):
        winner = None

        end_of_the_match = False
        white_points = 0
        black_points = 0

        black_points = np.sum(self.grid == BLACK)
        white_points = np.sum(self.grid == WHITE)

        if white_points < black_points:

            winner = "black"

        elif white_points > black_points:

            winner = "white"

        if white_points + black_points == BOARD_SIZE * BOARD_SIZE:
            end_of_the_match = True

            if white_points == black_points:
                winner = "draw"

        return end_of_the_match, white_points, black_points, winner

    def check_for_valid_show(self, current_turn, directions_to_check):
        valid_moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):

                if self.grid[row, col] != EMPTY:
                    continue

                for dx, dy in directions_to_check:
                    px, py = row + dx, col + dy
                    to_flip = []

                    while self.check_if_on_board(px, py) and self.grid[px, py] != 0 and self.grid[
                        px, py] != current_turn:
                        to_flip.append([px, py])
                        px += dx
                        py += dy
                    if self.check_if_on_board(px, py) and self.grid[px, py] == current_turn and to_flip:
                        valid_moves.append([row, col])
                        break
        return valid_moves

    def valid_moves_board(self, current_turn, directions_to_check):
        valid_moves = self.check_for_valid_show(current_turn, directions_to_check)
        board_valid_moves = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for row, col in valid_moves:
            board_valid_moves[row, col] = 1.0
        return board_valid_moves

    def flip(self, col, row, current_turn, directions_to_check):
        self.grid[row, col] = current_turn
        for x, y in directions_to_check:
            px, py = row + y, col + x
            to_flip = []

            while self.check_if_on_board(px, py) == True and self.grid[px, py] != 0 and self.grid[
                px, py] != current_turn:
                to_flip.append([px, py])
                px += y
                py += x

            if self.check_if_on_board(px, py) == True and self.grid[px, py] == current_turn:
                for px, py in to_flip:
                    self.grid[px, py] = current_turn

    def draw_valid_move(self, valid_moves):
        for row, col in valid_moves:
            pygame.draw.circle(screen, (250, 200, 152),
                               (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                               SQUARE_SIZE // 4)

    def draw_board(self):
        screen.fill((0, 128, 0))
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                pygame.draw.rect(screen, (0, 0, 0), (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 1)
                if self.grid[row, col] == WHITE:
                    pygame.draw.circle(screen, (255, 255, 255),
                                       (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                                       SQUARE_SIZE // 2 - 5)

                elif self.grid[row, col] == BLACK:
                    pygame.draw.circle(screen, (0, 0, 0),
                                       (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                                       SQUARE_SIZE // 2 - 5)

    def draw_points(self, white_points, black_points, current_turn):
        self.font = pygame.font.Font(None, 48)

        if current_turn == 1:
            current_turn = "White"
        elif current_turn == -1:
            current_turn = "Black"

        white_points, black_points = str(white_points), str(black_points)
        text_surface = self.font.render(f"WHITE:{white_points}  BLACK:{black_points}       {current_turn}", True,
                                        (255, 255, 255))
        screen.blit(text_surface, (200, 800))

    def evaluate_player(self):
        # funkce hodnotí pozici na desce
        white_points = 0
        black_points = 0

        # vyvolá funkci pomocí které upravuje Weighted board (jak jsou hodnocené jednotlivé pozice) podle toho,
        # v jaké části je hra (kolik tahů bylo odehráno)
        self.WEIGHTED_BOARD = update_weighted_board(self)

        #  použije funkci np.where která změní 0 a protihráčova (podle toho, jestli počítá bílé nebo černé body) pole
        #  na False a nepočítá je a hráčovo na True a počítá hodnotu která je na nich podle Weighted board, a ty potom
        #  sečte pomocí funkce sum a uloží do proměnné bodů které zrovna počítá

        black_points = np.sum(np.where(self.grid == BLACK, self.WEIGHTED_BOARD, 0))
        white_points = np.sum(np.where(self.grid == WHITE, self.WEIGHTED_BOARD, 0))

        # zde jsem se snažil zlepšit funkci tak, aby podle rohových bodů počítala kamínky které se nedají převrátito
        # funkce je podle mě funkční, ale nevyřešil jsem problém toho, že hodnotila rohové a zbylé kamínky moc dobře a kvůli tomu hrála špatně
        # chtěl bych ale toto zlepšení a opravit v budoucnu

        # stable_pieces = []
        # for row,col in borders:
        # if self.grid[row,col] == current_turn:
        # stable_pieces.append([row,col])
        # for dx, dy in directions_to_check:
        # nx, ny = row + dx, col + dy

        # while self.check_if_on_board(nx, ny) and self.grid[nx,ny] == current_turn:
        # around_stable_pieces = []
        # for x,y in directions_to_check:
        # px, py = nx + x, ny + y
        # if not self.check_if_on_board(px,py) or self.grid[px,py] == current_turn:
        # around_stable_pieces.append([px,py])
        # if len(around_stable_pieces) == 8:
        # stable_pieces.append([nx,ny])

        # nx += dx
        # ny += dy

        # for pieces in stable_pieces:
        # if current_turn == 1:

        # white_points += 10
        # else:
        # black_points += 10

        # přidává body podle toho, kolik je možných tahů, čím více tahů tím více bodů

        # konečné sčítání bodů
        mobility_weight = 5
        my_moves = len(self.check_for_valid_show(WHITE, directions_to_check))
        opp_moves = len(self.check_for_valid_show(BLACK, directions_to_check))
        mobility_term = mobility_weight * (my_moves - opp_moves)
        # konzistentní skóre: vždy WHITE - BLACK + mobilita(se znam.)
        return (white_points - black_points) + mobility_term

    def clone_board(self):
        # funkce která kopíruje desku a vytváří z ní samostatný objekt který neovlivnuje originál
        clone = Board()
        clone.grid = np.copy(self.grid)

        return clone

    def minimax(self, depth, current_turn, no_valid_move_counter, alfa, beta, borders, maximizing_player):
        end_of_the_match, white_points, black_points, winner = self.end_of_match()
        maximizing_player = (current_turn == WHITE)
        # nastavil jsem základní hodnotu ohodnocení na kladné a záporné nekonečno
        best = float('-inf')

        valid_moves = self.check_for_valid_show(current_turn, directions_to_check)
        # zjistím jestli algoritmus došel na konec hry nebo došel do hloubky nula
        # pokud ano, funkce vrátí konečné ohodnocení desky
        # zjistím jestli skončila hra a  spustím také funkci valid_moves (aby funkce evaluate mohla fungovat)
        if end_of_the_match or depth == 0:
            if winner == WHITE:
                white_points = 10000
                return white_points
            elif winner == BLACK:
                black_points = -10000
                return black_points
            return self.evaluate_player()

        # jestli je možné hrát a pokud ani jeden nemůže hrát ukončí hru
        # jednotná stop podmínka
        if end_of_the_match or depth == 0 or no_valid_move_counter >= 2:
            return self.evaluate_player()

        if not valid_moves:
            # Simply pass turn and increment counter
            return self.minimax(depth, -current_turn, no_valid_move_counter + 1, alfa, beta, borders,
                                not maximizing_player)

        no_valid_move_counter = 0
        # minimax algoritmus hraje za bílého neboli 1
        if maximizing_player:  # Proper maximizing logic
            best = float('-inf')
            for move in valid_moves:
                row, col = move
                # udělám kopii stavu desky
                grid_copy = np.copy(self.grid)
                # tah který algoritmus zahrál načteme na hrací desku (toto je v pořádku, protože hned potom je hrací
                # deska daná do nového objektu který neovlivnuje originální hrací desku. Nejsou spuštěny žádné funkce
                # které by ovlivňovaly originální hrací desku)
                grid_copy[row, col] = current_turn
                # vytvoříme prázdnou (bez __init__) hrací desku, která neovlivnuje originální desku
                cloned_board = Board.__new__(Board)
                # vloží stav hrací desky
                cloned_board.grid = grid_copy
                # přidá font z __init__, jelikož jsme tuto část neskopírovali

                # cloned_board.font = self.font

                # zavoláme flip funkci
                cloned_board.flip(col, row, current_turn, directions_to_check)
                # zavoláme funkci minimax kde ale změníme hloubku na hloubku -1 a změníme hráče na protihráče a naopak
                # (pokud skončí hra nebo funkce dojde na hloubku 0, řetěz funkcí se přeruší a funkce vrátí nejlepší hodnotu)
                eval = cloned_board.minimax(depth - 1, -current_turn, no_valid_move_counter, alfa, beta, borders,
                                            maximizing_player)
                # zhodnotí zdali je skore lepší než nejlepší dosavadní skore
                best = max(best, eval)
                # vyhodnotí zda-li je nový tah lepší než dosavadní nejlepší tah
                alfa = max(alfa, eval)
                # zjistíme jestli max hráč má lepší skore než minimální skore min hráče a pokud ano, tuto větev
                # můžeme odstranit, jelikož min hráč si ji nikdy nevybere, takže není důvod dále počítat možné tahy v této větvi
                if alfa >= beta:
                    break
            # vrátí nejlepší skore max hráče v této hloubce
            return best

        else:
            best = float('+inf')
            for move in valid_moves:
                # stejné jako max jenom hledáme minimum místo maxima
                row, col = move

                grid_copy = np.copy(self.grid)
                grid_copy[row, col] = current_turn

                cloned_board = Board.__new__(Board)
                cloned_board.grid = grid_copy
                # cloned_board.font = self.font
                cloned_board.flip(col, row, current_turn, directions_to_check)

                eval = cloned_board.minimax(depth - 1, -current_turn, no_valid_move_counter, alfa, beta, borders,
                                            maximizing_player)
                best = min(best, eval)
                beta = min(beta, eval)

                if alfa >= beta:
                    break

            return best


class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon):
        self.q_table = {}  # Dictionary for state-action pairs
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

    def get_state_key(self, board):
        # Convert board to tuple for dictionary key
        return tuple(map(tuple, board.grid))

    def train(self, num_episodes, filename):
        self.load_qtable(filename)
        # Initialize tracking variables
        total_wins = 0
        total_losses = 0
        total_draws = 0

        for episode in range(num_episodes):
            # Reset game for new episode
            board = Board()
            current_turn = BLACK  # Starting player
            no_valid_move_counter = 0

            while True:  # Game loop
                # Get current state and valid moves
                current_state = self.get_state_key(board)
                valid_moves = board.check_for_valid_show(current_turn, directions_to_check)

                if not valid_moves:
                    if no_valid_move_counter == 1:
                        break  # End game if both players can't move
                    current_turn = -current_turn
                    no_valid_move_counter += 1
                    continue

                # Reset counter if moves exist
                no_valid_move_counter = 0

                # Get action (either explore or exploit)
                move = self.get_action(board, valid_moves)

                # Make the move and get new state
                row, col = move
                board.grid[row, col] = current_turn
                board.flip(col, row, current_turn, directions_to_check)

                # Now get next state and valid moves AFTER the flip
                next_state = self.get_state_key(board)
                next_valid_moves = board.check_for_valid_show(-current_turn, directions_to_check)

                # Get reward for this state
                reward = board.evaluate_player(None, valid_moves, directions_to_check, current_turn, borders)

                # Create state-action pair and learn
                state_action = (current_state, tuple(move))
                self.learn(state_action, reward, next_state, next_valid_moves)

                # Switch players
                current_turn = -current_turn

                # Check if game is over
                end_of_match, white_points, black_points, winner = board.end_of_match()
                if end_of_match:
                    # Track results
                    if winner == "black":
                        total_wins += 1
                    elif winner == "white":
                        total_losses += 1
                    elif winner == "draw":
                        total_draws = +1
                    break

            # Decay epsilon after each episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if episode % 100 == 0:
                print(f"Episode {episode}: Wins: {total_wins}, Losses: {total_losses}, Draws: {total_draws}")

            # Save Q-table periodically (every 100 episodes)

        self.save_qtable(filename)

    def get_action(self, board, valid_moves):
        state = self.get_state_key(board)
        # for row in state:
        # print( "\n", row )

        # Exploration: random move
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        # Exploitation: best known move
        best_value = float('-inf')
        best_move = random.choice(valid_moves)  # Default

        for move in valid_moves:
            state_action = (state, tuple(move))
            value = self.q_table.get(state_action, 0)
            if value > best_value:
                best_value = value
                best_move = move

        return best_move

    def learn(self, state_action, reward, next_state, next_valid_moves):
        # Save the reward (evaluation score) in q_table
        self.q_table[state_action] = reward

    def save_qtable(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
            print(f"saved to the file {filename}")

    def load_qtable(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
                print(f"Successfully loaded Q-table from {filename}")
                # Print some statistics about the loaded Q-table
                print(f"Number of state-action pairs: {len(self.q_table)}")
                # Print a sample of the Q-table (first 3 entries)
                print("Sample of Q-table contents:")
                for i, (state_action, value) in enumerate(self.q_table.items()):
                    if i < 3:  # Show first 3 entries
                        print(f"State-Action: {state_action}, Value: {value}")
                    else:
                        break


        except FileNotFoundError:
            print(f"No existing Q-table found at {filename}")
        except Exception as e:
            print(f"Error loading Q-table: {str(e)}")


class Neural_agent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_size = 193  # 3*8*8 + 1

        self.body = nn.Sequential(
            # the amount of inputs, the amount of outputs (neurons)
            nn.Linear(input_size, 256),
            # ReLU function discareds neurons that recognize a pattern that usually lead to a bad game ---> bad outcome
            #                            ---> it makes the neural network faster
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)

        )

    def forward(self, input):
        # tenser is a method of storing data,weights,biasis that makes the calculation faster
        # flatten the board so it can be inputied to the neural network
        # board_tenser = input.view(-1,193), #-1 means that the function should figure out the correct number of rows in this case 1,
        # already did that in the tenser convertion function

        # send the data input in to the layer
        body_output = self.body(input)
        # send the output in to the other part of the hiden layer
        value = self.value_head(body_output)
        # converts the last output in to number betten 1 and -1 for better traking of the score and how good the move was
        # it does it thoru hyporbolic tangent
        final_output = torch.tanh(value)

        return final_output

    def load(self, save_path):
        print(f"Loading model weights from {save_path}...")
        self.load_state_dict(torch.load(save_path))
        print("completed")


if __name__ == '__main__':
    total_games_turnament = 1000
    total_games = 100
    num_processes = 5
    depth = 1
    num_epochs = 10
    learning_rate = 1e-4
    # run_data_generation(dataset_path_test, total_games,num_processes,depth)y
    # print("\nData generation script finished successfully.")

    model_1 = Neural_agent()
    model_2 = Neural_agent()
    model_3 = Neural_agent()

    # training_on_data(model_1, dataset_path_depth_1, num_epochs, learning_rate, save_path_quality_1_1)
    # print("1")
    # training_on_data(model_2, dataset_path_depth_2, num_epochs, learning_rate, save_path_quality_2_1)
    # print("2")
    # training_on_data(model_3, dataset_path_depth_3, num_epochs, learning_rate, save_path_quality_3_1)
    # print("3")

    # AI_self_trainig_full_function(model_1,num_epochs)

    white_wins, black_wins, draws = turnament(save_path_quality_1_1, save_path_quality_1_1, total_games_turnament)
    print(f"Game{total_games_turnament}  white wins:{white_wins} black wins:{black_wins} draws:{draws}")

    borders = get_borders(BOARD_SIZE)
    board = Board()
    mouse_x, mouse_y = 0, 0

    play = False
    no_valid_move_counter = 0

    model = Neural_agent()
    model.load(save_path)
    model.eval()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, window_size))
    pygame.display.set_caption('Reversi')
    font = pygame.font.Font(None, 48)

    while True:
        pygame.display.flip()
        if game_mode == "playing":
            valid_moves = board.check_for_valid_show(current_turn, directions_to_check)

            for event in pygame.event.get():
                valid_moves = board.check_for_valid_show(current_turn, directions_to_check)

                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

                if not valid_moves:
                    if current_turn == 1:
                        current_turn = -current_turn
                        no_valid_move_counter += 1


                    elif current_turn == -1:
                        current_turn = -current_turn
                        no_valid_move_counter += 1

                        no_valid_move_counter = 0

                    if no_valid_move_counter == 2:
                        end_of_the_match, white_points, black_points, winner = board.end_of_match()
                        games_played += 1
                        print(
                            f"Game{games_played} winner:{winner} white wins:{white_wins} black wins:{black_wins} draws:{draws}")

                        if winner == "white":
                            white_wins = white_wins + 1
                        elif winner == "black":
                            black_wins = black_wins + 1
                        else:
                            draws = draws + 1

                        board = Board()
                        current_turn = random.choice([1, -1])
                        no_valid_move_counter = 0

                if current_turn == 1:
                    move = AI_plays(model, current_turn, board)
                    for move_valid in valid_moves:
                        if move == move_valid:
                            row, col = move
                            board.grid[row, col] = current_turn
                            board.flip(col, row, current_turn, directions_to_check)
                            current_turn = -current_turn

                pygame.display.flip()

                if current_turn == -1:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        # kliknutí se převede na pole
                        mouse_x, mouse_y = event.pos
                        row, col = mouse_y // SQUARE_SIZE, mouse_x // SQUARE_SIZE
                        # pokud je pole, na které hráč kliknul, v seznamu možných polí, změní se toto pole na jeho a spustí se funkce na převracení protihráčových polí
                        if [row, col] in valid_moves:
                            board.grid[row, col] = current_turn
                            board.flip(col, row, current_turn, directions_to_check)
                            # po odehrání hraje protihráč
                            if current_turn == 1:
                                current_turn = -1
                            elif current_turn == -1:
                                current_turn = 1
                        else:
                            print("this move is impossible")

                pygame.display.flip()

            pygame.display.flip()
            board.draw_board()
            board.draw_valid_move(valid_moves)
            end_of_the_match, white_points, black_points, winner = board.end_of_match()

            board.draw_points(white_points, black_points, current_turn)
            if end_of_the_match:
                print(f"white:{white_points}   black:{black_points}       winner:{winner}")
                board = Board()
                current_turn = -1
                game_mode = "playing"

        pygame.display.flip()

# self play
# AI model vs AI model

import pygame
import sys
import numpy as np
import random
import pickle

header = 30
SCREEN_SIZE = 800
BOARD_SIZE = 8
SQUARE_SIZE = SCREEN_SIZE // BOARD_SIZE
WHITE, BLACK, EMPTY = 1, -1, 0
window_size = SCREEN_SIZE + header
filename = "C:/Users/micha/Documents/reversi_q_table.pkl"
directions_to_check = [(-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, 1), (1, 0), (0, -1)]

pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, window_size))
pygame.display.set_caption('Reversi')
font = pygame.font.Font(None, 48)
white_points = 0
black_points = 0
game_mode = "playing"
current_turn = - 1

# inicializace základních proměnných pro minimax algoritmus
maximizing_player = True
depth = 3

alfa = float('-inf')
beta = float('inf')

# Jak moc meni nove informace stare informace
alpha = 0.9
# Jak moc dulezite jsou dlouhodobe odmeny
gamma = 0.95
# Jak moc bude nas agent delat nahodne tahy
epsilon = 0.5
# Jak moc se epsilon zmensuje casem
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 1000

q_table = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

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


class Board:
    def __init__(self):

        self.font = pygame.font.Font(None, 48)
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

        if current_turn == 1:
            current_turn = "White"
        elif current_turn == -1:
            current_turn = "Black"

        white_points, black_points = str(white_points), str(black_points)
        text_surface = self.font.render(f"WHITE:{white_points}  BLACK:{black_points}       {current_turn}", True,
                                        (255, 255, 255))
        screen.blit(text_surface, (200, 800))

    def evaluate_player(self, winner, valid_moves, directions_to_check, current_turn, borders):
        # funkce hodnotí pozici na desce
        white_points = 0
        black_points = 0
        score = 0
        # vyvolá funkci pomocí které upravuje Weighted board (jak jsou hodnocené jednotlivé pozice) podle toho, v jaké
        # části je hra (kolik tahů bylo odehráno)
        self.WEIGHTED_BOARD = update_weighted_board(board)

        # použije funkci np.where která změní 0 a protihráčova (podle toho, jestli počítá bílé nebo černé body) pole na
        # False a nepočítá je a hráčovo na True a počítá hodnotu která je na nich podle Weighted board, a ty potom sečte
        # pomocí funkce sum a uloží do proměnné bodů které zrovna počítá

        black_points = np.sum(np.where(self.grid == BLACK, self.WEIGHTED_BOARD, 0))
        white_points = np.sum(np.where(self.grid == WHITE, self.WEIGHTED_BOARD, 0))

        # zde jsem se snažil zlepšit funkci tak, aby podle rohových bodů počítala kamínky které se nedají převrátit
        # funkce je podle mě funkční, ale nevyřešil jsem problém toho, že hodnotila rohové a zbylé kamínky moc dobře a
        # kvůli tomu hrála špatně

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

        # number_of_moves = 0
        # for move in valid_moves:
        # number_of_moves += 1

        # konečné sčítání bodů

        # if current_turn == 1:
        # white_points += (number_of_moves * 35)
        # else:
        # black_points += (number_of_moves * 35)

        if current_turn == 1:
            score = white_points - black_points
            return score

        else:
            score = black_points - white_points
            return score

    def clone_board(self):
        # funkce která kopíruje desku a vytváří z ní samostatný objekt který neovlivnuje originál
        clone = Board()
        clone.grid = np.copy(self.grid)

        return clone

    def minimax(self, depth, current_turn, no_valid_move_counter, alfa, beta, borders):
        # nastavil jsem základní hodnotu ohodnocení na kladné a záporné nekonečno
        max_eval = float('-inf')
        min_eval = float('inf')

        # zjistím jestli skončila hra a spustím také funkci valid_moves (aby funkce evaluate mohla fungovat)
        end_of_the_match, white_points, black_points, winner = self.end_of_match()
        valid_moves = self.check_for_valid_show(current_turn, directions_to_check)
        # zjistím jestli algoritmus došel na konec hry nebo došel do hloubky nula
        # pokud ano, funkce vrátí konečné ohodnocení desky
        if end_of_the_match or depth == 0:
            return self.evaluate_player(winner, valid_moves, directions_to_check, current_turn, borders)

        valid_moves = self.check_for_valid_show(current_turn, directions_to_check)
        # jestli je možné hrát a pokud ani jeden nemůže hrát ukončí hru
        if not valid_moves:
            if current_turn == 1:
                current_turn = -current_turn
                no_valid_move_counter += 1

            elif current_turn == -1:
                current_turn = -current_turn
                no_valid_move_counter += 1
            if no_valid_move_counter == 2:
                return self.evaluate_player(winner, valid_moves, directions_to_check, current_turn, borders)

            return self.minimax(depth, -current_turn, no_valid_move_counter, alfa, beta, borders)

        if valid_moves:
            no_valid_move_counter = 0
        # minimax algoritmus hraje za bílého neboli 1
        if current_turn == 1:
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
                cloned_board.font = self.font
                # zavoláme flip funkci
                cloned_board.flip(col, row, current_turn, directions_to_check)
                # zavoláme funkci minimax kde ale změníme hloubku na hloubku -1 a změníme hráče na protihráče a naopak
                # (pokud skončí hra nebo funkce dojde na hloubku 0, řetěz funkcí se přeruší a funkce vrátí nejlepší hodnotu)
                eval = cloned_board.minimax(depth - 1, -current_turn, no_valid_move_counter, alfa, beta, borders)
                # zhodnotí zdali je skore lepší než nejlepší dosavadní skore
                max_eval = max(max_eval, eval)
                # vyhodnoti zda-li je nový tah lepší než dozatimní nejlepší tah
                alfa = max(alfa, eval)
                # zjistíme jestli max hráč má lepší skore než minimální skore min hráče a pokud ano, tuto větev
                # můžeme odstranit, jelikož min hráč si ji nikdy nevybere, takže není důvod dále počítat možné tahy v této větvi
                if alfa >= beta:
                    break
            # vrátí nejlepší skore max hráče v této hloubce
            return max_eval

        else:
            for move in valid_moves:
                # stejné jako max jenom hledáme minimum místo maxima
                row, col = move

                grid_copy = np.copy(self.grid)
                grid_copy[row, col] = current_turn

                cloned_board = Board.__new__(Board)
                cloned_board.grid = grid_copy
                cloned_board.font = self.font
                cloned_board.flip(col, row, current_turn, directions_to_check)

                eval = cloned_board.minimax(depth - 1, -current_turn, no_valid_move_counter, alfa, beta, borders)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)

                if alfa >= beta:
                    break

            return min_eval


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
        agent.load_qtable(filename)
        # Initialize tracking variables
        total_wins = 0
        total_losses = 0

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
                    break

            # Decay epsilon after each episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if episode % 100 == 0:
                print(f"Episode {episode}: Wins: {total_wins}, Losses: {total_losses}")

            # Save Q-table periodically (every 100 episodes)

        self.save_qtable(filename)

    def get_action(self, board, valid_moves):
        state = self.get_state_key(board)

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


borders = get_borders(BOARD_SIZE)
board = Board()
mouse_x, mouse_y = 0, 0
play = False
no_valid_move_counter = 0
agent = QLearningAgent(alpha, gamma, epsilon)
agent.train(num_episodes, filename)
while True:
    if game_mode == "playing":
        agent_pro = QLearningAgent(alpha, gamma, epsilon)
        agent_pro.load_qtable(filename)
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

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if not valid_moves:
                if current_turn == 1:
                    current_turn = -current_turn
                    no_valid_move_counter += 1
                    print(no_valid_move_counter)
                elif current_turn == -1:
                    current_turn = -current_turn
                    no_valid_move_counter += 1
                    print(no_valid_move_counter)

            if no_valid_move_counter == 2:
                print(no_valid_move_counter)
                end_of_the_match, white_points, black_points, winner = board.end_of_match()
                print(f"white:{white_points}   black:{black_points}       winner:{winner}")
                board = Board()
                current_turn = -1
                game_mode = "playing"
                break

            if valid_moves:
                no_valid_move_counter = 0

            end_of_the_match, white_points, black_points, winner = board.end_of_match()

            while not end_of_the_match:
                print(tuple(map(tuple, board.grid)), "map")
                print(q_table, "table")

                current_state = agent.get_state_key(board)
                move = agent.get_action(board, valid_moves)

                row, col = move
                board.grid[row, col] = current_turn
                board.flip(col, row, current_turn, directions_to_check)
                reward = board.evaluate_player(None, valid_moves, directions_to_check, current_turn, borders)

                next_state = agent.get_state_key(board)
                next_valid_moves = board.check_for_valid_show(-current_turn, directions_to_check)

                state_action = (current_state, tuple(move))
                agent.learn(state_action, reward, next_state, next_valid_moves)

                current_turn = -current_turn
            num_episodes = num_episodes - 1
            if num_episodes == 0:
                pygame.quit
                sys.exit

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


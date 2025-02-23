import pygame
import sys
import numpy as np

header = 30
SCREEN_SIZE = 800
BOARD_SIZE = 8
SQUARE_SIZE = SCREEN_SIZE // BOARD_SIZE
WHITE, BLACK, EMPTY = 1, -1, 0
window_size = SCREEN_SIZE + header

directions_to_check = [(-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, 1), (1, 0), (0, -1)]

pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, window_size))
pygame.display.set_caption('Reversi')
font = pygame.font.Font(None, 48)
white_points = 0
black_points = 0
game_mode = "playing"
current_turn = - 1

# inicializace základních proměných pro minimax algoritmus
maximizing_player = True
depth = 3

alfa = float('-inf')
beta = float('inf')

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
        # vyvolá funkci pomocí které upravuje Weigted board(jak jsou hodnocené jednotlivé pozice) podle toho v jaké části je hra(kolik tahů bylo
        # odehráno)
        self.WEIGHTED_BOARD = update_weighted_board(board)

        # použije funkci np.where která změní 0 a protihráčova(podle toho jestli počítá bílé nebo černé body) pole na False a nepočítá je a
        # hráčovo na True a počítá hodnotu která je na nich podle Weighted board a ty potom sečte pomocí funkce sum a uloží do proměné bodů
        # které zrovna počítá

        black_points = np.sum(np.where(self.grid == BLACK, self.WEIGHTED_BOARD, 0))
        white_points = np.sum(np.where(self.grid == WHITE, self.WEIGHTED_BOARD, 0))

        # zde jsem se snažil zlepšit funkci tak aby podle rohovích bodů počítala kamínky které se nedají převrátit.
        # funkce je podle mě funkční ale neviřešil jsem problé toho že hodnotila rohové a zbylé kamínky moc dobře a kvůli tomu hrála špatně
        # chtěl bych ale toto zlepšení opravit v budoucnu

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

        # přidává body podle toho kolik je možných tahů čím více tahů tím více bodů
        number_of_moves = 0
        for move in valid_moves:
            number_of_moves += 1
        # konečné sčítání bodů
        if current_turn == 1:
            white_points += (number_of_moves * 35)
        else:
            black_points += (number_of_moves * 35)

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
        # nastavil jsem zákldní hodnotu ohodnocení na kladné a záporné nekonečno
        max_eval = float('-inf')
        min_eval = float('inf')

        # zjistím, jsetli skončila hra a valid_moves(aby funkce evaluate mohla fungovat)
        end_of_the_match, white_points, black_points, winner = self.end_of_match()
        valid_moves = self.check_for_valid_show(current_turn, directions_to_check)
        # zjistím, jestli algoritmus došel na konec hry nebo došel do hloubky nula
        # pokud ano funkce vrátí konečné ohodnocení desky
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
        # minimax algoritmus hraje za bílá neboli 1
        if current_turn == 1:
            for move in valid_moves:
                row, col = move
                # udělám kopiji stavu desky
                grid_copy = np.copy(self.grid)
                # tah který algoritmus zahrál načteme na hrací desku(toto je v pořádku protože hned potom je hrací deska daná do nového objektu
                # který neovlivnuje originální hrací desku.Nejsou spuštěny žádné funkce které by ovlivnovali originální hrací desku)
                grid_copy[row, col] = current_turn
                # vytvoříme prázdnou(bez __init__) hrací desku která neovlivnuje originální desku
                cloned_board = Board.__new__(Board)
                # vloží stav hrací desky
                cloned_board.grid = grid_copy
                # přidá font z __inti__ jelikož jsme tuto část neskopírovali
                cloned_board.font = self.font
                # zavoláme flip funkci
                cloned_board.flip(col, row, current_turn, directions_to_check)
                # zavoláme funkci minimax z kde ale změníme hloubku na hloubku -1 a změníme hráče na protihráče a naopak
                # (pokud skončí hra nebo funkce dojde na hloubku 0 řetěz funkcí se přeruší a funkce vrátí nejlepší hodnotu)
                eval = cloned_board.minimax(depth - 1, -current_turn, no_valid_move_counter, alfa, beta, borders)
                # zhodnotí zda-li je skore lepší než dosavadní nejlepší skore
                max_eval = max(max_eval, eval)
                # vyhodnoti, zda-li je nový tah lepší než dosud nejlepší tah
                alfa = max(alfa, eval)
                # zjistíme jsetli max hráč má lepší skore než minnimální skore min hráče a pokud ano tuto větev
                # můžeme odendat, jelikož min hráč jsi ji nikdy nevybere, takže není důvod dále počítat možné tahy v této větvi
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


borders = get_borders(BOARD_SIZE)
board = Board()
mouse_x, mouse_y = 0, 0
play = False
no_valid_move_counter = 0

while True:
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

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_w:
                    board = 1

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
                end_of_the_match, white_points, black_points, winner = board.end_of_match()
                print(no_valid_move_counter)
                print(f"white:{white_points}   black:{black_points}       winner:{winner}")
                board = Board()
                current_turn = -1
                game_mode = "playing"
                break

            if valid_moves:
                no_valid_move_counter = 0

            if current_turn == -1:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    row, col = mouse_y // SQUARE_SIZE, mouse_x // SQUARE_SIZE

                    if [row, col] in valid_moves:
                        board.grid[row, col] = current_turn
                        board.flip(col, row, current_turn, directions_to_check)

                        if current_turn == -1:
                            current_turn = -current_turn

                    else:
                        print("this move is impossible")

            elif current_turn == 1:
                best_move = None
                best_score = float('-inf')
                depth = update_depth(board)

                for move in valid_moves:
                    row, col = move
                    # naklonujeme desku s kterou potom vyvoláme minimax algoritmus
                    cloned_board = board.clone_board()
                    cloned_board.grid[row, col] = current_turn
                    cloned_board.flip(col, row, current_turn, directions_to_check)

                    score = cloned_board.minimax(depth, -current_turn, no_valid_move_counter, alfa, beta, borders)
                    # zhodnotíme zda-li je score lepší než předchozí když ano uložíme nejlepší tah a score
                    if best_score < score:
                        best_move = move
                        best_score = score

                if best_move:
                    # potom co projdem všechny možné tahy nejlepší tah zahrajeme
                    row, col = best_move
                    board.grid[row, col] = current_turn
                    board.flip(col, row, current_turn, directions_to_check)
                    current_turn = -current_turn

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








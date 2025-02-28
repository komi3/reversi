import pygame
import sys
import numpy as np

# inicializace základních proměnných
game_mode = "main menu"

header = 30
SCREEN_SIZE = 800
BOARD_SIZE = 8
SQUARE_SIZE = SCREEN_SIZE // BOARD_SIZE
WHITE, BLACK, EMPTY = 1, -1, 0
window_size = SCREEN_SIZE + header

# directions are in vectors (y,x) for example (-1,0) ↑
# directions jsou vektory (y,x), například (-1,0) ↑
directions_to_check = [(-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, 1), (1, 0), (0, -1)]

white_points = 0
black_points = 0

total_white_points = 0
total_black_points = 0

black_wins = 0
white_wins = 0

mouse_x, mouse_y = 0, 0
play = False
current_turn = 1
no_valid_move_counter = 0

minimax_mode = False
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
        return 3
    elif occupied_squares <= 40:
        return 3
    else:
        return 3


# inicializace pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, window_size))
pygame.display.set_caption('Reversi')


class Board:
    def __init__(self):
        # Vytvoří hrací desku a základní pozice
        self.font = pygame.font.Font(None, 48)
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

        self.grid[3, 3] = WHITE
        self.grid[3, 4] = BLACK
        self.grid[4, 3] = BLACK
        self.grid[4, 4] = WHITE

    def check_if_on_board(self, row, col):
        # kontroluje, zda-li je na hrací desce
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return True
        else:
            return False

    def end_of_match(self):
        winner = None
        # sečte celkový počet bodů obou hráčů, porovná je a určí, kdo vyhrál. Pokud je součet počtu bodů roven velikosti hrací desky, hra se ukončí.

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
        # tato funkce projde celou hrací desku a vybere ta pole, kde hráč může hrát, aby protihráči vzal kamínky (to je jedno z pravidel hry), potom vrátí seznam s pozicemi
        valid_moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                # zkontroluje, jestli pole je prázdné, a pokud není prázdné, přeskočí na další a pokračuje dál
                if self.grid[row, col] != EMPTY:
                    continue
                # prochází seznam se směry a postupně přičítá k pozici. Vytváří také prázdný seznam s pozicemi, které by v případě úspěšného tahu změnila na své.
                for dx, dy in directions_to_check:
                    px, py = row + dx, col + dy
                    to_flip = []
                    # pokud je pozice na hrací desce a zároveň je to pole protihráče, přidá tuto pozici na list a přičte směr k dané pozici
                    while self.check_if_on_board(px, py) and self.grid[px, py] == -current_turn:
                        to_flip.append([px, py])
                        px += dx
                        py += dy
                    # pokud je na poli hráč (který zrovna hraje) a zároveň list s pozicemi, které budou změněny na hráčovo, není prázdný, přidá základní pozici na list s možnými pozicemi, které může hráč zahrát
                    if self.check_if_on_board(px, py) and self.grid[px, py] == current_turn and to_flip:
                        valid_moves.append([row, col])
                        break
        return valid_moves

    def flip(self, col, row, current_turn, directions_to_check):
        # prochází list se směry a postupně přičítá k pozici. Vytváří také prázdný list s pozicemi, které by v případě úspěšného tahu změnila na své.
        for x, y in directions_to_check:
            px, py = row + y, col + x
            to_flip = []
            # pokud je pozice na hrací desce a zároveň je to pole protihráče, přidá tuto pozici na list a přičte směr k dané pozici
            while self.check_if_on_board(px, py) == True and self.grid[px, py] == -current_turn:
                to_flip.append([px, py])
                px += y
                py += x
            # pokud je na poli hráč (který zrovna hraje) a zároveň je na desce, projde celý list a všechny pozice změní na pozice, které vlastní hráč
            if self.check_if_on_board(px, py) == True and self.grid[px, py] == current_turn:
                for px, py in to_flip:
                    self.grid[px, py] = current_turn

    def draw_valid_move(self, valid_moves):
        # nakreslí kruhy tam, kde hráč může hrát podle listu s pozicemi
        for row, col in valid_moves:
            pygame.draw.circle(screen, (250, 200, 152),
                               (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                               SQUARE_SIZE // 4)

    def draw_board(self):
        # nakreslí hrací desku a hráče
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
        # nakreslí body a kdo je zrovna na řadě

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

        # zjistím jsetli skončila hra a valid_moves(aby funkce evaluate mohla fungovat)
        end_of_the_match, white_points, black_points, winner = self.end_of_match()
        valid_moves = self.check_for_valid_show(current_turn, directions_to_check)
        # zjistím jestli algoritmus došel na konec hry nebo došel do hloubky nula
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
                # zhodnotí zdali je skore lepší než nejlepší dozatimní skore
                max_eval = max(max_eval, eval)
                # vyhodnoti zda-li je nový tah lepší než dozatimní nejlepší tah
                alfa = max(alfa, eval)
                # zjistíme jsetli max hráč má lepší skore než minnimální skore min hráče a pokud ano tuto větev
                # můžeme odendat jelikož min hráč jsi ji nikdy nevybere takže není důvod dále počítat možné tahy v této větvi
                if alfa >= beta:
                    break
            # vrátí nejlepší skore max hráče v této hloupce
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


class MainMenu:
    # hlavní menu obrazovka
    def __init__(self):
        self.font = pygame.font.Font(None, 48)

    def main_menu(self, mouse_x, mouse_y):

        play = False
        stats_mode = False
        Ai = False
        end_game = False

        screen.fill((0, 0, 0))

        button_rect = pygame.Rect(SCREEN_SIZE // 3, window_size // 2, SCREEN_SIZE // 3, 50)
        pygame.draw.rect(screen, (173, 216, 230), button_rect)
        button_text = self.font.render("Play", True, (0, 0, 0))
        screen.blit(button_text, (button_rect.x + 10, button_rect.y + 10))

        button_rect2 = pygame.Rect(SCREEN_SIZE // 3, window_size // 2 + 60, SCREEN_SIZE // 3, 50)
        pygame.draw.rect(screen, (173, 216, 230), button_rect2)
        button_text2 = self.font.render("AI", True, (0, 0, 0))
        screen.blit(button_text2, (button_rect2.x + 10, button_rect2.y + 10))

        button_rect3 = pygame.Rect(SCREEN_SIZE // 3, window_size // 2 + 120, SCREEN_SIZE // 3, 50)
        pygame.draw.rect(screen, (173, 216, 230), button_rect3)
        button_text3 = self.font.render("Stats", True, (0, 0, 0))
        screen.blit(button_text3, (button_rect3.x + 10, button_rect3.y + 10))

        button_rect4 = pygame.Rect(SCREEN_SIZE // 3, window_size // 2 + 180, SCREEN_SIZE // 3, 50)
        pygame.draw.rect(screen, (173, 216, 230), button_rect4)
        button_text4 = self.font.render("Quit", True, (0, 0, 0))
        screen.blit(button_text4, (button_rect4.x + 10, button_rect4.y + 10))

        if button_rect.collidepoint((mouse_x, mouse_y)):
            play = True
        elif button_rect2.collidepoint((mouse_x, mouse_y)):
            Ai = True
        elif button_rect3.collidepoint((mouse_x, mouse_y)):
            stats_mode = True
        elif button_rect4.collidepoint((mouse_x, mouse_y)):
            end_game = True

        pygame.display.flip()
        return play, Ai, stats_mode, end_game


class End:
    # obrazovka, která se ukáže po dohrání
    def __init__(self):
        self.font = pygame.font.Font(None, 48)

    def end_screen(self, white_points, black_points, winner, mouse_x, mouse_y, total_white_points, total_black_points,
                   white_wins, black_wins):
        show_game = False
        play_again = False
        stats_mode = False
        back = False

        total_white_points = total_white_points + white_points
        total_black_points = total_black_points + black_points

        if winner == "black":
            black_wins += 1
        elif winner == "white":
            white_wins += 1

        screen.fill((0, 0, 0))
        white_points, black_points = str(white_points), str(black_points)
        text_surface = self.font.render(f"WHITE: {white_points}  BLACK: {black_points}  Winner: {winner}", True,
                                        (255, 255, 255))
        screen.blit(text_surface, (SCREEN_SIZE // 6, window_size // 3))

        button_rect = pygame.Rect(SCREEN_SIZE // 3, window_size // 2, SCREEN_SIZE // 3, 50)
        pygame.draw.rect(screen, (173, 216, 230), button_rect)
        button_text = self.font.render("Play Again", True, (0, 0, 0))
        screen.blit(button_text, (button_rect.x + 10, button_rect.y + 10))

        button_rect2 = pygame.Rect(SCREEN_SIZE // 3, window_size // 2 + 60, SCREEN_SIZE // 3, 50)
        pygame.draw.rect(screen, (173, 216, 230), button_rect2)
        button_text2 = self.font.render("Stats", True, (0, 0, 0))
        screen.blit(button_text2, (button_rect2.x + 10, button_rect2.y + 10))

        button_rect3 = pygame.Rect(SCREEN_SIZE // 3, window_size // 2 + 120, SCREEN_SIZE // 3, 50)
        pygame.draw.rect(screen, (173, 216, 230), button_rect3)
        button_text3 = self.font.render("Show game", True, (0, 0, 0))
        screen.blit(button_text3, (button_rect3.x + 10, button_rect3.y + 10))

        button_rect4 = pygame.Rect(SCREEN_SIZE // 3, window_size // 2 + 180, SCREEN_SIZE // 3, 50)
        pygame.draw.rect(screen, (173, 216, 230), button_rect4)
        button_text4 = self.font.render("Back", True, (0, 0, 0))
        screen.blit(button_text4, (button_rect4.x + 10, button_rect4.y + 10))

        if button_rect.collidepoint((mouse_x, mouse_y)):
            play_again = True

        elif button_rect3.collidepoint((mouse_x, mouse_y)):
            show_game = True

        elif button_rect2.collidepoint((mouse_x, mouse_y)):
            stats_mode = True

        elif button_rect4.collidepoint((mouse_x, mouse_y)):
            back = True

        pygame.display.flip()
        return play_again, show_game, stats_mode, total_black_points, total_white_points, white_wins, black_wins, back


class Stats:
    # obrazovka, která ukazuje celkový počet bodů a výher
    def __init__(self):
        self.font = pygame.font.Font(None, 48)

    def stats(self, white_wins, black_wins, total_white_points, total_black_points, screen, mouse_x, mouse_y):
        play_again = False
        back = False

        screen.fill((0, 0, 0))
        white_wins, black_wins, total_white_points, total_black_points = (str(white_wins), str(black_wins),
                                                                          str(total_white_points),
                                                                          str(total_black_points))
        text_surface = self.font.render(f"White  wins:{white_wins}     Black wins:{black_wins}", True,
                                        (255, 255, 255))
        screen.blit(text_surface, (SCREEN_SIZE // 4, window_size // 4))
        text_surface2 = self.font.render(
            f"White total points:{total_white_points}      Black total points:{total_black_points} ", True,
            (255, 255, 255))
        screen.blit(text_surface2, (SCREEN_SIZE // 20, window_size // 5))

        button_rect = pygame.Rect(SCREEN_SIZE // 3, window_size // 2 + 120, SCREEN_SIZE // 3, 50)
        pygame.draw.rect(screen, (173, 216, 230), button_rect)
        button_text = self.font.render("Play Again", True, (0, 0, 0))
        screen.blit(button_text, (button_rect.x + 10, button_rect.y + 10))

        button_rect2 = pygame.Rect(SCREEN_SIZE // 3, window_size // 2 + 180, SCREEN_SIZE // 3, 50)
        pygame.draw.rect(screen, (173, 216, 230), button_rect2)
        button_text2 = self.font.render("Back", True, (0, 0, 0))
        screen.blit(button_text2, (button_rect2.x + 10, button_rect2.y + 10))

        if button_rect.collidepoint((mouse_x, mouse_y)):
            play_again = True

        elif button_rect2.collidepoint((mouse_x, mouse_y)):
            back = True

        pygame.display.flip()

        return back, play_again


class AI:
    def __init__(self):
        self.font = pygame.font.Font(None, 48)

    def AI_menu(self):
        screen.fill((0, 0, 0))
        minimax = False
        back = False

        button_rect = pygame.Rect(SCREEN_SIZE // 3, window_size // 2, SCREEN_SIZE // 3, 50)
        pygame.draw.rect(screen, (173, 216, 230), button_rect)
        button_text = self.font.render("Minimax", True, (0, 0, 0))
        screen.blit(button_text, (button_rect.x + 10, button_rect.y + 10))

        button_rect2 = pygame.Rect(SCREEN_SIZE // 3, window_size // 2 + 60, SCREEN_SIZE // 3, 50)
        pygame.draw.rect(screen, (173, 216, 230), button_rect2)
        button_text2 = self.font.render("Back", True, (0, 0, 0))
        screen.blit(button_text2, (button_rect2.x + 10, button_rect2.y + 10))

        if button_rect.collidepoint((mouse_x, mouse_y)):
            minimax = True
        elif button_rect2.collidepoint((mouse_x, mouse_y)):
            back = True

        pygame.display.flip()

        return minimax, back


board = Board()
main_menu = MainMenu()
end_screen = End()
stats = Stats()
Ai_screen = AI()
borders = get_borders(BOARD_SIZE)
while True:

    if game_mode == "main menu":
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            main_menu.main_menu(mouse_x, mouse_y)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                play, Ai, stats_mode, end_game = main_menu.main_menu(mouse_x, mouse_y)
                if play:
                    minimax_mode = False
                    board = Board()
                    current_turn = BLACK
                    game_mode = "playing"

                elif Ai:
                    game_mode = "AI"

                elif stats_mode:
                    game_mode = "Stats"

                elif end_game:
                    pygame.quit()
                    sys.exit()

    if game_mode == "playing":
        valid_moves = board.check_for_valid_show(current_turn, directions_to_check)

        for event in pygame.event.get():
            valid_moves = board.check_for_valid_show(current_turn, directions_to_check)

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
            # pokud není žádná pozice možná, hraje druhý hráč, a pokud ani on nemůže hrát, hra končí
            if not valid_moves:
                if no_valid_move_counter == 2:
                    end_of_the_match, white_points, black_points, winner = board.end_of_match(winner)
                    game_mode = "end_screen"
                    break

                if current_turn == 1:
                    current_turn = -1

                    no_valid_move_counter = no_valid_move_counter + 1
                    print(no_valid_move_counter)
                    break

                elif current_turn == -1:
                    current_turn = 1

                    no_valid_move_counter = no_valid_move_counter + 1
                    print(no_valid_move_counter)
                    break
            # pokud hráč může hrát, resetuje se počet kol, kde některý z hráčů nemohl hrát
            if valid_moves:
                no_valid_move_counter = 0


            if minimax_mode and current_turn == 1:
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

            else:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # kliknutí se převede na pole
                    mouse_x, mouse_y = event.pos
                    row, col = mouse_y // SQUARE_SIZE, mouse_x // SQUARE_SIZE
                    # pokud je pole, na které hráč kliknul, v seznamu možných polí, změní se toto pole na jeho a spustí se funkce na převracení protihráčových polí
                    if [row, col] in valid_moves:
                        board.grid[row, col] = current_turn
                        board.flip(col, row, current_turn, directions_to_check)
                        ## po odehrání hraje protihráč
                        if current_turn == 1:
                            current_turn = -1
                        elif current_turn == -1:
                            current_turn = 1
                        pygame.display.flip()
                    else:
                        print("this move is impossible")


        # Vykreslení hrací desky
        board.draw_board()
        board.draw_valid_move(valid_moves)
        end_of_the_match, white_points, black_points, winner = board.end_of_match()
        board.draw_points(white_points, black_points, current_turn)
        # pokud je end_of_the_match = True, skončí hra a změní se na konečnou obrazovku
        if end_of_the_match:
            game_mode = "end_screen"

        pygame.display.flip()

    if game_mode == "end_screen":
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            end_screen.end_screen(white_points, black_points, winner, mouse_x, mouse_y, total_white_points,
                                  total_black_points, white_wins, black_wins)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()

                play_again, show_game, stats_mode, total_black_points, total_white_points, white_wins, black_wins, back = end_screen.end_screen(
                    white_points, black_points, winner, mouse_x, mouse_y, total_white_points, total_black_points,
                    white_wins, black_wins)

                if play_again:
                    board = Board()
                    current_turn = BLACK
                    game_mode = "playing"
                elif show_game:
                    game_mode = "playing"

                elif stats_mode:
                    game_mode = "Stats"

                elif back:
                    game_mode = "main menu"

        pygame.display.flip()

    if game_mode == "Stats":
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            stats.stats(white_wins, black_wins, total_white_points, total_black_points, screen, mouse_x, mouse_y)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                back, play_again = stats.stats(white_wins, black_wins, total_white_points, total_black_points, screen,
                                               mouse_x, mouse_y)
                if play_again:
                    board = Board()
                    current_turn = BLACK
                    game_mode = "playing"

                if back:
                    game_mode = "main menu"

    if game_mode == "AI":
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            Ai_screen.AI_menu()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                minimax, back = Ai_screen.AI_menu()

                if minimax:
                    minimax_mode = True
                    board = Board()
                    current_turn = BLACK
                    game_mode = "playing"
                    print('minimax')

                elif back:
                    game_mode = "main menu"

# if minimax_mod:
# ... it shoud triger  when 1 [White]
# ...minimax algoritm activates
# else:
# ...normal game










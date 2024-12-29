import pygame
import sys
import numpy as np

game_mode = "main menu"

header = 30
SCREEN_SIZE = 800
BOARD_SIZE = 8
SQUARE_SIZE = SCREEN_SIZE // BOARD_SIZE
WHITE, BLACK, EMPTY = 1, -1, 0
window_size = SCREEN_SIZE + header
# directions are in vectors (y,x) for example (-1,0) ↑
directions_to_check = [(-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, 1), (1, 0), (0, -1)]

pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, window_size))
pygame.display.set_caption('Reversi')
winner = ""
white_points = 0
black_points = 0
total_white_points = 0
total_black_points = 0
black_wins = 0
white_wins = 0


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

    def end_of_match(self, winner):
        end_of_the_match = False
        white_points = 0
        black_points = 0
        for x in self.grid.flatten():
            if x == 1:
                white_points = white_points + 1
            elif x == -1:
                black_points = black_points + 1

            else:
                continue

            if white_points < black_points:

                winner = "black"

            elif white_points > black_points:

                winner = "white"

            if white_points + black_points == BOARD_SIZE * BOARD_SIZE:

                end_of_the_match = True

        return end_of_the_match, white_points, black_points, winner

    def check_for_valid_show(self,current_turn, directions_to_check):
        valid_moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):

                if self.grid[row, col] != EMPTY:
                    continue

                for dx, dy in directions_to_check:
                    px, py = row + dx, col + dy
                    to_flip = []

                    while self.check_if_on_board(px, py) and self.grid[px, py] != 0 and self.grid[px, py] != current_turn:
                        to_flip.append([px, py])
                        px += dx
                        py += dy
                    if self.check_if_on_board(px, py) and self.grid[px, py] == current_turn and to_flip:
                        valid_moves.append([row, col])
                        break
        return valid_moves

    def flip(self,col, row, current_turn, directions_to_check):
        for x, y in directions_to_check:
            px, py = row + y, col + x
            to_flip = []

            while self.check_if_on_board(px, py) == True and self.grid[px, py] != 0 and self.grid[px, py] != current_turn:
                to_flip.append([px, py])
                px += y
                py += x

            if self.check_if_on_board(px, py) == True and self.grid[px, py] == current_turn:
                for px, py in to_flip:
                    self.grid[px, py] = current_turn

    def draw_valid_move(self,valid_moves):
        for row, col in valid_moves:
            pygame.draw.circle(screen, (250, 200, 152),
                               (col * SQUARE_SIZE + SQUARE_SIZE // 2,  row * SQUARE_SIZE + SQUARE_SIZE // 2),
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
        text_surface = self.font.render(f"WHITE:{white_points}  BLACK:{black_points}       {current_turn}", True, (255, 255, 255))
        screen.blit(text_surface, (200, 800))

    def minimax(self, depth, winner):
        maximizing_player = 1
        #if winner == 1:
             #continue







class MainMenu:
    def __init__(self):
        self.font = pygame.font.Font(None, 48)

    def main_menu(self,mouse_x, mouse_y):

        play = False
        stats_mode = False
        AI = False
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
            AI = True
        elif button_rect3.collidepoint((mouse_x, mouse_y)):
            stats_mode = True
        elif button_rect4.collidepoint((mouse_x, mouse_y)):
            end_game = True

        pygame.display.flip()
        return play, AI, stats_mode, end_game


class End:
    def __init__(self):
        self.font = pygame.font.Font(None, 48)

    def end_screen(self,white_points, black_points,  winner, mouse_x, mouse_y, total_white_points, total_black_points, white_wins, black_wins):
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
        text_surface = self.font.render(f"WHITE: {white_points}  BLACK: {black_points}  Winner: {winner}", True,(255, 255, 255))
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
        text_surface2 = self.font.render(f"White total points:{total_white_points}      Black total points:{total_black_points} ", True,
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





board = Board()
main_menu = MainMenu()
end_screen = End()
stats = Stats()

mouse_x, mouse_y = 0, 0
play = False
current_turn = 1
no_valid_move_counter = 0

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
                play, AI, stats_mode, end_game = main_menu.main_menu(mouse_x, mouse_y)
                if play:
                    board = Board()
                    current_turn = BLACK
                    game_mode = "playing"

                elif AI:
                    continue
                    
                elif stats_mode:
                    game_mode = "Stats"

                elif end_game:
                    pygame.quit()
                    sys.exit()

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

            if valid_moves:
                no_valid_move_counter = 0

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                row, col = mouse_y // SQUARE_SIZE, mouse_x // SQUARE_SIZE

                if not valid_moves:
                    if current_turn == 1:
                        current_turn = -1
                    elif current_turn == -1:
                        current_turn = 1

                if [row, col] in valid_moves:
                    board.grid[row, col] = current_turn
                    board.flip(col, row, current_turn, directions_to_check)

                    if current_turn == 1:
                        current_turn = -1
                    elif current_turn == -1:
                        current_turn = 1

                else:
                    print("this move is impossible")

        board.draw_board()
        board.draw_valid_move(valid_moves)
        end_of_the_match, white_points, black_points, winner = board.end_of_match(winner)

        board.draw_points(white_points, black_points, current_turn)
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

            end_screen.end_screen(white_points, black_points,  winner, mouse_x, mouse_y,total_white_points, total_black_points, white_wins, black_wins)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()

                play_again, show_game, stats_mode, total_black_points, total_white_points,white_wins, black_wins, back = end_screen.end_screen(white_points, black_points, winner, mouse_x, mouse_y,total_white_points, total_black_points, white_wins, black_wins)

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
                back, play_again = stats.stats(white_wins, black_wins, total_white_points, total_black_points, screen, mouse_x, mouse_y)
                if play_again:
                    board = Board()
                    current_turn = BLACK
                    game_mode = "playing"

                if back:
                    game_mode = "main menu"












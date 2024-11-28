import pygame
import sys
import numpy as np


SCREEN_SIZE = 800
BOARD_SIZE = 8
SQUARE_SIZE = SCREEN_SIZE // BOARD_SIZE
WHITE, BLACK, EMPTY = 1, -1, 0

#directions are in vectors (y,x) for example (-1,0) ↑
directions_to_check = [(-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, 1), (1, 0), (0, -1)]

pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption('Reversi')
white_points = 0
black_points = 0

class Board:
    def __init__(self):
        BOARD_SIZE = 8
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

        self.grid[3, 3] = WHITE
        self.grid[3, 4] = BLACK
        self.grid[4, 3] = BLACK
        self.grid[4, 4] = WHITE

    def check_if_on_board(self,row, col):

            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                return True
            else:
                return False


    def end_of_match(self,white_points,black_points):
        for x in self.grid.flatten():
            if x == 1:
                white_points = white_points + 1
            elif x == -1:
                black_points = black_points + 1

            else:
                continue

            if white_points + black_points == 64:
                print("end of the game")
                if white_points < black_points:
                    print(f"white:{white_points} black:{black_points}")
                    print("black won the game ")

                elif white_points > black_points:
                    print(f"white:{white_points} black:{black_points}")
                    print("white won the game ")




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


board = Board()
current_turn = BLACK

while True:
    for event in pygame.event.get():
        valid_moves = board.check_for_valid_show(current_turn, directions_to_check)

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                row, col = mouse_y // SQUARE_SIZE, mouse_x // SQUARE_SIZE


                if board.grid[row, col] == 0:
                    board.grid[row, col] = current_turn
                    board.flip(col, row, current_turn, directions_to_check)
                    if current_turn == 1:
                        current_turn = -1
                    else:
                        current_turn = 1

                else:
                    print("this move is impossible")


    board.end_of_match(white_points, black_points)
    board.draw_board()
    board.draw_valid_move(valid_moves)



    # Update the display
    pygame.display.flip()


# # end game screen



import time

from numba import njit
from numba import jit
import numpy as np

from Action import Action
from Checkers import Board

time_possible = 0
time_if = 0
time_own_logic = 0
time_create_coords = 0
time_get_board = 0
time_check_possible_moves = 0


class CheckersPawn:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.player = 1 if color == "black" else 2
        self.is_king = 0

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def get_position(self):
        return self.x, self.y

    def convert_to_king(self):
        self.is_king = 1

    def possible_moves(self, board: Board):
        if self.player == 2 and not self.is_king:
            nearby_coords = [(self.x + 1, self.y + 1), (self.x - 1, self.y + 1)]
        elif self.player == 1 and not self.is_king:
            nearby_coords = [(self.x + 1, self.y - 1), (self.x - 1, self.y - 1)]
        else:
            nearby_coords = [(self.x + 1, self.y + 1),
                             (self.x - 1, self.y + 1),
                             (self.x - 1, self.y - 1),
                             (self.x + 1, self.y - 1)]
        possible_moves = []
        jump_moves = []
        state = board.get_board()
        for x, y in nearby_coords:
            if 0 <= x < len(state) and 0 <= y < len(state[0]):
                if state[x][y] == 0:
                    possible_moves.append(Action(x, y, CheckersPawn(self.x, self.y, self.color)))
                elif state[x][y] != state[self.x][self.y]:
                    projectory_x, projectory_y = (x - self.x) + x, (y - self.y) + y
                    if 0 <= projectory_x < len(state) and 0 <= projectory_y < len(state[0]) \
                            and state[projectory_x, projectory_y] == 0:
                        jump_moves.append(Action(projectory_x, projectory_y, CheckersPawn(self.x, self.y, self.color)))
        return possible_moves, jump_moves

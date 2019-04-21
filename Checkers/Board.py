from numba import njit

import Action
import numpy as np
import Abstract_Game
from Checkers import Checkers_Pawn
import copy


def check_jump(action: Action):
    pawn = action.player
    if action.x == pawn.x-2 and action.y == pawn.y-2:
        return 1, -1, -1
    if action.x == pawn.x+2 and action.y == pawn.y-2:
        return 1, 1, -1
    if action.x == pawn.x-2 and action.y == pawn.y+2:
        return 1, -1, 1
    if action.x == pawn.x+2 and action.y == pawn.y+2:
        return 1, 1, 1
    return 0, 0, 0


class Board(Abstract_Game.Game):

    def __init__(self, actions, b_size, debug=0, pawns=None):
        super().__init__(actions, b_size)
        self.np_array = np.zeros((self.board_size, self.board_size))
        self.turns_with_no_kill = 0
        if pawns is None:
            self.pawns = []
            pawns = self.create_pawns()
        self.debug = debug
        self.pawns = pawns
        self.reds = 0
        self.blacks = 0
        for pawn in self.pawns:
            self.np_array[pawn.x][pawn.y] = pawn.player
            if pawn.color == "red":
                self.reds += 1
            else:
                self.blacks += 1

    def create_pawns(self):
        pawns = []
        x1 = 1
        x2 = 1
        for y in range((self.board_size // 2)-1):
            for x in range(self.board_size):
                if not x % 2 == 0:
                    checker = Checkers_Pawn.CheckersPawn(x-x1, y, "red")
                    self.place(Action.Action(x-x1, y, checker))
                    pawns.append(checker)

            for x in range(self.board_size):
                if x % 2 == 0:
                    checker = Checkers_Pawn.CheckersPawn(x+x2, self.board_size - y - 1, "black")
                    self.place(Action.Action(x+x2, self.board_size - y - 1, checker))
                    pawns.append(checker)
            x1 = 1 if x1 == 0 else 0
            x2 = 1 if x2 == 0 else 0
        return pawns

    def kill_pawn(self, x, y):
        if self.find_player(x,y):
            self.np_array[x][y] = 0
            self.pawns.remove(self.find_player(x,y))

    def delete_action(self, x, y, player):
        for action in self.actions:
            if action.x == x and action.y == y and action.player == player:
                self.actions.remove(action)

    def print_pawns(self):
        for pawn in self.pawns:
            print("x: "+str(pawn.x)+", y: "+str(pawn.y))

    def place(self, new_action: Action):
        if new_action.x >= self.board_size or new_action.x <= -1 or \
                new_action.y >= self.board_size or new_action.y <= -1:
            print("Not a board position! Try again...")
            return 0
        b = self.get_board()
        if not b[new_action.x][new_action.y]:
            is_jump, x_remove, y_remove = check_jump(new_action)
            if is_jump:
                if new_action.player.color == "black":
                    self.reds -= 1
                else:
                    self.blacks -= 1
                self.kill_pawn(new_action.player.x + x_remove, new_action.player.y + y_remove)
                self.turns_with_no_kill = 0

            self.kill_pawn(new_action.player.x, new_action.player.y)
            new_action.player.x = new_action.x
            new_action.player.y = new_action.y

            if not new_action.player.is_king:
                if new_action.player.player == 1 and (not new_action.y or not new_action.player.y):
                    new_action.player.is_king = 1
                if new_action.player.player == 2 and (new_action.y == self.board_size-1 or new_action.player.y == self.board_size-1):
                    new_action.player.is_king = 1

            self.pawns.append(new_action.player)
            self.np_array[new_action.player.x][new_action.player.y] = new_action.player.player
            self.actions.append(new_action)
            self.turns_with_no_kill += 1
            return 1
        else:
            print(self.get_board())
            print("Invalid Position! Try again...")
            return 0

    def get_board(self):
        return self.np_array

    def possible_moves(self, player, is_second_call=0):
        actions = []
        jump_actions = []
        for pawn in self.pawns:
            if pawn.player == player:
                pawn_moves, pawn_jumps = pawn.possible_moves(self)
                actions += pawn_moves
                jump_actions += pawn_jumps
        if len(jump_actions):
            return jump_actions
        else:
            return actions
        """elif is_second_call:
            exec("NO POSSIBLE MOVES FOUND")
        else:
            print("calling for opponent")
            opponent = 1 if player - 1 else 2
            act = self.possible_moves(opponent, 1)
            return act"""

    def is_terminal_state(self):
        if self.turns_with_no_kill == 100:
            return 1
        if self.is_winner(1) or self.is_winner(2) or (not len(self.possible_moves(1)) and not len(self.possible_moves(2))):
            return 1
        return 0

    def is_winner(self, player):
        if player == 1 and self.reds == 0:
            return 1
        elif player == 2 and self.blacks == 0:
            return 1
        return 0

    def copy_pawns(self):
        new_pawns = []
        for pawn in self.pawns:
            new_pawns.append(copy.copy(pawn))
        return new_pawns

    def copy_actions(self, actions):
        new_actions = []
        for action in actions:
            new_actions.append(Action.Action(action.x, action.y, copy.copy(action.player)))
        return new_actions

    def create_copy(self, actions, size):
        return Board(self.copy_actions(self.actions), self.board_size, self.debug, self.copy_pawns())

    def is_legal_field(self, x, y):
        if y <= -1 or y >= self.board_size or x <= -1 or x >= self.board_size:
            return 0
        else:
            return 1

    def find_player(self, x, y):
        for pawn in self.pawns:
            if pawn.x == x and pawn.y == y:
                return pawn
        return None

    def get_turn(self, player):
        new_player = 2 if player == 1 else 1
        if not len(self.possible_moves(new_player)):
            return player
        else:
            return new_player

    def create_base_copy(self):
        return Board([], self.board_size, 0)


@njit()
def is_legal_field(x, y, board_size):
    if y <= -1 or y >= board_size or x <= -1 or x >= board_size:
        return 0
    else:
        return 1

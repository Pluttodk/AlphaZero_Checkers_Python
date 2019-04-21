import Action
import numpy as np
import Abstract_Game


class Board(Abstract_Game.Game):
    def get_turn(self, player):
        return 2 if player == 1 else 1

    def __init__(self, actions, b_size, debug = 0):
        super().__init__(actions, b_size)
        self.debug = debug

    def place(self, new_action: Action):
        should_place = 1
        if new_action.x >= self.board_size or new_action.x <= -1 or new_action.y >= self.board_size or new_action.y <= -1:
            print("Not a board position! Try again...")
            return 0
        for action in self.actions:
            if action.x == new_action.x and action.y == new_action.y:
                print(action.x, action.y, new_action.x, new_action.y, self.get_board())
                should_place = 0
                break
        if should_place:
            self.actions.append(new_action)
            return 1
        else:
            print("Invalid Position! Try again...")
            return 0

    def get_board(self):
        state = np.zeros((self.board_size, self.board_size))
        for action in self.actions:
            state[action.x][action.y] = action.player
        return state

    def possible_moves(self, player):
        actions = []
        board = self.get_board()
        x, y = 0, 0
        for pos in board:
            y = 0
            for value in pos:
                if not value:
                    actions.append(Action.Action(x, y, player))
                y += 1
            x += 1
        return actions

    def is_terminal_state(self):
        if len(self.actions) >= self.board_size * self.board_size or self.is_winner(1) or self.is_winner(2):
            return 1
        else:
            return 0

    def is_winner(self, player):
        board = self.get_board()
        if self.debug:
            print("Checking rows for player" + str(player))
        # Checking rows
        done = 0
        for i in range(0, self.board_size):
            in_row = 0
            for j in range(0, self.board_size):
                if board[i][j] == player:
                    in_row += 1
                if in_row == self.board_size:
                    done = 1
                    return done

        if self.debug:
            print("Checking columns for player" + str(player))
        # Checking columns
        for i in range(0, self.board_size):
            in_column = 0
            for j in range(0, self.board_size):
                if board[j][i] == player:
                    in_column += 1
                if in_column == self.board_size:
                    done = 1
                    return done

        if self.debug:
            print("Checking diagonals for player" + str(player))
        # Checking diagonals
        in_diagonal = 0
        for i in range(0, self.board_size):
            if board[i][i] == player:
                in_diagonal += 1
            if in_diagonal == self.board_size:
                done = 1
                return done
        in_diagonal = 0
        j = self.board_size - 1
        for i in range(0, self.board_size):
            if board[j][i] == player:
                in_diagonal += 1
            if in_diagonal == self.board_size:
                done = 1
                return done
            j -= 1
        return 0

    def create_copy(self, actions, size):
        return Board(actions, size, 0)

    def create_base_copy(self):
        return Board([], self.board_size, 0)



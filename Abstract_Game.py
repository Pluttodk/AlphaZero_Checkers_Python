from abc import ABC, abstractmethod
import Action


class Game(ABC):
    def __init__(self, actions, board_size):
        super().__init__()
        self.board_size = board_size
        self.actions = actions

    @abstractmethod
    def place(self, new_action: Action):
        pass

    @abstractmethod
    def get_board(self):
        pass

    @abstractmethod
    def get_turn(self, player):
        pass

    @abstractmethod
    def is_terminal_state(self):
        pass

    @abstractmethod
    def possible_moves(self, player):
        pass

    @abstractmethod
    def is_winner(self, player):
        pass

    @abstractmethod
    def create_copy(self, actions, size):
        pass

    @abstractmethod
    def create_base_copy(self):
        pass

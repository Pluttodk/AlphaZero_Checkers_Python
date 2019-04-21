from Checkers.View import View
from Checkers import Board
import AI
import time
from time import sleep
import time
import threading
import NN

board_size = 6
debug = 0


class Solo_Ai(AI.AI):

    def __init__(self, view):
        super().__init__()
        self.view = view

    def decide_move(self, board, value):
        while not self.view.has_placed:
            pass
        self.view.has_placed = 0

    def decide_move_two(self, board, value, x, y):
        pass


def parse_input(game_style, view):
    splitter = game_style.split(" ")
    switcher = {
        "solo": Solo_Ai(view),
        "minmax": AI.MINMAX(),
        "mcts": AI.MCTS(),
        "rnd": AI.VariousRnd(),
        "NN": NN.BetaOne()
    }
    return switcher[splitter[0]], switcher[splitter[1]]


def parse_settings(game_settings):
    splitter = game_settings.split(" ")
    switcher = {
        "y": "yes",
        "n": "no"
    }
    return switcher[splitter[0]]


def get_moves(play1, play2, v, b, enable_gui, game_style, game_settings, number_of_games):
    player = 1

    t0 = time.time()

    while not b.is_terminal_state():
        player = b.get_turn(player)
        if player == 1:
            play1.decide_move(b, 1)
            v.update()
            sleep(0.3)
        elif player == 2:
            play2.decide_move(b, 2)
            v.update()
            sleep(0.3)
    print(b.get_board())
    if debug:
        print(threading.active_count())
        print(threading.enumerate())
        print(threading.current_thread())
    t1 = time.time()
    total = t1 - t0
    print("Game time: "+str(total)+" seconds.")
    if number_of_games > 1:
        v.tk.quit()
        run_game(game_style, game_settings, number_of_games-1)


def run_game(game_style, game_settings, number_of_games):
    b = Board.Board([], board_size, debug)
    v = View(board_size, b)

    play1, play2 = parse_input(game_style, v)
    enable_gui = 1 if parse_settings(game_settings) == "yes" else 0

    playing = threading.Thread(target=get_moves, args=(play1, play2, v, b, enable_gui,
                                                       game_style, game_settings, number_of_games))
    playing.start()
    if enable_gui:
        v.draw_players(b.get_board())
        v.run()


def play_game():
    game_style = input("WELCOME TO CHECKERS THE GAME OF THE YEAR \n"
                       "Type as position as (x,y) \n "
                       "\"solo\" = no ai \n "
                       "\"minmax\" = play as minmax ai \n"
                       "\"mcts\" =  play as mcts \n"
                       "e.g. \"solo minmax\" will let you play against a minmax\n")

    game_settings = input("Do you want to enable GUI?\n"
                          "Type y/n.\n")
    print("How many games do you want to play?\n"
          "Type a number eg. 10\n")
    number_of_games = int(input())

    run_game(game_style, game_settings, number_of_games)

if __name__ == '__main__':
    play_game()
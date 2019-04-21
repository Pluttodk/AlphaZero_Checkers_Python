from TicTacToe import View
from TicTacToe.Board import Board
import AI
import NN

board_size = 3
debug = 0


def Network_from_file():
    value = input("Write a version model")
    return AI.Network_MCTS(f"Output/model_V{value}00.h5")


def parse_input(game_style):
    splitter = game_style.split(" ")
    switcher = {
        "solo": AI.SOLO(),
        "minmax": AI.MINMAX(),
        "mcts": AI.MCTS(),
        "NN": NN.BetaOne(),
        "File": Network_from_file()
    }
    return switcher[splitter[0]], switcher[splitter[1]]


def play_game():
    game_style = input("WELCOME TO TIC TAC TOE THE GAME OF THE YEAR \n"
                       "Type as position as (x,y) \n "
                       "\"solo\" = no ai \n "
                       "\"minmax\" = play as minmax ai \n"
                       "\"mcts\" =  play as mcts \n"
                       "e.g. \"solo minmax\" will let you play against a minmax\n")
    play1, play2 = parse_input(game_style)
    print("Please write your desired board size,\ne.g. 5, which will make a 5x5 board.")
    board_size = int(input())
    b = Board([], board_size, debug)
    if debug:
        print("Board size = " + str(board_size))
    print("-------------")
    View.print_board(b)
    player_turn = 2
    while not b.is_terminal_state():
        player_turn = b.get_turn(player_turn)
        player = play1 if player_turn == 1 else play2

        result = 0
        while not result:
            result = player.decide_move(b, player_turn)
        View.print_board(b)
    View.print_winner(player_turn)

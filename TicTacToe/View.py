from TicTacToe.Board import Board
import tkinter
import numpy as np
import sys
import AI

icons = {
    0: "-",
    1: "X",
    2: "O"
}

buttons = np.zeros((5, 5), object)
player_turn = 1
board = Board([], 0)


def start(b, board_size, play1, play2):
    global buttons
    global board

    board = b
    buttons = np.zeros((board_size, board_size), object)
    root = tkinter.Tk()

    frame = tkinter.Frame(root)
    tkinter.Grid.rowconfigure(root, 0, weight=1)
    tkinter.Grid.columnconfigure(root, 0, weight=1)
    frame.grid(row=0, column=0, sticky=tkinter.N + tkinter.S + tkinter.E + tkinter.W)
    grid = tkinter.Frame(frame)
    grid.grid(sticky=tkinter.N + tkinter.S + tkinter.E + tkinter.W, column=0, row=7, columnspan=2)
    tkinter.Grid.rowconfigure(frame, 7, weight=1)
    tkinter.Grid.columnconfigure(frame, 0, weight=1)

    # example values
    for x in range(board_size):
        for y in range(board_size):
            btn = tkinter.Button(frame, height=3, width=10,
                                 command=lambda x=x, y=y: on_click(x, y, play1, play2), text=" ")
            btn.grid(column=x, row=y, sticky=tkinter.N + tkinter.S + tkinter.E + tkinter.W)
            buttons[x][y] = btn

    for x in range(board_size):
        tkinter.Grid.columnconfigure(frame, x, weight=1)

    for y in range(board_size):
        tkinter.Grid.rowconfigure(frame, y, weight=1)

    return root


def on_click(x, y, play1, play2):
    get_player(play1, play2, x, y)


def set_button_text(x, y, text):
    buttons[x][y].config(text=text, state=tkinter.DISABLED)


def get_player(play1, play2, x, y):
    player = play1 if player_turn == 1 else play2

    result = 0
    while not result:
        result = player.decide_move_two(board, player_turn, y, x)
    newest_action = board.actions[-1]
    set_button_text(newest_action.y, newest_action.x, get_text())
    print_board(board)
    if board.is_terminal_state():
        if board.is_winner(1):
            print_winner(1)
        if board.is_winner(2):
            print_winner(2)
        print("Well played!")
        sys.exit(0)
    set_player()
    # print("player_turn: " + str(player_turn) + ", player1: " + str(play1) + ", player2: " + str(play2))
    if (player_turn == 1 and isinstance(play1, AI.MINMAX)) or (player_turn == 2 and isinstance(play2, AI.MINMAX)):
        # print("AI turn!")
        get_player(play1, play2, 0, 0)


def set_player():
    global player_turn

    if player_turn == 1:
        player_turn = 2
    else:
        player_turn = 1


def get_text():
    if player_turn == 1:
        return "X"
    else:
        return "O"


def print_board(b: Board):
    state = b.get_board()
    for x in range(0, b.board_size):
        s = "[ "
        for y in range(0, b.board_size):
            s += icons[state[x][y]]
            s += " ; " if y < b.board_size - 1 else ""
        s += " ]"
        print(s)
    print("-------------")


def print_winner(player):
    print("{} has won the game".format(icons[player]))

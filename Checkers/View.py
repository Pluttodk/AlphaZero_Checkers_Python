from tkinter import *
from Action import Action
import numpy as np

Title = "The BetaOne Checkers"

width, height = 500, 500

class View:
    def __init__(self, size, board):
        self.tk = Tk()
        self.cv = Canvas(self.tk, width=width, height=height, bg="antique white")
        self.cv.pack()
        self.dimension = width / size
        self.players = []
        self.init_board(size)
        self.board = board
        self.has_placed = 0

    def init_board(self, size):
        tk = self.tk
        tk.title(Title)
        black = 1
        dimension = self.dimension
        cv = self.cv
        for x in range(size):
            black = not black
            for y in range(size):
                black = not black
                bg = "saddle brown" if black else "LightGoldenrod1"
                cv.create_rectangle(dimension * x, dimension * y, dimension * (x + 1), dimension * (y + 1), fill=bg)
        cv.bind("<Button-1>", self.find_player)

    def find_player(self, event):
        x = int(event.x / self.dimension)
        y = int(event.y / self.dimension)
        pawn_to_move = self.board.find_player(x,y)
        #TODO: Make it so that you can only click you player
        self.cv.unbind("<Buton-1>")
        self.cv.bind("<Button-1>",
                     lambda event, a=pawn_to_move:
                     self.parce_place(event, a))

    def draw_players(self, board):
        self.cv.delete("player")
        positions = np.transpose(np.nonzero(board))

        for x,y in positions:
            player_color = self.get_color(board[x][y])
            size = self.dimension / 4
            player = self.draw_player(x, y, player_color, size)
            self.players.append((player,(x,y)))
        self.tk.update()
        self.tk.update_idletasks()

    def draw_player(self, x, y, color, size):
        startx, starty, endx, endy = self.get_pos(x, y, size)
        return self.cv.create_oval(startx, starty, endx, endy, fill=color, tags="player")

    def get_pos(self, x, y, size):
        return (x * self.dimension + size,
                y * self.dimension + size,
                (x + 1) * self.dimension - size,
                (y + 1) * self.dimension - size)

    def run(self):
        self.tk.mainloop()

    def print_board(self, board):
        pass

    def start(self, board, board_size, player1, player2):
        pass

    def print_winner(self, player):
        pass

    def get_color(self, player):
        return "OrangeRed2" if player - 1 else "gray16"

    def parce_place(self, event, a):
        x = int(event.x / self.dimension)
        y = int(event.y / self.dimension)
        self.board.place(Action(x,y,a))
        self.has_placed = 1
        self.cv.unbind("<Button-1>")
        self.cv.bind("<Button-1>", self.find_player)
        self.draw_players(self.board.get_board())
        self.tk.update_idletasks()
        self.tk.update()

    def update(self):
        self.draw_players(self.board.get_board())

import AI
import time

import Config
from Checkers import Checkers_Pawn
from Checkers.Board import Board
import matplotlib.pyplot as plt
import NN
from Network_MCTS import NN_MCTS

number_of_games = 10

# A python file made simply to test mcts during multiple iterations
def Start_test():
    outcome = []
    print("Test start...")
    t_start = time.time()

    number_of_wins = 0
    number_of_losses = 0
    number_of_draws = 0

    play1,play2 = NN.BetaOne(), AI.VariousRnd()
    for i in range(0, 11):
        #play2 = NN_MCTS(f"Output/model_CheckersSoftmax3_V{i*50}.h5")
        board_size = 6
        b = Board([], board_size, 0)
        #boards = [Board([], board_size, 0) for _ in range(number_of_games)]
        #board_to_use = boards
        t1 = time.time()

        player = 2
        while not b.is_terminal_state():
            turn = b.get_turn(player)
            if turn-1:
                play2.decide_move(b, 2)
            else:
                play1.decide_move(b,1)
            player = 1 if player-1 else 2
        """turns = [2]*number_of_games
        while len(board_to_use):
            net_board = []
            rnd_board = []
            for i, b in enumerate(board_to_use):
                if b.get_turn(turns[i]) == 1:
                    net_board.append(b)
                else:
                    rnd_board.append(b)
            if len(net_board):
                play1.decide_move_multiple_actors(net_board, [1]*len(net_board))
            for b in rnd_board:
                if len(b.possible_moves(2)):
                    play2.decide_move(b, 2)
                else:
                    print(b.get_board())
            n_turns = []
            for i, b in enumerate(board_to_use):
                if b.is_terminal_state():
                    board_to_use.remove(b)
                else:
                    opponent = 1 if turns[i]-1 else 2
                    n_turns.append(b.get_turn(opponent))
            turns = n_turns
            print(f"Played one move {len(board_to_use)}")
        for b in boards:"""
        if b.is_winner(1):
            number_of_wins += 1
            outcome.append(1)
        elif b.is_winner(2):
            number_of_losses += 1
            outcome.append(-1)
        else:
            number_of_draws += 1
            outcome.append(0)
        print(f"{(i+1)} games finished. and resulted in {outcome[-1]}")
        t2 = time.time()
        total = t2 - t1
        print("Game time: " + str(total) + " seconds.")
    # Creates a histogram of the results. Proves how well it does
    t_end = time.time()
    print("Average time per game: "+str((t_end-t_start)/number_of_games))
    print("MCTS wins: "+str(number_of_wins)+", RandomAI wins: "+str(number_of_losses)+", draws: "+str(number_of_draws))
    plt.plot(outcome)
    print(outcome)
    plt.show()

    print("Test end...")

if __name__ == '__main__':
    Start_test()
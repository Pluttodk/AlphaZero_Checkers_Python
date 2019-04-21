from Checkers import Game, Checkers_Test
from TicTacToe import Game as Ga
from TicTacToe import testing
import numpy as np
from matplotlib import pyplot as plt
import sys
import matplotlib.patches as mpatches
from Config import Config

def show_loss():
    loss = np.load(f"Output/{Config.LOSS_NAME}.npy")
    general_loss, loss1, loss2 = [i[0] for i in loss[0]], [i[1] for i in loss[0]],[i[2] for i in loss[0]]
    game_resx, yaxis = [i[0] for i in loss[1]], [i[1] for i in loss[1]]
    game_rnd, game_mcts, game_prev = [g[1] for g in game_resx], [g[0] for g in game_resx], [g[2] for g in game_resx]
    plt.subplot(2,2,1)
    plt.plot(loss1)
    plt.title("Value_Loss")
    plt.subplot(2,2,2)
    plt.plot(loss2)
    plt.title("Policy_Loss")
    plt.subplot(2,2,3)
    plt.plot(general_loss)
    plt.title("General_Loss")
    plt.subplot(2,2,4)

    red_patch = mpatches.Patch(color="red", label="RND")
    blue_patch = mpatches.Patch(color="blue", label="MCTS")
    green_patch = mpatches.Patch(color="green", label="PREV")
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    plt.plot(yaxis, game_rnd, label="RND", color="red")
    plt.plot(yaxis, game_mcts, label="MCTS", color="blue")
    plt.plot(yaxis, game_prev, label="PREV", color="green")
    plt.title("Game vs opoonent")
    plt.show()

if __name__ == '__main__':

    args = sys.argv
    if len(args) > 1:
        if args[1] == "-c1":
            Game.play_game()
        elif args[1] == "-c2":
            Checkers_Test.Start_test()
        elif args[1] == "-t1":
            Ga.play_game()
        elif args[1] == "-t2":
            testing.start_test()
        elif args[1] == "-p":
            show_loss()
    else:
        print("Please input one of the following args on next run \n"
              "'-c1' => Running checkers with GUI \n"
              "'-c2' => Running checkers with a test \n"
              "'-t1' => Running TicTacToe with GUI \n"
              "'-t2' => Running TicTacToe with a test \n")

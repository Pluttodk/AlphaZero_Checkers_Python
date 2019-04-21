import AI
from Action import Action
from Config import Config
from Network import Network
from Network_MCTS import NN_MCTS
from TicTacToe.Board import Board
import matplotlib.pyplot as plt
import NN
import time
import tensorflow as tf
import numpy as np
def start_test():
    """def train_on_stored_network():
    # A python file made simply to test mcts during multiple iterations
    outcome = []
    for i in range(0, 10):
        play1, play2 = AI.Network_MCTS(f"model_V{(i + 1) * 10}.h5"), AI.VariousRnd()
        res = 0
        for i in range(0, 10):
            start = time.time()

            board_size = 3
            b = Board([], board_size, 0)
            player_turn = 2
            while not b.is_terminal_state():
                player_turn = 1 if player_turn == 2 else 2
                player = play1 if player_turn == 1 else play2

                player.decide_move(b, player_turn)
            if b.is_winner(1):
                res += 1
            end = time.time()
            print(f"Played {i} game and it took {end - start}")
        # Creates a histogram of the results. Proves how well it does
        outcome.append(res / 10)
    plt.hist(outcome)
    print(outcome)
    plt.show()"""


    """if __name__ == '__main__':
    network = Network((3, 3, 1))
    buffer = np.load("tic_tac_toe_4000.npy")
    plot = []
    for _ in range(10):
        move_sum = float(sum(len(g.history) for g in buffer))
        games = np.random.choice(
            buffer,
            size=Config.BATCH_SIZE,
            p=[len(g.history) / move_sum for g in buffer]
        )
        game_pos = [(g, np.random.randint(len(g.history))) for g in games]
        batch = [(g.history[i], g.result) for (g, i) in game_pos]
        batch_image, expected_out = np.array([i[0] for i in batch]), np.array([i[1] for i in batch])
        batch_image = batch_image.reshape((-1, 3, 3, 1))

        loss = network.model.fit(batch_image, expected_out, epochs=Config.TRAINING_STEPS)
        plot = plot + loss.history["loss"]
    plt.plot(plot)
    plt.show()
    network.model.save("TEST_TIC_TAC_TOE_NETWORK.h5")"""
# A python file made simply to test mcts during multiple iterations
    outcome = []
    play1, play2 = NN.BetaOne(), AI.VariousRnd()
    # play1, play2 = NN.BetaOne(), AI.VariousRnd()
    for i in range(0, 50):
        #play2 = NN_MCTS(f"Output/model_V{i*Config.CHECKPOINT_INTERVAL}.h5")
        start = time.time()
        board_size = 4
        b = Board([], board_size, 0)
        player_turn = 1
        while not b.is_terminal_state():
            player = play1 if player_turn == 1 else play2
            output = player.decide_move(b, player_turn)
            player_turn = 1 if player_turn == 2 else 2
            """if not player_turn-1:
                image = b.get_board().reshape((-1, b.board_size, b.board_size, 1))
                res = play1.network.predict(image)
                val, policy_logits = res[0][0][0], res[1][0]
                print(val, "\n", policy_logits)
                print(b.get_board())
                print(output[1].value)"""
        if b.is_winner(1):
            outcome.append(1)
        elif b.is_winner(2):
            outcome.append(-1)
        else:
            outcome.append(0)
        print(b.get_board())
        end = time.time()
        print(f"Played {i} game and it took {end - start}")
    # Creates a histogram of the results. Proves how well it does
    plt.hist(outcome)
    #plt.plot(outcome, [i*Config.CHECKPOINT_INTERVAL for i in range(12)])
    print(outcome)
    plt.show()
    """play1, play2 = AI.Network_MCTS("Output/model_V3000.h5"), AI.VariousRnd()
    board_size = 4
    b = Board([], board_size, 0)
    image = b.get_board().reshape((-1, b.board_size, b.board_size, 1))
    res = play1.network.predict(image)
    val, policy_logits = res[0][0][0], res[1][0]
    print(val, "\n", policy_logits)

    b = Board([Action(0,0,1), Action(1,1,1), Action(2,2,1), Action(3,3,1), Action(0,1,2), Action(0,2,2), Action(0,3,2)], board_size, 0)
    image = b.get_board().reshape((-1, b.board_size, b.board_size, 1))
    res = play1.network.predict(image)
    val, policy_logits = res[0][0][0], res[1][0]
    print(val, "\n", policy_logits)"""



if __name__ == '__main__':
    start_test()

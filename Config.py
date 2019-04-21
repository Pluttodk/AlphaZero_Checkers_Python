class Config:
    LOSS_NAME = "TicTacTestNewGamesHighSampling"
    NETWORK_NAME = "model_TicTac10HigheSampling_V"
    BOARD_WIDTH = 4
    TEST_GAMES_PR_CHECKPOINT = 5
    NUM_SAME_AGENTS = 25
    EPOCHS_PR_TRAINING = 10
    ROOT_EXPLORATION_FRACTION = 0.25
    ROOT_ALPHA = 0.2 #can flucturate a lot
    ACTION_SPACE = BOARD_WIDTH*BOARD_WIDTH # possible actions in 4x4 TicTacToe
    NUM_SAMPLING_MOVES = 2 # Is there to avoid it going down the exact same path in the begining
    PB_C = 19652
    PB_C_INIT = 1.25
    TRAINING_LOOPS = 10
    EPOCHS = 30
    WINDOW_SIZE = 2000
    BATCH_SIZE = 256
    BUFFER_SIZE = 50
    MCTS_NN_SIMULATION = 100
    MCTS_SIMULATION = 100
    ACTORS = 2
    CHECKPOINT_INTERVAL = 1000
    TRAINING_STEPS = 200000
    LEARNING_RATIO = 0.1
    DECAY = 0.0004
    MOMENTUM = 0.9
    REG_VAL = 0.0001
    # TODO: MAKE THIS HERE TWO VARIABLES INSTEAD OF THIS BULLCRAB
    HIDDEN_CNN_LAYERS = [
        {'filters': 128, 'kernel_size': (4,4)}
        , {'filters': 128, 'kernel_size': (4,4)}
        , {'filters': 128, 'kernel_size': (4,4)}
        , {'filters': 128, 'kernel_size': (4,4)}
        , {'filters': 128, 'kernel_size': (4,4)}
        , {'filters': 128, 'kernel_size': (4, 4)}
        , {'filters': 128, 'kernel_size': (4, 4)}
        , {'filters': 128, 'kernel_size': (4, 4)}
    ]

def Print_Progress_Bar(iteration, total, prefix="", suffix="", decimals=1, fill="#"):
    percentage = f"{round(100 * (iteration / total), decimals)}"
    filled = int(50 * iteration // total)
    bar = fill * filled + ' ' * (50 - filled)
    print(f"\r {prefix} [{bar}] {percentage}% {suffix}", end="\r")
    # Print New Line on Complete
    if iteration == total:
        print()

if __name__ == '__main__':
    from time import sleep

    # A List of Items
    items = list(range(0, 57))
    l = len(items)

    # Initial call to print 0% progress
    Print_Progress_Bar(0, l, prefix='Progress:', suffix='Complete')
    for i, item in enumerate(items):
        # Do stuff...
        sleep(0.1)
        # Update Progress Bar
        Print_Progress_Bar(i + 1, l, prefix='Progress:', suffix='Complete')
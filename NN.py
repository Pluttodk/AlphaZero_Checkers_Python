import time

from AI import *

from multiprocessing import Process
from multiprocessing.connection import wait
import multiprocessing
from Config import Config
from matplotlib import pyplot
from Network_MCTS import *


class NN_Game:
    def __init__(self, org_player):
        """
        A special NN game that only hold the important information regarding the game
        :param games: A numpy array of all the states of a given game
        :param result: The outcome of the game
        """
        self.history = []
        self.result = 0
        self.child_visits = []
        self.score = []
        self.org_player = org_player

    def store_search_statistics(self, root: NN_Node):
        sum_visits = sum(child.number_of_simulations for child in root.children)
        width = Config.BOARD_WIDTH
        actions_pos = {width*a.actions[-1].x+a.actions[-1].y: a.number_of_simulations/sum_visits for a in root.children}
        search_statistic = [0]*Config.ACTION_SPACE
        for i in range(Config.ACTION_SPACE):
            if i in actions_pos.keys():
                search_statistic[i] = actions_pos[i]
        self.child_visits.append(search_statistic)

    def add_history(self, hist, turn):
        outcome = np.zeros((2, len(hist), len(hist[0])))
        for y in range(len(hist)):
            for x in range(len(hist[0])):
                opponent = 1 if turn - 1 else 2
                if hist[y][x] == turn:
                    outcome[0][y][x] = 1
                elif hist[y][x] == opponent:
                    outcome[1][y][x] = 1
        self.history.append((outcome, turn))

    def add_result(self, res):
        self.result = res

    def make_label(self, index: int):
        res = self.result if self.history[index][1] != self.org_player else -self.result
        score = self.score[index] if self.history[index][1] != self.org_player else -self.score[index]
        output = (score+res)/2
        return output, self.child_visits[index]

    def add_part_time_result(self, score):
        self.score.append(score)


class Buffer:
    def __init__(self):
        self.windows_size = Config.WINDOW_SIZE
        self.buffer = []

    def save_game(self, game: NN_Game):
        if len(self.buffer) > self.windows_size:
            self.buffer = self.buffer[:len(self.buffer) // 2]
        self.buffer.append(game)

    def sample_batch(self):
        move_sum = float(sum(len(g.history) for g in self.buffer))
        games = np.random.choice(
            self.buffer,
            size=Config.BATCH_SIZE,
            p=[len(g.history) / move_sum for g in self.buffer]
        )
        game_pos = [(g, numpy.random.randint(len(g.history))) for g in games]
        return [(g.history[i][0], g.make_label(i)) for (g, i) in game_pos]


class Storage:

    def __init__(self):
        self.networks = {}

    def get_latest_network(self, size) -> Network:
        if len(self.networks):
            return self.networks[max(self.networks.keys())]
        else:
            net = Network((2, size, size), (size * size))
            self.networks[0] = net
            return net

    def save_network(self, network):
        self.networks[0] = network

    def store_network(self, i, network):
        network.save_network(i)


class BetaOne(AI):

    def __init__(self, storage=Storage(), buffer=Buffer()):
        super().__init__()
        self.num_actors = Config.ACTORS
        self.storage = storage
        self.board_size = 6
        self.board = None
        self.buffer = buffer
        self.has_trained = 0
        self.child_pipe = None
        self.turn = 0

    def launch_job(self, pipe):
        while True:
            game = self.play_game(pipe)
            pipe.send(["save_game", game])

    def start_actor(self, child_pipe):
        # self.launch_job()
        process = Process(target=self.launch_job, args=(child_pipe,))
        process.start()

    def handler(self, pipe):
        self.run_handler(pipe)

    def run_handler(self, pipe):
        # TODO: COMPRESS CODE
        buffer = Buffer()
        storage = Storage()
        parser = {
            "evaluate": 3,
            "print": 1,
            "get_sample_batch": 0,
            "save_game": 2,
            "get_size_buffer": 5,
            "train_network": 6,
            "done_training": 7,
            "clear_buffer": 8,
            "predict_multiple_roots": 9
        }
        i = 0
        plots = [[],[]]
        buffer_size_at_last_training = 0
        while True:
            for parent in wait(pipe):
                val = parent.recv()
                if parser[val[0]] == 3:
                    net = storage.get_latest_network(self.board_size)
                    value = net.evaluate(val[1], self.board_size, val[2])
                    parent.send(value)
                elif parser[val[0]] == 1:
                    print(len(buffer.buffer))
                elif parser[val[0]] == 2:
                    for game in val[1]:
                        buffer.save_game(game)
                    buffer_size_at_last_training += len(val[1])
                elif parser[val[0]] == 0:
                    parent.send(buffer.sample_batch())
                elif parser[val[0]] == 5:
                    parent.send(buffer_size_at_last_training)
                elif parser[val[0]] == 9:
                    net = storage.get_latest_network(self.board_size)
                    parent.send(net.evaluate_multiple(val[1]))
            if buffer_size_at_last_training >= Config.BUFFER_SIZE:
                buffer_size_at_last_training = 0
                if not i:
                    plots = self.train_network(buffer, storage, i, plots)
                    i+=1
                for j in range(i, i + Config.EPOCHS_PR_TRAINING):
                    plots = self.train_network(buffer, storage, j, plots)
                i += Config.EPOCHS_PR_TRAINING
            if i >= Config.TRAINING_STEPS:
                print(i)
                pyplot.plot(plots)
                pyplot.show()
                break

    def decide_move(self, board, value):
        if self.has_trained:
            # nn = self.storage.get_latest_network(self.board_size)
            player = NN_MCTS()
            return player.decide_move(board, value, self.child_pipe)[0]
        self.board_size = board.board_size
        self.board = board
        self.turn = value
        # Creates the pipe to send forward the buffer and storage

        parent_pipes = []
        for _ in range(Config.ACTORS):
            parent_pipe, child_pipe = multiprocessing.Pipe()
            parent_pipes.append(parent_pipe)
            self.start_actor(child_pipe)
        parent_pipe, child_pipe = multiprocessing.Pipe()
        parent_pipes.append(parent_pipe)
        self.handler(parent_pipes)
        print("started handler")

        # nn = self.storage.get_latest_network(self.board_size)
        player = NN_MCTS()
        self.has_trained = 1
        self.child_pipe = child_pipe
        return player.decide_move(board, value, child_pipe)[0]

    def decide_move_two(self, board, value, x, y):
        pass

    def play_game(self, child_pipe):
        start = time.time()
        boards = [self.board.create_copy((self.board.actions + []), self.board_size) for _ in range(Config.NUM_SAME_AGENTS)]
        turns = [self.turn] * Config.NUM_SAME_AGENTS
        games = [NN_Game(self.turn) for _ in range(Config.NUM_SAME_AGENTS)]
        board_to_use = [(g,b, turn) for g,b,turn in zip(games, boards, turns)]
        play1 = NN_MCTS()
        while len(board_to_use):
            roots = play1.decide_move_multiple_actors([b[1] for b in board_to_use], turns, child_pipe)
            for i,(g, board, turn) in enumerate(board_to_use):
                g.add_history(board.get_board(), turn)
                g.store_search_statistics(roots[i])
                g.add_part_time_result(roots[i].get_score())
            turns = [b[1].get_turn(b[2]) for b in board_to_use]
            board_to_use = []
            for g, board, turn in zip(games, boards, turns):
                if not board.is_terminal_state():
                    board_to_use.append((g,board,turn))
            # play1.print(roots[0])

        for board, game in zip(boards,games):
            opponent = 1 if self.turn - 1 else 2
            res = board.is_winner(self.turn) if board.is_winner(self.turn) else -1
            game.add_result(res)
        end = time.time()
        print(f"Playing a game with {Config.NUM_SAME_AGENTS} Agents and {Config.ACTORS} Resulted in a total play time of {end-start} seconds")
        return games

    def train_network(self, buffer, storage, i, plot):
        """
        Trains the network on one given batch
        :param buffer: The buffer were it can fetch the played games from
        :param storage: A storage containing all the networks
        :param i: Current number of time it has trained
        :param plot: A list of all the loss
        :return: The updated loss
        """
        network = storage.get_latest_network(self.board_size)
        Print_Progress_Bar(i, Config.TRAINING_STEPS, f"Epoch {i} ")
        print()
        batch = buffer.sample_batch()
        batch_image, expected_out = np.array([i[0] for i in batch]), [i[1] for i in batch]
        expected_res, expected_img = np.array([i[0] for i in expected_out]), np.array([i[1] for i in expected_out])
        batch_image = batch_image.reshape((-1, 2, self.board_size, self.board_size))
        loss = network.model.train_on_batch(batch_image, (expected_res, expected_img))
        plot[0].append(loss)

        #TODO: Make an arena between the best network and this current to keep on using the best possible network

        storage.save_network(network)
        if not i % Config.CHECKPOINT_INTERVAL:
            storage.store_network(i, network)
            result = [-1,-1,-1] if not i else self.play_on_that_network(network, i)
            print(f"Started at loss {plot[0][0]} now at {plot[0][-1]}")
            plot[1].append((result, i))
            np.save(f"Output/{Config.LOSS_NAME}.npy", plot)
        return plot

    def play_on_that_network(self, network, i):
        result = []
        player = NN_MCTS()
        opponents = [MCTS(), VariousRnd(), NN_MCTS(f"Output/{Config.NETWORK_NAME}{i-Config.CHECKPOINT_INTERVAL}.h5")]
        opponent_turn = 1 if self.turn-1 else 2
        for opponent in opponents:
            res = 0
            for i in range(Config.TEST_GAMES_PR_CHECKPOINT):
                board = self.board.create_base_copy()
                turn = self.turn
                while not board.is_terminal_state():
                    if turn == self.turn:
                        player.decide_move(board, turn, network=network)
                    else:
                        opponent.decide_move(board, turn)
                    turn = board.get_turn(turn)
                res += 1 if board.is_winner(self.turn) else -board.is_winner(opponent_turn)
            result.append(res/Config.TEST_GAMES_PR_CHECKPOINT)
        return result

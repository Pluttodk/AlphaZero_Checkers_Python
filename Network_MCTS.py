import copy
import time

import numpy as np
from numba import njit
from pptree import Node as PP_Node
from pptree import *

from Action import Action
from Config import Config, Print_Progress_Bar
from Network import Network, softmax_cross_entropy_logits
import cProfile
import tensorflow as tf


@njit()
def reshape_one_board(state, output, turn, i):
    for y in range(len(state)):
        for x in range(len(state[0])):
            opponent = 1 if turn - 1 else 2
            if state[y][x] == turn:
                output[i][0][y][x] = 1
            elif state[y][x] == opponent:
                output[i][1][y][x] = 1
    return output


@njit()
def reshape_one_board(state, output, turn, i):
    for y in range(len(state)):
        for x in range(len(state[0])):
            opponent = 1 if turn - 1 else 2
            if state[y][x] == turn:
                output[i][0][y][x] = 1
            elif state[y][x] == opponent:
                output[i][1][y][x] = 1
    return output


def reshape_boards(boards, turns):
    output = np.zeros((len(boards), 2, boards[0].board_size, boards[0].board_size))
    for i, (board, turn) in enumerate(zip(boards, turns)):
        state = board.get_board()
        output = reshape_one_board(state, output, turn, i)
    return output


class NN_Node:
    def __init__(self, score_sum, actions, org_player, turn, prior, parent=None):
        """

        :param score_sum: The value/score
        :param actions: All the actions taken so far
        :param org_player: the original player
        :param turn: The current player
        :param prior: The prior value
        :param parent: It's parent
        """
        self.prior = prior
        self.org_player = org_player
        self.turn = turn
        self.actions = actions
        self.score_sum = score_sum
        self.number_of_simulations = 0
        self.children = []
        self.parent = parent

    def get_score(self):
        return 0 if not self.number_of_simulations else self.score_sum / self.number_of_simulations


class NN_MCTS:

    def __init__(self, network_path=None):
        self.board = None
        self.board_size = 0
        self.prof = cProfile.Profile()
        self.network = Network(network_path=network_path) if network_path else None

    def decide_move(self, board, value, child_pipe=None, network=None):
        self.board = board
        self.board_size = board.board_size
        # TODO: Figure out if get_turn should be used here
        root = NN_Node(0, board.actions, value, value, 0)
        if child_pipe:
            self.simulate(board, root, child_pipe)
            self.add_exploration_noise(root)
        else:
            if self.network:
                network = self.network
            self.simulate(board, root, network=network)

        for i in range(0, Config.MCTS_NN_SIMULATION):
            node = self.select(root)
            board_clone = self.board.create_copy(node.actions, self.board_size)
            if board_clone.is_terminal_state():
                opponent = 1 if value - 1 else 2
                score = board.is_winner(value) if board.is_winner(value) else -board.is_winner(opponent)
                self.backpropagate(node, score)
            elif child_pipe:
                score = self.simulate(board_clone, node, child_pipe)
                self.backpropagate(node, score)
            else:
                score = self.simulate(board_clone, node, network=network)
                self.backpropagate(node, score)
        if network:
            return board.place(self.select_child(board, root, 0)), root
        return board.place(self.select_child(board, root)), root

    def decide_move_multiple_actors(self, boards, turns, child_pipe=None):
        self.prof.enable()
        board_size = boards[0].board_size
        self.board_size = board_size
        roots = [NN_Node(0, boards[i].actions + [], turns[i], turns[i], 0)
                 for i in range(len(boards))]
        if child_pipe:
            prediction = self.predict_multiple(boards, roots, child_pipe)
        else:
            network = self.network if self.network else None
            prediction = self.predict_multiple(boards, roots, network=network)
        j = 0
        for i, policy_logits in enumerate(prediction[1]):
            self.expand(boards[i], roots[i], policy_logits)
            if not len(roots[i].children):
                print(f"WTF JUST HAPPENED \n --------------- \n {boards[i].get_board()}")
                j+=1
            self.add_exploration_noise(roots[i])
        for i in range(0, Config.MCTS_NN_SIMULATION):
            leafs = [self.select(node) for node in roots]
            board_clones = [board.create_copy(leaf.actions + [], board_size) for leaf, board in zip(leafs, boards)]
            if child_pipe:
                scores = self.predict_multiple(board_clones, leafs, child_pipe)
            else:
                scores = self.predict_multiple(board_clones, leafs, network=network)
            for board, val, policy_logits, node, org_turn in zip(boards, scores[0], scores[1], leafs, turns):
                self.expand(board, node, policy_logits)
                if board.is_terminal_state():
                    opponent = 1 if org_turn-1 else 2
                    score = board.is_winner(org_turn) if board.is_winner(org_turn) else -1
                    self.backpropagate(node, score)
                else:
                    self.backpropagate(node, val[0])
            Print_Progress_Bar(i, Config.MCTS_NN_SIMULATION, "Running NN_MCTS", f" playing with {Config.NUM_SAME_AGENTS}")
        for board, root in zip(boards, roots):
            action = self.select_child(board, root)
            board.place(action)


            # self.print(root)
        self.prof.disable()
        return roots

    def backpropagate(self, node: NN_Node, value):
        """
            Recusively goes back through all the parents and updates
            the number of simulations and the result
            :param node: The node to update
            :param value: the valDue to yupdate it with
            """
        node.number_of_simulations += 1
        node.score_sum += value if node.turn != node.org_player else -value
        if node.parent:
            self.backpropagate(node.parent, value)

    def predict_multiple(self, boards, nodes, child_pipe=None, network=None):
        state = reshape_boards(boards, [node.turn for node in nodes])
        if child_pipe:
            child_pipe.send(["predict_multiple_roots", state])
            return child_pipe.recv()
        else:
            return network.evaluate_multiple(state)

    def predict(self, board, node, child_pipe=None, network=None):
        if child_pipe:
            child_pipe.send(["evaluate", board.get_board(), node.turn])
            return child_pipe.recv()
        else:
            return network.evaluate(board.get_board(), board.board_size, node.turn)

    def simulate(self, board, node, child_pipe=None, network=None):
        if child_pipe:
            val, policy_logits = self.predict(board, node, child_pipe=child_pipe)
        else:
            val, policy_logits = self.predict(board, node, network=network)
        self.expand(board, node, policy_logits)
        return val

    def expand(self, board, node, policy_logits):
        state = board.get_board()
        width = len(state)
        actions = board.possible_moves(node.turn)
        if not len(actions):
            opponent = 1 if node.turn - 1 else 2
            actions = board.possible_moves(opponent)
        policy = [(np.exp(policy_logits[(width * a.x + a.y)]), a) for a in actions]
        policy_sum = sum(v[0] for v in policy)
        clones = [board.create_copy(board.actions + [], self.board_size) for _ in range(len(policy))]
        i = 0
        for p, action in policy:
            new_actions = node.actions + [Action(action.x, action.y, copy.copy(action.player))]
            clones[i].place(action)
            node.children.append(
                NN_Node(0, new_actions, node.org_player,
                        clones[i].get_turn(node.turn),
                        p / policy_sum, node))
            i+=1

    def puct(self, node: NN_Node, child: NN_Node):
        """
        Upper confidence tree search
        A formular used to ensure both expantation and exploration
        :param node: The node to test on
        :return: The UCT
        """
        prior_score = puct(node.number_of_simulations, child.number_of_simulations,
                           child.prior, Config.PB_C, Config.PB_C_INIT)
        value_score = child.get_score()
        return prior_score + value_score

    def select(self, node: NN_Node):
        """
        Select the node with the highest UCT
        Unless there is a node that have not yet been visited,
        that way it will choose that ode
        to ensure overfitting it
        Runs recursively until an unexpanded node is reached
        :param node: The node to select from
        :return: a leaf node
        """
        if len(node.children) > 0:
            values = [(self.puct(node, child), child) for child in node.children]
            _, child = max(values, key=lambda child: child[0])
            return self.select(child)
        else:
            return node

    def select_child(self, board, node, use_softmax_sample=1):
        visit_counts = [(child.number_of_simulations, child.actions[-1]) for child in node.children]
        if len(board.actions) < Config.NUM_SAMPLING_MOVES and use_softmax_sample:
            action = self.softmax_sample(visit_counts)
        else:
            if len(visit_counts):
                score, action = max(visit_counts, key=lambda child: child[0])
            else:
                action = node.children[-1].actions[-1]
        return action

    def softmax_sample(self, visit_counts):
        # TODO: Optimize to a beter version maybe
        sum_visits = sum([c[0] for c in visit_counts])
        visits = [c[0] / sum_visits for c in visit_counts]
        visits = self.safer_softmax1D(visits, alpha=1.5)
        actions = [c[1] for c in visit_counts]
        choice = np.random.choice(actions, p=visits)
        return choice

    def safer_softmax1D(self, values, alpha=1.0):
        values = values.copy()
        max_v = max(values)
        sum_exp = 0
        for i, x in enumerate(values):
            v = np.exp((x - max_v) * alpha)
            sum_exp += v
            values[i] = v
        for i, x in enumerate(values):
            values[i] = x / sum_exp
        return values

    def print(self, root):
        """
        A method to visualize the tree
        :param root:
        :return:
        """
        childs = []
        new = PP_Node(f"{root.get_score()} , {root.number_of_simulations}")
        for child in root.children:
            next = PP_Node(f"{child.get_score()}, {child.number_of_simulations}", new)
            childs.append(next)
            for child2 in child.children:
                next2 = PP_Node(f"{child2.get_score()} , {child2.number_of_simulations}", next)
                childs.append(next2)
        print_tree(new)

    def add_exploration_noise(self, node: NN_Node):
        # TODO: Maybe normalize the exploration values
        actions = [i for i in range(len(node.children))]
        noise = np.random.gamma(Config.ROOT_ALPHA, 1, len(actions))
        frac = Config.ROOT_EXPLORATION_FRACTION
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


@njit()
def puct(number_of_simulations, child_number_of_simulations1, prior, PB_C, PB_C_INIT):
    pb_c = np.log((number_of_simulations + PB_C + 1) /
                  PB_C) + PB_C_INIT
    pb_c *= np.sqrt(number_of_simulations) / (child_number_of_simulations1 + 1)

    return pb_c * prior

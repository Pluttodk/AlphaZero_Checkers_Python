from abc import ABC, abstractmethod

from numba import njit

import Abstract_Game
from Action import Action
import numpy
import re
import math
from time import sleep
import tensorflow as tf
from Config import Config


class AI(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def decide_move(self, board, value):
        pass

    @abstractmethod
    def decide_move_two(self, board, value, x, y):
        pass


class CHECKERS_SOLO(AI):
    def decide_move_two(self, board, value, x, y):
        pass

    def decide_move(self, board, value):
        answer = input("type a pawn to move) \n")
        pos = self.getPos(answer)
        answer2 = input("where to move?) \n")
        pos2 = self.getPos(answer2)
        action = Action(int(pos2[0]), int(pos2[1]), board.find_player(int(pos[0]), int(pos[1])))
        return board.place(action)

    def getPos(self, answer):
        p = re.compile(r'\d+')
        m = p.findall(answer)
        if len(m) == 2:
            return m
        else:
            print("NOT CORRECT FORMAT")

class SOLO(AI):

    def decide_move(self, board: Abstract_Game, value):
        answer = input("type a position e.g. (x,y) \n")
        pos = self.getPos(answer)
        action = Action(int(pos[0]), int(pos[1]), value)
        return board.place(action)

    def decide_move_two(self, board: Abstract_Game, value, x, y):
        action = Action(x, y, value)
        return board.place(action)

    def getPos(self, answer):
        p = re.compile(r'\d+')
        m = p.findall(answer)
        if len(m) == 2:
            return m
        else:
            print("NOT CORRECT FORMAT")


class MINMAX(AI):

    def decide_move(self, board: Abstract_Game, value):
        current_actions = [] + board.actions
        b = board.create_copy(current_actions, board.board_size)
        best_score = float("-inf")
        best_move = None
        for action in b.possible_moves(value):
            new_state = [action] + current_actions
            score = self.min(board.create_copy(new_state, board.board_size), self.map_person(value), 1)
            if score > best_score:
                best_score = score
                best_move = action
        return board.place(best_move)

    def decide_move_two(self, board, value, x, y):
        current_actions = [] + board.actions
        b = board.create_copy(current_actions, board.board_size)
        best_score = float("-inf")
        best_move = None
        for action in b.possible_moves(value):
            new_state = [action] + current_actions
            score = self.min(board.create_copy(new_state, board.board_size), self.map_person(value), 1)
            if score > best_score:
                best_score = score
                best_move = action
        x = best_move.x
        y = best_move.y
        return board.place(best_move)

    def max(self, board, maximum, depth):
        if depth < 9 and not board.is_terminal_state():
            best_score = float("-inf")
            for action in board.possible_moves(maximum):
                new_state = [action] + board.actions
                new_board = board.create_copy(new_state, board.board_size)
                score = self.min(new_board, self.map_person(maximum), depth + 1)
                if score > best_score:
                    best_score = score
            return best_score
        else:
            return -10 if board.is_winner(self.map_person(maximum)) else 0

    def min(self, board, minimum, depth):
        if depth < 9 and not board.is_terminal_state():
            best_score = float("inf")
            for action in board.possible_moves(minimum):
                new_state = [action] + board.actions
                new_board = board.create_copy(new_state, board.board_size)
                score = self.max(new_board, self.map_person(minimum), depth + 1)
                if score < best_score:
                    best_score = score
            return best_score
        else:
            return 10 if board.is_winner(self.map_person(minimum)) else 0

    def map_person(self, value):
        return {
            1: 2,
            2: 1
        }[value]


class Node:
    def __init__(self, value, actions, org_player, turn, parent=None):
        self.org_player = org_player
        self.turn = turn
        self.actions = actions
        self.value = value
        self.number_of_simulations = 0
        self.children = []
        self.parent = parent
        self.is_fully_expanded = 0


class MCTS(AI):

    def decide_move_two(self, board, value, x, y):
        result = self.decide_move(board, value)
        x = board.actions[-1].x
        y = board.actions[-1].y

        return result

    def __init__(self):
        super().__init__()
        self.board = None
        self.board_size = 0

    def map_to_opponent(self, person):
        """Maps a person between the two states
        A method that mostlikey should be moved somewhere else
        late on in the process"""
        return 2 if person == 1 else 1

    def decide_move(self, board, value):
        """
        The following method simulates a playthrough N number of times
        than from that result it choses the move from the root with the
        highest possibility of a win
        :param board: The current board state
        :param value: The value of the current player
        :return: The board after a given action
        """
        self.board = board
        self.board_size = board.board_size
        root = Node(0, board.actions, value, self.map_to_opponent(value))
        number_of_simulations = 800
        for i in range(0, number_of_simulations):
            self.play(root)
        best_node = root
        best_score = float("-inf")
        for child in root.children:
            if child.number_of_simulations > 0:
                score = child.value / child.number_of_simulations
                if score > best_score:
                    best_score = score
                    best_node = child
            else:
                if board.is_winner(value):
                    return board.place(child.actions[-1])
        return board.place(best_node.actions[-1])

    # Method used to go through the tree (should be caled N number of times to simulate)
    def play(self, root):
        """
        The base premise of the MCTS
        Select -> Expand -> Simulate -> Backpropogate
        :param board:
        :param root: The current root node, which it is going to select from
        """
        # SELECT
        leaf = self.select(root)
        #b = self.board.create_copy(leaf.actions, self.board_size)
        b = self.board
        if not b.is_terminal_state():
            node = leaf
            if leaf.number_of_simulations > 0:
                node = self.expand(leaf)

            # Simulate which also calls backpropagate
            score = self.simulate(node)
            self.backpropagate(node, score)

    def uct(self, node: Node):
        """
        Upper confidence tree search
        A formular used to ensure both expantation and exploration
        :param node: The node to test on
        :return: The UCT
        """
        left_site = node.value / node.number_of_simulations
        right_site = numpy.sqrt(2) * numpy.sqrt(
            numpy.log(node.parent.number_of_simulations) / node.number_of_simulations)
        return left_site + right_site

    def select(self, node: Node):
        """
        Select the node with the highest UCT
        Unless there is a node that have not yet been visited,
        that way it will choose that ode
        to snure overfitting it
        Runs recursively until an unexpanded node is reached
        :param node: The node to select from
        :return: a leaf node
        """
        if len(node.children) > 0:
            chosen_one = node.children[0]
            best_uct = float("-inf")
            for child in node.children:
                if child.number_of_simulations > 0:
                    uct = self.uct(child)
                    if uct > best_uct:
                        best_uct = uct
                        chosen_one = child
                else:
                    # returns first child that hasn't been simulated yet
                    return child
            return self.select(chosen_one)
        else:
            return node

    # Expands all the children
    def expand(self, node: Node):
        """
        Takes all given action from that state and creates a node
        :param node: a node to expand
        :return: a random one of the children to be simulated
        """
        #currentBoard = self.board.create_copy(node.actions, self.board_size)
        currentBoard = self.board
        for action in currentBoard.possible_moves(currentBoard.get_turn(node.turn)):
            new_node = Node(0, node.actions + [action], node.org_player, self.map_to_opponent(node.turn), node)
            node.children.append(new_node)
        rnd = int(numpy.random.random_sample() * len(node.children))
        return node.children[rnd]

    def simulate(self, node: Node):
        """
        Runs all the way through a given state until it reaches
        a terminal state
        :param node: the node to simulate from
        :return: the result of the simulation unless it is a loss it returns negative 1
        """
        board = self.board.create_copy(node.actions + [], self.board_size)
        player = node.turn
        while not board.is_terminal_state():
            player = board.get_turn(player)
            possible_moves = board.possible_moves(player)
            if len(possible_moves) == 1:
                board.place(possible_moves[0])
            else:
                if len(possible_moves) > 0:
                    rnd = numpy.random.randint(len(possible_moves))
                    board.place(possible_moves[int(rnd)])
        result = board.is_winner(node.org_player)
        opponent_result = board.is_winner(self.map_to_opponent(node.org_player))
        return result if result else -opponent_result

    def backpropagate(self, node: Node, value):
        """
        Recusively goes back through all the parents and updates
        the number of simulations and the result
        :param node: The node to update
        :param value: the valDue to yupdate it with
        :param value: the value to yupdate it with
        """
        node.number_of_simulations += 1
        node.value += -value if node.turn != node.org_player else value
        if node.parent:
            self.backpropagate(node.parent, value)

    # Worth experimenting with
    def policy_child(self, children):
        """
        A policy that should be altered to try out different things
        :param children: a list of childrens
        :return: a random one of those childrens
        """
        rnd = numpy.random.random_sample() * len(children)
        return children[int(rnd)]


class VariousRnd(AI):

    def decide_move_two(self, board, value, x, y):
        pass

    def decide_move(self, board, value):
        moves = board.possible_moves(value)
        rnd = numpy.random.randint(len(moves))
        return board.place(moves[rnd])


from NN import NN_Node
from Network import softmax_cross_entropy_logits

@njit()
def puct(number_of_simulations, child_number_of_simulations1, prior, PB_C, PB_C_INIT):
    pb_c = numpy.log((number_of_simulations + PB_C + 1) /
                     PB_C) + PB_C_INIT
    pb_c *= numpy.sqrt(number_of_simulations) / (child_number_of_simulations1 + 1)

    return pb_c * prior


class Network_MCTS:

    def __init__(self, network):
        self.mcts = MCTS()
        self.board = None
        self.board_size = 0
        self.network = tf.keras.models.load_model(network, custom_objects={'softmax_cross_entropy_logits': softmax_cross_entropy_logits})

    def decide_move(self, board, value):
        self.board = board
        self.board_size = board.board_size
        self.mcts.board = board
        self.turn = value
        self.mcts.board_size = self.board_size
        root = NN_Node(0, board.actions, value, board.get_turn(value), 0)
        self.simulate(board, root)
        for i in range(0, Config.MCTS_NN_SIMULATION):
            node = self.select(root)
            board_clone = self.board.create_copy(node.actions, self.board_size)
            if not board_clone.is_terminal_state():
                score = self.simulate(board_clone, node)
                self.backpropagate(node, score)
        return board.place(self.select_child(board, root)), root

    def backpropagate(self, node: NN_Node, value):
        """
            Recusively goes back through all the parents and updates
            the number of simulations and the result
            :param node: The node to update
            :param value: the valDue to yupdate it with
            """
        node.number_of_simulations += 1
        node.score_sum += -value if node.turn != node.org_player else value
        if node.parent:
            self.backpropagate(node.parent, value)

    def simulate(self, board, node):
        size = self.board_size
        image = board.get_board()
        turn = node.turn
        outcome = numpy.zeros((2, size, size))
        for y in range(len(image)):
            for x in range(len(image[0])):
                opponent = 1 if turn - 1 else 2
                if image[y][x] == turn:
                    outcome[0][y][x] = 1
                elif image[y][x] == opponent:
                    outcome[1][y][x] = 1

        outcome = outcome.reshape((-1, 2, size, size))
        res = self.network.predict(outcome)
        val, policy_logits = res[0][0][0], res[1][0]

        state = board.get_board()
        width = len(state)
        actions = board.possible_moves(board.get_turn(node.turn))
        values = [width * a.x + a.y for a in actions]
        policy = {v: (numpy.exp(policy_logits[v]), a) for a, v in zip(actions, values)}
        policy_sum = sum(v for v, _ in policy.values())
        for p, action in policy.values():
            # x = action // width
            # y = action % width
            # print(f"Expecting to move something from {actions[i].x}, {actions[i].y} to {x}, {y}")
            node.children.append(
                NN_Node(0, node.actions + [action], node.org_player, board.get_turn(node.turn),
                        p / policy_sum, node))
        return val

    def puct(self, node: NN_Node, child: NN_Node):
        """
        Upper confidence tree search
        A formular used to ensure both expantation and exploration
        :param node: The node to test on
        :return: The UCT
        """
        prior_score = puct(node.number_of_simulations, child.number_of_simulations,
                           child.prior, Config.PB_C, Config.PB_C_INIT)
        value_score = child.score_sum
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

    def select_child(self, board, node):
        visit_counts = [(child.number_of_simulations, child.actions[-1]) for child in node.children]
        score, action = max(visit_counts, key=lambda child: child[0])
        return action

    def add_exploration_noise(self, node: NN_Node):
        # TODO: Maybe normalize the exploration values
        actions = [i for i in range(len(node.children))]
        noise = numpy.random.gamma(Config.ROOT_ALPHA, 1, len(actions))
        frac = Config.ROOT_EXPLORATION_FRACTION
        for a, n in zip(actions, noise):
            node.children[a].prior *= ((1 - frac) + n * frac)

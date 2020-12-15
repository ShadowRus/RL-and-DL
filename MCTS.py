

import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

# https://github.com/igavriil/two-player-ai/blob/66abcadafc68940ade57fa62f51b68c95fca54a3/src/two_player_ai/alpha_zero/mcts.py
# https://github.com/yanjingang/pigchess/blob/022d2e967c0b8af89b95909ba17308a3f410cb6d/player.py

class MCTSTreeNode(object):

    def __init__(self, env, board, parent=None):
        self.state_env = TicTacToe(env.n_rows, env.n_cols, env.n_win)
        self.state_env.board = copy.deepcopy(board)
        self.state_env.isTerminal()
        self.parent = parent
        self.children = []
        self.n = 0
        self.q = 0
        self.actions = list(self.state_env.getEmptySpaces())

    def best_child(self, c_param=1):
        choices_weights = [
            (-c.q / c.n) + 
                c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return np.argmax(choices_weights)

    def expand(self):
        action = self.actions.pop(0)
        next_board = copy.deepcopy(self.state_env.board)
        next_board[action[0], action[1]] = self.state_env.curTurn
        child_node = MCTSTreeNode(self.state_env, next_board, parent=self)
        child_node.state_env.curTurn = -self.state_env.curTurn
        self.children.append(child_node)
        return child_node

    def rollout(self):
        rollout_env = TicTacToe(self.state_env.n_rows, 
                                self.state_env.n_cols, 
                                self.state_env.n_win)
        rollout_env.board = copy.deepcopy(self.state_env.board)
        rollout_env.curTurn = self.state_env.curTurn
        reward = rollout_env.isTerminal()
        done = rollout_env.gameOver
        random_actions = list(np.random.permutation(rollout_env.getEmptySpaces()))
        while not done:
            action = random_actions.pop()
            observation, reward, done, info = rollout_env.step(action)
        return reward * self.state_env.curTurn

    def backprop(self, result):
        self.n += 1.
        self.q += result
        if self.parent:
            self.parent.backprop(-result)
			
			
			
class MCTSAgent(object):
    def __init__(self, environment, n_simulations):
        self.index_best_action = True
        self.env = environment
        self.n_simulations = n_simulations
        
    def get_best_action(self, s, n_actions):
        board = np.array([float(n) for n in s]).reshape(
            (self.env.n_rows, self.env.n_cols)) - 1
        
        tree_root = MCTSTreeNode(self.env, board)
 
        for _ in range(self.n_simulations):            
            v = tree_root
            while not v.state_env.gameOver:
                if len(v.actions) > 0:
                    v = v.expand()
                else:
                    v = v.children[v.best_child()]
            reward = v.rollout()
            v.backprop(reward)
        return tree_root.best_child(c_param=0)
# -*- coding: utf-8 -*-
"""
Environment крестики-нолики для DoubleDQN

"""



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

# https://github.com/Silvan-M/Game-TARS/tree/a405f4a0146b94fd071e956a6a36672c866ac0ca


class TicTacToeDoubleDQN(TicTacToeDQN):
    def __init__(self, n_rows=4, n_cols=4, n_win=4, model_class=Network_4_4, gamma=0.8, device=device):
        super(TicTacToeDoubleDQN, self).__init__(n_rows=n_rows, n_cols=n_cols, n_win=n_win, model_class=model_class, gamma=gamma, device=device)
        self.target_models = {-1: model_class().to(device), 1: model_class().to(device)}
        self.episodes_learned = {-1: 0, 1: 0}

    def learn(self, cur_turn):
        if np.min([len(self.memories[cur_turn]), len(self.memories[-cur_turn])]) < self.batch_size:
            return
        transitions = self.memories[cur_turn].sample(self.batch_size)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        batch_state = Variable(torch.stack(batch_state).to(self.device))
        batch_action = Variable(torch.cat(batch_action).to(self.device))
        batch_reward = Variable(torch.cat(batch_reward).to(self.device))
        batch_next_state = Variable(torch.stack(batch_next_state).to(self.device))
        Q = self.models[cur_turn](batch_state)
        Q = Q.gather(1, batch_action).reshape([self.batch_size])
        Qmax = self.target_models[cur_turn](batch_next_state).detach()
        Qmax = Qmax.max(1)[0]
        Qnext = batch_reward + (self.gamma * Qmax)
        loss = F.smooth_l1_loss(Q, Qnext)
        self.optimizers[cur_turn].zero_grad()
        loss.backward()
        
        self.optimizers[cur_turn].step()
        
        self.episodes_learned[cur_turn] += 1
        if self.episodes_learned[cur_turn] % 500:
            self.target_models[cur_turn].load_state_dict(self.models[cur_turn].state_dict())
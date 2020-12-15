# -*- coding: utf-8 -*-
"""
Environment крестики-нолики для DQN

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

class TicTacToeDQN():
    def __init__(self, n_rows=4, n_cols=4, n_win=4, model_class=Network_4_4, gamma=0.8, device=device):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.env = TicTacToe(n_rows, n_cols, n_win)
        self.device = device
        self.models = {-1: model_class().to(device), 1: model_class().to(device)}
        self.memories = {-1: ReplayMemory(1000000), 1: ReplayMemory(1000000)}
        self.optimizers = {-1: optim.Adam(self.models[-1].parameters(), lr=0.0001, weight_decay=0.001),
                            1: optim.Adam(self.models[1].parameters(), lr=0.0001, weight_decay=0.001)}
        self.previous_states = {-1: None, 1: None}
        self.previous_actions = {}
        self.steps_done = 0
        
        self.gamma = gamma
        self.batch_size = 512
        
        self.eps_init, self.eps_final, self.eps_decay = 0.9, 0.05, 100000
        self.num_step = 0

    def select_greedy_action(self, state, cur_turn):
        return self.models[cur_turn](state.unsqueeze(0)).data.max(1)[1].view(1, 1)

    def select_action(self, state, cur_turn):
        sample = random.random()
        self.num_step += 1
        eps_threshold = self.eps_final + (self.eps_init - self.eps_final) * math.exp(-1. * self.num_step / self.eps_decay)
        if sample > eps_threshold:
            return self.select_greedy_action(state, cur_turn)
        else:
            return torch.tensor([[random.randrange(self.n_rows * self.n_cols)]], dtype=torch.int64)
        
    def run_episode(self, e=0, do_learning=True, greedy=False):
        self.env.reset()
        self.previous_states = {-1: None, 1: None}
        self.previous_actions = {}
        state, _, cur_turn = self.env.getState()
        while True:
            state_tensor = s_to_tensor(state)
            with torch.no_grad():
                if greedy:
                    action_idx = self.select_greedy_action(state_tensor.to(self.device), cur_turn).cpu()
                else:
                    action_idx = self.select_action(state_tensor.to(self.device), cur_turn).cpu()
            self.previous_states[cur_turn] = state_tensor
            self.previous_actions[cur_turn] = action_idx
            action = self.env.action_from_int(action_idx.numpy()[0][0])
            (next_state, empty_spaces, cur_turn), reward, done, _ = self.env.step(action)
            next_state_tensor = s_to_tensor(next_state)
            if reward == -10:
                transition = (state_tensor, action_idx, next_state_tensor, torch.tensor([reward], dtype=torch.float32))
                self.memories[cur_turn].store(transition)
            else:
                if self.previous_states[cur_turn] is not None:
                    if reward == -cur_turn: 
                        transition = (self.previous_states[-cur_turn], 
                                      self.previous_actions[-cur_turn], 
                                      next_state_tensor, 
                                      torch.tensor([1.0], dtype=torch.float32)
                                     )
                        self.memories[-cur_turn].store(transition)
                    transition = (self.previous_states[cur_turn], 
                                  self.previous_actions[cur_turn], 
                                  next_state_tensor, 
                                  torch.tensor([reward * cur_turn], dtype=torch.float32)
                                 )
                    self.memories[cur_turn].store(transition)

            
            if do_learning:
                self.learn(cur_turn)

            state = next_state

            if done:
                break

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
        Qmax = self.models[cur_turn](batch_next_state).detach()
        Qmax = Qmax.max(1)[0]
        Qnext = batch_reward + (self.gamma * Qmax)
        loss = F.smooth_l1_loss(Q, Qnext)
        self.optimizers[cur_turn].zero_grad()
        loss.backward()
        
        self.optimizers[cur_turn].step()
        
    def test_strategy(self, player, n_episodes=1000):
        rewards = []
        for _ in range(n_episodes):
            self.env.reset()
            state, empty_spaces, cur_turn = self.env.getState()
            done = False
            while not done:
                if cur_turn == player:
                    idx = self.select_greedy_action(s_to_tensor(state).to(device), player)
                    action = self.env.action_from_int(idx)
                else:
                    idx = np.random.randint(len(empty_spaces))
                    action = empty_spaces[idx]
                (state, empty_spaces, cur_turn), reward, done, _ = self.env.step(action)
            if reward != -10:
                rewards.append(reward * player)
            else:
                if cur_turn == player:
                    rewards.append(reward)
        return np.array(rewards)
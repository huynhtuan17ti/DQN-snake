from collections import deque
import numpy as np
import random
import torch
from trainer import Trainer
from utils import calc_state
from typing import Dict

class Agent:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.n_games = 0
        self.memory = deque(maxlen=cfg['max_memory'])
        self.trainer = Trainer(cfg)

    def get_state(self, game):
        return calc_state(game)

    def save_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.cfg['batch_size']:
            mini_batch = random.sample(self.memory, self.cfg['batch_size'])
        else:
            mini_batch = self.memory
        
        mini_batch = zip(*mini_batch)
        self.trainer.train_step(mini_batch)

    def get_action(self, state):
        # eps_ed + (eps_st - eps_ed)*e^(-1.0 * n_games/decay)
        eps_threshold = self.cfg['eps_end'] + (self.cfg['eps_start'] - self.cfg['eps_end']) \
                        * np.exp(-1.0 * self.n_games / self.cfg['eps_decay'])
        
        action = [0, 0, 0]
        rd = np.random.random()
        if rd > eps_threshold:
            with torch.no_grad():
                torch_state = torch.tensor(state, dtype = torch.float).unsqueeze(0)
                qvals = self.trainer.model.forward(torch_state)
                move = np.argmax(qvals.cpu().detach().numpy())
                action[move] = 1
        else:
            move = random.randrange(self.cfg['n_actions'])
            action[move] = 1
        
        return action


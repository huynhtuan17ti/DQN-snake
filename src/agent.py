from collections import deque
import numpy as np
import random
import torch
from trainer import Trainer
from utils import get_device
from typing import Dict
import wandb

class Agent:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.n_games = 0
        self.memory = deque(maxlen=cfg['max_memory'])
        self.trainer = Trainer(cfg)
        self.device = get_device(cfg['device'])
        self.total_loss = 0
        self.n_iters = 0
        wandb.init(project='dqn-snake')

    def save_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.total_loss += self.trainer.train_step((state, action, reward, next_state, done))
        self.n_iters += 1

    def train_long_memory(self):
        if len(self.memory) > self.cfg['batch_size']:
            mini_batch = random.sample(self.memory, self.cfg['batch_size'])
        else:
            mini_batch = self.memory
        
        mini_batch = zip(*mini_batch)
        self.total_loss += self.trainer.train_step(mini_batch)
        self.n_iters += 1
        wandb.log({'loss': self.total_loss/self.n_iters})


    def get_action(self, state):
        # using e-greedy
        eps_threshold = self.cfg['e_greedy']['eps_end'] + (self.cfg['e_greedy']['eps_start'] - self.cfg['e_greedy']['eps_end']) \
                        * np.exp(-1.0 * self.n_games / self.cfg['e_greedy']['eps_decay'])
        
        action = [0, 0, 0]
        rd = np.random.random()
        if rd > eps_threshold:
            with torch.no_grad():
                torch_state = torch.tensor(state, dtype = torch.float).unsqueeze(0).to(self.device)
                qvals = self.trainer.policy_net(torch_state)
                move = np.argmax(qvals.cpu().detach().numpy())
                action[move] = 1
        else:
            move = random.randrange(self.cfg['n_actions'])
            action[move] = 1
        
        return action


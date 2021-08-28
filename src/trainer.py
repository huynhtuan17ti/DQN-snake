import torch
from torch.optim import Adam
from torch import nn
from typing import Dict

class Trainer:
    def __init__(self, model: nn.Module, cfg: Dict) -> None:
        self.lr = cfg['lr']
        self.gamma = cfg['gamma']
        self.model = model
        self.optimizer = Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, torch.float)
        next_state = torch.tensor(next_state, torch.float)

        action = torch.tensor(action, torch.long)
        reward = torch.tensor(reward, torch.float)

        
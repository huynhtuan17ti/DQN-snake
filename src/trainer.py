import torch
from torch.optim import Adam
from torch import nn
from typing import Dict
from utils import get_device
from network import DuelingDQN, LinearQN

class Trainer:
    def __init__(self, cfg: Dict) -> None:
        self.lr = cfg['lr']
        self.gamma = cfg['gamma']
        self.device = get_device(cfg['device'])
        self.model = LinearQN(cfg['input_size'], cfg['output_size']).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()


    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.tensor(states, dtype = torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype = torch.float).to(self.device)

        actions = torch.tensor(actions, dtype = torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype = torch.float).to(self.device)

        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            next_states = torch.unsqueeze(next_states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            dones = (dones, )

        preds = self.model.forward(states)
        targets = preds.clone()

        for idx in range(len(dones)):
            nextQ = rewards[idx]
            if not dones[idx]:
                nextQ = rewards[idx] + self.gamma * torch.max(self.model.forward(next_states[idx]))
            targets[idx][torch.argmax(actions[idx]).item()] = nextQ

        loss = self.criterion(preds, targets)
        return loss


    def train_step(self, batch):
        loss = self.compute_loss(batch)

        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
from torch import nn 
from torch.nn import functional as F
import torch.autograd as autograd
import os
import torch
from typing import Tuple

def save_model(state_dict, file_name: str) -> None:
    model_folder_path = './saved_model'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    file_name = os.path.join(model_folder_path, file_name)
    torch.save(state_dict, file_name)


class DuelingDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.feauture_layer = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )


    def forward(self, x):
        features = self.feauture_layer(x)

        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals

    def save(self, file_name: str = 'model.pth'):
        save_model(self.state_dict(), file_name)
    

class LinearQN(nn.Module):
    def __init__(self, input_dim: Tuple[int, int, int], output_dim: int):
        super().__init__()
        c, h, w = input_dim
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        x = self.online(x)
        return x

    def save(self, file_name: str = 'model.pth'):
        save_model(self.state_dict(), file_name)
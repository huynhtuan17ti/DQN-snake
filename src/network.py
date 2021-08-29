from torch import nn 
from torch.nn import functional as F
import torch.autograd as autograd
import os
import torch

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
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc_adv1 = nn.Linear(3136, 512)
        self.fc_adv2 = nn.Linear(512, output_dim)

        self.fc_val1 = nn.Linear(3136, 512)
        self.fc_val2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        val = F.relu(self.fc_val1(x))
        val = self.fc_val2(val)

        adv = F.relu(self.fc_adv1(x))
        adv = self.fc_adv2(adv)

        return val + adv - adv.mean()

    def save(self, file_name: str = 'model.pth'):
        save_model(self.state_dict(), file_name)
    

class LinearQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name: str = 'model.pth'):
        save_model(self.state_dict(), file_name)
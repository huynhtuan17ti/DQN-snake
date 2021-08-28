from collections import deque
import numpy as np
import random
import torch
from utils import calc_state
from typing import Dict

class Agent:
    def __init__(self, cfg: Dict) -> None:
        self.n_games = 0
        self.gamma = 0.8 # discounted factor
        self.memory = deque(maxlen=cfg['max_memory'])
        self.model = None #TODO: NotImplemented
        self.trainer = None #TODO:NotImplemented

    def get_state(self, game):
        return calc_state(game)

    def save_memory(self, old_state: np.ndarray, action, reward: float, new_state: np.ndarray, done: bool):
        pass

    def train_short_memory(self):
        pass

    def train_long_memory(self):
        pass

    def get_action(self):
        pass

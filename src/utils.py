from game.machine import SnakeGameAI, Direction, Point
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import pygame
from typing import Dict

def get_game_screen(cfg: Dict, game: SnakeGameAI) -> np.ndarray:
    screen = game.get_screen()
    normalize = T.Compose(
        [
            T.ToPILImage(),
            T.Grayscale(1),
            T.Resize(cfg['DQN']['resize_size'], interpolation=Image.BILINEAR),
        ]
    )
    screen = np.rot90(pygame.surfarray.array3d(screen))[::-1]
    screen = np.array(normalize(screen), dtype=np.float32) / 255.0
    return screen


def calc_cur_state(cfg: Dict, game: SnakeGameAI) -> np.ndarray:
    screen = get_game_screen(cfg, game)
    state = np.stack([screen.astype(np.float32) for _ in range(cfg['DQN']['num_channel'])], axis=0)
    return state


def calc_next_state(cfg: Dict, state: np.ndarray, game: SnakeGameAI) -> np.ndarray:
    screen = get_game_screen(cfg, game)
    new_state = np.stack([state[1, :], state[2, :], state[3, :], screen], axis=0)
    return new_state


def get_device(device: str):
    return torch.device(device)


import matplotlib.pyplot as plt
from IPython import display


plt.ion()
def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('N. Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
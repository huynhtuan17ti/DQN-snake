from torchvision.transforms.functional import normalize
from game.machine import SnakeGameAI, Direction, Point, BLOCK_SIZE
import numpy as np
import torch
import torchvision.transforms as T
import pygame
from typing import Dict
from PIL import Image
import cv2

def get_game_screen(cfg: Dict, game: SnakeGameAI) -> np.ndarray:
    screen = game.get_screen()
    normalize = T.Compose([
        T.ToPILImage(),
        T.Grayscale(1),
        T.Resize(cfg['resize_size'], interpolation=Image.BILINEAR)
    ])
    screen = np.rot90(pygame.surfarray.array3d(screen))[::-1]
    screen = normalize(screen)
    return np.array(screen, dtype=np.float32) / 255.0

def get_state(cfg: Dict, game: SnakeGameAI) -> np.ndarray:
    screen = get_game_screen(cfg, game)
    state = np.stack([screen.astype(np.float32) for _ in range(4)], axis=0)
    return state

def get_next_state(state: np.ndarray, cfg: Dict, game: SnakeGameAI) -> np.ndarray:
    screen = get_game_screen(cfg, game)
    state = np.stack([state[1, :], state[2, :], state[3, :], screen], axis=0)
    return state

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
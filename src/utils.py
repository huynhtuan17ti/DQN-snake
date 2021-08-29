from game.machine import SnakeGameAI, Direction, Point
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import pygame
import cv2

cnt = 0

def calc_state(game: SnakeGameAI):
    global cnt
    screen = game.get_screen()
    normalize = T.Compose(
        [
            T.ToPILImage(),
            T.Grayscale(1),
            T.Resize(84, interpolation=Image.BILINEAR),
        ]
    )
    screen = np.rot90(pygame.surfarray.array3d(screen))[::-1]
    cv2.imwrite('img/img{}.png'.format(cnt), screen)
    cnt += 1

    screen = np.array(normalize(screen), dtype=np.float32) / 255.0
    state = np.stack([screen.astype(np.float32) for _ in range(1)], axis=0)
    return state

def calc_state_v1(game: SnakeGameAI) -> np.ndarray:
    '''
        calculate current state of the game base on the snake's position
        state = [danger_straight, danger_right, danger_left] + [Move direction (one-hot encoding)] + [Food location (base on snake)]
        ----------
        Parameters:
            game: SnakeGameAI
        ----------
        Return:
            state: np.ndarray
        ----------
    '''
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)
        
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger straight
        (dir_r and game.is_collision(point_r)) or 
        (dir_l and game.is_collision(point_l)) or 
        (dir_u and game.is_collision(point_u)) or 
        (dir_d and game.is_collision(point_d)),

        # Danger right
        (dir_u and game.is_collision(point_r)) or 
        (dir_d and game.is_collision(point_l)) or 
        (dir_l and game.is_collision(point_u)) or 
        (dir_r and game.is_collision(point_d)),

        # Danger left
        (dir_d and game.is_collision(point_r)) or 
        (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or 
        (dir_l and game.is_collision(point_d)),
            
        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,
            
        # Food location 
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y  # food down
    ]

    return np.array(state, dtype=int)

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
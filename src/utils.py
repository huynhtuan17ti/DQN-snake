from game.machine import SnakeGameAI, Direction, Point, BLOCK_SIZE
import numpy as np
import torch
import math
from typing import Dict

def calc_dist(a: int, b: int, limit: int) -> float:
    '''
        calculate dist and normalize it
    '''
    if a - b < 0:
        return 1 # limit/limit
    return (a - b)/limit


def dist_wall(direction: Direction, pt: Point, game: SnakeGameAI) -> float:
    if pt.x > game.w - BLOCK_SIZE or pt.x < 0 or pt.y > game.h - BLOCK_SIZE or pt.y < 0:
        return 0
    if direction == Direction.RIGHT:
        return (game.w - pt.x)/game.w
    if direction == Direction.LEFT:
        return pt.x/game.w
    if direction == Direction.UP:
        return pt.y/game.h
    if direction == Direction.DOWN:
        return (game.h - pt.y)/game.h


def dist_apple(direction: Direction, pt: Point, game: SnakeGameAI) -> float:
    if game.food == None:
        return 0

    if direction == Direction.RIGHT:
        return calc_dist(game.food.x, pt.x, game.w)
    if direction == Direction.LEFT:
        return calc_dist(pt.x, game.food.x, game.w)
    if direction == Direction.UP:
        return calc_dist(pt.y, game.food.y, game.h)
    if direction == Direction.DOWN:
        return calc_dist(game.food.y, pt.y, game.h)


def dist_itself(direction: Direction, pt: Point, game: SnakeGameAI) -> int:
    if direction == Direction.RIGHT:
        min_dist = 1
        for p in game.snake:
            if p.y == pt.y and p != pt: 
                min_dist = min(min_dist, calc_dist(p.x, pt.x, game.w))
        return min_dist

    if direction == Direction.LEFT:
        min_dist = 1
        for p in game.snake:
            if p.y == pt.y and p != pt: 
                min_dist = min(min_dist, calc_dist(pt.x, p.x, game.w))
        return min_dist

    if direction == Direction.UP:
        min_dist = 1
        for p in game.snake:
            if p.x == pt.x and p != pt: 
                min_dist = min(min_dist, calc_dist(pt.y, p.y, game.h))
        return min_dist

    if direction == Direction.DOWN:
        min_dist = 1
        for p in game.snake:
            if p.x == pt.x and p != pt: 
                min_dist = min(min_dist, calc_dist(p.y, pt.y, game.h))
        return min_dist


def calc_cur_state(cfg: Dict, game: SnakeGameAI) -> np.ndarray:
    head = game.snake[0]
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)
    
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,

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

        # Food location 
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y  # food down        
    ]

    return np.array(state, dtype=np.float32)


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
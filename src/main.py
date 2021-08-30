import numpy as np
from agent import Agent
from game.machine import SnakeGameAI
from typing import Dict
from utils import plot
from utils import calc_cur_state, calc_next_state
import yaml

class Run:
    def __init__(self, cfg: Dict) -> None:
        self.agent = Agent(cfg)
        self.game = SnakeGameAI(cfg)
        self.best_score = 0
        self.cfg = cfg
    
    def train(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0

        while True:
            # get current state
            state = calc_cur_state(self.cfg, self.game)

            # get action
            action = self.agent.get_action(state)

            # apply action and get new state
            reward, done, score = self.game.play_step(action)
            new_state = calc_cur_state(self.cfg, self.game)
            # train short memory
            self.agent.train_short_memory(state, action, reward, new_state, done)

            # saving
            self.agent.save_memory(state, action, reward, new_state, done)

            if done:
                # train long memory
                self.game.reset()
                self.agent.n_games += 1
                self.agent.train_long_memory()

                if score > self.best_score:
                    self.best_score = score
                    self.agent.trainer.policy_net.save()

                if self.agent.n_games % self.cfg['target_update_step'] == 0:
                    print('Update target net!')
                    self.agent.trainer.save_target_net()

                print('Game:', self.agent.n_games, 'Score:', score, 'Best:', self.best_score)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / self.agent.n_games
                plot_mean_scores.append(mean_score)

                plot(plot_scores, plot_mean_scores)
            
if __name__ == '__main__':
    cfg = yaml.safe_load(open('config.yaml', 'r'))
    run = Run(cfg)
    run.train()
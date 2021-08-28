from agent import Agent
from game.machine import SnakeGameAI
from typing import Dict

class Run:
    def __init__(self, cfg: Dict) -> None:
        self.agent = Agent()
        self.game = SnakeGameAI()
        self.n_iters = cfg['num_iters']
    
    def train(self):
        for iter in range(self.n_iters):
            # get old state
            old_state = self.agent.get_state(self.game)

            # get action
            action = self.agent.get_action()

            # apply action and get new state
            reward, done, score = self.game.play_step(action)
            new_state = self.agent.get_state(self.game)

            # train short memory
            self.agent.train_short_memory(old_state, action, reward, new_state, done)

            # saving
            self.agent.save_memory(old_state, action, reward, new_state, done)

            if done:
                # train long memory
                self.game.reset()
                self.agent.n_games += 1
                self.agent.train_long_memory()

            # TODO: plot and print the score

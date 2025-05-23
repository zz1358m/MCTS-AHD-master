import numpy as np
import pickle
import sys

from gpt import choose_action as choose_action


from typing import Any
import gym


def evaluate(seed) -> float:
    """Evaluate heuristic function on car mountain problem."""
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 500
    observation, _ = env.reset(seed= seed)  # initialization
    action = 1  # initial action, stay static

    for i in range(env._max_episode_steps):
        action = choose_action(observation[0], observation[1], action)
        observation, reward, done, truncated, info = env.step(action)

        if done:
            return (i / env._max_episode_steps)  # succeed

        if truncated:
            return (max(0.5 - observation[0], 0) + 1)  # failed



if __name__ == "__main__":
    import os
    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2] # reserved for compatibility
    mood = sys.argv[3]
    score = 0
    for seed in range(10):
        score += evaluate(seed)

    print("[*] Average:")
    print(score/10)
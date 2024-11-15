from __future__ import annotations

import glob
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

from pettingzoo.butterfly import knights_archers_zombies_v10


class RewardLoggerCallback(BaseCallback):
    """
    Callback pour enregistrer les récompenses moyennes et générer une courbe d'apprentissage.
    """

    def __init__(self, check_freq: int, verbose: int = 0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []
        self.max_rewards = []

    def _on_step(self) -> bool:
        # Ajouter la récompense moyenne toutes les `check_freq` étapes
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.locals["rollout_buffer"].rewards)
            self.rewards.append(mean_reward)
            max_reward = np.max(self.locals["rollout_buffer"].rewards)
            self.max_rewards.append(max_reward)
        return True

    def plot_rewards(self):
        plt.plot(self.rewards)
        plt.title("Courbe d'apprentissage")
        plt.xlabel(f"Step (x{format(self.check_freq)})")
        plt.ylabel("Reward moyenne")
        plt.grid()
        plt.show()

    def plot_max_rewards(self):
        plt.plot(self.max_rewards)
        plt.title("Courbe d'apprentissage (Reward Max)")
        plt.xlabel(f"Step (x{format(self.check_freq)})")
        plt.ylabel("Reward maximale")
        plt.grid()
        plt.show()


def train(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    # Train a single model to play as each agent in an AEC environment
    env = env_fn.parallel_env(**env_kwargs)

    # Add black death wrapper so the number of agents stays constant
    # MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True
    env = ss.black_death_v3(env)

    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    # Use a CNN policy if the observation space is visual
    model = PPO(
        CnnPolicy if visual_observation else MlpPolicy,
        env,
        verbose=3,
        batch_size=512,
        n_steps=1024,
    )

    reward_logger = RewardLoggerCallback(check_freq=10)

    model.learn(total_timesteps=steps, callback=reward_logger)
    model.save(
        f"trainning_saves/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    )

    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    # Afficher la courbe d'apprentissage
    reward_logger.plot_rewards()
    reward_logger.plot_max_rewards()

    env.close()


if __name__ == "__main__":
    env_fn = knights_archers_zombies_v10

    # Set vector_state to false in order to use visual observations (significantly longer training time)
    env_kwargs = dict(max_cycles=100, max_zombies=4, vector_state=True)

    # Train a model (takes ~5 minutes on a laptop CPU)
    train(env_fn, steps=1_000_000, seed=0, **env_kwargs)

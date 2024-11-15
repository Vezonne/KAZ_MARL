from __future__ import annotations

import glob
import os
import time

import numpy as np
from pettingzoo.utils import BaseParallelWrapper
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


class DistanceRewardWrapper(BaseParallelWrapper):
    def __init__(
        self, env, target_type="zombie", distance_threshold=5, reward_bonus=0.1
    ):
        super().__init__(env)
        self.target_type = (
            target_type  # Type de cible pour la distance, par exemple "zombie"
        )
        self.distance_threshold = (
            distance_threshold  # Distance pour laquelle la récompense est maximisée
        )
        self.reward_bonus = reward_bonus  # Bonus de récompense basé sur la distance

    def step(self, action):
        # Exécute l'action et récupère les résultats de base
        obs, reward, termination, truncation, info = self.env.step(action)

        # Appliquer le bonus de récompense de distance
        agent_position = self.get_agent_position(self.env.agent_selection)
        target_positions = self.get_target_positions(self.target_type)

        # Calculer le bonus de récompense basé sur la distance
        if target_positions:
            closest_distance = min(
                np.linalg.norm(agent_position - pos) for pos in target_positions
            )
            if closest_distance <= self.distance_threshold:
                # Ajoute un bonus proportionnel à la proximité de la cible
                reward += (
                    self.reward_bonus
                    * (self.distance_threshold - closest_distance)
                    / self.distance_threshold
                )

        return obs, reward, termination, truncation, info

    def get_agent_position(self, agent):
        # Récupère la position de l'agent depuis l'observation
        # (Supposant que l'observation contient la position, par ex. un tableau [x, y])
        obs = self.env.observe(agent)
        if isinstance(obs, dict) and "position" in obs:
            return np.array(obs["position"])
        else:
            raise ValueError(
                "L'observation de l'agent ne contient pas d'information de position"
            )

    def get_target_positions(self, target_type):
        # Récupère les positions des cibles, par exemple des zombies
        # (Supposant que les observations de l'environnement contiennent les positions des zombies)
        target_positions = []
        for agent in self.env.agents:
            if target_type in agent:  # Ex: 'zombie' dans 'zombie_1'
                obs = self.env.observe(agent)
                if isinstance(obs, dict) and "position" in obs:
                    target_positions.append(np.array(obs["position"]))
        return target_positions


class FilterAgentWrapper(BaseParallelWrapper):
    def __init__(self, env, agent_type="archer", nb_agents=2):
        super().__init__(env)
        self.agent_type = agent_type
        self.filtered_agents = {
            f"{self.agent_type}_{i}" for i in range(nb_agents)
        }  # Utiliser un ensemble pour vérification rapide

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # Retourner les observations de tous les agents
        return obs, info

    def step(self, actions):
        # Filtrer les actions pour inclure uniquement celles des agents filtrés
        filtered_actions = {
            agent: actions[agent] if agent in self.filtered_agents else 0
            for agent in self.env.agents
        }

        # Exécuter les actions filtrées
        obs, reward, termination, truncation, info = super().step(filtered_actions)

        # Filtrer les sorties pour ne retourner que les données des agents sélectionnés
        # obs = {agent: obs[agent] for agent in self.filtered_agents if agent in obs}
        # reward = {agent: reward.get(agent, 0) for agent in self.filtered_agents}
        # termination = {
        #     agent: termination.get(agent, False) for agent in self.filtered_agents
        # }
        # truncation = {
        #     agent: truncation.get(agent, False) for agent in self.filtered_agents
        # }
        # info = {agent: info.get(agent, {}) for agent in self.filtered_agents}

        return obs, reward, termination, truncation, info


class SharedRewardWrapper(BaseParallelWrapper):
    def __init__(self, env, share_fraction=0.1):
        super().__init__(env)
        self.share_fraction = share_fraction  # Fraction de la récompense à partager avec les autres agents

    def step(self, action):
        # Effectuer une étape dans l'environnement original
        obs, rewards, terminations, truncations, infos = self.env.step(action)

        # Calculer les récompenses partagées
        updated_rewards = rewards.copy()

        for agent, reward in rewards.items():
            if reward > 0:  # Seuls les gains positifs sont partagés
                shared_reward = reward * self.share_fraction
                for other_agent in self.env.agents:
                    if other_agent != agent:  # Ne pas inclure l'agent lui-même
                        updated_rewards[other_agent] += shared_reward

        return obs, updated_rewards, terminations, truncations, infos


def train_shared_reward(
    env_fn,
    steps: int = 10_000,
    seed: int | None = 0,
    share_fraction: float = 0.1,
    **env_kwargs,
):
    # Train a single model to play as each agent in an AEC environment
    env = env_fn.parallel_env(**env_kwargs)

    # Appliquer le wrapper de récompense partagée
    env = SharedRewardWrapper(env, share_fraction=share_fraction)

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
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")

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
        f"trainning_multi_saves/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    )

    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    # Afficher la courbe d'apprentissage
    reward_logger.plot_rewards()
    reward_logger.plot_max_rewards()

    env.close()


def train_archers(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    # Environnement pour les archers
    env = env_fn.parallel_env(**env_kwargs)
    env = FilterAgentWrapper(env, agent_type="archer")
    env = ss.black_death_v3(env)

    # Pré-traitement pour observations visuelles
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    # Vérification avant le reset

    env.reset(seed=seed)

    print(f"Starting archers training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")

    # Entraîner le modèle PPO pour les archers
    archer_model = PPO(
        CnnPolicy if visual_observation else MlpPolicy,
        env,
        verbose=3,
        batch_size=512,
        n_steps=1024,
    )

    # Callback pour suivre les récompenses
    reward_logger = RewardLoggerCallback(check_freq=10)

    archer_model.learn(total_timesteps=steps, callback=reward_logger)
    archer_model.save(f"trainning_multi_saves/archer_model")

    print("Model has been saved.")
    print(f"Finished archers training on {str(env.unwrapped.metadata['name'])}.")

    # Afficher la courbe d'apprentissage
    reward_logger.plot_rewards()
    reward_logger.plot_max_rewards()

    env.close()


def train_knights(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    # Environnement pour les archers
    env = env_fn.parallel_env(**env_kwargs)
    env = FilterAgentWrapper(env, agent_type="knight")
    env = ss.black_death_v3(env)

    # Pré-traitement pour observations visuelles
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    env.reset(seed=seed)

    print(f"Starting knights training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")

    # Entraîner le modèle PPO pour les archers
    archer_model = PPO(
        CnnPolicy if visual_observation else MlpPolicy,
        env,
        verbose=3,
        batch_size=512,
        n_steps=1024,
    )

    # Callback pour suivre les récompenses
    reward_logger = RewardLoggerCallback(check_freq=10)

    archer_model.learn(total_timesteps=steps, callback=reward_logger)
    archer_model.save(f"trainning_multi_saves/knight_model")

    print("Model has been saved.")
    print(f"Finished knights training on {str(env.unwrapped.metadata['name'])}.")

    # Afficher la courbe d'apprentissage
    reward_logger.plot_rewards()
    reward_logger.plot_max_rewards()

    env.close()


if __name__ == "__main__":
    # Créer l'environnement et appliquer le wrapper de récompense de distance
    base_env = knights_archers_zombies_v10

    env_kwargs = dict(max_cycles=100, max_zombies=4, vector_state=True)

    # Train a model for all agents
    train_shared_reward(
        base_env, steps=1_000_000, seed=0, share_fraction=0.1, **env_kwargs
    )

    # Train a model for archers
    train_archers(base_env, steps=1_000_000, seed=0, **env_kwargs)

    # Train a model for knights
    train_knights(base_env, steps=1_000_000, seed=0, **env_kwargs)

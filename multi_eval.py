from __future__ import annotations

import glob
import os
from tqdm import tqdm

import numpy as np
from pettingzoo.utils.conversions import parallel_to_aec

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

from pettingzoo.butterfly import knights_archers_zombies_v10

from multi_train import SharedRewardWrapper


def eval_shared_reward(
    env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs
):
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    # env = parallel_to_aec(SharedRewardWrapper(env, share_fraction=0.1))

    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"trainning_multi_saves/{env.metadata['name']}*.zip"),
            key=os.path.getctime,
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    for i in tqdm(range(num_games), desc="Evaluating games"):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                # if agent == env.possible_agents[0]:
                #     act = env.action_space(agent).sample()
                # else:
                #     act = model.predict(obs, deterministic=True)[0]
                act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }

    print(f"Avg rewards: \t{avg_reward_per_agent}")
    print("Full rewards: \t", rewards)
    return avg_reward


def eval_random_agents(env_fn, num_games: int = 100, **env_kwargs):
    """
    Évalue les performances de tous les agents en suivant un comportement aléatoire.

    Arguments:
    - env_fn: fonction de création de l'environnement PettingZoo (ex. knights_archers_zombies_v10).
    - num_games: nombre de parties à jouer pour l'évaluation.
    - env_kwargs: paramètres supplémentaires pour l'environnement.

    Retourne:
    - avg_rewards: dictionnaire contenant la moyenne des récompenses par agent.
    - std_rewards: dictionnaire contenant l'écart-type des récompenses par agent.
    """
    # Créer l'environnement
    env = env_fn.env(**env_kwargs)

    # env = parallel_to_aec(SharedRewardWrapper(env, share_fraction=0.1))

    # Pré-traitement pour les observations visuelles, si nécessaire
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    print(
        f"\nStarting random evaluation on {str(env.metadata['name'])} (num_games={num_games})"
    )

    # Initialiser les récompenses
    rewards = {agent: 0 for agent in env.possible_agents}

    # Boucle sur le nombre de parties à jouer
    for i in tqdm(range(num_games), desc="Evaluating random agents"):
        env.reset(seed=i)

        # Boucle de jeu en utilisant des actions aléatoires
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Vérifiez si l'agent est "mort" (terminé) et passez une action `None` si c'est le cas
            if termination or truncation:
                env.step(None)
            else:
                # Utiliser une action aléatoire lorsque l'agent est actif
                action = env.action_space(agent).sample()
                env.step(action)

            # Enregistrer la récompense pour chaque agent
            for a in env.agents:
                rewards[a] += env.rewards[a]

    env.close()

    # Calcul des statistiques par agent
    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }

    # Affichage des résultats
    print(f"Avg rewards: \t{avg_reward_per_agent}")
    print("Full rewards: \t", rewards)

    # Visualisation de la distribution des récompenses
    # for agent in env.possible_agents:
    #     plt.hist(rewards[agent], bins=20, alpha=0.6, label=f"{agent}")
    # plt.title("Reward Distribution per Agent (Random Actions)")
    # plt.xlabel("Reward")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.show()

    return avg_reward


def eval_archers(
    env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs
):
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    print(
        f"\nStarting archers evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        archer_policy = max(
            glob.glob(f"trainning_multi_saves/archer_model*.zip"),
            key=os.path.getctime,
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    archer_model = PPO.load(archer_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Action neutre pour les non-archers
    no_action_dict = {
        agent: 0 for agent in env.possible_agents if "archer" not in agent
    }

    # Boucle sur le nombre de parties à jouer
    for i in tqdm(range(num_games), desc="Evaluating archers"):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            elif "archer" in agent:
                action, _ = archer_model.predict(obs, deterministic=True)
            else:
                action = no_action_dict[agent]
            env.step(action)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }

    print(f"Avg rewards: \t{avg_reward_per_agent}")
    print("Full rewards: \t", rewards)
    return avg_reward


def eval_knights(
    env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs
):
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    print(
        f"\nStarting knights evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        knight_policy = max(
            glob.glob(f"trainning_multi_saves/knight_model*.zip"),
            key=os.path.getctime,
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    knight_model = PPO.load(knight_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Action neutre pour les non-knights
    no_action_dict = {
        agent: 0 for agent in env.possible_agents if "knight" not in agent
    }

    # Boucle sur le nombre de parties à jouer
    for i in tqdm(range(num_games), desc="Evaluating knights"):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            elif "knight" in agent:
                action, _ = knight_model.predict(obs, deterministic=True)
            else:
                action = no_action_dict[agent]
            env.step(action)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }

    print(f"Avg rewards: \t{avg_reward_per_agent}")
    print("Full rewards: \t", rewards)
    return avg_reward


def eval_multi(
    env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs
):
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    print(
        f"\nStarting multi evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        archer_policy = max(
            glob.glob(f"trainning_multi_saves/archer_model*.zip"),
            key=os.path.getctime,
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    archer_model = PPO.load(archer_policy)

    try:
        knight_policy = max(
            glob.glob(f"trainning_multi_saves/knight_model*.zip"),
            key=os.path.getctime,
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    knight_model = PPO.load(knight_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Boucle sur le nombre de parties à jouer
    for i in tqdm(range(num_games), desc="Evaluating multi policys"):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            elif "archer" in agent:
                action, _ = archer_model.predict(obs, deterministic=True)
            elif "knight" in agent:
                action, _ = knight_model.predict(obs, deterministic=True)
            env.step(action)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }

    print(f"Avg rewards: \t{avg_reward_per_agent}")
    print("Full rewards: \t", rewards)
    return avg_reward


if __name__ == "__main__":
    print("Evaluation for multi trainning:")

    # Créer l'environnement et appliquer le wrapper de récompense de distance
    env_fn = knights_archers_zombies_v10

    env_kwargs = dict(max_cycles=100, max_zombies=4, vector_state=True)

    # Evaluate 1000 random games
    eval_random_agents(env_fn, num_games=1000, **env_kwargs)

    # Evaluate 1000 games (takes ~10 seconds on a laptop CPU)
    eval_shared_reward(env_fn, num_games=1000, render_mode=None, **env_kwargs)

    # Evaluate 1000 games with trained archers
    eval_archers(env_fn, num_games=1000, render_mode=None, **env_kwargs)

    # Evaluate 1000 games with trained knights
    eval_knights(env_fn, num_games=1000, render_mode=None, **env_kwargs)

    # Evaluate 1000 games
    eval_multi(env_fn, num_games=1000, render_mode=None, **env_kwargs)

    # Watch 2 games (takes ~10 seconds on a laptop CPU)
    eval(env_fn, num_games=2, render_mode="human", **env_kwargs)

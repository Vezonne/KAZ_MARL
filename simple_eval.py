from __future__ import annotations

import glob
import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

from pettingzoo.butterfly import knights_archers_zombies_v10


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

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
            glob.glob(f"trainning_saves/{env.metadata['name']}*.zip"),
            key=os.path.getctime,
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(num_games):
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
    for i in range(num_games):
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

    return avg_reward


if __name__ == "__main__":
    print("Evaluation for simple trainning:")

    env_fn = knights_archers_zombies_v10

    # Set vector_state to false in order to use visual observations (significantly longer training time)
    env_kwargs = dict(max_cycles=100, max_zombies=4, vector_state=True)

    # Evaluate 1000 random games
    eval_random_agents(env_fn, num_games=1000, **env_kwargs)

    # Evaluate 1000 games
    eval(env_fn, num_games=1000, render_mode=None, **env_kwargs)

    # Watch 2 games (takes ~10 seconds on a laptop CPU)
    eval(env_fn, num_games=2, render_mode="human", **env_kwargs)

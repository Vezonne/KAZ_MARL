L’objectif de ce projet a été d’explorer l’apprentissage par renforcement dans un environ-
nement multi-agent en utilisant des approches personnalisées pour entraîner des agents
distincts et observer leur comportement collectif. 
Nous avons choisi de travailler avec
l’environnement Knights-Archers-Zombies (KAZ), une simulation multijoueurs de type
"PettingZoo" qui permet d’expérimenter des situations de coopération et de compétition
entre agents.
Pour l’entraînement des agents, nous utilisons Stable-Baselines3 avec l'algorithme Proximal Policy Optimization (PPO).

Il y a deux types d'éxecutions: simple ou multi.
L'éxecution simple permet d'entrainner les agents sans politiques supplémentaires et d'évaluer leurs apprentissage.
l'éxecution multi permet d'entrainner les agents dans un premier temps avec des récompenses partagées puis séparément et de les évaluer.

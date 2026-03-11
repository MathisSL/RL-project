import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import random
import pygame
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

global list_reward
list_reward = []
# Création de l'environnement personnalisé
class ThermalNavigationEnv(gym.Env): # Héritage de la classe gym.Env pour créer un environnement personnalisé
    """
    Environnement Gym personnalisé pour la navigation avec contraintes thermiques, obstacle et goal.
    """
    def __init__(self, grid_size=10, max_temp=40): # Initialisation de l'environnement avec une grille de 10x10 et une température maximale de 100°C pour l'environement
        super(ThermalNavigationEnv, self).__init__()
        
        # Initialiser Pygame mixer pour le son
        pygame.mixer.init()
        pygame.display.init()  # Initialize Pygame display
        self.screen = pygame.display.set_mode((grid_size * 50, grid_size * 50))  # Create display surface
        # Charger l'image de l'agent
        self.agent_image = pygame.image.load(r"C:\Users\mathi\OneDrive\Bureau\A faire\Prog_sys_emb\robot_head.png")  # Remplacez par le chemin de votre image
        self.agent_image = pygame.transform.scale(self.agent_image, (40, 40))

        # Charger l'image du black hole
        self.hole_image = pygame.image.load(r"C:\Users\mathi\OneDrive\Bureau\A faire\Prog_sys_emb\black_hole.png")  # Remplacez par le chemin de votre image
        self.hole_image = pygame.transform.scale(self.hole_image, (50, 50))

        # Charger le son pour le trou noir
        self.hole_sound = pygame.mixer.Sound(r"C:\Users\mathi\OneDrive\Bureau\A faire\Prog_sys_emb\bang-140381.mp3")

        # Charger le son pour l'objectif
        self.goal_sound = pygame.mixer.Sound(r"C:\Users\mathi\OneDrive\Bureau\A faire\Prog_sys_emb\cinematic-designed-sci-fi-whoosh-transition-nexawave-228295.mp3")


        # Taille de la grille et paramètres thermiques
        self.grid_size = grid_size
        self.max_temp = max_temp # Température maximale de l'environnement
        self.current_temperature = 20 # Température aléatoire de l'agent
        self.time_spent = 0  # Initialiser le temps passé
        self.agent_temp_history = []  # Initialiser l'historique des températures de l'agent

        # Définir l'espace d'actions et d'observations en utilisant la classe spaces.Box et spaces.Discrete de Gymnasium
        '''
        crée un espace d'observation continu pour l'environnement, où chaque cellule de la grille contient deux valeurs : 
        la température et la position, avec des valeurs comprises entre 0 et la température maximale.
        '''
        self.action_space = spaces.Discrete(4)  # 0: Haut, 1: Bas, 2: Gauche, 3: Droite => 4 actions
        self.observation_space = spaces.Box( 
            low=0, high=max_temp,  # La valeur minimale pour chaque dimension de l'espace d'observation est 0, la valeur maximale est la température maximale
            shape=(grid_size, grid_size, 2),  # Température + position
            dtype=np.float32
        )

        # Position de départ et zone d'objectif
        self.agent_position = [0, 0] # Position de départ en haut à gauche
        self.goal_position = [grid_size - 1, grid_size - 1] # Position de l'objectif en bas à droite

        # Modélisation de la température ambiante dans différentes zones
        self.temperature_map = self._generate_temperature_map()

        # Initialiser Pygame
        self.cell_size = 50 # Taille de la cellule
        self.screen_size = self.grid_size * self.cell_size # Taille de l'écran
        pygame.init() # Initialiser Pygame
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size)) # Taille de la fenêtre
        pygame.display.set_caption("Projet système embarqué") # Titre de la fenêtrea
        self.clock = pygame.time.Clock() # Horloge pour contrôler la vitesse de rendu

    def _generate_temperature_map(self):
        """
        Génère une carte de température aléatoire pour l'environnement.
        """
        # Fixer la seed pour la reproductibilité
        np.random.seed(42)
        return np.random.uniform(20, 50, (self.grid_size, self.grid_size)) # Température aléatoire entre 20 et 50°C

    def reset(self):
        """
        Réinitialise l'environnement à son état initial.
        """
        self.agent_position = [0, 0] # RAZ de la position de l'agent
        self.agent_temp_history = []  # Réinitialiser l'historique des températures
        return self._get_observation()

    def _get_observation(self):
        """
        Retourne l'observation actuelle de l'environnement.
        """
        obs = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float32) # Initialisation de l'observation
        obs[:, :, 0] = self.temperature_map # Température ambiante de la zone
        obs[self.agent_position[0], self.agent_position[1], 1] = 1 # Position de l'agent 0 = x et 1 = y
        return obs 
    def _update_agent_temperature(self, zone_temp, dt=1): 
        """
        Met à jour la température interne de l'agent.
        
        :param zone_temp: Température ambiante de la zone actuelle.
        :param dt: Intervalle de temps écoulé.
        """

        # Température de la zone actuelle
        zone_temp = self.temperature_map[self.agent_position[0], self.agent_position[1]] 
        # Mise à jour de la température interne
        self._update_agent_temperature(zone_temp, dt=1)  # dt à ajuster voir si on le garde à 1


        # Ajouter la température actuelle à l'historique
        self.agent_temp_history.append(self.current_temperature)


    def step(self, action):
        """
        Effectue une action et retourne les résultats.
        """
        global list_reward
        # Mouvement de l'agent
        if action == 0:  # Haut
            self.agent_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 1:  # Bas
            self.agent_position[0] = min(self.grid_size - 1, self.agent_position[0] + 1)
        elif action == 2:  # Gauche
            self.agent_position[1] = max(0, self.agent_position[1] - 1)
        elif action == 3:  # Droite
            self.agent_position[1] = min(self.grid_size - 1, self.agent_position[1] + 1)

        # Mise à jour de la température interne
        zone_temp = self.temperature_map[self.agent_position[0], self.agent_position[1]] # Température de la zone actuelle
        self.time_spent += 1  # Incrémenter le temps passé
        self.current_temperature += (zone_temp - self.current_temperature) * 0.1 * 1  # Ajouter le temps passé au facteur de mise à jour
        self.agent_temp_history.append(self.current_temperature)  # Ajouter la température de l'agent à l'historique

        # Calcul de la récompense
        done = self.agent_position == self.goal_position
        distance_to_goal = np.linalg.norm(
            np.array(self.agent_position) - np.array(self.goal_position)
        )
    
        reward = 0
    
        # Recompensas
        if done:
            reward += 200  # Reward pour avoir atteint l'objectif
            pygame.mixer.Sound.play(self.goal_sound) # Jouer le son de l'objectif atteint
        else:
            # Reward pour se rapprocher de l'objectif
            reward += 10 * (1 / (1 + distance_to_goal))  
            # 
            reward -= abs(zone_temp - 37) * 0.1
    
        # Reward négative si la température dépasse la température maximale
        if self.current_temperature > self.max_temp:
            reward -= (self.current_temperature - 40) * 0.5 # Penalisation si la température dépasse 40°C
    
        # Reward négatif si l'agent tombe dans le trou noir
        if self.agent_position == [4, 4]:
            reward -= 100
            pygame.mixer.Sound.play(self.hole_sound)
            done = True
    
        # Pénalité de temps
        reward -= 0.1
    
        list_reward.append(reward)
        return self._get_observation(), reward, done, list_reward, {}


    def render(self, mode='human'):
        """
        Visualisation de l'environnement avec Pygame.
        """
        if not hasattr(self, 'screen') or self.screen is None:
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.grid_size * 50, self.grid_size * 50))
        
        if pygame.display.get_init():
            self.screen.fill((255, 255, 255))  # Fond blanc
            # ...existing rendering code...
            pygame.display.flip()
        else:
            print("Pygame display is not initialized or has been closed.")

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Couleur en fonction de la température (bleu = froid, rouge = chaud)
                temp = self.temperature_map[i, j]
                color = (
                    int(255 * min(1, (temp - 20) / 30)),  # Rouge
                    0,  # Vert
                    int(255 * max(0, 1 - (temp - 20) / 30))  # Bleu
                )
                pygame.draw.rect(
                    self.screen,
                    color,
                    (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                )

                # Bordures des cellules
                pygame.draw.rect(
                    self.screen,
                    (200, 200, 200),
                    (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size),
                    1
                )

        # Dessiner le trou noir
        black_hole_x = 200
        black_hole_y = 200
        self.screen.blit(self.hole_image, (black_hole_x, black_hole_y))


        # Dessiner l'objectif
        goal_color = (0, 255, 0)  # Vert
        pygame.draw.rect(
            self.screen,
            goal_color,
            (self.goal_position[1] * self.cell_size, self.goal_position[0] * self.cell_size, self.cell_size, self.cell_size)
        )

        # Dessiner l'agent (image)
        agent_x = self.agent_position[1] * self.cell_size + self.cell_size // 4
        agent_y = self.agent_position[0] * self.cell_size + self.cell_size // 4
        self.screen.blit(self.agent_image, (agent_x, agent_y))


        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()

    def plot(self):
        # plot de la température de l'agent
        plt.plot(self.agent_temp_history)
        plt.axhline(y=30, color='r', linestyle='--', label='Température limite')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.title('Temperature of the agent over time')
        plt.grid()
        plt.show()

# Créer le modèle DQN
class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(np.prod(input_dim), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Applatir les observations
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
learning_rate = 0.001
batch_size = 64
memory_size = 10000

# Créer l'environnement
env = ThermalNavigationEnv(grid_size=10, max_temp=100)
state_size = env.observation_space.shape
action_size = env.action_space.n

# Initialiser le modèle et les outils d'entraînement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(state_size, action_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Mémoire d'expérience
memory = deque(maxlen=memory_size)

# Entraînement de l'agent avec DQN
def train_dqn(env, num_episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, lr=0.001, batch_size=64):
    # Initialiser le réseau Q et le réseau cible
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Mémoire pour le replay
    replay_memory = deque(maxlen=10000)

    epsilon = epsilon_start
    rewards_history = []
    final_temperature_history = []
    exploration_exploitation_history = []
    exploration_count = 0 # Initialisation du compteur d'exploration
    exploitation_count = 0 # Initialisation du compteur d'exploitation

    for episode in range(num_episodes):
        state = env.reset()
        state = state.flatten()
        episode_reward = 0
        done = False

        while not done:
            # Choisir une action
            if random.random() < epsilon:
                action = env.action_space.sample()
                exploration_count += 1
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = torch.argmax(policy_net(state_tensor)).item()
                    exploitation_count += 1

            # Effectuer l'action
            next_state, reward, done, _, _ = env.step(action)
            next_state = next_state.flatten()

            # Ajouter la transition à la mémoire
            replay_memory.append((state, action, reward, next_state, done))

            state = next_state
            episode_reward += reward

            # Entraîner le réseau
            if len(replay_memory) >= batch_size:
                batch = random.sample(replay_memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                # Calculer la cible Q
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)

                # Calculer les Q-values prédites
                current_q_values = policy_net(states).gather(1, actions).squeeze()

                # Calculer la perte
                loss = loss_fn(current_q_values, target_q_values)

                # Optimiser le modèle
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Mettre à jour epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Mettre à jour le réseau cible périodiquement
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        final_temperature_history.append(env.current_temperature)
        rewards_history.append(episode_reward)
        exploration_exploitation_history.append((exploration_count, exploitation_count))
        print(f"\rEpisode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}", end="")

    return policy_net, rewards_history, final_temperature_history, exploration_exploitation_history

# Utilisation de l'environnement et du DQN
env = ThermalNavigationEnv()
trained_policy, rewards, temperature, exploration_exploitation_history = train_dqn(env)

# Affichage de la température de l'agent final au fil des épisodes
plt.plot(temperature)
plt.axhline(y=40, color='r', linestyle='--', label='Température limite')
plt.xlabel('Episode')
plt.ylabel('Final Temperature')
plt.title("Température de l'agent en fonction des épisodes")
plt.show()

# Affichage des récompenses cumulées
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards au fil des épisodes')
plt.show()

# Sauvegarde du modèle
torch.save(model.state_dict(), "dqn_model.pth")
print("Modèle sauvegardé sous 'dqn_model.pth'.")

# Tracer du graphique d'exploration/exploitation
exploration_counts = [x[0] for x in exploration_exploitation_history] # Récupération des valeurs d'exploration
exploitation_counts = [x[1] for x in exploration_exploitation_history] # Récupération des valeurs d'exploitation

plt.figure(figsize=(12, 6))
plt.plot(exploration_counts, label='Exploration')
plt.plot(exploitation_counts, label='Exploitation')
plt.xlabel('Episodes')
plt.ylabel('Count')
plt.title('Exploration vs Exploitation')
plt.legend()
plt.show()
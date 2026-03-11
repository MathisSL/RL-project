import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import pygame
import matplotlib.pyplot as plt
import time
from tqdm import tqdm



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
        plt.axhline(y=40, color='r', linestyle='--', label='Température limite')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.title('Temperature of the agent over time')
        plt.grid()
        plt.show()

        
def sarsa(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    SARSA: On-policy TD control algorithm.
    
    Args:
        env: Environnement Gym personnalisé.
        num_episodes: Nombre total d'épisodes d'entraînement.
        alpha: Taux d'apprentissage.
        gamma: Facteur de discount.
        epsilon: Probabilité d'explorer au lieu d'exploiter.
    
    Returns:
        Q: Table des valeurs d'action.
        rewards_per_episode: Liste des récompenses obtenues par épisode.
    """
    # Initialisation de la table Q pour toutes les actions possibles à toutes les positions
    Q = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    
    # Stocker les récompenses par épisode
    rewards_per_episode = []
    
    for episode in tqdm(range(num_episodes), desc="Training SARSA"):
        # Réinitialiser l'environnement et obtenir l'état initial
        state = env.reset()
        x, y = env.agent_position
        action = choose_action(Q, x, y, epsilon, env.action_space.n)
        
        total_reward = 0
        done = False
        
        while not done:
            # Effectuer une action et observer la récompense et le nouvel état
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            next_x, next_y = env.agent_position
            
            # Choisir la prochaine action selon la politique ε-greedy
            next_action = choose_action(Q, next_x, next_y, epsilon, env.action_space.n)
            
            # Mettre à jour la valeur de Q pour (state, action)
            Q[x, y, action] += alpha * (
                reward + gamma * Q[next_x, next_y, next_action] - Q[x, y, action]
            )
            
            # Passer à l'état suivant
            x, y, action = next_x, next_y, next_action
            
        rewards_per_episode.append(total_reward)
    
    return Q, rewards_per_episode

def choose_action(Q, x, y, epsilon, num_actions):
    """
    Choisit une action selon la politique ε-greedy.
    """
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(num_actions))  # Explorer
    else:
        return np.argmax(Q[x, y])  # Exploiter

def test_agent(env, Q):
    """
    Teste l'agent en utilisant la table de valeurs Q apprise.
    """
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        x, y = env.agent_position
        action = np.argmax(Q[x, y])
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.1)
    
    return total_reward

if __name__ == "__main__":
    env = ThermalNavigationEnv(grid_size=10, max_temp=40)
    Q, rewards_per_episode = sarsa(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
    total_reward = test_agent(env, Q)
    print(f"Total reward: {total_reward}")
    env.plot()
    env.close()
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.grid()

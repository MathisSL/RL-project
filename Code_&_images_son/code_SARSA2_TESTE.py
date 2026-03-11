import numpy as np
import pygame
import random
import matplotlib.pyplot as plt


class SimpleEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.agent_position = [0, 0]
        self.goal_position = [grid_size - 1, grid_size - 1]

        # Génération des obstacles
        self.obstacle_map = np.zeros((grid_size, grid_size))
        self.generate_obstacles()

        self.cell_size = 50
        self.screen_size = self.grid_size * self.cell_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("SARSA Navigation")
        self.clock = pygame.time.Clock()

        self.action_space = 4  # Haut, Bas, Gauche, Droite

    def generate_obstacles(self):
        self.obstacle_map[0, :3] = 1
        middle = self.grid_size // 2
        for row in [4, 5, 6]:
            self.obstacle_map[row, middle - 1:middle + 2] = 1
        for i in range(3):
            self.obstacle_map[self.grid_size - 3 + i, self.grid_size - 1] = 1

        self.obstacle_map[self.agent_position[0], self.agent_position[1]] = 0
        self.obstacle_map[self.goal_position[0], self.goal_position[1]] = 0

    def render(self):
        self.screen.fill((255, 255, 255))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = (200, 200, 200)
                if self.obstacle_map[i, j] == 1:
                    color = (150, 150, 150)
                pygame.draw.rect(
                    self.screen,
                    color,
                    (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                )
                pygame.draw.rect(
                    self.screen,
                    (0, 0, 0),
                    (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size),
                    1
                )

        goal_color = (0, 255, 0)
        pygame.draw.rect(
            self.screen,
            goal_color,
            (self.goal_position[1] * self.cell_size, self.goal_position[0] * self.cell_size, self.cell_size, self.cell_size)
        )

        agent_color = (255, 255, 0)
        pygame.draw.circle(
            self.screen,
            agent_color,
            (self.agent_position[1] * self.cell_size + self.cell_size // 2,
             self.agent_position[0] * self.cell_size + self.cell_size // 2),
            self.cell_size // 4
        )

        pygame.display.flip()
        self.clock.tick(10)

    def step(self, action):
        if action == 0:
            self.agent_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 1:
            self.agent_position[0] = min(self.grid_size - 1, self.agent_position[0] + 1)
        elif action == 2:
            self.agent_position[1] = max(0, self.agent_position[1] - 1)
        elif action == 3:
            self.agent_position[1] = min(self.grid_size - 1, self.agent_position[1] + 1)

        if self.obstacle_map[self.agent_position[0], self.agent_position[1]] == 1:
            reward = -10
            done = True
        elif self.agent_position == self.goal_position:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        return self.get_observation(), reward, done, {}

    def get_observation(self):
        return self.agent_position

    def reset(self):
        self.agent_position = [0, 0]
        return self.get_observation()

    def close(self):
        pygame.quit()


class SARSAAgent:
    def __init__(self, env, alpha=0.22, gamma=0.3, epsilon=0.85):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.random.uniform(low=-1, high=1, size=(env.grid_size, env.grid_size, env.action_space))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.env.action_space - 1)
        x, y = state
        return np.argmax(self.q_table[x, y])

    def learn(self, state, action, reward, next_state, next_action, done):
        x, y = state
        next_x, next_y = next_state
        q_target = reward + self.gamma * self.q_table[next_x, next_y, next_action] if not done else reward
        self.q_table[x, y, action] += self.alpha * (q_target - self.q_table[x, y, action])
        self.epsilon = max(0.1, self.epsilon * 0.99)


def train_sarsa_agent(env, agent, episodes=1000):
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        done = False
        total_reward = 0

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.learn(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Récompense totale = {total_reward}")

    plt.plot(total_rewards)
    plt.xlabel("Épisodes")
    plt.ylabel("Récompense totale")
    plt.title("Évolution des récompenses par épisode")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    env = SimpleEnv(grid_size=10)
    agent = SARSAAgent(env)

    train_sarsa_agent(env, agent, episodes=1000)

    done = False
    state = env.reset()
    while not done:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        env.render()

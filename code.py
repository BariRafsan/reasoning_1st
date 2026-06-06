import pygame
import numpy as np
from collections import defaultdict
import random

pygame.init()

# ==========================================================
# SETTINGS
# ==========================================================

WIDTH = 1000
HEIGHT = 700

ROWS = 7
COLS = 10

CELL_SIZE = 70

MARGIN_X = 100
MARGIN_Y = 100

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 120, 255)
GREEN = (50, 200, 50)
RED = (255, 70, 70)
GRAY = (220, 220, 220)
YELLOW = (255, 220, 0)

FONT = pygame.font.SysFont("arial", 24)

screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Windy Gridworld - SARSA")


# ==========================================================
# ENVIRONMENT
# ==========================================================


class WindyGridworld:

    def __init__(self, kings=False, stochastic=False):

        self.width = 10
        self.height = 7

        self.start = (0, 3)
        self.goal = (7, 3)

        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        self.kings = kings
        self.stochastic = stochastic

        if kings:

            self.actions = [
                (1, 0),
                (0, -1),
                (-1, 0),
                (0, 1),
                (1, -1),
                (-1, -1),
                (1, 1),
                (-1, 1),
            ]

        else:

            self.actions = [(1, 0), (0, -1), (-1, 0), (0, 1)]

        self.num_actions = len(self.actions)

        self.reset()

    def reset(self):

        self.state = self.start

        return self.state

    def step(self, action):

        x, y = self.state

        dx, dy = self.actions[action]

        x += dx
        y += dy

        wind = self.wind[x] if 0 <= x < self.width else 0

        if self.stochastic:

            wind += random.choice([-1, 0, 1])

        y -= wind

        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))

        self.state = (x, y)

        reward = -1

        done = self.state == self.goal

        return self.state, reward, done


# ==========================================================
# SARSA AGENT
# ==========================================================


class SARSAAgent:

    def __init__(self, env, alpha=0.5, epsilon=0.1, gamma=1.0):

        self.env = env

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.Q = defaultdict(lambda: np.zeros(env.num_actions))

    def choose_action(self, state):

        if np.random.rand() < self.epsilon:

            return np.random.randint(self.env.num_actions)

        return np.argmax(self.Q[state])

    def train(self, episodes=3000):

        for ep in range(episodes):

            state = self.env.reset()

            action = self.choose_action(state)

            done = False

            steps = 0

            while not done and steps < 500:

                next_state, reward, done = self.env.step(action)

                next_action = self.choose_action(next_state)

                td_target = reward

                if not done:

                    td_target += self.gamma * self.Q[next_state][next_action]

                td_error = td_target - self.Q[state][action]

                self.Q[state][action] += self.alpha * td_error

                state = next_state
                action = next_action

                steps += 1

    def get_policy(self):

        policy = {}

        for state in self.Q:

            policy[state] = np.argmax(self.Q[state])

        return policy


# ==========================================================
# DRAW GRID
# ==========================================================

ARROWS = {0: "→", 1: "↑", 2: "←", 3: "↓", 4: "↗", 5: "↖", 6: "↘", 7: "↙"}


def draw_grid(env, policy, agent_pos):

    screen.fill(WHITE)

    for row in range(ROWS):

        for col in range(COLS):

            x = MARGIN_X + col * CELL_SIZE
            y = MARGIN_Y + row * CELL_SIZE

            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            pygame.draw.rect(screen, GRAY, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)

            # WIND
            wind_text = FONT.render(str(env.wind[col]), True, RED)

            screen.blit(wind_text, (x + 50, y + 5))

            # START
            if (col, row) == env.start:

                pygame.draw.rect(screen, YELLOW, rect)

                txt = FONT.render("S", True, BLACK)

                screen.blit(txt, (x + 25, y + 20))

            # GOAL
            elif (col, row) == env.goal:

                pygame.draw.rect(screen, GREEN, rect)

                txt = FONT.render("G", True, BLACK)

                screen.blit(txt, (x + 25, y + 20))

            # POLICY
            elif (col, row) in policy:

                arrow = ARROWS[policy[(col, row)]]

                txt = FONT.render(arrow, True, BLACK)

                screen.blit(txt, (x + 22, y + 18))

    # AGENT
    ax = MARGIN_X + agent_pos[0] * CELL_SIZE + CELL_SIZE // 2
    ay = MARGIN_Y + agent_pos[1] * CELL_SIZE + CELL_SIZE // 2

    pygame.draw.circle(screen, BLUE, (ax, ay), 20)

    # TITLE
    title = FONT.render("Windy Gridworld - SARSA", True, BLACK)

    screen.blit(title, (350, 30))

    pygame.display.update()


# ==========================================================
# MAIN
# ==========================================================


def main():

    env = WindyGridworld(kings=True, stochastic=True)

    agent = SARSAAgent(env)

    print("Training...")

    agent.train(episodes=5000)

    print("Training Finished")

    policy = agent.get_policy()

    state = env.reset()

    clock = pygame.time.Clock()

    running = True

    while running:

        clock.tick(3)

        draw_grid(env, policy, state)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False

        # FOLLOW POLICY

        if state in policy:

            action = policy[state]

            next_state, reward, done = env.step(action)

            state = next_state

            if done:

                pygame.time.delay(1000)

                state = env.reset()

    pygame.quit()


if __name__ == "__main__":

    main()
